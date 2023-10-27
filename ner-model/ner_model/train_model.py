import yaml
import json
import logging
import pandas as pd
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForTokenClassification
from typing import Any, Tuple, List
from seqeval.metrics import classification_report


def tokenize_and_preserve_labels(
    sentence: str, text_labels: str, tokenizer: Any
) -> Tuple[List[str], List[str]]:
    """
    Word piece tokenization makes it difficult to match word labels
    back up with individual word pieces. This function tokenizes each
    word one at a time so that it is easier to preserve the correct
    label for each subword. It is, of course, a bit slower in processing
    time, but it will help our model achieve higher accuracy.
    """

    tokenized_sentence = []
    labels = []

    sentence = sentence.strip()

    for word, label in zip(sentence.split(), text_labels.split(",")):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels


class dataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: Any, max_len: int, label2id: dict):
        self.len = len(df)
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = label2id

    def __getitem__(self, index: int) -> dict:
        # step 1: tokenize (and adapt corresponding labels)
        sentence = self.df.sentence[index]
        word_labels = self.df.word_labels[index]
        tokenized_sentence, labels = tokenize_and_preserve_labels(
            sentence, word_labels, self.tokenizer
        )

        # step 2: add special tokens (and corresponding labels)
        tokenized_sentence = (
            ["[CLS]"] + tokenized_sentence + ["[SEP]"]
        )  # add special tokens
        labels.insert(0, "O")  # add outside label for [CLS] token
        labels.insert(-1, "O")  # add outside label for [SEP] token

        # step 3: truncating/padding
        maxlen = self.max_len

        if len(tokenized_sentence) > maxlen:
            # truncate
            tokenized_sentence = tokenized_sentence[:maxlen]
            labels = labels[:maxlen]
        else:
            # pad
            tokenized_sentence = tokenized_sentence + [
                "[PAD]" for _ in range(maxlen - len(tokenized_sentence))
            ]
            labels = labels + ["O" for _ in range(maxlen - len(labels))]

        # step 4: obtain the attention mask
        attn_mask = [1 if tok != "[PAD]" else 0 for tok in tokenized_sentence]

        # step 5: convert tokens to input ids
        ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)

        label_ids = [self.label2id[label] for label in labels]
        # the following line is deprecated
        # label_ids = [label if label != 0 else -100 for label in label_ids]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(attn_mask, dtype=torch.long),
            "targets": torch.tensor(label_ids, dtype=torch.long),
        }

    def __len__(self) -> int:
        return self.len


def create_train_test(
    df: pd.DataFrame,
    train_size: float,
    tokenizer: Any,
    max_length: int,
    seed: int,
    label2id: dict,
    df_sample_size: float = 1.0,
    verbose: bool = True,
) -> Tuple[Dataset, Dataset]:
    df = df.sample(frac=df_sample_size, random_state=seed)
    train_dataset = df.sample(frac=train_size, random_state=seed)
    test_dataset = df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    if verbose:
        print("FULL Dataset: {}".format(df.shape))
        print("TRAIN Dataset: {}".format(train_dataset.shape))
        print("TEST Dataset: {}".format(test_dataset.shape))

    training_set = dataset(train_dataset, tokenizer, max_length, label2id)
    testing_set = dataset(test_dataset, tokenizer, max_length, label2id)

    return training_set, testing_set


def train(
    training_loader: torch.utils.data.DataLoader,
    model: Any,
    epochs: int,
    max_grad_norm: int,
    optimizer: torch.optim.Adam,
    device: str,
    verbose: bool = True,
) -> Any:
    for epoch in range(epochs):
        print(f"Training epoch: {epoch + 1}")

        tr_loss, tr_accuracy = 0, 0
        nb_tr_examples, nb_tr_steps = 0, 0
        tr_preds, tr_labels = [], []
        model.train()

        for idx, batch in enumerate(training_loader):

            ids = batch["ids"].to(device, dtype=torch.long)
            mask = batch["mask"].to(device, dtype=torch.long)
            targets = batch["targets"].to(device, dtype=torch.long)

            outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
            loss, tr_logits = outputs.loss, outputs.logits
            tr_loss += loss.item()

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            if idx % 100 == 0:
                loss_step = tr_loss / nb_tr_steps
                if verbose:
                    print(f"Training loss per 100 training steps: {loss_step}")

            # compute training accuracy
            flattened_targets = targets.view(-1)  # shape (batch_size * seq_len,)
            active_logits = tr_logits.view(
                -1, model.num_labels
            )  # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(
                active_logits, axis=1
            )  # shape (batch_size * seq_len,)
            active_accuracy = mask.view(-1) == 1
            targets = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)

            tr_preds.extend(predictions)
            tr_labels.extend(targets)

            tmp_tr_accuracy = accuracy_score(
                targets.cpu().numpy(), predictions.cpu().numpy()
            )
            tr_accuracy += tmp_tr_accuracy

            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=max_grad_norm
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss = tr_loss / nb_tr_steps
        tr_accuracy = tr_accuracy / nb_tr_steps
        print(f"Training loss epoch: {epoch_loss}")
        print(f"Training accuracy epoch: {tr_accuracy}")

    return model


def valid(
    model: Any,
    testing_loader: torch.utils.data.DataLoader,
    id2label: dict,
    device: str,
    verbose: bool = True,
) -> Tuple[List[int], List[int]]:
    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels = [], []

    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):

            ids = batch["ids"].to(device, dtype=torch.long)
            mask = batch["mask"].to(device, dtype=torch.long)
            targets = batch["targets"].to(device, dtype=torch.long)

            outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
            loss, eval_logits = outputs.loss, outputs.logits

            eval_loss += loss.item()

            nb_eval_steps += 1
            nb_eval_examples += targets.size(0)

            if idx % 100 == 0:
                loss_step = eval_loss / nb_eval_steps
                if verbose:
                    print(f"Validation loss per 100 evaluation steps: {loss_step}")

            # compute evaluation accuracy
            flattened_targets = targets.view(-1)  # shape (batch_size * seq_len,)
            active_logits = eval_logits.view(
                -1, model.num_labels
            )  # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(
                active_logits, axis=1
            )  # shape (batch_size * seq_len,)
            # now, use mask to determine where we should compare predictions
            # # with targets (includes [CLS] and [SEP] token predictions)
            active_accuracy = (
                mask.view(-1) == 1
            )  # active accuracy is also of shape (batch_size * seq_len,)
            targets = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)

            eval_labels.extend(targets)
            eval_preds.extend(predictions)

            tmp_eval_accuracy = accuracy_score(
                targets.cpu().numpy(), predictions.cpu().numpy()
            )
            eval_accuracy += tmp_eval_accuracy

    labels = [id2label[id.item()] for id in eval_labels]
    predictions = [id2label[id.item()] for id in eval_preds]

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps
    print(f"Validation Loss: {eval_loss}")
    print(f"Validation Accuracy: {eval_accuracy}")

    return labels, predictions


def main():
    logging.basicConfig(
        filename="ner_model.log",
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("Starting main function")

    try:
        with open("ner_model/static.yaml", "r") as f:
            config = yaml.safe_load(f.read())
    except Exception as e:
        logging.error(f"Error reading static.yaml: {e}")
        return

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    max_length = config["max_length"]
    train_batch_size = config["train_batch_size"]
    valid_batch_size = config["valid_batch_size"]
    epochs = config["epochs"]
    learning_rate = config["learning_rate"]
    max_gradient_norm = config["max_gradient_norm"]
    df_sample_size = config["df_sample_size"]
    train_size = config["train_size"]
    seed = config["seed"]
    tokenizer = BertTokenizer.from_pretrained(config["hugging_face_model_path"])

    df = pd.read_csv(config["preprocess_data_path"])
    with open(config["label2id_path"], "r") as f:
        label2id = json.load(f)
    with open(config["id2label_path"], "r") as f:
        id2label = json.load(f)
        id2label = {int(k): v for k, v in id2label.items()}

    training_set, testing_set = create_train_test(
        df=df,
        train_size=train_size,
        tokenizer=tokenizer,
        max_length=max_length,
        seed=seed,
        label2id=label2id,
        df_sample_size=df_sample_size,
        verbose=True,
    )
    train_params = {"batch_size": train_batch_size, "shuffle": True, "num_workers": 0}
    test_params = {"batch_size": valid_batch_size, "shuffle": True, "num_workers": 0}
    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    logging.info("Loaded data")

    model = BertForTokenClassification.from_pretrained(
        config["hugging_face_model_path"],
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )
    model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    trained_model = train(
        training_loader,
        model,
        epochs,
        max_gradient_norm,
        optimizer,
        device,
        verbose=True,
    )

    logging.info("Trained model")

    labels, predictions = valid(model, testing_loader, id2label, device, verbose=True)
    report = classification_report([labels], [predictions])
    print(report)

    try:
        with open(config["classification_report_path"], "w") as f:
            f.write(report)
    except Exception as e:
        logging.error(f"Error writing to classification report: {e}")

    tokenizer.save_pretrained(config["tokenizer_path"])
    trained_model.save_pretrained(config["model_path"])

    logging.info("Saved model and tokenizer")


if __name__ == "__main__":
    main()

    # model inference
