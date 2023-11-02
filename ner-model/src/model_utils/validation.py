from sklearn.metrics import accuracy_score
import torch
from transformers import BertForTokenClassification
from typing import Tuple, List


def valid(
    model: BertForTokenClassification,
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

            # compute validation accuracy
            flattened_targets = targets.view(-1)  # shape (batch_size * seq_len,)
            active_logits = eval_logits.view(
                -1, model.num_labels
            )  # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(
                active_logits, axis=1
            )  # shape (batch_size * seq_len,)
            # use mask to determine where we should compare predictions
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
