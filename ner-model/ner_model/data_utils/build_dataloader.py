import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizer
from typing import List, Tuple


class dataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, tokenizer: BertTokenizer, max_len: int, label2id: dict
    ):
        self.len = len(df)
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = label2id

    def _tokenize_and_preserve_labels(
        self, sentence: str, text_labels: str, tokenizer: BertTokenizer
    ) -> Tuple[List[str], List[str]]:
        """
        Word piece tokenization makes it difficult to match word labels
        back up with individual word pieces. This function tokenizes each
        word one at a time so that it is easier to preserve the correct
        label for each subword. It is, of course, a bit slower in processing
        time, but it will help our model achieve higher accuracy.

        Steps:
        - Tokenize the word and count # of subwords the word is broken into
        - Add the tokenized word to the final tokenized word list
        - Add the same label to the new list of labels `n_subwords` times
        """

        tokenized_sentence = []
        labels = []

        sentence = sentence.strip()

        for word, label in zip(sentence.split(), text_labels.split(",")):
            tokenized_word = tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)
            tokenized_sentence.extend(tokenized_word)
            labels.extend([label] * n_subwords)

        return tokenized_sentence, labels

    def __getitem__(self, index: int) -> dict:
        # step 1: tokenize (and adapt corresponding labels)
        sentence = self.df.sentence[index]
        word_labels = self.df.word_labels[index]
        tokenized_sentence, labels = self._tokenize_and_preserve_labels(
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
