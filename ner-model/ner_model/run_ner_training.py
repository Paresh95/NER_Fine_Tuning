import logging
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForTokenClassification
from seqeval.metrics import classification_report
from ner_model.data_utils.utils import (
    create_train_test,
    read_yaml_config,
    load_df_and_label_dicts,
)
from ner_model.model_utils.training import train
from ner_model.model_utils.validation import valid
from abc import ABC, abstractmethod
from typing import Tuple, Dict, List


class BaseModelTraining(ABC):
    def __init__(self):
        logging.basicConfig(
            filename="logs/ner_train_model.log",
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging
        self.config = read_yaml_config(path="ner_model/static.yaml")

    @abstractmethod
    def training_logic(self) -> None:
        raise NotImplementedError


class TrainNerModel(BaseModelTraining):
    """Trains Named Entity Recognition model and saves model and tokeniser to local."""

    def __init__(self):
        super().__init__()
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.max_length = self.config["max_length"]
        self.train_batch_size = self.config["train_batch_size"]
        self.valid_batch_size = self.config["valid_batch_size"]
        self.epochs = self.config["epochs"]
        self.learning_rate = self.config["learning_rate"]
        self.max_gradient_norm = self.config["max_gradient_norm"]
        self.df_sample_size = self.config["df_sample_size"]
        self.train_size = self.config["train_size"]
        self.seed = self.config["seed"]
        self.hugging_face_model_path = self.config["hugging_face_model_path"]
        self.model_save_path = self.config["model_save_path"]
        self.tokenizer_save_path = self.config["tokenizer_save_path"]

    def get_data(self) -> Tuple[pd.DataFrame, Dict, Dict]:
        df, label2id, id2label = load_df_and_label_dicts(self.config)
        return df, label2id, id2label

    def split_data(self, df: pd.DataFrame, label2id: Dict) -> Tuple[Dataset, Dataset]:
        self.tokenizer = BertTokenizer.from_pretrained(self.hugging_face_model_path)
        training_set, testing_set = create_train_test(
            df=df,
            train_size=self.train_size,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            seed=self.seed,
            label2id=label2id,
            df_sample_size=self.df_sample_size,
            verbose=True,
        )
        return training_set, testing_set

    def create_dataloaders(
        self, training_set: Dataset, testing_set: Dataset
    ) -> Tuple[DataLoader, DataLoader]:
        train_params = {
            "batch_size": self.train_batch_size,
            "shuffle": True,
            "num_workers": 0,
        }
        test_params = {
            "batch_size": self.valid_batch_size,
            "shuffle": True,
            "num_workers": 0,
        }
        training_loader = DataLoader(training_set, **train_params)
        testing_loader = DataLoader(testing_set, **test_params)
        return training_loader, testing_loader

    def train_model(
        self, label2id: Dict, id2label: Dict, training_loader: DataLoader
    ) -> BertForTokenClassification:
        model = BertForTokenClassification.from_pretrained(
            self.hugging_face_model_path,
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
        )
        model.to(self.device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=self.learning_rate)
        self.trained_model = train(
            training_loader,
            model,
            self.epochs,
            self.max_gradient_norm,
            optimizer,
            self.device,
            verbose=True,
        )
        return self.trained_model

    def validate_model(
        self,
        trained_model: BertForTokenClassification,
        testing_loader: DataLoader,
        id2label: Dict,
    ) -> Tuple[List[int], List[int]]:
        labels, predictions = valid(
            trained_model, testing_loader, id2label, self.device, verbose=True
        )
        return labels, predictions

    def get_results(
        self, labels: List[int], predictions: List[int], verbose: bool = True
    ):
        report = classification_report([labels], [predictions])
        if verbose:
            print(report)

        try:
            with open(self.config["classification_report_path"], "w") as f:
                f.write(report)
        except Exception as e:
            self.logger.error(f"Error writing to classification report: {e}")

    def save_model_artifacts(self) -> None:
        self.tokenizer.save_pretrained(self.model_save_path)
        self.trained_model.save_pretrained(self.tokenizer_save_path)
        return None

    def training_logic(self):
        self.logger.info("Get data")
        df, label2id, id2label = self.get_data()

        self.logger.info("Split data")
        training_set, testing_set = self.split_data(df, label2id)

        self.logger.info("Create dataloaders")
        training_loader, testing_loader = self.create_dataloaders(
            training_set, testing_set
        )

        self.logger.info("Train model")
        trained_model = self.train_model(label2id, id2label, training_loader)

        self.logger.info("Validate model")
        labels, predictions = self.validate_model(
            trained_model, testing_loader, id2label
        )

        self.logger.info("Get results")
        self.get_results(labels, predictions)

        self.logger.info("Save model artifacts")
        self.save_model_artifacts()


if __name__ == "__main__":
    TrainNerModel().training_logic()


# how to provide options for yaml or arg parser
# create helpers to load yaml and data files - take out of other files too
# run poetry shell to make sure this works
