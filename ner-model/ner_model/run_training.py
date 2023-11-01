import argparse
import logging
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForTokenClassification
from seqeval.metrics import classification_report
from ner_model.data_utils.utils import (
    create_train_test,
    read_yaml_config,
    load_id2label,
    load_label2id,
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

    @abstractmethod
    def training_logic(self) -> None:
        raise NotImplementedError


class TrainNerModel(BaseModelTraining):
    """Trains Named Entity Recognition model and saves model and tokeniser to local."""

    def __init__(self, args):
        super().__init__()
        self.device = args.device
        self.max_length = args.max_length
        self.train_batch_size = args.train_batch_size
        self.valid_batch_size = args.valid_batch_size
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.max_gradient_norm = args.max_gradient_norm
        self.df_sample_size = args.df_sample_size
        self.train_size = args.train_size
        self.seed = args.seed
        self.hugging_face_model_path = args.hugging_face_model_path
        self.model_save_path = args.model_save_path
        self.tokenizer_save_path = args.tokenizer_save_path
        self.data_path = args.data_path
        self.label2id_path = args.label2id_path
        self.id2label_path = args.id2label_path

    def get_data(self) -> Tuple[pd.DataFrame, Dict, Dict]:
        df = pd.read_csv(self.data_path)
        label2id = load_label2id(self.label2id_path)
        id2label = load_id2label(self.id2label_path)
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
        self.logger.info("Starting training logic")
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

        self.logger.info("Finished training logic")


if __name__ == "__main__":
    config = read_yaml_config(path="ner_model/static.yaml")
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        "-d",
        default="mps" if torch.backends.mps.is_available() else "cpu",
        action="store",
        help="Compute device to use with PyTorch model",
    )
    parser.add_argument(
        "--max_length",
        "-ml",
        default=config["max_length"],
        action="store",
        help="Max length of token sequences",
    )
    parser.add_argument(
        "--train_batch_size",
        "-tbs",
        default=config["train_batch_size"],
        action="store",
        help="Training data batch size",
    )
    parser.add_argument(
        "--valid_batch_size",
        "-vbs",
        default=config["valid_batch_size"],
        action="store",
        help="Validation data batch size",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        default=config["epochs"],
        action="store",
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        "-lr",
        default=config["learning_rate"],
        action="store",
        help="Learning rate for optimizer",
    )
    parser.add_argument(
        "--max_gradient_norm",
        "-mgn",
        default=config["max_gradient_norm"],
        action="store",
        help="Maxium gradient normalisation parameter for gradient clipping",
    )
    parser.add_argument(
        "--df_sample_size",
        "-dss",
        default=config["df_sample_size"],
        action="store",
        help="Percentage of df to use for training model",
    )
    parser.add_argument(
        "--train_size",
        "-ts",
        default=config["train_size"],
        action="store",
        help="Percentage data to use for training vs test data",
    )
    parser.add_argument(
        "--seed",
        "-s",
        default=config["seed"],
        action="store",
        help="Seed value",
    )
    parser.add_argument(
        "--hugging_face_model_path",
        "-hfmp",
        default=config["hugging_face_model_path"],
        action="store",
        help="Path to hugging face model on hugging face hub",
    )
    parser.add_argument(
        "--model_save_path",
        "-msp",
        default=config["model_save_path"],
        action="store",
        help="Path to save model locally",
    )
    parser.add_argument(
        "--tokenizer_save_path",
        "-tsp",
        default=config["tokenizer_save_path"],
        action="store",
        help="Path to save tokenizer locally",
    )
    parser.add_argument(
        "--data_path",
        "-dp",
        default=config["preprocess_data_path"],
        action="store",
        help="Path to local data",
    )
    parser.add_argument(
        "--label2id_path",
        "-l2idp",
        default=config["label2id_path"],
        action="store",
        help="Path to local label2id dictionary",
    )
    parser.add_argument(
        "--id2label_path",
        "-i2ldp",
        default=config["id2label_path"],
        action="store",
        help="Path to local id2label dictionary",
    )

    args = parser.parse_args()

    TrainNerModel(args=args).training_logic()
