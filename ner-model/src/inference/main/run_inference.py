import torch
import argparse
from typing import Tuple, List
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    BertTokenizer,
    BertForTokenClassification,
)
from src.inference.utils import manual_inference_pipeline
from src.data_utils.utils import read_yaml_config
from src.inference.base import BaseModelInference

class InferenceNerModel(BaseModelInference):
    """Conducts inference on pre-trained Named Entity Recognition model."""

    def __init__(self, logging_file_path, args):
        super().__init__(logging_file_path)
        self.device = args.device
        self.tokenizer_save_path = args.tokenizer_save_path
        self.model_save_path = args.model_save_path
        self.max_length = args.max_length
        self.sentence = args.sentence

    def load_model_artifacts(self) -> Tuple[BertTokenizer, BertForTokenClassification]:
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_save_path)
        model = AutoModelForTokenClassification.from_pretrained(self.model_save_path)
        return tokenizer, model

    def score_data(
        self, tokenizer: BertTokenizer, model: BertForTokenClassification
    ) -> List:

        output = manual_inference_pipeline(
            tokenizer=tokenizer,
            model=model,
            sentence=self.sentence,
            max_length=self.max_length,
            device=self.device,
        )
        return output

    def inference_logic(self) -> List:
        self.logger.info("Starting inference logic")
        self.logger.info("Loading model artifacts")
        tokenizer, model = self.load_model_artifacts()

        self.logger.info("Score data")
        output = self.score_data(tokenizer, model)

        self.logger.info("Finished inference logic")

        return output


if __name__ == "__main__":
    sentence = "England has a capital called London. On wednesday, the Prime Minister will give a presentation"
    config = read_yaml_config(path="src/static.yaml")
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        "-d",
        default="mps" if torch.backends.mps.is_available() else "cpu",
        action="store",
        help="Compute device to use with PyTorch model",
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
        "--max_length",
        "-ml",
        default=config["max_length"],
        action="store",
        help="Max length of token sequences",
    )
    parser.add_argument(
        "--label2id_path",
        "-l2ip",
        default=config["label2id_path"],
        action="store",
        help="label2id_path local path",
    )
    parser.add_argument(
        "--sentence",
        "-s",
        default=sentence,
        action="store",
        help="Sentence to be scored",
    )
    args = parser.parse_args()
    print(
        InferenceNerModel(
            logging_file_path=config["inference_model_log_path"], args=args
        ).inference_logic()
    )
