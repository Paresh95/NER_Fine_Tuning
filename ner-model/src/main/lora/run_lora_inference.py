import torch
import argparse
from typing import Tuple, List
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    BertTokenizer,
    BertForTokenClassification,
)
from src.model_utils.inference import manual_inference_pipeline
from src.data_utils.utils import read_yaml_config, load_id2label, load_label2id
from src.model_utils.base_inference import BaseModelInference
from peft import PeftConfig, PeftModel


class InferenceNerLoraModel(BaseModelInference):
    """Conducts inference on pre-trained Named Entity Recognition model."""

    def __init__(self, logging_file_path, args):
        super().__init__(logging_file_path)
        self.device = args.device
        self.tokenizer_save_path = args.tokenizer_save_path
        self.lora_model_save_path = args.lora_model_save_path
        self.max_length = args.max_length
        self.sentence = args.sentence
        self.label2id_path = args.label2id_path
        self.id2label_path = args.id2label_path

    def load_model_artifacts(self) -> Tuple[BertTokenizer, BertForTokenClassification]:
        label2id = load_label2id(self.label2id_path)
        id2label = load_id2label(self.id2label_path)
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_save_path)
        lora_config = PeftConfig.from_pretrained(self.lora_model_save_path)
        base_model = AutoModelForTokenClassification.from_pretrained(
            lora_config.base_model_name_or_path,
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
        )
        lora_model = PeftModel.from_pretrained(base_model, self.lora_model_save_path)
        return tokenizer, lora_model

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
        "--lora_model_save_path",
        "-lmsp",
        default=config["lora_model_save_path"],
        action="store",
        help="Path to save lora model locally",
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
        "--sentence",
        "-s",
        default=sentence,
        action="store",
        help="Sentence to be scored",
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
    print(
        InferenceNerLoraModel(
            logging_file_path=config["inference_model_log_path"], args=args
        ).inference_logic()
    )
