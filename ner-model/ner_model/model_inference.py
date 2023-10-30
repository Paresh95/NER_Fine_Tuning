import logging
import yaml
import torch
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import Any, List
from transformers import pipeline
import re


def auto_inference_pipeline(
    tokenizer: Any, model: Any, sentence: str, device: int = -1
) -> dict:
    """Uses CPU by default. Change device to 0 to use CUDA GPU."""
    pipe = pipeline(
        task="token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=device,
    )
    return pipe(sentence)


def manual_inference_pipeline(
    tokenizer: Any, model: Any, sentence: str, max_length: int, device: str
) -> List:
    inputs = tokenizer(
        sentence,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    model.to(device)
    ids = inputs["input_ids"].to(device)
    mask = inputs["attention_mask"].to(device)
    outputs = model(ids, mask)
    logits = outputs[0]
    active_logits = logits.view(-1, model.num_labels)
    flattened_predictions = torch.argmax(active_logits, axis=1)
    tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
    token_predictions = [id2label[i] for i in flattened_predictions.cpu().numpy()]
    wp_preds = list(zip(tokens, token_predictions))

    word_level_predictions = []
    for pair in wp_preds:
        if (pair[0].startswith(" ##")) or (pair[0] in ["[CLS]", "[SEP]", "[PAD]"]):
            continue
        else:
            word_level_predictions.append(pair[1])

    words = re.findall(r"\b\w+\b", sentence)
    return list(zip(words, word_level_predictions))


if __name__ == "__main__":
    logging.basicConfig(
        filename="logs/ner_model_inference.log",
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("Starting main")

    try:
        with open("ner_model/static.yaml", "r") as f:
            config = yaml.safe_load(f.read())
    except yaml.YAMLError as e:
        logging.error(f"Error parsing static.yaml: {e}")
    except Exception as e:
        logging.error(f"Unexpected error reading static.yaml: {e}")

    logging.info("Importing key parameters, model and tokenizer")

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"])
    model = AutoModelForTokenClassification.from_pretrained(config["model_path"])
    max_length = config["max_length"]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    with open(config["id2label_path"], "r") as f:
        id2label = json.load(f)
        id2label = {int(k): v for k, v in id2label.items()}

    logging.info("Scoring data")
    sentence = "England has a capital called London. On wednesday, the Prime Minister will give a presentation"
    output = manual_inference_pipeline(
        tokenizer=tokenizer,
        model=model,
        sentence=sentence,
        max_length=max_length,
        device=device,
    )
    print(output)
