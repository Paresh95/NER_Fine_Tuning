import torch
from transformers import BertTokenizer, BertForTokenClassification
from typing import List
from transformers import pipeline
import re
from src.data_utils.utils import read_yaml_config, load_id2label


def auto_inference_pipeline(
    tokenizer: BertTokenizer,
    model: BertForTokenClassification,
    sentence: str,
    device: int = -1,
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
    tokenizer: BertTokenizer,
    model: BertForTokenClassification,
    sentence: str,
    max_length: int,
    device: str,
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
    config = read_yaml_config(path="src/static.yaml")
    id2label = load_id2label(config["id2label_path"])
    token_predictions = [id2label[i] for i in flattened_predictions.cpu().numpy()]
    wp_preds = list(zip(tokens, token_predictions))

    word_level_predictions = []
    for pair in wp_preds:
        if (pair[0].startswith(" ##")) or (pair[0] in ["[CLS]", "[SEP]", "[PAD]"]):
            continue
        else:
            word_level_predictions.append(pair[1])

    words = re.findall(r"\b\w+\b", sentence)
    result = list(zip(words, word_level_predictions))
    return result