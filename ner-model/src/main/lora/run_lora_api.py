import torch
import logging
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)
from peft import PeftConfig, PeftModel
from src.data_utils.utils import read_yaml_config, load_label2id, load_id2label
from src.model_utils.inference import manual_inference_pipeline

logging.basicConfig(level=logging.INFO)
app = FastAPI()

config = read_yaml_config(path="src/static.yaml")
tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_save_path"])
label2id = load_label2id(config['label2id_path'])
id2label = load_id2label(config['id2label_path'])
lora_config = PeftConfig.from_pretrained(config['lora_model_save_path'])
base_model = AutoModelForTokenClassification.from_pretrained(
    lora_config.base_model_name_or_path, num_labels=len(id2label), id2label=id2label, label2id=label2id
)
lora_model = PeftModel.from_pretrained(base_model, config['lora_model_save_path'])
device = "mps" if torch.backends.mps.is_available() else "cpu"


class Item(BaseModel):
    text: str


@app.post("/test/")
def test():
    return {"device": device}


@app.post("/predict/")
def predict(item: Item):
    try:
        prediction = manual_inference_pipeline(
            tokenizer=tokenizer,
            model=lora_model,
            sentence=item.text,
            max_length=config["max_length"],
            device=device,
        )
    except Exception as e:
        logging.error(f"Error during inference: {str(e)}")
        prediction = {}
    return {"prediction": prediction}


if __name__ == "__main__":
    uvicorn.run("src.main.lora.run_lora_api:app", reload=True, port=8000, host="0.0.0.0")
