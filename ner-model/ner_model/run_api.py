import torch
import logging
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)
from ner_model.data_utils.utils import read_yaml_config
from ner_model.model_utils.inference import manual_inference_pipeline

logging.basicConfig(level=logging.INFO)
app = FastAPI()

config = read_yaml_config(path="ner_model/static.yaml")
tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_save_path"])
model = AutoModelForTokenClassification.from_pretrained(config["model_save_path"])
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
            model=model,
            sentence=item.text,
            max_length=config["max_length"],
            device=device,
        )
    except Exception as e:
        logging.error(f"Error during inference: {str(e)}")
        prediction = {}
    return {"prediction": prediction}


if __name__ == "__main__":
    uvicorn.run("ner_model.run_api:app", reload=True, port=8000, host="0.0.0.0")
