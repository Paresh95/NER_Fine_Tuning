import yaml
import json
import pandas as pd
from typing import Tuple, Dict
from ner_model.data_utils.build_dataloader import dataset
from torch.utils.data import Dataset
from transformers import BertTokenizer


def read_yaml_config(path: str) -> Dict:
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f.read())
    except yaml.YAMLError as e:
        print(f"Error parsing static.yaml: {e}")
    except Exception as e:
        print(f"Unexpected error reading static.yaml: {e}")
    return {}


def write_json_to_disk(json_file: Dict, path: str) -> None:
    with open(path, "w") as fp:
        json.dump(json_file, fp)
    return None


def load_id2label(path: str) -> Dict:
    with open(path, "r") as f:
        id2label = json.load(f)
        id2label = {int(k): v for k, v in id2label.items()}
        return id2label


def load_label2id(path: str) -> Dict:
    with open(path, "r") as f:
        label2id = json.load(f)
    return label2id


def load_df_and_label_dicts(config: Dict) -> Tuple[pd.DataFrame, Dict, Dict]:
    df = pd.read_csv(config["preprocess_data_path"])
    label2id = load_label2id(config["label2id_path"])
    id2label = load_id2label(config["id2label_path"])
    return df, label2id, id2label


def create_train_test(
    df: pd.DataFrame,
    train_size: float,
    tokenizer: BertTokenizer,
    max_length: int,
    seed: int,
    label2id: dict,
    df_sample_size: float = 1.0,
    verbose: bool = True,
) -> Tuple[Dataset, Dataset]:
    df = df.sample(frac=df_sample_size, random_state=seed)
    train_dataset = df.sample(frac=train_size, random_state=seed)
    test_dataset = df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    if verbose:
        print("FULL Dataset: {}".format(df.shape))
        print("TRAIN Dataset: {}".format(train_dataset.shape))
        print("TEST Dataset: {}".format(test_dataset.shape))

    training_set = dataset(train_dataset, tokenizer, max_length, label2id)
    testing_set = dataset(test_dataset, tokenizer, max_length, label2id)

    return training_set, testing_set
