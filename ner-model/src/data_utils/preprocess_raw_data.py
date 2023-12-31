import pandas as pd
from typing import Dict, Tuple
from src.data_utils.utils import read_yaml_config, write_json_to_disk


def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int], Dict[int, str]]:
    entities_to_remove = ["B-art", "I-art", "B-eve", "I-eve", "B-nat", "I-nat"]
    df = df[~df.Tag.isin(entities_to_remove)]
    df = df.fillna(method="ffill")
    df["sentence"] = (
        df[["Sentence #", "Word", "Tag"]]
        .groupby(["Sentence #"])["Word"]
        .transform(lambda x: " ".join(x))
    )
    df["word_labels"] = (
        df[["Sentence #", "Word", "Tag"]]
        .groupby(["Sentence #"])["Tag"]
        .transform(lambda x: ",".join(x))
    )
    label2id = {k: v for v, k in enumerate(df.Tag.unique())}
    id2label = {int(v): k for v, k in enumerate(df.Tag.unique())}
    df = df[["sentence", "word_labels"]].drop_duplicates().reset_index(drop=True)
    return df, label2id, id2label


if __name__ == "__main__":
    config = read_yaml_config(path="src/static.yaml")
    df = pd.read_csv(config["raw_data_path"], encoding="unicode_escape")
    print(df.shape)
    df, label2id, id2label = preprocess(df)
    df.to_csv(config["preprocess_data_path"])
    write_json_to_disk(label2id, config["label2id_path"])
    write_json_to_disk(id2label, config["id2label_path"])
