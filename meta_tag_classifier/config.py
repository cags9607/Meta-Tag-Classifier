from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class Config:
    raw_data: str
    clean_data: str
    artifacts_dir: str
    text_column: str
    label_column: str
    embedding: dict
    model: dict
    train: dict

    @staticmethod
    def load(path):
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        return Config(**d)
