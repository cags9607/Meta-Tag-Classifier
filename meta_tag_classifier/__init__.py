# src/meta_tag_classifier/__init__.py  (or meta_tag_classifier/__init__.py if not using src/)
__all__ = [
    "__version__",
    "clean_dataframe",
    "embed_texts",
    "train",
    "load_predictor",
    "Predictor",
    "svm_predictor",
]

__version__ = "0.1.1"

from .api import (
    clean_dataframe,
    embed_texts,
    train,
    load_predictor,
    Predictor,
    svm_predictor,
)
