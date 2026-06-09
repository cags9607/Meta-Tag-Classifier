__all__ = [
    "__version__",
    "clean_dataframe",
    "load_predictor",
    "Predictor",
    "template_predictor",
    "svm_predictor",
]

__version__ = "0.2.0"

from .api import (
    clean_dataframe,
    load_predictor,
    Predictor,
    template_predictor,
    svm_predictor,
)
