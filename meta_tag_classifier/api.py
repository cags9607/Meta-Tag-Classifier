
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from joblib import parallel_backend

from .data.clean import clean_metas
from .utils import ensure_dir, save_artifact, load_artifact

__all__ = [
    "clean_dataframe",
    "embed_texts",
    "train",
    "load_predictor",
    "Predictor",
]

_model_cache: Dict[str, SentenceTransformer] = {}

def _get_embedder(model_name: str) -> SentenceTransformer:
    m = _model_cache.get(model_name)
    if m is None:
        m = SentenceTransformer(model_name)
        _model_cache[model_name] = m
    return m

def embed_texts(texts: Iterable[str], model_name: str = "distiluse-base-multilingual-cased-v2") -> np.ndarray:
    model = _get_embedder(model_name)
    embs = model.encode(list(texts), show_progress_bar=True, convert_to_numpy=True)
    return embs

def clean_dataframe(df: pd.DataFrame,
                    nav_thr: float = 2.0,
                    prefer_title: bool = True,
                    short_title_word_threshold: int = 8) -> pd.DataFrame:
    return clean_metas(
        df.copy(),
        nav_thr=nav_thr,
        prefer_title=prefer_title,
        short_title_word_threshold=short_title_word_threshold,
    )

@dataclass
class TrainResult:
    artifacts_dir: Path
    best_params: Dict[str, Any]
    report_text: str
    classes: List[Any]
    embedding_model: str

def train(df: pd.DataFrame,
          label_column: str,
          artifacts_dir: str | Path = "models/artifacts",
          embedding_model: str = "distiluse-base-multilingual-cased-v2",
          C_grid: Optional[List[float]] = None,
          test_size: float = 0.2,
          random_state: int = 42) -> TrainResult:
    if C_grid is None:
        C_grid = [0.0001, 0.001, 0.01, 0.1, 1.0]

    if label_column not in df.columns:
        raise ValueError(f"Missing label column: {label_column}")

    df_sel = clean_dataframe(df)
    df_sel = df_sel[df_sel["selected_text"].notna() & (df_sel["selected_text"] != "")].copy()

    X_text = df_sel["selected_text"].tolist()
    y = df_sel[label_column].values

    X = embed_texts(X_text, embedding_model)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95)),
        ("linear_svc", LinearSVC(dual=False, class_weight="balanced"))
    ])
    param_grid = {"linear_svc__C": C_grid}
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring="f1_macro", n_jobs=-1)

    with parallel_backend("loky", inner_max_num_threads=1):
        grid.fit(X_train, y_train)

    y_pred = grid.best_estimator_.predict(X_test)
    report = classification_report(y_test, y_pred, digits=4)

    art_dir = ensure_dir(artifacts_dir)
    save_artifact(grid.best_estimator_, Path(art_dir) / "pipeline.joblib")
    meta = {
        "embedding_model": embedding_model,
        "best_params": grid.best_params_,
        "classes_": sorted(pd.unique(y).tolist()),
    }
    (Path(art_dir) / "meta.json").write_text(
        __import__("json").dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    (Path(art_dir) / "report.txt").write_text(report, encoding="utf-8")

    return TrainResult(
        artifacts_dir=Path(artifacts_dir),
        best_params=grid.best_params_,
        report_text=report,
        classes=meta["classes_"],
        embedding_model=embedding_model,
    )

class Predictor:
    def __init__(self, artifacts_dir: str | Path):
        artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir = artifacts_dir
        meta_path = artifacts_dir / "meta.json"
        if meta_path.exists():
            self.meta = __import__("json").loads(meta_path.read_text(encoding="utf-8"))
        else:
            self.meta = {"embedding_model": "distiluse-base-multilingual-cased-v2"}
        self.pipe = load_artifact(artifacts_dir / "pipeline.joblib")
        self.model = _get_embedder(self.meta.get("embedding_model", "distiluse-base-multilingual-cased-v2"))

    def predict_texts(self, texts: Iterable[str]) -> Tuple[np.ndarray, np.ndarray]:
        X = self.model.encode(list(texts), show_progress_bar=True, convert_to_numpy=True)
        preds = self.pipe.predict(X)
        proba = np.full(shape=(len(preds),), fill_value=np.nan, dtype=float)
        return preds, proba

    def predict_dataframe(self, df: pd.DataFrame, text_col: Optional[str] = None) -> pd.DataFrame:
        if text_col is None or text_col not in df.columns:
            proc = clean_dataframe(df.copy())
            proc = proc[proc["selected_text"].notna() & (proc["selected_text"] != "")].copy()
            texts = proc["selected_text"].tolist()
            out = proc[["selected_text"]].copy()
            if "target_domain" in proc.columns:
                out.insert(0, "target_domain", proc["target_domain"])
        else:
            texts = df[text_col].fillna("").tolist()
            out = df[[text_col]].rename(columns={text_col: "selected_text"}).copy()
            if "target_domain" in df.columns:
                out.insert(0, "target_domain", df["target_domain"])

        preds, proba = self.predict_texts(texts)
        out["predicted_label"] = preds
        out["predicted_proba"] = proba
        return out

def load_predictor(artifacts_dir: str | Path) -> Predictor:
    return Predictor(artifacts_dir)
