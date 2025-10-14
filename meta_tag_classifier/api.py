# src/meta_tag_classifier/api.py
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from importlib.resources import as_file, files
from joblib import parallel_backend
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .data.clean import clean_metas
from .utils import ensure_dir  # simple mkdir helper


# -------------------------- Embedding cache & utils --------------------------

_model_cache: Dict[str, SentenceTransformer] = {}


def _get_embedder(model_name: str) -> SentenceTransformer:
    """Get (and cache) a SentenceTransformer embedder."""
    m = _model_cache.get(model_name)
    if m is None:
        m = SentenceTransformer(model_name)
        _model_cache[model_name] = m
    return m


def embed_texts(
    texts: Iterable[str],
    model_name: str = "distiluse-base-multilingual-cased-v2",
    show_progress_bar: bool = True,
) -> np.ndarray:
    """Encode texts into embeddings using SentenceTransformers."""
    model = _get_embedder(model_name)
    embs = model.encode(
        list(texts),
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
    )
    return embs


# ------------------------------ Cleaning wrapper -----------------------------

def clean_dataframe(
    df: pd.DataFrame,
    nav_thr: float = 2.0,
    prefer_title: bool = True,
    short_title_word_threshold: int = 8,
) -> pd.DataFrame:
    """
    Run the repository's meta cleaning + selection.
    Adds `selected_text` and diagnostics (nav/quality scores) to the returned DataFrame.
    Accepts raw meta fields (title / og:title / description / og:description / twitter:...).
    """
    return clean_metas(
        df.copy(),
        nav_thr=nav_thr,
        prefer_title=prefer_title,
        short_title_word_threshold=short_title_word_threshold,
    )


# ---------------------------- Training (optional) ----------------------------

@dataclass
class TrainResult:
    artifacts_dir: Path
    best_params: Dict[str, Any]
    report_text: str
    classes: List[Any]
    embedding_model: str
    pipeline_filename: str = "pipeline.pkl"
    pipeline_format: str = "pickle"  # "pickle" or "joblib"


def _save_pipeline(obj: Any, path: Path, fmt: str = "pickle") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "pickle" or path.suffix.lower() == ".pkl":
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        import joblib
        joblib.dump(obj, path)


def train(
    df: pd.DataFrame,
    label_column: str,
    artifacts_dir: str | Path = "models/artifacts",
    embedding_model: str = "distiluse-base-multilingual-cased-v2",
    C_grid: Optional[List[float]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    pipeline_filename: str = "pipeline.pkl",     # default to .pkl as requested
    pipeline_format: str = "pickle",             # "pickle" or "joblib"
) -> TrainResult:
    """
    Clean/select text, embed with SentenceTransformers, then train:
      StandardScaler -> PCA(0.95) -> LinearSVC (GridSearch on C, f1_macro)
    Saves pipeline + meta to `artifacts_dir`.
    """
    if C_grid is None:
        C_grid = [0.0001, 0.001, 0.01, 0.1, 1.0]

    if label_column not in df.columns:
        raise ValueError(f"Missing label column: {label_column!r}")

    df_sel = clean_dataframe(df)
    df_sel = df_sel[df_sel["selected_text"].notna() & (df_sel["selected_text"] != "")].copy()

    X_text = df_sel["selected_text"].tolist()
    y = df_sel[label_column].values

    X = embed_texts(X_text, model_name=embedding_model)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95)),
            ("linear_svc", LinearSVC(dual=False, class_weight="balanced")),
        ]
    )
    param_grid = {"linear_svc__C": C_grid}
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring="f1_macro", n_jobs=-1)

    with parallel_backend("loky", inner_max_num_threads=1):
        grid.fit(X_train, y_train)

    y_pred = grid.best_estimator_.predict(X_test)
    report = classification_report(y_test, y_pred, digits=4)

    art_dir = ensure_dir(artifacts_dir)
    # Save pipeline (default .pkl)
    _save_pipeline(grid.best_estimator_, Path(art_dir) / pipeline_filename, fmt=pipeline_format)

    # Save meta.json with embedder + file info
    meta = {
        "embedding_model": embedding_model,
        "pipeline_filename": pipeline_filename,
        "pipeline_format": pipeline_format,
        "best_params": grid.best_params_,
        "classes_": sorted(pd.unique(y).tolist()),
    }
    (Path(art_dir) / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (Path(art_dir) / "report.txt").write_text(report, encoding="utf-8")

    return TrainResult(
        artifacts_dir=Path(artifacts_dir),
        best_params=grid.best_params_,
        report_text=report,
        classes=meta["classes_"],
        embedding_model=embedding_model,
        pipeline_filename=pipeline_filename,
        pipeline_format=pipeline_format,
    )


# --------------------- Packaged-artifacts loading helpers --------------------

def _packaged_artifacts_dir() -> Path | Any:
    """
    Locate artifacts bundled inside the installed package:
      meta_tag_classifier/artifacts/{meta.json, pipeline.*}
    Returns a Traversable (PEP 302/451) or a materialized Path via as_file().
    """
    try:
        return files("meta_tag_classifier") / "artifacts"
    except Exception:
        # Fallback for odd environments
        return Path(__file__).resolve().parent / "artifacts"


def _load_pipeline_any(path: Path, fmt: Optional[str] = None) -> Any:
    """
    Load a pipeline saved as pickle (.pkl) or joblib (.joblib/.pkl),
    preferring 'fmt' when provided, otherwise infer from suffix.
    """
    suffix = path.suffix.lower()
    if fmt == "pickle" or suffix == ".pkl":
        with open(path, "rb") as f:
            return pickle.load(f)
    # Else fallback to joblib
    import joblib
    return joblib.load(path)


# --------------------------------- Predictor ---------------------------------

class Predictor:
    """
    Convenience predictor:
      - Loads packaged artifacts by default (no artifacts_dir needed).
      - Or point to an external artifacts directory if desired.
    """

    def __init__(self, artifacts_dir: str | Path | None = None):
        if artifacts_dir is None:
            # Use artifacts bundled with the package
            art = _packaged_artifacts_dir()
            with as_file(art / "meta.json") as m_path:
                meta = json.loads(Path(m_path).read_text(encoding="utf-8"))
            pipe_name = meta.get("pipeline_filename", "pipeline.pkl")
            with as_file(art / pipe_name) as p_path:
                self.pipe = _load_pipeline_any(Path(p_path), meta.get("pipeline_format"))
        else:
            # Use artifacts from a provided folder
            artifacts_dir = Path(artifacts_dir)
            meta_path = artifacts_dir / "meta.json"
            meta = (
                json.loads(meta_path.read_text(encoding="utf-8"))
                if meta_path.exists()
                else {"embedding_model": "distiluse-base-multilingual-cased-v2", "pipeline_filename": "pipeline.pkl"}
            )
            pipe_path = artifacts_dir / meta.get("pipeline_filename", "pipeline.pkl")
            self.pipe = _load_pipeline_any(pipe_path, meta.get("pipeline_format"))

        self.meta = meta or {"embedding_model": "distiluse-base-multilingual-cased-v2"}

        # Initialize embedder
        self.model = _get_embedder(self.meta.get("embedding_model", "distiluse-base-multilingual-cased-v2"))

    def predict_texts(self, texts: Iterable[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict for a list of strings.
        Returns (labels, probs) where probs are NaN for LinearSVC.
        """
        X = self.model.encode(list(texts), show_progress_bar=True, convert_to_numpy=True)
        preds = self.pipe.predict(X)
        proba = np.full(shape=(len(preds),), fill_value=np.nan, dtype=float)  # LinearSVC has no predict_proba
        return preds, proba

    def predict_dataframe(self, df: pd.DataFrame, text_col: Optional[str] = None) -> pd.DataFrame:
        """
        Predict for a DataFrame.
        If `text_col` is provided, uses it directly; otherwise auto-cleans raw meta fields to `selected_text`.
        Returns a DataFrame with `selected_text`, `predicted_label`, `predicted_proba`,
        and passes through `target_domain` if present.
        """
        if text_col is None or text_col not in df.columns:
            proc = clean_dataframe(df.copy())
            out = proc[["selected_text"]].copy()
            if "target_domain" in proc.columns:
                out.insert(0, "target_domain", proc["target_domain"])
        else:
            out = df[[text_col]].rename(columns={text_col: "selected_text"}).copy()
            if "target_domain" in df.columns:
                out.insert(0, "target_domain", df["target_domain"])

        # Mask non-empty texts
        mask = out["selected_text"].fillna("").astype(str).str.strip() != ""
        if not mask.any():
            # Nothing to predict; return NaNs with same shape
            out["predicted_label"] = np.nan
            out["predicted_proba"] = np.nan
            return out

        texts = out.loc[mask, "selected_text"].tolist()
        preds, proba = self.predict_texts(texts)

        # Initialize with NaNs, then fill only where we predicted
        out["predicted_label"] = np.nan
        out["predicted_proba"] = np.nan
        out.loc[mask, "predicted_label"] = preds
        out.loc[mask, "predicted_proba"] = proba
        return out


def load_predictor(artifacts_dir: str | Path | None = None) -> Predictor:
    """
    Factory: return a Predictor that loads packaged artifacts by default,
    or from `artifacts_dir` if provided.
    """
    return Predictor(artifacts_dir)


# ------------------------------ One-liner wrapper ----------------------------

def svm_predictor(
    raw_data,
    artifacts_dir: str | Path | None = None,
    text_col: str | None = None,
    output: str = "df",
):
    """
    One-call prediction.

    Args:
        raw_data:
            - pd.DataFrame with raw meta fields (title/og:title/description/...) or a pre-cleaned `text_col`
            - list[str] of texts
            - list[dict] (each dict forms one row, keys are column names)
        artifacts_dir: directory containing {meta.json, pipeline.(pkl|joblib)}.
                       If None (default), uses the package-bundled artifacts.
        text_col: if your DataFrame already has a pre-cleaned text column to use.
        output: "df" (default) returns a DataFrame; "labels" returns a 1D numpy array.

    Returns:
        DataFrame or numpy.ndarray depending on `output`.
    """
    pred = load_predictor(artifacts_dir)

    if isinstance(raw_data, pd.DataFrame):
        df_out = pred.predict_dataframe(raw_data.copy(), text_col=text_col)
        return df_out if output == "df" else df_out["predicted_label"].to_numpy()

    if isinstance(raw_data, list):
        if len(raw_data) == 0:
            return (
                pd.DataFrame(columns=["selected_text", "predicted_label"])
                if output == "df"
                else np.array([])
            )
        first = raw_data[0]
        if isinstance(first, str):
            labels, _ = pred.predict_texts(raw_data)
            if output == "labels":
                return labels
            return pd.DataFrame({"selected_text": raw_data, "predicted_label": labels})
        if isinstance(first, dict):
            df = pd.DataFrame(raw_data)
            df_out = pred.predict_dataframe(df, text_col=text_col)
            return df_out if output == "df" else df_out["predicted_label"].to_numpy()

    raise TypeError("raw_data must be a pandas DataFrame, list[str], or list[dict].")
