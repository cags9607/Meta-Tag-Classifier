# meta_tag_classifier/api.py
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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
from .data.ingest import long_to_wide_meta
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
    Run meta cleaning + selection.
    Expects df to already have title_meta / description_meta (or empty strings).
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
    pipeline_filename: str = "pipeline.pkl",     # default to .pkl
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
    _save_pipeline(grid.best_estimator_, Path(art_dir) / pipeline_filename, fmt=pipeline_format)

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
    """Locate artifacts bundled inside the installed package."""
    try:
        return files("meta_tag_classifier") / "artifacts"
    except Exception:
        return Path(__file__).resolve().parent / "artifacts"

def _load_pipeline_any(path: Path, fmt: Optional[str] = None) -> Any:
    """Load a pipeline saved as pickle (.pkl) or joblib (.joblib/.pkl)."""
    suffix = path.suffix.lower()
    if fmt == "pickle" or suffix == ".pkl":
        with open(path, "rb") as f:
            return pickle.load(f)
    import joblib
    return joblib.load(path)


# ----------------------------- softmax for margins ---------------------------

def _rowwise_softmax(m: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along axis=1."""
    z = m - m.max(axis=1, keepdims=True)
    np.exp(z, out=z)
    z_sum = z.sum(axis=1, keepdims=True)
    z /= z_sum
    return z


# -------------- Fallbacks for composing selected_text when empty -------------

_FALLBACK_FIELDS = [
    "title", "og:title", "twitter:title",
    "description", "og:description", "twitter:description",
    "meta_description"
]

def _fallback_selected_text(df: pd.DataFrame) -> pd.Series:
    def pick(row):
        for c in _FALLBACK_FIELDS:
            v = row.get(c, "")
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""
    return df.apply(pick, axis=1)

def _ensure_wide(
    df: pd.DataFrame,
    domain_col: str = "target_domain",
    name_col: str = "name",
    value_col: str = "content_latest",
) -> pd.DataFrame:
    """
    If input is long-format (has domain/name/value), pivot to wide and build
    title_meta / description_meta like your Colab. Otherwise pass through and
    ensure required columns exist for the cleaner.
    """
    if {domain_col, name_col, value_col}.issubset(df.columns):
        return long_to_wide_meta(
            df,
            domain_col=domain_col,
            name_col=name_col,
            value_col=value_col,
        )

    # Wide path: make sure the cleaner has what it needs
    out = df.copy()
    if "target_domain" not in out.columns:
        # try to build from the provided domain_col, else empty
        out["target_domain"] = out[domain_col].astype(str) if domain_col in out.columns else ""
    for col in ["title_meta", "description_meta"]:
        if col not in out.columns:
            out[col] = ""
    return out


# --------------------------------- Predictor ---------------------------------

class Predictor:
    """
    Loads packaged artifacts by default; or from `artifacts_dir`.
    """

    def __init__(self, artifacts_dir: str | Path | None = None):
        if artifacts_dir is None:
            art = _packaged_artifacts_dir()
            with as_file(art / "meta.json") as m_path:
                meta = json.loads(Path(m_path).read_text(encoding="utf-8"))
            pipe_name = meta.get("pipeline_filename", "pipeline.pkl")
            with as_file(art / pipe_name) as p_path:
                self.pipe = _load_pipeline_any(Path(p_path), meta.get("pipeline_format"))
        else:
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
        self.model = _get_embedder(self.meta.get("embedding_model", "distiluse-base-multilingual-cased-v2"))

    def predict_dataframe(
        self,
        df: pd.DataFrame,
        text_col: Optional[str] = None,
        *,
        # column-name overrides for long-format ingestion
        domain_col: str = "target_domain",
        meta_tag_name: str = "name",
        meta_tag_value: str = "content_latest",
        # name of the domain column to expose in the output
        output_domain_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        If `text_col` is provided, uses it directly; otherwise:
          - auto-detects long CSV layout using (domain_col, meta_tag_name, meta_tag_value)
          - pivots to wide and builds title_meta/description_meta
          - cleans to `selected_text`
          - falls back to composing from raw fields when selected_text is empty
          - predicts with LinearSVC pipeline
          - returns: domain column (see `output_domain_col`), selected_text, predicted_label,
                     predicted_proba (NaN), proba_pseudo.
        """
        if output_domain_col is None:
            output_domain_col = domain_col  # mirror user's chosen name

        base = df.copy()

        # 1) Prepare text to embed
        if text_col is None or text_col not in base.columns:
            wide = _ensure_wide(
                base,
                domain_col=domain_col,
                name_col=meta_tag_name,
                value_col=meta_tag_value,
            )
            proc = clean_dataframe(wide)
            out = proc[["selected_text"]].copy()

            # domain pass-through under user's chosen name
            if "target_domain" in proc.columns:
                out.insert(0, output_domain_col, proc["target_domain"])

            # fallback for empties
            empty_mask = out["selected_text"].fillna("").astype(str).str.strip() == ""
            if empty_mask.any():
                out.loc[empty_mask, "selected_text"] = _fallback_selected_text(wide.loc[empty_mask])
        else:
            out = base[[text_col]].rename(columns={text_col: "selected_text"}).copy()
            if domain_col in base.columns:
                out.insert(0, output_domain_col, base[domain_col])

        # 2) Predict only for non-empty texts; keep all rows in output
        mask = out["selected_text"].fillna("").astype(str).str.strip() != ""

        # dtype-safe initialization (avoid FutureWarning)
        out["predicted_label"] = pd.Series(index=out.index, dtype="object")       # labels may be str/int
        out["predicted_proba"] = pd.Series(np.nan, index=out.index, dtype="float64")
        out["proba_pseudo"]   = pd.Series(np.nan, index=out.index, dtype="float64")

        if mask.any():
            texts = out.loc[mask, "selected_text"].tolist()
            X = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

            # labels
            preds = self.pipe.predict(X)
            out.loc[mask, "predicted_label"] = preds

            # pseudo-proba via decision_function -> softmax
            if hasattr(self.pipe, "decision_function"):
                margins = self.pipe.decision_function(X)  # (n, K) or (n,)
                if margins.ndim == 1:  # binary safety
                    margins = np.column_stack([-margins, margins])
                probs = _rowwise_softmax(margins)
                # map predicted class -> index
                try:
                    classes = self.pipe.named_steps["linear_svc"].classes_
                except Exception:
                    classes = np.unique(preds)
                cls_to_ix = {c: i for i, c in enumerate(classes)}
                pred_ix = np.array([cls_to_ix[c] for c in preds])
                p_hat = probs[np.arange(len(texts)), pred_ix]
                out.loc[mask, "proba_pseudo"] = p_hat

        return out


def load_predictor(artifacts_dir: str | Path | None = None) -> Predictor:
    """Factory: Predictor that loads packaged artifacts by default."""
    return Predictor(artifacts_dir)


# ------------------------------ One-liner wrapper ----------------------------

def svm_predictor(
    raw_data,
    artifacts_dir: str | Path | None = None,
    text_col: str | None = None,
    output: str = "df",
    *,
    # column-name overrides for ingestion (long format)
    domain_col: str = "target_domain",
    meta_tag_name: str = "name",
    meta_tag_value: str = "content_latest",
    # customize output domain column name
    output_domain_col: Optional[str] = None,
):
    """
    One-call prediction.

    raw_data:
      - DataFrame in long (domain_col/meta_tag_name/meta_tag_value) OR wide schema;
      - list[str] of texts;
      - list[dict] rows.

    Column name params let you match your CSV:
      - domain_col      (default "target_domain")
      - meta_tag_name   (default "name")
      - meta_tag_value  (default "content_latest")
      - output_domain_col (default mirrors domain_col)

    Returns:
      DataFrame with {output_domain_col}, selected_text, predicted_label,
      predicted_proba (NaN), proba_pseudo; or labels ndarray if output="labels".
    """
    pred = load_predictor(artifacts_dir)

    if isinstance(raw_data, pd.DataFrame):
        df_out = pred.predict_dataframe(
            raw_data.copy(),
            text_col=text_col,
            domain_col=domain_col,
            meta_tag_name=meta_tag_name,
            meta_tag_value=meta_tag_value,
            output_domain_col=output_domain_col,
        )
        return df_out if output == "df" else df_out["predicted_label"].to_numpy()

    if isinstance(raw_data, list):
        if len(raw_data) == 0:
            return (pd.DataFrame(columns=["selected_text", "predicted_label", "predicted_proba", "proba_pseudo"])
                    if output == "df" else np.array([]))
        first = raw_data[0]
        if isinstance(first, str):
            tmp = pd.DataFrame({"selected_text": raw_data})
            df_out = pred.predict_dataframe(
                tmp, text_col="selected_text",
                domain_col=domain_col, meta_tag_name=meta_tag_name,
                meta_tag_value=meta_tag_value,
                output_domain_col=output_domain_col,
            )
            return df_out if output == "df" else df_out["predicted_label"].to_numpy()
        if isinstance(first, dict):
            df = pd.DataFrame(raw_data)
            df_out = pred.predict_dataframe(
                df, text_col=text_col,
                domain_col=domain_col, meta_tag_name=meta_tag_name,
                meta_tag_value=meta_tag_value,
                output_domain_col=output_domain_col,
            )
            return df_out if output == "df" else df_out["predicted_label"].to_numpy()

    raise TypeError("raw_data must be a pandas DataFrame, list[str], or list[dict].")
