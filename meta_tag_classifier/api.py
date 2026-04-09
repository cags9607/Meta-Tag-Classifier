from __future__ import annotations

import json
import pickle
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
from .utils import ensure_dir


_model_cache: Dict[str, SentenceTransformer] = {}


def _get_embedder(model_name: str) -> SentenceTransformer:
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
    model = _get_embedder(model_name)
    embs = model.encode(
        list(texts),
        show_progress_bar = show_progress_bar,
        convert_to_numpy = True,
    )
    return embs


def clean_dataframe(
    df: pd.DataFrame,
    nav_thr: float = 2.0,
    prefer_title: bool = True,
    short_title_word_threshold: int = 8,
) -> pd.DataFrame:
    return clean_metas(
        df.copy(),
        nav_thr = nav_thr,
        prefer_title = prefer_title,
        short_title_word_threshold = short_title_word_threshold,
    )


class TrainResult:
    def __init__(
        self,
        artifacts_dir: Path,
        best_params: Dict[str, Any],
        report_text: str,
        classes: List[Any],
        embedding_model: str,
        pipeline_filename: str = "pipeline.pkl",
        pipeline_format: str = "pickle",
    ):
        self.artifacts_dir = artifacts_dir
        self.best_params = best_params
        self.report_text = report_text
        self.classes = classes
        self.embedding_model = embedding_model
        self.pipeline_filename = pipeline_filename
        self.pipeline_format = pipeline_format


def _save_pipeline(obj: Any, path: Path, fmt: str = "pickle") -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    if fmt == "pickle" or path.suffix.lower() == ".pkl":
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol = pickle.HIGHEST_PROTOCOL)
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
    pipeline_filename: str = "pipeline.pkl",
    pipeline_format: str = "pickle",
) -> TrainResult:
    if C_grid is None:
        C_grid = [0.0001, 0.001, 0.01, 0.1, 1.0]

    if label_column not in df.columns:
        raise ValueError(f"Missing label column: {label_column!r}")

    df_sel = clean_dataframe(df)
    df_sel = df_sel[df_sel["selected_text"].notna() & (df_sel["selected_text"] != "")].copy()

    X_text = df_sel["selected_text"].tolist()
    y = df_sel[label_column].values

    X = embed_texts(X_text, model_name = embedding_model)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = test_size, random_state = random_state, stratify = y
    )

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components = 0.95)),
            ("linear_svc", LinearSVC(dual = False, class_weight = "balanced")),
        ]
    )
    param_grid = {"linear_svc__C": C_grid}
    grid = GridSearchCV(pipe, param_grid, cv = 5, scoring = "f1_macro", n_jobs = -1)

    with parallel_backend("loky", inner_max_num_threads = 1):
        grid.fit(X_train, y_train)

    y_pred = grid.best_estimator_.predict(X_test)
    report = classification_report(y_test, y_pred, digits = 4)

    art_dir = ensure_dir(artifacts_dir)
    _save_pipeline(grid.best_estimator_, Path(art_dir) / pipeline_filename, fmt = pipeline_format)

    meta = {
        "embedding_model": embedding_model,
        "pipeline_filename": pipeline_filename,
        "pipeline_format": pipeline_format,
        "best_params": grid.best_params_,
        "classes_": sorted(pd.unique(y).tolist()),
    }
    (Path(art_dir) / "meta.json").write_text(
        json.dumps(meta, ensure_ascii = False, indent = 2),
        encoding = "utf-8",
    )
    (Path(art_dir) / "report.txt").write_text(report, encoding = "utf-8")

    return TrainResult(
        artifacts_dir = Path(artifacts_dir),
        best_params = grid.best_params_,
        report_text = report,
        classes = meta["classes_"],
        embedding_model = embedding_model,
        pipeline_filename = pipeline_filename,
        pipeline_format = pipeline_format,
    )


def _packaged_artifacts_dir() -> Path | Any:
    try:
        return files("meta_tag_classifier") / "artifacts"
    except Exception:
        return Path(__file__).resolve().parent / "artifacts"


def _load_pipeline_any(path: Path, fmt: Optional[str] = None) -> Any:
    suffix = path.suffix.lower()
    if fmt == "pickle" or suffix == ".pkl":
        with open(path, "rb") as f:
            return pickle.load(f)
    import joblib
    return joblib.load(path)


def _rowwise_softmax(m: np.ndarray) -> np.ndarray:
    z = m - m.max(axis = 1, keepdims = True)
    np.exp(z, out = z)
    z_sum = z.sum(axis = 1, keepdims = True)
    z /= z_sum
    return z


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
    return df.apply(pick, axis = 1)


def _ensure_wide(
    df: pd.DataFrame,
    domain_col: str = "target_domain",
    name_col: str = "name",
    value_col: str = "content_latest",
) -> pd.DataFrame:
    if {domain_col, name_col, value_col}.issubset(df.columns):
        return long_to_wide_meta(
            df,
            domain_col = domain_col,
            name_col = name_col,
            value_col = value_col,
        )

    out = df.copy()
    if "target_domain" not in out.columns:
        out["target_domain"] = out[domain_col].astype(str) if domain_col in out.columns else ""
    for col in ["title_meta", "description_meta"]:
        if col not in out.columns:
            out[col] = ""
    return out


class Predictor:
    def __init__(self, artifacts_dir: str | Path | None = None):
        if artifacts_dir is None:
            art = _packaged_artifacts_dir()
            with as_file(art / "meta.json") as m_path:
                meta = json.loads(Path(m_path).read_text(encoding = "utf-8"))
            pipe_name = meta.get("pipeline_filename", "pipeline.pkl")
            with as_file(art / pipe_name) as p_path:
                self.pipe = _load_pipeline_any(Path(p_path), meta.get("pipeline_format"))
        else:
            artifacts_dir = Path(artifacts_dir)
            meta_path = artifacts_dir / "meta.json"
            meta = (
                json.loads(meta_path.read_text(encoding = "utf-8"))
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
        domain_col: str = "target_domain",
        meta_tag_name: str = "name",
        meta_tag_value: str = "content_latest",
        output_domain_col: Optional[str] = None,
    ) -> pd.DataFrame:
        if output_domain_col is None:
            output_domain_col = domain_col

        base = df.copy()

        if text_col is None or text_col not in base.columns:
            wide = _ensure_wide(
                base,
                domain_col = domain_col,
                name_col = meta_tag_name,
                value_col = meta_tag_value,
            )
            proc = clean_dataframe(wide)

            keep_cols = ["selected_text"]
            if "selected_name" in proc.columns:
                keep_cols.append("selected_name")

            out = proc[keep_cols].copy()

            if "target_domain" in proc.columns:
                out.insert(0, output_domain_col, proc["target_domain"])

            empty_mask = out["selected_text"].fillna("").astype(str).str.strip() == ""
            if empty_mask.any():
                out.loc[empty_mask, "selected_text"] = _fallback_selected_text(wide.loc[empty_mask])

            if "selected_name" not in out.columns:
                out["selected_name"] = pd.Series(index = out.index, dtype = "object")
        else:
            out = base[[text_col]].rename(columns = {text_col: "selected_text"}).copy()
            if domain_col in base.columns:
                out.insert(0, output_domain_col, base[domain_col])
            out["selected_name"] = pd.Series(index = out.index, dtype = "object")

        mask = out["selected_text"].fillna("").astype(str).str.strip() != ""

        out["predicted_label"] = pd.Series(index = out.index, dtype = "object")
        out["predicted_proba"] = pd.Series(np.nan, index = out.index, dtype = "float64")
        out["proba_pseudo"] = pd.Series(np.nan, index = out.index, dtype = "float64")

        if mask.any():
            texts = out.loc[mask, "selected_text"].tolist()
            X = self.model.encode(texts, show_progress_bar = True, convert_to_numpy = True)

            preds = self.pipe.predict(X)
            out.loc[mask, "predicted_label"] = preds

            if hasattr(self.pipe, "decision_function"):
                margins = self.pipe.decision_function(X)
                if margins.ndim == 1:
                    margins = np.column_stack([-margins, margins])
                probs = _rowwise_softmax(margins)
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
    return Predictor(artifacts_dir)


def svm_predictor(
    raw_data,
    artifacts_dir: str | Path | None = None,
    text_col: str | None = None,
    output: str = "df",
    *,
    domain_col: str = "target_domain",
    meta_tag_name: str = "name",
    meta_tag_value: str = "content_latest",
    output_domain_col: Optional[str] = None,
):
    pred = load_predictor(artifacts_dir)

    if isinstance(raw_data, pd.DataFrame):
        df_out = pred.predict_dataframe(
            raw_data.copy(),
            text_col = text_col,
            domain_col = domain_col,
            meta_tag_name = meta_tag_name,
            meta_tag_value = meta_tag_value,
            output_domain_col = output_domain_col,
        )
        return df_out if output == "df" else df_out["predicted_label"].to_numpy()

    if isinstance(raw_data, list):
        if len(raw_data) == 0:
            return (
                pd.DataFrame(
                    columns = [
                        "selected_text",
                        "selected_name",
                        "predicted_label",
                        "predicted_proba",
                        "proba_pseudo",
                    ]
                )
                if output == "df"
                else np.array([])
            )

        first = raw_data[0]

        if isinstance(first, str):
            tmp = pd.DataFrame({"selected_text": raw_data})
            df_out = pred.predict_dataframe(
                tmp,
                text_col = "selected_text",
                domain_col = domain_col,
                meta_tag_name = meta_tag_name,
                meta_tag_value = meta_tag_value,
                output_domain_col = output_domain_col,
            )
            return df_out if output == "df" else df_out["predicted_label"].to_numpy()

        if isinstance(first, dict):
            df = pd.DataFrame(raw_data)
            df_out = pred.predict_dataframe(
                df,
                text_col = text_col,
                domain_col = domain_col,
                meta_tag_name = meta_tag_name,
                meta_tag_value = meta_tag_value,
                output_domain_col = output_domain_col,
            )
            return df_out if output == "df" else df_out["predicted_label"].to_numpy()

    raise TypeError("raw_data must be a pandas DataFrame, list[str], or list[dict].")
