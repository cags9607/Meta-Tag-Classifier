from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from importlib.resources import as_file, files

from .data.clean import clean_metas
from .data.ingest import long_to_wide_meta
from .models.qwen_infer import QwenSequenceClassifier


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


_FALLBACK_FIELDS = [
    "title",
    "og:title",
    "twitter:title",
    "description",
    "og:description",
    "twitter:description",
    "meta_description",
]


def _fallback_selected_text(df: pd.DataFrame) -> pd.Series:
    def pick(row):
        for c in _FALLBACK_FIELDS:
            v = row.get(c, "")
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    return df.apply(pick, axis = 1)


def _packaged_artifacts_dir() -> Path | Any:
    try:
        return files("meta_tag_classifier") / "artifacts"
    except Exception:
        return Path(__file__).resolve().parent / "artifacts"


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
    """
    Production predictor for the Qwen meta-tag template-family classifier.

    The public dataframe contract intentionally matches the previous SVM version:
    clean meta tags -> select selected_text -> output predicted_label/proba_pseudo.
    The model input is only selected_text.
    """

    def __init__(
        self,
        artifacts_dir: str | Path | None = None,
        *,
        batch_size: int | None = None,
        max_length: int | None = None,
        load_in_4bit: bool | None = None,
    ):
        if artifacts_dir is None:
            art = _packaged_artifacts_dir()
            with as_file(art) as art_path:
                artifacts_path = Path(art_path)
                self.backend = QwenSequenceClassifier(
                    artifacts_path,
                    batch_size = batch_size,
                    max_length = max_length,
                    load_in_4bit = load_in_4bit,
                )
        else:
            self.backend = QwenSequenceClassifier(
                artifacts_dir,
                batch_size = batch_size,
                max_length = max_length,
                load_in_4bit = load_in_4bit,
            )

        self.meta = self.backend.config

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

            keep_cols = ["selected_text", "selected_name"]
            out = proc[keep_cols].copy()

            if "target_domain" in proc.columns:
                out.insert(0, output_domain_col, proc["target_domain"])

            empty_mask = out["selected_text"].fillna("").astype(str).str.strip() == ""
            if empty_mask.any():
                out.loc[empty_mask, "selected_text"] = _fallback_selected_text(wide.loc[empty_mask])
        else:
            out = base[[text_col]].rename(columns = {text_col: "selected_text"}).copy()

            if domain_col in base.columns:
                out.insert(0, output_domain_col, base[domain_col])

            out["selected_name"] = pd.Series(index = out.index, dtype = "object")

        out["selected_text"] = out["selected_text"].fillna("").astype(str).str.strip()
        mask = out["selected_text"] != ""

        out["predicted_label"] = pd.Series(index = out.index, dtype = "object")
        out["predicted_proba"] = pd.Series(np.nan, index = out.index, dtype = "float64")
        out["proba_pseudo"] = pd.Series(np.nan, index = out.index, dtype = "float64")

        if mask.any():
            texts = out.loc[mask, "selected_text"].tolist()
            preds, confs = self.backend.predict_texts(texts)

            out.loc[mask, "predicted_label"] = preds
            out.loc[mask, "predicted_proba"] = confs
            out.loc[mask, "proba_pseudo"] = confs

        return out


def load_predictor(
    artifacts_dir: str | Path | None = None,
    *,
    batch_size: int | None = None,
    max_length: int | None = None,
    load_in_4bit: bool | None = None,
) -> Predictor:
    return Predictor(
        artifacts_dir,
        batch_size = batch_size,
        max_length = max_length,
        load_in_4bit = load_in_4bit,
    )


def template_predictor(
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
            empty = pd.DataFrame(
                columns = [
                    "selected_text",
                    "selected_name",
                    "predicted_label",
                    "predicted_proba",
                    "proba_pseudo",
                ]
            )
            return empty if output == "df" else np.array([])

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


# Backward-compatible alias for older notebooks/scripts.
svm_predictor = template_predictor
