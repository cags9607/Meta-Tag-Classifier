"""
Core wrapper for the meta-tag template classifier.
Cleaning + selection are done internally by the library.

This wrapper adds one convenience field:
- selected_name

Because the underlying library returns selected_text but not the chosen tag name,
we infer the chosen tag name from the original long-form input passed to the model.
That keeps processor.py simple and makes the queue output exact/stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from meta_tag_classifier.api import svm_predictor


PREDICTION_INPUT_TAGS = {"title", "description", "og:title", "og:description"}
NAME_PRIORITY = ["title", "og:title", "description", "og:description"]


def _normalize_text_for_match(x: Any) -> str:
    return " ".join(str(x or "").strip().split()).lower()


def _pick_selected_name(df_long: pd.DataFrame, selected_text: Any) -> Optional[str]:
    """
    Infer which meta-tag name produced selected_text.

    Expected df_long columns:
    - target_domain
    - name
    - content_latest
    """
    selected_norm = _normalize_text_for_match(selected_text)
    if not selected_norm:
        return None

    if df_long is None or df_long.shape[0] == 0:
        return None

    candidates = df_long.copy()

    required = {"name", "content_latest"}
    if not required.issubset(candidates.columns):
        return None

    candidates["name"] = candidates["name"].astype(str)
    candidates["name_norm"] = candidates["name"].str.strip().str.lower()
    candidates["content_latest"] = candidates["content_latest"].fillna("").astype(str)

    candidates = (
        candidates
        .loc[lambda x: x["name_norm"].isin(PREDICTION_INPUT_TAGS)]
        [["name", "name_norm", "content_latest"]]
        .drop_duplicates()
        .reset_index(drop = True)
    )

    if candidates.shape[0] == 0:
        return None

    candidates["content_norm"] = candidates["content_latest"].map(_normalize_text_for_match)

    exact = candidates.loc[candidates["content_norm"] == selected_norm].copy()
    if exact.shape[0] > 0:
        exact["priority"] = exact["name_norm"].map({name: i for i, name in enumerate(NAME_PRIORITY)})
        exact = exact.sort_values(["priority", "name"])
        return str(exact.iloc[0]["name"])

    contains = candidates.loc[
        candidates["content_norm"].map(lambda x: bool(x) and (selected_norm in x or x in selected_norm))
    ].copy()

    if contains.shape[0] > 0:
        contains["priority"] = contains["name_norm"].map({name: i for i, name in enumerate(NAME_PRIORITY)})
        contains["len_delta"] = (contains["content_norm"].str.len() - len(selected_norm)).abs()
        contains = contains.sort_values(["len_delta", "priority", "name"])
        return str(contains.iloc[0]["name"])

    return None


@dataclass
class MetaTagTemplateClassifier:
    artifacts_dir: Path

    def __post_init__(self):
        self.artifacts_dir = Path(self.artifacts_dir)

    def predict_dataframe(self, df_long: pd.DataFrame) -> pd.DataFrame:
        """
        df_long must include:
          - target_domain
          - name
          - content_latest

        Returns domain-level DataFrame with:
          - target_domain
          - selected_text
          - selected_name
          - predicted_label
          - predicted_proba
          - proba_pseudo
        """
        df_out = svm_predictor(df_long, artifacts_dir = self.artifacts_dir, output = "df").copy()

        if df_out.shape[0] == 0:
            return df_out

        df_out["selected_name"] = df_out["selected_text"].map(
            lambda x: _pick_selected_name(df_long, x)
        )

        return df_out

    def predict_one_domain(self, df_long: pd.DataFrame) -> Dict[str, Any]:
        """
        Convenience method: returns first row as dict.
        """
        df_out = self.predict_dataframe(df_long)
        if df_out.shape[0] == 0:
            return {}
        return df_out.iloc[0].to_dict()
