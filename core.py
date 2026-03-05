"""
Core wrapper for the meta-tag template classifier.
Cleaning + selection are done internally by the library.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from meta_tag_classifier.api import svm_predictor


@dataclass
class MetaTagTemplateClassifier:
    artifacts_dir: Path

    def __post_init__(self):
        self.artifacts_dir = Path(self.artifacts_dir)

    def predict_dataframe(self, df_long: pd.DataFrame) -> pd.DataFrame:
        """
        df_long must include (defaults):
          - target_domain
          - name
          - content_latest
        Returns domain-level DataFrame with:
          target_domain, selected_text, predicted_label, predicted_proba (NaN), proba_pseudo
        """
        return svm_predictor(df_long, artifacts_dir = self.artifacts_dir, output = "df")

    def predict_one_domain(self, df_long: pd.DataFrame) -> Dict[str, Any]:
        """
        Convenience method: returns first row as dict.
        """
        df_out = self.predict_dataframe(df_long)
        if df_out.shape[0] == 0:
            return {}
        return df_out.iloc[0].to_dict()
