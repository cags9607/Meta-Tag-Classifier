from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from meta_tag_classifier.api import load_predictor


@dataclass
class MetaTagTemplateClassifier:
    artifacts_dir: Path

    def __post_init__(self):
        self.artifacts_dir = Path(self.artifacts_dir)
        self.predictor = load_predictor(self.artifacts_dir)

    def predict_dataframe(
        self,
        df_long: pd.DataFrame,
        *,
        domain_col: str = "target_domain",
        meta_tag_name: str = "name",
        meta_tag_value: str = "content_latest",
        output_domain_col: Optional[str] = None,
    ) -> pd.DataFrame:
        return self.predictor.predict_dataframe(
            df_long.copy(),
            domain_col = domain_col,
            meta_tag_name = meta_tag_name,
            meta_tag_value = meta_tag_value,
            output_domain_col = output_domain_col,
        )

    def predict_one_domain(self, df_long: pd.DataFrame) -> Dict[str, Any]:
        df_out = self.predict_dataframe(df_long)

        if df_out.shape[0] == 0:
            return {}

        return df_out.iloc[0].to_dict()
