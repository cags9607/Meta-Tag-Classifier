from __future__ import annotations

from pathlib import Path

import pandas as pd

from meta_tag_classifier.api import load_predictor


def predict_csv(
    input_csv: str | Path,
    text_col: str | None,
    artifacts_dir: str | Path | None,
    output_csv: str | Path,
) -> Path:
    df = pd.read_csv(input_csv)
    pred = load_predictor(artifacts_dir)
    out = pred.predict_dataframe(df, text_col = text_col)

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents = True, exist_ok = True)
    out.to_csv(out_path, index = False)
    return out_path
