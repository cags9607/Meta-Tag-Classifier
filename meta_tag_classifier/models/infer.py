from pathlib import Path
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from meta_tag_classifier.data.clean import clean_metas

def _load_meta(artifacts_dir: str | Path):
    mpath = Path(artifacts_dir) / "meta.json"
    if not mpath.exists():
        return {"embedding_model": "distiluse-base-multilingual-cased-v2", "pipeline_filename": "pipeline.pkl"}
    return json.loads(mpath.read_text(encoding="utf-8"))

def _load_pipeline_any(path: Path, fmt: str | None):
    if (fmt == "pickle") or (path.suffix.lower() == ".pkl"):
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
    import joblib
    return joblib.load(path)

def predict_csv(input_csv: str | Path, text_col: str | None, artifacts_dir: str | Path | None, output_csv: str | Path) -> Path:
    df = pd.read_csv(input_csv)

    if text_col is None or text_col not in df.columns:
        df_proc = clean_metas(df.copy())
        df_proc = df_proc[df_proc["selected_text"].notna() & (df_proc["selected_text"] != "")].reset_index(drop=True)
        texts = df_proc["selected_text"].tolist()
        base_out = df_proc[["selected_text"]].copy()
        if "target_domain" in df_proc.columns:
            base_out.insert(0, "target_domain", df_proc["target_domain"])
    else:
        texts = df[text_col].fillna("").tolist()
        base_out = df[[text_col]].rename(columns={text_col: "selected_text"}).copy()
        if "target_domain" in df.columns:
            base_out.insert(0, "target_domain", df["target_domain"])

    if artifacts_dir is None:
        # use packaged artifacts
        from importlib.resources import files, as_file
        art = files("meta_tag_classifier") / "artifacts"
        with as_file(art / "meta.json") as m_path:
            meta = json.loads(Path(m_path).read_text(encoding="utf-8"))
        with as_file(art / meta.get("pipeline_filename", "pipeline.pkl")) as p_path:
            pipe = _load_pipeline_any(Path(p_path), meta.get("pipeline_format"))
    else:
        meta = _load_meta(artifacts_dir)
        pipe = _load_pipeline_any(Path(artifacts_dir) / meta.get("pipeline_filename", "pipeline.pkl"),
                                  meta.get("pipeline_format"))

    model = SentenceTransformer(meta.get("embedding_model", "distiluse-base-multilingual-cased-v2"))
    X = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    preds = pipe.predict(X)
    proba = np.full(shape=(len(preds),), fill_value=np.nan, dtype=float)

    out = base_out.copy()
    out["predicted_label"] = preds
    out["predicted_proba"] = proba

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out_path
