from pathlib import Path
import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from joblib import parallel_backend
from sentence_transformers import SentenceTransformer

from meta_tag_classifier.config import Config
from meta_tag_classifier.data.clean import clean_metas
from meta_tag_classifier.utils import ensure_dir

def _embed_texts(texts, model_name: str):
    model = SentenceTransformer(model_name)
    return model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

def train_from_config(config_path: str | Path):
    cfg = Config.load(config_path)

    df_raw = pd.read_csv(cfg.raw_data)
    df_sel = clean_metas(df_raw.copy())
    df_sel = df_sel[df_sel["selected_text"].notna() & (df_sel["selected_text"] != "")].copy()
    if cfg.label_column not in df_sel.columns:
        raise ValueError(f"Missing label column: {cfg.label_column}")

    X_text = df_sel["selected_text"].tolist()
    y = df_sel[cfg.label_column].values

    model_name = cfg.embedding.get("model_name", "distiluse-base-multilingual-cased-v2")
    X = _embed_texts(X_text, model_name)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.train.get("test_size", 0.2),
        random_state=cfg.train.get("random_state", 42), stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95)),
        ("linear_svc", LinearSVC(dual=False, class_weight="balanced"))
    ])

    C_grid = cfg.model.get("C_grid", [0.0001, 0.001, 0.01, 0.1, 1.0])
    grid = GridSearchCV(pipeline, {"linear_svc__C": C_grid}, cv=5, scoring="f1_macro", n_jobs=-1)

    with parallel_backend("loky", inner_max_num_threads=1):
        grid.fit(X_train, y_train)

    y_pred = grid.best_estimator_.predict(X_test)
    report = classification_report(y_test, y_pred, digits=4)
    print(report)

    art_dir = ensure_dir(cfg.artifacts_dir)
    # Save as pickle by default
    with open(Path(art_dir) / "pipeline.pkl", "wb") as f:
        import pickle; pickle.dump(grid.best_estimator_, f, protocol=pickle.HIGHEST_PROTOCOL)

    meta = {
        "embedding_model": model_name,
        "pipeline_filename": "pipeline.pkl",
        "pipeline_format": "pickle",
        "best_params": grid.best_params_,
        "classes_": np.unique(y).tolist()
    }
    (Path(art_dir) / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (Path(art_dir) / "report.txt").write_text(report, encoding="utf-8")

    Path(cfg.clean_data).parent.mkdir(parents=True, exist_ok=True)
    df_sel.to_csv(cfg.clean_data, index=False)

    return art_dir
