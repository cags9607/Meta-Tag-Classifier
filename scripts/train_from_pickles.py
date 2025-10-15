#!/usr/bin/env python3
# scripts/train_linear_svc_from_pickles.py
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from joblib import parallel_backend
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def load_pickle(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser(
        description="Train LinearSVC (Scaler → PCA(0.95) → LinearSVC) from precomputed embeddings."
    )
    # === Defaults now point to repo paths ===
    parser.add_argument(
        "--x-train",
        default="data/embeddings/20251015_X_train_svm.pkl",
        help="Path to X_train embeddings pickle within the repo",
    )
    parser.add_argument(
        "--y-train",
        default="data/embeddings/20251015_y_train_svm.pkl",
        help="Path to y_train labels pickle within the repo",
    )
    parser.add_argument(
        "--x-test",
        default="data/embeddings/20251015_X_test_svm.pkl",
        help="Path to X_test embeddings pickle within the repo",
    )
    parser.add_argument(
        "--y-test",
        default="data/embeddings/20251015_y_test_svm.pkl",
        help="Path to y_test labels pickle within the repo",
    )
    parser.add_argument(
        "--artifacts-out",
        default="meta_tag_classifier/artifacts",
        help="Directory to write pipeline.pkl, meta.json, report.txt",
    )
    parser.add_argument(
        "--embedding-model",
        default="distiluse-base-multilingual-cased-v2",
        help="ST model used for the embeddings (stored in meta.json)",
    )
    parser.add_argument(
        "--pkl-name",
        default="pipeline.pkl",
        help="Filename for the saved pipeline (inside artifacts-out)",
    )
    parser.add_argument(
        "--grid-c",
        nargs="*",
        type=float,
        default=[0.0001, 0.001, 0.01, 0.1, 1.0],
        help="Grid of C values for LinearSVC",
    )
    parser.add_argument("--cv", type=int, default=5, help="Number of CV folds")
    args = parser.parse_args()

    # --- Load data ---
    print("Loading pickles from repo paths...")
    X_train = load_pickle(args.x_train)
    y_train = load_pickle(args.y_train)
    X_test = load_pickle(args.x_test)
    y_test = load_pickle(args.y_test)

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)

    print(f"X_train: {X_train.shape} | y_train: {len(y_train)}")
    print(f"X_test : {X_test.shape} | y_test : {len(y_test)}")

    # --- Build pipeline & grid ---
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95)),
        ("linear_svc", LinearSVC(dual=False, class_weight="balanced")),
    ])

    param_grid = {"linear_svc__C": args.grid_c}
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=args.cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
    )

    # --- Fit ---
    print(f"Fitting GridSearchCV with C grid: {args.grid_c} and cv={args.cv} ...")
    with parallel_backend("loky", inner_max_num_threads=1):
        grid_search.fit(X_train, y_train)

    print(f"Best params: {grid_search.best_params_}")
    print(f"Best CV f1_macro: {grid_search.best_score_:.6f}")

    # --- Evaluate on test ---
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    f1 = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    print("\n=== Test Metrics ===")
    print(f"Accuracy  : {acc:.6f}")
    print(f"F1 (macro): {f1:.6f}")
    print("\nClassification report:\n")
    print(report)

    # --- Save artifacts for the library ---
    out_dir = Path(args.artifacts_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline_path = out_dir / args.pkl_name
    print(f"\nSaving pipeline to: {pipeline_path}")
    save_pickle(best_model, pipeline_path)

    meta = {
        "embedding_model": args.embedding_model,
        "pipeline_filename": args.pkl_name,
        "pipeline_format": "pickle",
        "best_params": grid_search.best_params_,
        "classes_": sorted(set(y_train)),
        "cv_f1_macro": grid_search.best_score_,
        "test_f1_macro": float(f1),
        "test_accuracy": float(acc),
    }
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved meta.json to: {meta_path}")

    report_path = out_dir / "report.txt"
    report_path.write_text(
        f"== Grid Search ==\n"
        f"Param grid: {param_grid}\n"
        f"Best params: {grid_search.best_params_}\n"
        f"Best CV f1_macro: {grid_search.best_score_:.6f}\n\n"
        f"== Test Metrics ==\n"
        f"Accuracy: {acc:.6f}\n"
        f"F1 (macro): {f1:.6f}\n\n"
        f"{report}\n",
        encoding="utf-8",
    )
    print(f"Saved report to: {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
