#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("--csv")
    p.add_argument("--label")
    p.add_argument("--artifacts", default="models/artifacts")
    p.add_argument("--embedding-model", default="distiluse-base-multilingual-cased-v2")
    p.add_argument("--c-grid", nargs="*", type=float, default=[0.0001, 0.001, 0.01, 0.1, 1.0])
    args = p.parse_args()

    if args.config:
        from meta_tag_classifier.models.train import train_from_config
        art = train_from_config(args.config)
        print(f"Artifacts written to: {art}")
        return

    if not (args.csv and args.label):
        raise SystemExit("Provide either --config or both --csv and --label.")

    from meta_tag_classifier import train
    df = pd.read_csv(args.csv)
    res = train(df, label_column=args.label, artifacts_dir=args.artifacts,
                embedding_model=args.embedding_model, C_grid=args.c_grid,
                pipeline_filename="pipeline.pkl", pipeline_format="pickle")
    print("Best params:", json.dumps(res.best_params, indent=2))
    print("Artifacts:", Path(res.artifacts_dir).resolve())

if __name__ == "__main__":
    main()
