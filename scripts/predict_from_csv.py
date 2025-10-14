#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--text-col", default=None)
    p.add_argument("--artifacts", default=None)   # None => packaged model
    args = p.parse_args()

    from meta_tag_classifier import load_predictor

    df = pd.read_csv(args.input)
    pred = load_predictor(args.artifacts)
    out = pred.predict_dataframe(df, text_col=args.text_col)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote predictions for {len(out):,} rows to {args.output}")

if __name__ == "__main__":
    main()
