#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--nav-thr", type=float, default=2.0)
    p.add_argument("--prefer-title", action="store_true", default=True)
    p.add_argument("--short-title-word-threshold", type=int, default=8)
    args = p.parse_args()

    from meta_tag_classifier import clean_dataframe

    df = pd.read_csv(args.input)
    out = clean_dataframe(df, nav_thr=args.nav_thr,
                          prefer_title=args.prefer_title,
                          short_title_word_threshold=args.short_title_word_threshold)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out):,} rows to {args.output}")

if __name__ == "__main__":
    main()
