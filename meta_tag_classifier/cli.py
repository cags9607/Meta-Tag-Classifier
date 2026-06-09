import argparse


def main():
    parser = argparse.ArgumentParser(prog = "meta-tag-classifier")
    sub = parser.add_subparsers(dest = "cmd", required = True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--config", required = True)

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--artifacts", required = True)
    p_pred.add_argument("--input", required = True)
    p_pred.add_argument("--text-col", required = False, default = None)
    p_pred.add_argument("--output", required = True)

    p_clean = sub.add_parser("clean")
    p_clean.add_argument("--input", required = True)
    p_clean.add_argument("--output", required = True)

    args = parser.parse_args()

    if args.cmd == "train":
        raise SystemExit(
            "Training is intentionally not implemented in this deployment repo. "
            "Fine-tune Qwen in Colab, then copy the saved adapter folder and metadata into artifacts/."
        )

    if args.cmd == "predict":
        from meta_tag_classifier.models.infer import predict_csv

        predict_csv(args.input, args.text_col, args.artifacts, args.output)
        return

    if args.cmd == "clean":
        from pathlib import Path
        import pandas as pd
        from meta_tag_classifier.data.clean import clean_metas

        df = pd.read_csv(args.input)
        df_out = clean_metas(df)
        Path(args.output).parent.mkdir(parents = True, exist_ok = True)
        df_out.to_csv(args.output, index = False)
        return


if __name__ == "__main__":
    main()
