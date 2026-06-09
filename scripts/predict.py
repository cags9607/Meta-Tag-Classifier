from meta_tag_classifier.cli import main
if __name__ == "__main__":
    import sys
    sys.argv = [
        "mtc", "predict",
        "--artifacts", "models/artifacts",
        "--input", "data/processed/infer_input.csv",
        "--output", "data/processed/preds.csv",
    ]
    main()
