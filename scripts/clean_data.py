from meta_tag_classifier.cli import main
if __name__ == "__main__":
    import sys
    sys.argv = ["mtc", "clean", "--input", "data/raw/train.csv", "--output", "data/processed/train_clean.csv"]
    main()
