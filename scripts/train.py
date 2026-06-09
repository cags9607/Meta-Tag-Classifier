from meta_tag_classifier.cli import main
if __name__ == "__main__":
    import sys
    sys.argv = ["mtc", "train", "--config", "configs/default.yaml"]
    main()
