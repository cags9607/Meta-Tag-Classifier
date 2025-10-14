# Meta-Tag-Classifier

Colab-friendly pipeline: **clean → train → infer** for meta tag text classification. Works as a **library** and **CLI**.

## Install (editable)
```bash
pip install -e .
```

## Library usage
```python
import pandas as pd
from meta_tag_classifier import train, load_predictor, clean_dataframe

# 1) Clean + train
df = pd.read_csv("data/raw/train.csv")
res = train(df, label_column="label", artifacts_dir="models/artifacts")
print(res.best_params)
print(res.report_text)

# 2) Load predictor and run on raw meta fields (auto-clean)
pred = load_predictor("models/artifacts")
df_infer = pd.read_csv("data/processed/infer_input.csv")
df_out = pred.predict_dataframe(df_infer)     # auto-clean (title/og:title/...)
df_out.to_csv("data/processed/preds.csv", index=False)

# 3) Or predict from a list of texts
texts = ["example description", "another meta title"]
labels, probas = pred.predict_texts(texts)
```

## CLI
```bash
python -m meta_tag_classifier.cli train --config configs/default.yaml
python -m meta_tag_classifier.cli predict --artifacts models/artifacts --input data/processed/infer_input.csv --output data/processed/preds.csv
# or, after install:
mtc predict --artifacts models/artifacts --input data/processed/infer_input.csv --output data/processed/preds.csv
```
