# Meta-Tag-Classifier

Pipeline to get a classification of a templated domain using its homepage meta tags (description, title, og:description, og:title). The library implements an SVM Classifier trained on `distiluse-base-multilingual-cased-v2` multilingual embeddings of a dataset of ~`12k` meta tags from templated domains.

# Methods

The construction of the training data followed several steps. First, we took a random sample of `11923` templated domains and retrieved meta tags from the homepage using the `meta_tags` superset table. Then we used BERTopic to get soft labels for each domain, using `distiluse-base-multilingual-cased-v2` embeddings of the selected meta tag (after cleaning/processing). After that, several regex-based heuristics were applied to prune further the proposed labels. Once we had a curated labeled data set, we trained an SVM Classifier and tested the model using a data set of manually labeled templates (837 domains).

# Install (editable)

## Repository

```bash
git clone https://github.com/cags9607/Meta-Tag-Classifier.git
cd Meta-Tag-Classifier
pip install -e .
```

## Library

```bash
pip install -U pip
pip install "git+https://github.com/cags9607/Meta-Tag-Classifier.git"
```

# Usage

```python
import pandas as pd
from meta_tag_classifier import svm_predictor

# LONG format (rows per meta)
df_long = pd.DataFrame([
    {"target_domain":"https://fitjourney.example","name":"title","content_latest":"Fitjourney"},
    {"target_domain":"https://fitjourney.example","name":"description","content_latest":"Daily plans for strength and cardio."},
    {"target_domain":"http://shoestore.example","name":"og:title","content_latest":"Best running shoes"},
    {"target_domain":"http://shoestore.example","name":"og:description","content_latest":"Lightweight marathon trainers for all distances."},
])

out = svm_predictor(df_long)   # auto-pivot → clean → embed → predict
print(out)
```

|index|target\_domain|selected\_text|predicted\_label|predicted\_proba|proba\_pseudo|
|---|---|---|---|---|---|
|0|fitjourney\.example|Daily plans for strength and cardio\.|Content Farm|NaN|0\.40931944629041905|
|1|shoestore\.example|Best running shoes|Content Farm|NaN|0\.3182812910331625|


## Cleaning

- domain-word checks (does the domain name appear in the meta tags?)
- HTML/JS stripped,
- Unicode normalization
- Emoticons removed
- URLs/emails/phones/domains removed/masked
- De-duplication (2-3 grams, repeated punctuation)
- Domain-word removal
- Separator trimming
- Clean pass for distil embeddings (remove unknown characters, remove isolated non-word symbols, collapse whitespaces)
- Score noise (navigation lexicon)
- Select meta tag (pick the non-noisy, prioritize long enough titles)

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
