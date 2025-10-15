# Meta-Tag-Classifier

Pipeline to get a classification of a templated domain using its homepage meta tags (description, title, og:description, og:title). The library implements an SVM Classifier trained on `distiluse-base-multilingual-cased-v2` multilingual embeddings of a dataset of ~`12k` meta tags from templated domains.

# Methods

The construction of the training data followed several steps. First, we took a random sample of `11923` templated domains and retrieved meta tags (`title`, `description`, `og:title`, `og:description`) from their homepages using the `meta_tags` superset table. Then we used BERTopic (https://maartengr.github.io/BERTopic/index.html) to get soft labels for each domain, using `distiluse-base-multilingual-cased-v2` embeddings of the selected meta tag (after cleaning/processing, detailed in sections below). After that, several regex-based heuristics were applied to prune further the proposed labels. Once we had a curated labeled data set, we trained an SVM Classifier and tested the model using a data set of manually labeled templates (837 domains).

The process to select which meta tag to use was inspired by the paper **Hierarchical Contaminated Web Page Classification Based on Meta Tag Denoising Disposal** (see Acknowledgments).

# Install 

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

df_long = pd.DataFrame([
   
    # App Download Portal
    {"target_domain":"https://apkpremium.example","name":"title","content_latest":"YouTube MOD APK — Premium Unlocked"},
    {"target_domain":"https://apkpremium.example","name":"description","content_latest":"Download premium APKs with no ads and extra features."},

    # Content Farm
    {"target_domain":"https://healthtipsdaily.example","name":"title","content_latest":"10 Miracle Foods Doctors Hate!"},
    {"target_domain":"https://healthtipsdaily.example","name":"description","content_latest":"Shocking secrets to boost weight loss overnight with these tips."},

    # News Scraper
    {"target_domain":"https://topnewshub.example","name":"og:title","content_latest":"Trending News: World, Business, Tech"},
    {"target_domain":"https://topnewshub.example","name":"og:description","content_latest":"Live aggregated headlines updated every minute from hundreds of sources."},

    # Novels or Short Stories
    {"target_domain":"https://readshortstories.example","name":"title","content_latest":"The Lantern At Dusk — Chapter 1"},
    {"target_domain":"https://readshortstories.example","name":"description","content_latest":"A quiet village, a restless shadow, and a secret in the attic."},

    # Online Games & Trivia
    {"target_domain":"https://playtriviafree.example","name":"title","content_latest":"Play Free Online Games"},
    {"target_domain":"https://playtriviafree.example","name":"description","content_latest":"Casual puzzles, arcade, and multiplayer trivia — no download required."},

    # Parked/Holding
    {"target_domain":"https://get-this-domain.example","name":"title","content_latest":"This Domain Is For Sale"},
    {"target_domain":"https://get-this-domain.example","name":"description","content_latest":"parked"},

    # Tool
    {"target_domain":"https://giftoolkit.example","name":"title","content_latest":"GIF Maker — Free Animated Images"},
    {"target_domain":"https://giftoolkit.example","name":"description","content_latest":"Create, resize, and optimize GIFs right in your browser."},
])

out = svm_predictor(
    df_long,
    domain_col="target_domain",
    meta_tag_name="name",
    meta_tag_value="content_latest"
)  # auto-pivot → clean → embed → predict
out
```

|index|target\_domain|selected\_text|predicted\_label|predicted\_proba|proba\_pseudo|
|---|---|---|---|---|---|
|0|apkpremium\.example|Download premium APKs with no ads and extra features\.|App Download Portal|NaN|0\.3356097699714278|
|1|get-this-domain\.example|This Domain Is For Sale|Parked/Holding|NaN|0\.46050087376644117|
|2|giftoolkit\.example|Create, resize, and optimize GIFs right in your browser\.|Tool|NaN|0\.45386199338831246|
|3|healthtipsdaily\.example|Shocking secrets to boost weight loss overnight with these tips\.|Content Farm|NaN|0\.26464324793488536|
|4|playtriviafree\.example|Casual puzzles, arcade, and multiplayer trivia no download required\.|Online Games & Trivia|NaN|0\.7680241748220957|
|5|readshortstories\.example|A quiet village, a restless shadow, and a secret in the attic\.|Novels or Short Stories|NaN|0\.3355778725194999|
|6|topnewshub\.example|Live aggregated headlines updated every minute from hundreds of sources\.|News Scraper|NaN|0\.2563293300283744|


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

# Model Artifacts

The library will look in meta_tag_classifier/artifacts/ for:

```arduino
pipeline.pkl    # scikit-learn pipeline (pickle)
meta.json       # metadata (embedder name, filename, classes, etc.)
report.txt      # optional training report
```

To ship a new model, just overwrite `pipeline.pkl` and update `meta.json` accordingly.

# Training

To train a new model we can use:

```bash
scripts/train_from_pickle.py
```

with relative paths

```bash
data/embeddings/20251015_X_train_svm.pkl
data/embeddings/20251015_y_train_svm.pkl
data/embeddings/20251015_X_test_svm.pkl
data/embeddings/20251015_y_test_svm.pkl
```

Run:

```
python scripts/train_from_pickles.py \
  --artifacts-out meta_tag_classifier/artifacts

```

This writes `pipeline.pkl`, `meta.json`, and `report.txt` into meta_tag_classifier/artifacts/.

If you need to change file names/locations, pass `--x-train`, `--y-train`, `--x-test`, `--y-test` with proper paths.

# Notes and tips

- The first inference downloads the SentenceTransformers (~540mb). Then cache with `SENTENCE_TRANSFORMERS_HOME` so no need to redownload after every call.

- `predicted_proba` is `NaN` because there is not enough test data to calibrate the model (will change in the future). `proba_pseudo` (softmax over SVM margins) is included for ranking tasks.

- GPU will be used automatically by SentenceTransformers, if available.

# Acknowledgements

- SentenceTransformers (DistilUSE Multilingual)
- scikit-learn pipeline (StandardScaler -> PCA -> LinearSVC)
- BERTopic (https://maartengr.github.io/BERTopic/index.html)
- **Hierarchical Contaminated Web Page Classification Based on Meta Tag Denoising Disposal**. Retrieved from: https://onlinelibrary.wiley.com/doi/10.1155/2021/2470897
