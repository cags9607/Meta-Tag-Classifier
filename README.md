# Meta-Tag-Classifier

Default hosted model: `Trinotrotolueno/meta-tag-classifier/template_classifier`.

DeepSee queue/deployment repo for classifying a domain template family from homepage meta tags.

This version keeps the original production contract:

```text
long meta-tag rows
→ pivot title/description/og:title/og:description
→ clean and denoise meta text
→ select selected_text
→ classify one row per domain/job
```

The model backend has changed from `SentenceTransformer embeddings + SVM` to a Qwen sequence-classification fine-tune. The Qwen model input is **only `selected_text`**. The classifier does not add the domain, URL, meta-tag name, prompt text, or any other handcrafted prefix to the model input.

## What changed from the SVM version

- Removed SentenceTransformer/SVM inference from the production path.
- Added a Qwen + PEFT/LoRA inference backend.
- Added HuggingFace Hub artifact loading. `ARTIFACTS_DIR` can be a local path or a Hub repo/subfolder such as `Trinotrotolueno/meta-tag-classifier/template_classifier`.
- Added `label_mapping.json` support and a default 8-class meta-template mapping.
- Preserved `clean_metas`, `long_to_wide_meta`, `selected_text`, and `selected_name` behavior.
- Preserved the queue output schema:
  - `session_id`
  - `target_url`
  - `name`
  - `selected_text`
  - `predicted_label`
  - `predicted_proba`
  - `proba_pseudo`
  - `timestamp`
- `predicted_proba` and `proba_pseudo` are now the same value: the max softmax confidence from the Qwen classification head.
- Kept `svm_predictor` as a backward-compatible alias to `template_predictor`.

## Artifact layouts

The loader supports two layouts.

### Packaged repo layout

```text
meta_tag_classifier/artifacts/
  model_config.json              # preferred production config
  label_mapping.json             # canonical label/id mapping
  adapter/
    adapter_config.json
    adapter_model.safetensors
    tokenizer.json / tokenizer files
```

### HuggingFace Hub snapshot layout

This repo is configured by default to use your hosted model folder:

```text
Trinotrotolueno/meta-tag-classifier/template_classifier
```

That matches the Hub layout shown in your screenshot, where the model repo is `Trinotrotolueno/meta-tag-classifier` and the adapter files live inside the `template_classifier` folder:

```text
template_classifier/
  README.md
  adapter_config.json
  adapter_model.safetensors
  tokenizer.json / tokenizer files
  label_mapping.json
```

Use it locally like this:

```python
from meta_tag_classifier import load_predictor

predictor = load_predictor("Trinotrotolueno/meta-tag-classifier/template_classifier")
```

or in deployment:

```bash
ARTIFACTS_DIR=Trinotrotolueno/meta-tag-classifier/template_classifier
```

If the Hub repo is private, set the usual HuggingFace token environment variable before starting the worker, for example `HF_TOKEN`.

## Label mapping

The repo now includes this default `meta_tag_classifier/artifacts/label_mapping.json`:

```json
{
  "label2id": {
    "App Download Portal": 0,
    "Content Farm": 1,
    "News Scraper": 2,
    "Novels or Short Stories": 3,
    "Online Games & Trivia": 4,
    "Other": 5,
    "Parked/Holding": 6,
    "Tool": 7
  },
  "id2label": {
    "0": "App Download Portal",
    "1": "Content Farm",
    "2": "News Scraper",
    "3": "Novels or Short Stories",
    "4": "Online Games & Trivia",
    "5": "Other",
    "6": "Parked/Holding",
    "7": "Tool"
  }
}
```

At runtime, the loader **prefers `label_mapping.json`** when present. It falls back to `label2id`/`id2label` inside `model_config.json` or `training_metadata.json` only when no separate label mapping exists.

A minimal production `model_config.json` can therefore be:

```json
{
  "backend": "qwen_sequence_classifier",
  "base_model_name": "Qwen/Qwen2.5-3B-Instruct",
  "adapter_dir": "adapter",
  "input_text_source": "selected_text_only",
  "max_length": 512,
  "inference_batch_size": 8,
  "load_in_4bit": true,
  "label_mapping_file": "label_mapping.json"
}
```

The loader can infer `base_model_name` from `adapter_config.json` via `base_model_name_or_path` when using a HuggingFace-style artifact folder.

Do not commit large model files unless the deployment process expects the adapter to be shipped inside the repo. For local testing or dstack deployment, copy the trained adapter folder into `meta_tag_classifier/artifacts/adapter`, or point `ARTIFACTS_DIR` to `Trinotrotolueno/meta-tag-classifier/template_classifier`.

## Packaging Colab or HuggingFace artifacts

After training in Colab, the saved folder usually looks like:

```text
qwen3b_qlora_meta_tags_selected_text_only_no_cal/
  adapter/
  training_metadata.json
  anchor_unbiased_test_metrics.json
  anchor_unbiased_test_predictions.csv
```

A HuggingFace repo snapshot may instead look like:

```text
model_repo/
  adapter_config.json
  adapter_model.safetensors
  tokenizer.json
  tokenizer_config.json
  label_mapping.json
```

Package either layout into this repo format:

```bash
python scripts/package_qwen_artifacts.py \
  --trained-dir /path/to/qwen3b_qlora_meta_tags_selected_text_only_no_cal \
  --artifacts-out meta_tag_classifier/artifacts \
  --overwrite
```

The script writes:

```text
source adapter files -> meta_tag_classifier/artifacts/adapter
label_mapping.json   -> meta_tag_classifier/artifacts/label_mapping.json
model metadata       -> meta_tag_classifier/artifacts/model_config.json
```

It also ignores optimizer/trainer state files such as `optimizer.pt`, `scheduler.pt`, `rng_state.pth`, `trainer_state.json`, and `training_args.bin`, because they are not needed for inference.

## Install

```bash
pip install -U pip
pip install -e .
```

For GPU inference with 4-bit loading, the important dependencies are:

```text
torch
transformers
accelerate
peft
bitsandbytes
sentencepiece
safetensors
huggingface_hub
```

## Python usage

```python
import pandas as pd
from meta_tag_classifier import template_predictor

meta_rows = pd.DataFrame([
    {
        "target_domain": "example-games.com",
        "name": "title",
        "content_latest": "Play Free Online Games"
    },
    {
        "target_domain": "example-games.com",
        "name": "description",
        "content_latest": "Casual puzzles, arcade games, and multiplayer trivia in your browser."
    },
])

out = template_predictor(
    meta_rows,
    artifacts_dir = "meta_tag_classifier/artifacts",
    domain_col = "target_domain",
    meta_tag_name = "name",
    meta_tag_value = "content_latest",
)

print(out)
```

The dataframe returned by `template_predictor` includes:

```text
target_domain
selected_text
selected_name
predicted_label
predicted_proba
proba_pseudo
```

## CLI prediction

```bash
mtc predict \
  --artifacts meta_tag_classifier/artifacts \
  --input data/meta_tags.csv \
  --output data/predictions.csv
```

If `--text-col selected_text` is passed, the CLI skips long-to-wide meta-tag selection and sends that column directly to Qwen.

## Queue processor

The queue worker is still started with:

```bash
python processor.py
```

Important environment variables:

```bash
QUEUE_URL=http://100.98.79.5:4949/exchange-batch
QUEUE_API_KEY=...
QUEUE_KEY=META_TAGS_CLASSIFIER
BATCH_SIZE=5
EMPTY_QUEUE_SLEEP_SECONDS=60
ARTIFACTS_DIR=meta_tag_classifier/artifacts
# or ARTIFACTS_DIR=owner/model_repo for HuggingFace Hub
INFERENCE_BATCH_SIZE=8
MAX_LENGTH=512
LOAD_IN_4BIT=true
```

`BATCH_SIZE` controls how many queue jobs are popped at once. `INFERENCE_BATCH_SIZE` controls how many selected texts are sent through Qwen at once.

## dstack

The included `dstack.yml` installs requirements and runs `processor.py`. Qwen2.5-3B with a LoRA adapter should be run on GPU. Start conservatively with:

```text
INFERENCE_BATCH_SIZE=4 or 8
LOAD_IN_4BIT=true
```

Then increase the inference batch size only after checking VRAM usage.

## Training

Training is intentionally not implemented in this deployment repo.

Use the Colab Qwen training notebook/cell to create the QLoRA adapter. The important training design is:

- `AutoModelForSequenceClassification`
- base model: `Qwen/Qwen2.5-3B-Instruct`
- LoRA/QLoRA adapter
- input text: `selected_text` only
- no calibration split
- train/test split with `anchor_unbiased` as test

After training, package the artifacts with `scripts/package_qwen_artifacts.py` or upload the adapter files plus `label_mapping.json` to HuggingFace Hub and set `ARTIFACTS_DIR` to the repo id.

## Cleaning behavior

The original cleaning/selection logic remains in place:

- HTML/JS stripping
- Unicode normalization
- URL/email/phone/domain masking/removal
- repeated token and punctuation cleanup
- navigation/noise scoring
- title vs description selection
- fallback to title/description/OG fields when `selected_text` is empty

## Notes

- The softmax confidence is useful for ranking and triage, but it is not a calibrated probability unless you add a separate calibration procedure later.
- CPU inference is not recommended for production throughput.
- The model receives no target URL or domain context, by design.
