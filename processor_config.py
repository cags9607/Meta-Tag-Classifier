"""
Configuration for the META_TAGS queue processor.
"""

import os

QUEUE_API_KEY = os.getenv("QUEUE_API_KEY", "super-cool-api-key")
QUEUE_URL = os.getenv("QUEUE_URL", "http://100.98.79.5:4949/exchange-batch")
QUEUE_KEY = os.getenv("QUEUE_KEY", "META_TAGS_CLASSIFIER")

print(f"Using queue URL: {QUEUE_URL}")
print(f"Using queue key: {QUEUE_KEY}")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))
EMPTY_QUEUE_SLEEP_SECONDS = int(os.getenv("EMPTY_QUEUE_SLEEP_SECONDS", "60"))

ARTIFACTS_DIR = os.getenv(
    "ARTIFACTS_DIR",
    "Trinotrotolueno/meta-tag-classifier/template_classifier"
)
# ARTIFACTS_DIR may be a local path or a HuggingFace Hub model repo/subfolder.
# Default hosted model folder:
#   Trinotrotolueno/meta-tag-classifier/template_classifier

# Qwen inference settings are read by meta_tag_classifier.models.qwen_infer.
INFERENCE_BATCH_SIZE = int(os.getenv("INFERENCE_BATCH_SIZE", "8"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "512"))
LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "true").lower() in {"1", "true", "yes", "y", "on"}

print(f"Using artifacts dir: {ARTIFACTS_DIR}")
print(f"Using Qwen inference batch size: {INFERENCE_BATCH_SIZE}")
print(f"Using Qwen max length: {MAX_LENGTH}")
print(f"Using Qwen 4-bit loading: {LOAD_IN_4BIT}")
