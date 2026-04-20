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

ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "meta_tag_classifier/artifacts")
