"""
Configuration for the queue processor.
"""

import os

# Queue API configuration
QUEUE_API_KEY = os.getenv("QUEUE_API_KEY", "super-cool-api-key")
QUEUE_URL = os.getenv("QUEUE_URL", "http://100.98.79.5:4949/exchange-batch")

print(f"Using queue URL: {QUEUE_URL}")

# Processing configuration
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))
EMPTY_QUEUE_SLEEP_SECONDS = int(os.getenv("EMPTY_QUEUE_SLEEP_SECONDS", "60"))

# Meta-tag classifier configuration
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "meta_tag_classifier/artifacts")
