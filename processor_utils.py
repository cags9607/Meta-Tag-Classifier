import logging

import requests

from processor_config import QUEUE_API_KEY, QUEUE_KEY, QUEUE_URL

logger = logging.getLogger(__name__)


def pop(batch_size: int = 1) -> list:
    """Fetch jobs from the queue."""
    headers = {"x-api-key": QUEUE_API_KEY}
    data = {
        "key": QUEUE_KEY,
        "get": batch_size,
    }

    try:
        logger.debug(f"Requesting {batch_size} jobs from queue")
        response = requests.post(url = QUEUE_URL, json = data, headers = headers)
        response.raise_for_status()
        jobs = response.json()["data"]["jobs"]
        logger.debug(f"Received {len(jobs)} jobs from queue")
        return jobs
    except Exception as e:
        logger.error(f"Failed to fetch jobs from queue: {e}")
        raise


def push(processed_jobs: list):
    """Push processed jobs back to the queue."""
    headers = {"x-api-key": QUEUE_API_KEY}
    data = {
        "key": QUEUE_KEY,
        "put": processed_jobs,
    }

    try:
        logger.debug(f"Pushing {len(processed_jobs)} processed jobs to queue")
        response = requests.post(url = QUEUE_URL, json = data, headers = headers)
        response.raise_for_status()
        logger.debug("Successfully pushed processed jobs to queue")
    except Exception as e:
        logger.error(f"Failed to push processed jobs to queue: {e}")
        raise
