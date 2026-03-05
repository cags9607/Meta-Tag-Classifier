"""
Meta-tag template classification processor.

Pulls jobs from queue key META_TAG_CLASSIFIER, expects LONG format input:
  target_domain, name, content_latest

Uses proba_pseudo as the output probability (renamed to "probability" in results).

Robustness:
- Every result row includes job_id (and job_token) to avoid any misalignment ambiguity.
- Error and success rows share the exact same schema (prediction fields are None on error).
"""

from __future__ import annotations

import time
import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from core import MetaTagTemplateClassifier
from processor_utils import pop, push
from processor_config import BATCH_SIZE, EMPTY_QUEUE_SLEEP_SECONDS, ARTIFACTS_DIR

# Logging
logging.basicConfig(level = logging.INFO, format = "%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("urllib3").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

classifier: Optional[MetaTagTemplateClassifier] = None


# ----------------------------
# Payload parsing
# ----------------------------
def _coerce_long_df(job_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Supported payload shape:
    {
  "target_domain": "example.com",
  "timestamp": "2026-03-05T00:00:00Z",
  "rows": [
    {"name": "title", "content_latest": "Text from title meta tag"},
    {"name": "description", "content_latest": "Text from description meta tag"},
    {"name": "og:title", "content_latest": "Text from og:title meta tag"},
    {"name": "og:description", "content_latest": "Text from og:description"}
  ]
    }
   
    """
    if not isinstance(job_data, dict):
        raise TypeError("job['data'] must be a dict")

    # accept 'rows' (recommended) or 'meta_tags' as alias
    rows = job_data.get("rows", None)
    if rows is None:
        rows = job_data.get("meta_tags", None)

    if rows is None:
        # If upstream sends a single long row at the top-level (rare), allow it.
        if all(k in job_data for k in ["target_domain", "name", "content_latest"]):
            return pd.DataFrame([{
                "target_domain": job_data["target_domain"],
                "name": job_data["name"],
                "content_latest": job_data["content_latest"],
            }])
        raise KeyError("Missing 'rows' (or 'meta_tags') in job data")

    if not isinstance(rows, list):
        raise TypeError("'rows' must be a list of dicts")

    df = pd.DataFrame(rows)

    # If job_data provides target_domain and rows don't include it, add it.
    if "target_domain" in job_data and "target_domain" not in df.columns:
        df["target_domain"] = job_data["target_domain"]

    return df


# ----------------------------
# Result schema (stable columns)
# ----------------------------
_RESULT_KEYS = [
    "job_id",
    "job_token",
    "target_domain",
    "timestamp",
    "predicted_label",
    "probability",
    "selected_text",
    "status",
    "error",
]


def _blank_result(job_id: str, job_token: str, target_domain: str, timestamp: Optional[str]) -> Dict[str, Any]:
    """
    Start with a stable schema, fill in later.
    """
    return {
        "job_id": job_id,
        "job_token": job_token,
        "target_domain": target_domain,
        "timestamp": timestamp,
        "predicted_label": None,
        "probability": None,
        "selected_text": None,
        "status": None,   # "ok" or "error"
        "error": None,
    }


def _as_stable_schema(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure output dict has all keys (and only those keys), in a consistent order.
    """
    return {k: d.get(k, None) for k in _RESULT_KEYS}


def process_batch(batch_size: int = 1):
    global classifier

    if classifier is None:
        logger.info("Initializing MetaTagTemplateClassifier...")
        classifier = MetaTagTemplateClassifier(artifacts_dir = ARTIFACTS_DIR)
        logger.info(f"MetaTagTemplateClassifier initialized (artifacts_dir={ARTIFACTS_DIR})")

    jobs = pop(batch_size = batch_size)
    n_jobs = len(jobs)

    if n_jobs == 0:
        logger.info("No jobs received from queue. Sleeping.")
        time.sleep(EMPTY_QUEUE_SLEEP_SECONDS)
        return

    logger.info(f"Processing {n_jobs} jobs")

    results: List[Dict[str, Any]] = []

    for job in jobs:
        job_id = str(job.get("id", ""))
        job_token = str(job.get("token", ""))
        data = job.get("data", {}) or {}

        timestamp = data.get("timestamp", None)
        target_domain = data.get("target_domain", "") or ""

        # Create a stable result shell immediately (misalignment-proof + schema-stable)
        res = _blank_result(job_id = job_id, job_token = job_token, target_domain = target_domain, timestamp = timestamp)

        try:
            df_long = _coerce_long_df(data)

            # Validate required columns for defaults
            needed = {"target_domain", "name", "content_latest"}
            missing = sorted(list(needed - set(df_long.columns)))
            if missing:
                # best-effort target_domain inference
                if not res["target_domain"] and "target_domain" in df_long.columns and len(df_long) > 0:
                    res["target_domain"] = str(df_long["target_domain"].iloc[0])

                res["status"] = "error"
                res["error"] = f"missing_columns:{','.join(missing)}"
                results.append(_as_stable_schema(res))
                continue

            # Ensure we have a target_domain for output (domain-level)
            if not res["target_domain"] and len(df_long) > 0:
                res["target_domain"] = str(df_long["target_domain"].iloc[0])

            # Empty rows -> error
            if df_long.shape[0] == 0:
                res["status"] = "error"
                res["error"] = "no_meta_rows"
                results.append(_as_stable_schema(res))
                continue

            pred_row = classifier.predict_one_domain(df_long)

            if not pred_row:
                res["status"] = "error"
                res["error"] = "empty_prediction"
                results.append(_as_stable_schema(res))
                continue

            # Success path
            res["target_domain"] = pred_row.get("target_domain", res["target_domain"])
            res["predicted_label"] = pred_row.get("predicted_label", None)
            res["selected_text"] = pred_row.get("selected_text", None)

            # Rename proba_pseudo -> probability (canonical for now)
            res["probability"] = pred_row.get("proba_pseudo", None)

            res["status"] = "ok"
            res["error"] = None
            results.append(_as_stable_schema(res))

        except Exception as e:
            res["status"] = "error"
            res["error"] = f"exception:{type(e).__name__}:{e}"
            results.append(_as_stable_schema(res))

    filename = f"results_{int(time.time())}.json"

    processed_jobs = [
        {
            "jobs": [{"id": job.get("id", ""), "token": job.get("token", "")} for job in jobs],
            "filename": filename,
            "results": results,
        }
    ]

    push(processed_jobs)

    ok = sum(1 for r in results if r.get("status") == "ok")
    err = len(results) - ok
    logger.info(f"Pushed {len(results)} results (ok={ok}, error={err}) back to queue")


def main():
    logger.info("Starting meta-tag template classifier processor...")
    while True:
        try:
            process_batch(batch_size = BATCH_SIZE)
        except Exception as e:
            logger.error(f"Error in process_batch: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
