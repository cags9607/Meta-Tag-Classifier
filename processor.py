"""
Meta-tag template classification processor.

Pulls jobs from queue key META_TAG_CLASSIFIER, expects LONG format input:
  target_domain, name, content_latest

Domain-level only. No session_id. No target_url.
Uses proba_pseudo as the output probability (renamed to "probability" in results).
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


def _coerce_long_df(job_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Supports two payload shapes:

    Option 1 (preferred):
      { "target_domain": "...", "rows": [ {"name": "...", "content_latest": "..."}, ... ] }

    Option 2 (already-long rows):
      { "rows": [ {"target_domain": "...", "name": "...", "content_latest": "..."}, ... ] }
    """
    if not isinstance(job_data, dict):
        raise TypeError("job['data'] must be a dict")

    # accept 'rows' (recommended) or 'meta_tags' as alias
    rows = job_data.get("rows", None)
    if rows is None:
        rows = job_data.get("meta_tags", None)

    if rows is None:
        # If upstream sends long rows at the top-level (rare), allow it:
        # e.g. { "target_domain": "...", "name": "...", "content_latest": "..." } (single row)
        # but we strongly prefer list-of-rows.
        if all(k in job_data for k in ["target_domain", "name", "content_latest"]):
            df = pd.DataFrame([{
                "target_domain": job_data["target_domain"],
                "name": job_data["name"],
                "content_latest": job_data["content_latest"],
            }])
            return df
        raise KeyError("Missing 'rows' (or 'meta_tags') in job data")

    if not isinstance(rows, list):
        raise TypeError("'rows' must be a list of dicts")

    df = pd.DataFrame(rows)

    # If job_data provides target_domain and rows don't include it, add it.
    if "target_domain" in job_data and "target_domain" not in df.columns:
        df["target_domain"] = job_data["target_domain"]

    return df


def _build_error_result(target_domain: str, timestamp: Optional[str], error: str) -> Dict[str, Any]:
    return {
        "target_domain": target_domain,
        "timestamp": timestamp,
        "status": "error",
        "error": error,
    }


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
        data = job.get("data", {})
        timestamp = data.get("timestamp", None)

        # Try to infer target_domain early for better error reporting
        target_domain = data.get("target_domain", None)

        try:
            df_long = _coerce_long_df(data)

            # validate required columns for defaults
            needed = {"target_domain", "name", "content_latest"}
            missing = sorted(list(needed - set(df_long.columns)))
            if missing:
                td = target_domain or (df_long["target_domain"].iloc[0] if "target_domain" in df_long.columns and len(df_long) else "")
                results.append(_build_error_result(td, timestamp, f"missing_columns:{','.join(missing)}"))
                continue

            # ensure we have a target_domain for output (domain-level)
            if target_domain is None and len(df_long) > 0:
                target_domain = str(df_long["target_domain"].iloc[0])

            # empty rows -> error
            if df_long.shape[0] == 0:
                results.append(_build_error_result(target_domain or "", timestamp, "no_meta_rows"))
                continue

            pred_row = classifier.predict_one_domain(df_long)

            if not pred_row:
                results.append(_build_error_result(target_domain or "", timestamp, "empty_prediction"))
                continue

            # Rename proba_pseudo -> probability (canonical for now)
            probability = pred_row.get("proba_pseudo", None)

            results.append({
                "target_domain": pred_row.get("target_domain", target_domain),
                "timestamp": timestamp,
                "predicted_label": pred_row.get("predicted_label", None),
                "probability": probability,
                "selected_text": pred_row.get("selected_text", None),
                "status": "ok",
            })

        except Exception as e:
            results.append(_build_error_result(target_domain or "", timestamp, f"exception:{type(e).__name__}:{e}"))

    filename = f"results_{int(time.time())}.json"

    processed_jobs = [
        {
            "jobs": [{"id": job["id"], "token": job["token"]} for job in jobs],
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
