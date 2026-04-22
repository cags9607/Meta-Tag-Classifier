"""
META_TAGS classification processor.

Queue contract:
- one queue job represents one crawl
- each job contains N meta tags
- the worker runs one domain-level prediction per job
- the worker emits one flat result row per job
- the worker pushes the flat results array grouped with job ids/tokens

Expected row shape inside each job:
{ session_id, target_domain, name, content, timestamp }

Supported job payloads:
1) job["data"] is a list[dict] of rows
2) job["data"] is a dict containing one of:
   - rows
   - meta_tags
   - tags

Output rows:
- session_id
- target_url
- name
- selected_text
- predicted_label
- predicted_proba
- proba_pseudo
- timestamp
"""

from __future__ import annotations

import logging
import time
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core import MetaTagTemplateClassifier
from processor_config import ARTIFACTS_DIR, BATCH_SIZE, EMPTY_QUEUE_SLEEP_SECONDS
from processor_utils import pop, push

logging.basicConfig(level = logging.INFO, format = "%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("urllib3").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

classifier: Optional[MetaTagTemplateClassifier] = None

PREDICTION_INPUT_TAGS = {"title", "description", "og:title", "og:description"}


def safe_scalar(x: Any) -> Any:
    if x is None:
        return None

    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    if isinstance(x, np.integer):
        return int(x)

    if isinstance(x, np.floating):
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return None
        return x

    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return x

    if isinstance(x, pd.Timestamp):
        return x.isoformat()

    if isinstance(x, list):
        return [safe_scalar(v) for v in x]

    if isinstance(x, dict):
        return {k: safe_scalar(v) for k, v in x.items()}

    if isinstance(x, (str, int, bool)):
        return x

    return str(x)


def _extract_rows(job_data: Any) -> List[Dict[str, Any]]:
    if isinstance(job_data, list):
        return job_data

    if isinstance(job_data, dict):
        for key in ["rows", "meta_tags", "tags"]:
            rows = job_data.get(key)
            if rows is not None:
                if not isinstance(rows, list):
                    raise TypeError(f"job['data']['{key}'] must be a list")
                return rows

        if {"session_id", "target_domain", "name", "content", "timestamp"}.issubset(job_data.keys()):
            return [job_data]

    raise TypeError(
        "job['data'] must be either a list of meta-tag rows or a dict containing rows/meta_tags/tags"
    )


def _normalize_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows).copy()

    if df.shape[0] == 0:
        return pd.DataFrame(columns = ["session_id", "target_domain", "name", "content", "timestamp"])

    rename_map = {
        "content_latest": "content",
    }
    df = df.rename(columns = {k: v for k, v in rename_map.items() if k in df.columns})

    required = ["session_id", "target_domain", "name", "content", "timestamp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"missing_columns:{','.join(missing)}")

    df = df[required].copy()
    df["session_id"] = df["session_id"].astype(str)
    df["target_domain"] = df["target_domain"].astype(str)
    df["name"] = df["name"].astype(str)
    df["content"] = df["content"].fillna("").astype(str)
    df["timestamp"] = df["timestamp"].astype(str)
    df["name_norm"] = df["name"].str.strip().str.lower()

    return df


def _build_prediction_input(df_rows: pd.DataFrame) -> pd.DataFrame:
    if df_rows.shape[0] == 0:
        return pd.DataFrame(columns = ["target_domain", "name", "content_latest"])

    pred_df = (
        df_rows
        .loc[lambda x: x["name_norm"].isin(PREDICTION_INPUT_TAGS), ["target_domain", "name_norm", "content"]]
        .copy()
        .rename(columns = {
            "name_norm": "name",
            "content": "content_latest",
        })
        .drop_duplicates()
        .reset_index(drop = True)
    )

    return pred_df


def _job_to_result_row(df_rows: pd.DataFrame, pred_row: Dict[str, Any]) -> Dict[str, Any]:
    first_row = df_rows.iloc[0]

    return {
        "session_id": safe_scalar(first_row["session_id"]),
        "target_url": safe_scalar(first_row["target_domain"]),
        "name": safe_scalar(pred_row.get("selected_name")),
        "selected_text": safe_scalar(pred_row.get("selected_text")),
        "predicted_label": safe_scalar(pred_row.get("predicted_label")),
        "predicted_proba": safe_scalar(pred_row.get("predicted_proba")),
        "proba_pseudo": safe_scalar(pred_row.get("proba_pseudo")),
        "timestamp": safe_scalar(first_row["timestamp"]),
    }


def process_batch(batch_size: int = 1):
    global classifier

    if classifier is None:
        logger.info("Initializing MetaTagTemplateClassifier...")
        classifier = MetaTagTemplateClassifier(artifacts_dir = ARTIFACTS_DIR)
        logger.info(f"MetaTagTemplateClassifier initialized successfully (artifacts_dir={ARTIFACTS_DIR})")

    jobs = pop(batch_size = batch_size)
    n_jobs = len(jobs)

    if n_jobs == 0:
        logger.info("No jobs received from queue. Sleeping.")
        time.sleep(EMPTY_QUEUE_SLEEP_SECONDS)
        return

    logger.info(f"Processing {n_jobs} jobs")

    results: List[Dict[str, Any]] = []

    for job in jobs:
        try:
            rows_raw = _extract_rows(job.get("data", {}))
            df_rows = _normalize_rows(rows_raw)

            if df_rows.shape[0] == 0:
                logger.info(f"Job {job.get('id', '')}: no rows found in payload")
                continue

            pred_input = _build_prediction_input(df_rows)
            if pred_input.shape[0] == 0:
                logger.info(f"Job {job.get('id', '')}: no classifier-relevant tags were provided")
                continue

            pred_row = classifier.predict_one_domain(pred_input)
            if not pred_row:
                logger.info(f"Job {job.get('id', '')}: empty prediction returned")
                continue

            results.append(_job_to_result_row(df_rows, pred_row))

        except Exception as e:
            logger.error(f"Failed to process job {job.get('id', '')}: {type(e).__name__}: {e}")
            continue

    filename = f"meta_tag_results_{int(time.time())}.json"
    processed_jobs = [
        {
            "jobs": [{"id": job["id"], "token": job["token"]} for job in jobs],
            "filename": filename,
            "results": results,
        }
    ]

    processed_jobs = safe_scalar(processed_jobs)

    push(processed_jobs)
    logger.info(f"Pushed {len(results)} result rows for {n_jobs} jobs")


def main():
    logger.info("Starting META_TAGS processor...")
    while True:
        try:
            process_batch(batch_size = BATCH_SIZE)
        except Exception as e:
            logger.error(f"Error in process_batch: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
