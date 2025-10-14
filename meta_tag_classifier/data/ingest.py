# meta_tag_classifier/data/ingest.py
from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

def _clean_url_like(domain: str) -> str:
    if not isinstance(domain, str):
        return ""
    s = domain.strip()
    if s.startswith("https://"):
        s = s[len("https://"):]
    elif s.startswith("http://"):
        s = s[len("http://"):]
    return s.rstrip("/")

def long_to_wide_meta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a long metas CSV of the form:
        index, target_domain, name, content_latest, date_latest
    into a wide dataframe with columns:
        target_domain, title, og:title, description, og:description, ...
        plus title_meta, description_meta (title preferred over og:title, etc.)
    """
    pivot_df = df.copy()
    # normalize domain to match your Colab step
    pivot_df["target_domain"] = pivot_df["target_domain"].astype(str).map(_clean_url_like)

    # pivot wide (ignore date_latest if present)
    cols = ["target_domain", "name", "content_latest"]
    missing = [c for c in cols if c not in pivot_df.columns]
    if missing:
        raise ValueError(f"Missing required columns for long->wide: {missing}")

    pv = pivot_df.drop(columns=[c for c in ["date_latest"] if c in pivot_df.columns]) \
                 .pivot_table(index="target_domain",
                               columns="name",
                               values="content_latest",
                               aggfunc="first")

    # build title_meta / description_meta as you did in Colab
    # use .get with defaults to avoid KeyErrors
    title = pv.get("title")
    og_title = pv.get("og:title")
    desc = pv.get("description")
    og_desc = pv.get("og:description")

    # ensure columns exist to keep the fallback happy later
    if "title" not in pv.columns:
        pv["title"] = np.nan
        title = pv["title"]
    if "og:title" not in pv.columns:
        pv["og:title"] = np.nan
        og_title = pv["og:title"]
    if "description" not in pv.columns:
        pv["description"] = np.nan
        desc = pv["description"]
    if "og:description" not in pv.columns:
        pv["og:description"] = np.nan
        og_desc = pv["og:description"]

    pv["title_meta"] = np.where(title.isna() | (title.astype(str).str.strip() == ""),
                                og_title, title)
    pv["description_meta"] = np.where(desc.isna() | (desc.astype(str).str.strip() == ""),
                                      og_desc, desc)

    pv = pv.reset_index()
    # coerce to str where appropriate (cleaner handles empty strings)
    for c in ["title_meta", "description_meta", "title", "og:title", "description", "og:description"]:
        if c in pv.columns:
            pv[c] = pv[c].fillna("").astype(str)
    return pv
