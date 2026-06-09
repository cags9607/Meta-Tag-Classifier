# meta_tag_classifier/data/ingest.py
from __future__ import annotations

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

def long_to_wide_meta(
    df: pd.DataFrame,
    *,
    domain_col: str = "target_domain",
    name_col: str = "name",
    value_col: str = "content_latest",
) -> pd.DataFrame:
    """
    Convert a long metas CSV of the form:
        index, <domain_col>, <name_col>, <value_col>[, ...]
    into a wide dataframe with columns:
        target_domain, title, og:title, description, og:description, ...
        plus title_meta, description_meta (title preferred over og:title, etc.)
    """
    pivot_df = df.copy()

    # Validate required columns
    missing = [c for c in (domain_col, name_col, value_col) if c not in pivot_df.columns]
    if missing:
        raise ValueError(f"Missing required columns for long->wide: {missing}")

    # Normalize domain into canonical 'target_domain'
    pivot_df["target_domain"] = pivot_df[domain_col].astype(str).map(_clean_url_like)

    # Pivot wide (names become columns)
    pv = pivot_df[["target_domain", name_col, value_col]].pivot_table(
        index="target_domain",
        columns=name_col,
        values=value_col,
        aggfunc="first",
    )

    # Build title_meta / description_meta like your Colab
    title = pv.get("title")
    og_title = pv.get("og:title")
    desc = pv.get("description")
    og_desc = pv.get("og:description")

    # Ensure base columns exist to simplify downstream logic
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

    pv["title_meta"] = np.where(
        title.isna() | (title.astype(str).str.strip() == ""), og_title, title
    )
    pv["description_meta"] = np.where(
        desc.isna() | (desc.astype(str).str.strip() == ""), og_desc, desc
    )

    pv = pv.reset_index()

    # Coerce to str where appropriate (cleaner handles empty strings)
    for c in ["title_meta", "description_meta", "title", "og:title", "description", "og:description"]:
        if c in pv.columns:
            pv[c] = pv[c].fillna("").astype(str)

    return pv
