from __future__ import annotations

import re
from typing import Any

import pandas as pd


TITLE_CANDIDATES = ["title", "og:title", "twitter:title"]
DESCRIPTION_CANDIDATES = ["description", "og:description", "twitter:description", "meta_description"]


def _clean_text(x: Any) -> str:
    x = "" if pd.isna(x) else str(x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def _word_count(x: str) -> int:
    x = _clean_text(x)
    if not x:
        return 0
    return len(x.split())


def _pick_best_nonempty(row: pd.Series, candidates: list[str]) -> tuple[str, str]:
    pairs = []
    for name in candidates:
        val = _clean_text(row.get(name, ""))
        if val:
            pairs.append((name, val))

    if not pairs:
        return "", ""

    # Prefer the longest available candidate within the family
    return max(pairs, key = lambda x: len(x[1]))


def _pick_selected_pair(
    row: pd.Series,
    nav_thr: float = 2.0,
    prefer_title: bool = True,
    short_title_word_threshold: int = 8,
) -> tuple[str, str]:
    """
    Return (selected_name, selected_text).

    Current policy:
    - prefer a sufficiently informative title when available
    - otherwise fall back to description
    - if description exists but title is too short/generic, prefer description
    - if only one family exists, use it

    nav_thr is kept in the signature for compatibility with the existing API,
    even if not used directly here.
    """
    best_title_name, best_title = _pick_best_nonempty(row, TITLE_CANDIDATES)
    best_desc_name, best_desc = _pick_best_nonempty(row, DESCRIPTION_CANDIDATES)

    title_wc = _word_count(best_title)
    desc_wc = _word_count(best_desc)

    if prefer_title:
        if best_title and title_wc >= short_title_word_threshold:
            return best_title_name, best_title
        if best_desc:
            return best_desc_name, best_desc
        if best_title:
            return best_title_name, best_title
    else:
        if best_desc:
            return best_desc_name, best_desc
        if best_title:
            return best_title_name, best_title

    return "", ""


def clean_metas(
    df: pd.DataFrame,
    nav_thr: float = 2.0,
    prefer_title: bool = True,
    short_title_word_threshold: int = 8,
) -> pd.DataFrame:
    """
    Expect a wide dataframe with one row per domain/page and columns such as:
    - target_domain
    - title
    - description
    - og:title
    - og:description
    - twitter:title
    - twitter:description
    - meta_description

    Returns the same dataframe plus:
    - selected_name
    - selected_text
    """
    out = df.copy()

    if "target_domain" not in out.columns:
        out["target_domain"] = ""

    selected_pairs = out.apply(
        lambda row: _pick_selected_pair(
            row,
            nav_thr = nav_thr,
            prefer_title = prefer_title,
            short_title_word_threshold = short_title_word_threshold,
        ),
        axis = 1,
    )

    out["selected_name"] = selected_pairs.map(lambda x: x[0] if isinstance(x, tuple) else "")
    out["selected_text"] = selected_pairs.map(lambda x: x[1] if isinstance(x, tuple) else "")

    out["selected_name"] = out["selected_name"].replace("", pd.NA)
    out["selected_text"] = out["selected_text"].replace("", pd.NA)

    return out
