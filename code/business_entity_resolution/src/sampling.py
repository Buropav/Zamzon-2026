"""
sampling.py - density-preserving training sample.

Taking the first N rows (or random rows) of Source 1 and random Source 2/3 negatives gives a sparse
pool: the model rarely sees the look-alike neighbours (same street, same chain, same surname) that
fill every candidate list at test time, so validation F0.5 is inflated and thresholds are too loose.

Instead take EVERY record of a few whole regions (US states / Indian states), in every source, plus
all ground-truth matches of the chosen Source 1 records (so recall is measured honestly).
Region tags are matched on the raw address in all common spellings (code, name, native script).
"""
import re

import pandas as pd

REGION_PATTERNS = {
    # US: trailing state code or name
    "OR": ("US", r",\s*(?:OR|Oregon)\s*$"),
    "KY": ("US", r",\s*(?:KY|Kentucky)\s*$"),
    "AR": ("US", r",\s*(?:AR|Arkansas)\s*$"),
    "MO": ("US", r",\s*(?:MO|Missouri)\s*$"),
    "WI": ("US", r",\s*(?:WI|Wisconsin)\s*$"),
    # India: state names, codes and native-script names anywhere in the address
    "KERALA": ("India", r"kerala|,\s*KL\b|കേരളം"),
    "PUNJAB": ("India", r"punjab|panjab|,\s*PB\b|ਪੰਜਾਬ"),
    "HARYANA": ("India", r"haryana|hariyana|,\s*HR\b|हरियाणा"),
    "ORISSA": ("India", r"orissa|odisha|,\s*OD\b|ଓଡ଼ିଶା"),
    "RAJASTHAN": ("India", r"rajasthan|,\s*RJ\b|राजस्थान"),
}
DEFAULT_REGIONS = ("OR", "KY", "AR", "KERALA", "PUNJAB", "HARYANA")


def _mask(df, regions):
    m = pd.Series(False, index=df.index)
    ctry = df["country"].astype(str)
    addr = df["business_address"].astype(str)
    for r in regions:
        c, pat = REGION_PATTERNS[r]
        m |= (ctry == c) & addr.str.contains(pat, flags=re.I, regex=True)
    return m


def region_sample(s1, s2, s3, gt, regions=DEFAULT_REGIONS):
    """Returns (s1, s23, gt) restricted to whole regions (+ all GT matches of the chosen S1)."""
    s1r = s1[_mask(s1, regions)].reset_index(drop=True)
    ids = set(s1r["entity_id"])
    gtr = gt[gt.iloc[:, 0].isin(ids)].copy()
    need = {x.strip() for m in gtr.iloc[:, 1].fillna("") for x in str(m).split(",") if x.strip()}
    parts = [d[_mask(d, regions) | d["entity_id"].isin(need)] for d in (s2, s3)]
    s23 = pd.concat(parts, ignore_index=True).drop_duplicates("entity_id").reset_index(drop=True)
    return s1r, s23, gtr
