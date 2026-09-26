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
    # US: state code or name as a whole comma-separated component (any position)
    "OR": ("US", r"(?:^|,)\s*(?:OR|Oregon)\s*(?:,|$)"),
    "KY": ("US", r"(?:^|,)\s*(?:KY|Kentucky)\s*(?:,|$)"),
    "AR": ("US", r"(?:^|,)\s*(?:AR|Arkansas)\s*(?:,|$)"),
    "MO": ("US", r"(?:^|,)\s*(?:MO|Missouri)\s*(?:,|$)"),
    "WI": ("US", r"(?:^|,)\s*(?:WI|Wisconsin)\s*(?:,|$)"),
    # India: state names, codes and native-script names anywhere in the address
    "KERALA": ("India", r"kerala|,\s*KL\b|കേരളം"),
    "PUNJAB": ("India", r"punjab|panjab|,\s*PB\b|ਪੰਜਾਬ"),
    "HARYANA": ("India", r"haryana|hariyana|,\s*HR\b|हरियाणा"),
    "ORISSA": ("India", r"orissa|odisha|,\s*OD\b|ଓଡ଼ିଶା"),
    "RAJASTHAN": ("India", r"rajasthan|,\s*RJ\b|राजस्थान"),
}
DEFAULT_REGIONS = ("OR", "KY", "AR", "MO", "WI", "KERALA", "PUNJAB", "HARYANA", "ORISSA", "RAJASTHAN")


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


# ------------------------------------------------------------------ test-like pools (region keys)
# The blocker searches each Source 1 record in its (country, region) pool, and Source 2/3 records whose region is
# unknown (mostly EMPTY addresses) join every regional pool of their country. A training sample must look the
# same: whole regions by the blocker's own region key, plus ALL unknown-region Source 2/3 records of the country.
# Without them, an empty-address record only appears in training when it is a true match of a sampled Source 1,
# so the model learns "empty address + similar name = match" and over-matches at test time (measured on the
# Karnataka pool: 89% of false positives were empty-address records owned by a Source 1 in another state).
DEFAULT_REGION_KEYS = ("OR", "KY", "AR", "NC", "KL", "PB", "HR", "KA")  # incl. two dense ones (NC, KA)


def _region_keys_task(args):
    # <package-only>
    from .geo import region_key
    from .er_multilingual import normalize_ml
    from .text import canon_country
    # </package-only>
    countries, addrs = args
    return [region_key(canon_country(c), normalize_ml(a, "addr"), a) for c, a in zip(countries, addrs)]


def region_keys(df, n_jobs=None, fork_map=None):
    """Blocker region key of every record (parallel over CPU cores when fork_map is given)."""
    import numpy as np
    parts = np.array_split(np.arange(len(df)), max(1, (n_jobs or 1) * 8))
    items = [(df["country"].to_numpy()[i], df["business_address"].fillna("").to_numpy()[i]) for i in parts]
    res = fork_map(_region_keys_task, items, n_jobs) if fork_map and (n_jobs or 1) > 1 else [_region_keys_task(x) for x in items]
    return np.array([k for r in res for k in r], dtype=object)


def region_sample_keys(s1, s2, s3, gt, regions=DEFAULT_REGION_KEYS, n_jobs=None, fork_map=None, log=print,
                       orphan_frac=0.0, seed=0):
    """Test-like training sample -> (s1, s23, gt).

    Source 1: every record of the given region keys. Source 2/3: the records of those regions, the true matches
    of the chosen Source 1 wherever they are, and never-matched unknown-region records of the same countries at
    the sample's share of the country (so the sample has the full dataset's mix; an unknown-region record owned
    by a Source 1 OUTSIDE the sample would look unmatched here although its owner competes for it at test time).

    orphan_frac: share of the chosen Source 1 then REMOVED, their Source 2/3 records kept as unmatched
    distractors. Test Source 2/3 records are ~40% unmatched vs ~26% in train (5.8 vs 4.7 records per Source 1
    with the same records per business), i.e. ~19% of the businesses have no Source 1 record in test."""
    import numpy as np
    r1 = region_keys(s1, n_jobs, fork_map)
    s1r = s1[np.isin(r1, regions)].reset_index(drop=True)
    ctry = set(s1r["country"])
    share = (s1r["country"].value_counts() / s1["country"].value_counts()).dropna().to_dict()
    ids = set(s1r["entity_id"])
    gtr = gt[gt.iloc[:, 0].isin(ids)].copy()
    need = {x.strip() for m in gtr.iloc[:, 1].fillna("") for x in str(m).split(",") if x.strip()}
    owned = pd.Index(gt.iloc[:, 1].fillna("").astype(str).str.split(",").explode().str.strip().unique())
    rng = np.random.default_rng(seed)
    parts = []
    for d in (s2, s3):
        r = region_keys(d, n_jobs, fork_map)
        c = d["country"].to_numpy(object)
        lottery = rng.random(len(d)) < d["country"].map(share).fillna(0.0).to_numpy(float)
        unmatched = ~d["entity_id"].isin(owned).to_numpy()
        keep = (np.isin(r, regions) | d["entity_id"].isin(need).to_numpy()
                | ((r == "") & np.isin(c, list(ctry)) & unmatched & lottery))
        parts.append(d[keep])
    s23 = pd.concat(parts, ignore_index=True).drop_duplicates("entity_id").reset_index(drop=True)
    if orphan_frac > 0:
        drop = np.random.default_rng(seed + 1).random(len(s1r)) < orphan_frac
        s1r = s1r[~drop].reset_index(drop=True)
        gtr = gtr[gtr.iloc[:, 0].isin(set(s1r["entity_id"]))].copy()
    if log:
        mine = {x.strip() for m in gtr.iloc[:, 1].fillna("") for x in str(m).split(",") if x.strip()}
        unm = 1 - s23["entity_id"].isin(mine).mean()
        log(f"  region sample {regions}: S1={len(s1r):,} S23={len(s23):,} (orphaned {orphan_frac:.0%} of Source 1; "
            f"Source 2/3 without a Source 1 match: {unm:.1%}, test ~40%)")
    return s1r, s23, gtr


def context_owners(cands, s1p, s23p, s1_all, gt_all, min_name=0.8):
    """Source 1 records OUTSIDE the sample that own an unknown-region Source 2/3 record which some sampled
    Source 1 resembles by name (token-set ratio >= min_name, i.e. a real impostor threat). At test time every
    Source 1 of the country is present, so an empty-address record competes with its true owner; adding the
    owners as context records (scored, never trained on) gives training the same competition.
    -> (owners DataFrame, their ground-truth rows, positions in s23p of the relevant Source 2/3 records)."""
    import numpy as np
    from rapidfuzz import fuzz
    from rapidfuzz.process import cpdist
    unk = s23p["region"].to_numpy(object) == ""
    q, d = cands["qi"].to_numpy(), cands["di"].to_numpy()
    m = unk[d]
    q, d = q[m], d[m]
    sim = cpdist(s1p["norm_name"].to_numpy()[q], s23p["norm_name"].to_numpy()[d], scorer=fuzz.token_set_ratio,
                 workers=-1, dtype=np.float32) / 100.0
    di = np.unique(d[sim >= min_name])
    rset = set(s23p["entity_id"].to_numpy()[di])
    g = gt_all.iloc[:, 1].fillna("").astype(str).str.split(",")
    ex = pd.DataFrame({"a": gt_all.iloc[:, 0].to_numpy(), "b": g}).explode("b")
    ex = ex[ex["b"].isin(rset)]
    owners = set(ex["a"]) - set(s1p["entity_id"])
    ctx = s1_all[s1_all["entity_id"].isin(owners)].reset_index(drop=True)
    ctx_gt = gt_all[gt_all.iloc[:, 0].isin(owners)]
    return ctx, ctx_gt, di
