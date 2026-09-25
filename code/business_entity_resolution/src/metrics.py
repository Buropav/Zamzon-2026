import numpy as np
import pandas as pd

def parse_gt(gt):
    """Parses ground truth into {s1_id: set_of_matched_ids}."""
    return {
        a: {x.strip() for x in m.split(",") if x.strip()}
        for a, m in zip(gt.iloc[:, 0], gt.iloc[:, 1].fillna(""))
    }

def gt_diagnostics(gt_dict, s1, s23):
    """Computes dataset structural statistics."""
    pairs = [(a, b) for a, ms in gt_dict.items() for b in ms]
    claims = pd.Series([b for _, b in pairs], dtype=object).value_counts()
    c1 = dict(zip(s1.entity_id, s1.country))
    c23 = dict(zip(s23.entity_id, s23.country))
    n_match = pd.Series([len(gt_dict.get(e, ())) for e in s1.entity_id], index=s1.country.values)
    return {
        "s1_total": len(s1),
        "s23_total": len(s23),
        "singleton_share": float((n_match == 0).mean()),
        "singleton_share_by_country": (n_match == 0).groupby(level=0).mean().round(4).to_dict(),
        "matches_per_s1_hist": n_match.value_counts().sort_index().head(8).to_dict(),
        "s23_claimed_by_more_than_one_s1": int((claims > 1).sum()),
        "cross_country_true_pairs": int(sum(c1.get(a) != c23.get(b) for a, b in pairs)),
        "s23_never_matched_share": float(1 - len(claims) / max(len(s23), 1)),
        "gt_ids_missing_from_sources": int(sum(b not in c23 for _, b in pairs)),
    }

def macro_f05(pred, n_true, s1_ids):
    """Vectorized calculation of the official Entity-Level Macro F0.5.
    
    Includes singleton credit (f=1.0 for true singletons with 0 predictions).
    """
    agg = pred.groupby("qi")["y"].agg(["size", "sum"]) if not pred.empty else pd.DataFrame(columns=["size", "sum"])
    npred = agg["size"].reindex(s1_ids, fill_value=0).to_numpy(float)
    tp = agg["sum"].reindex(s1_ids, fill_value=0).to_numpy(float)
    nt = np.asarray(n_true, float)[s1_ids]
    
    p = np.divide(tp, npred, out=np.zeros_like(tp), where=npred > 0)
    r = np.divide(tp, nt, out=np.zeros_like(tp), where=nt > 0)
    den = 0.25 * p + r
    f = np.divide(1.25 * p * r, den, out=np.zeros_like(tp), where=den > 0)
    f[(nt == 0) & (npred == 0)] = 1.0
    return float(f.mean())

def select_matches(df, score_col, tau1, tau2, one_to_one=True):
    """Two-threshold policy with one-to-one S2/S3 assignment.
    
    - tau1: threshold for each S1's top candidate
    - tau2: threshold (>= tau1) for additional candidates
    - one_to_one: ensures each S2/S3 record is claimed only by the S1 scoring it highest
    """
    if df.empty:
        return df
    d = df
    if one_to_one:
        d = d.loc[d.groupby("di")[score_col].idxmax()]
    d = d[d[score_col] >= min(tau1, tau2)]
    if d.empty:
        return d
    rank = d.groupby("qi")[score_col].rank(ascending=False, method="first")
    keep = ((rank == 1) & (d[score_col] >= tau1)) | ((rank > 1) & (d[score_col] >= tau2))
    return d[keep]

def tune_policy(df, score_col, n_true, s1_ids, one_to_one=True,
                grid1=np.arange(0.20, 0.81, 0.05), grid2=np.arange(0.30, 0.96, 0.05)):
    """Grid searches optimal (tau1, tau2) on validation data using exact Macro F0.5."""
    best = (-1.0, 0.5, 0.6)
    for t1 in grid1:
        for t2 in grid2:
            if t2 < t1:
                continue
            sel = select_matches(df, score_col, t1, t2, one_to_one)
            f = macro_f05(sel, n_true, s1_ids)
            if f > best[0]:
                best = (f, float(t1), float(t2))
    return best
