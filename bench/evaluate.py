"""Entity-level macro F0.5, exactly as the challenge scores it."""
import numpy as np
import polars as pl


def macro_f05(pred_pairs, gt_pairs, s1_ids):
    """pred_pairs / gt_pairs: frames with columns (s1, s23) as entity-id strings.
    s1_ids: every S1 entity in the evaluation set (singletons included).
    Returns dict with macro F0.5, mean precision/recall and singleton stats.
    """
    base = pl.DataFrame({"s1": s1_ids}).unique()
    ids = set(base["s1"].to_list())
    pred = pred_pairs.select("s1", "s23").unique().filter(pl.col("s1").is_in(ids))
    gt = gt_pairs.select("s1", "s23").unique().filter(pl.col("s1").is_in(ids))
    tp = pred.join(gt, on=["s1", "s23"]).group_by("s1").len("tp")
    n_pred = pred.group_by("s1").len("np")
    n_true = gt.group_by("s1").len("nt")
    df = (base.join(n_pred, on="s1", how="left").join(n_true, on="s1", how="left")
          .join(tp, on="s1", how="left").fill_null(0))
    npred, ntrue, ntp = (df[c].to_numpy().astype(np.float64) for c in ("np", "nt", "tp"))
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(npred > 0, ntp / npred, 0.0)
        r = np.where(ntrue > 0, ntp / ntrue, 0.0)
        f = np.where((p + r) > 0, 1.25 * p * r / (0.25 * p + r), 0.0)
    f = np.where((npred == 0) & (ntrue == 0), 1.0, f)
    both = (npred > 0) & (ntrue > 0)
    return {
        "macro_f05": float(f.mean()),
        "precision": float(p[both].mean()) if both.any() else 0.0,
        "recall": float(r[both].mean()) if both.any() else 0.0,
        "n": int(len(f)),
        "singleton_acc": float(((npred == 0) & (ntrue == 0)).sum() / max((ntrue == 0).sum(), 1)),
        "pred_empty_rate": float((npred == 0).mean()),
    }
