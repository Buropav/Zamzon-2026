"""Where is macro F0.5 lost? Decomposes a run's loss (1 - F0.5 per S1, averaged over all S1) into recoverable causes:
  blocking_miss   true pair never in candidate_pairs.tsv          -> gain if those pairs were predicted
  model_reject    true pair was a candidate but not predicted      -> gain if predicted
  wrong_owned     predicted S2/S3 belongs to ANOTHER S1            -> gain if not predicted
  wrong_orphan    predicted S2/S3 belongs to no S1 (distractor)    -> gain if not predicted
Each gain is computed by fixing only that cause per entity (they overlap, so they do not sum exactly to the loss).
Segments use the raw test files: S2/S3 address empty, native-script name, country.
    python bench/error_analysis.py <output dir> <bench dir>"""
import re
import sys

import numpy as np
import polars as pl

out, bench = sys.argv[1], sys.argv[2]
rd = lambda p: pl.read_csv(p, separator="\t", infer_schema=False, quote_char=None)  # noqa: E731


def pairs(df, col):
    return (df.filter(pl.col(col).fill_null("") != "").with_columns(pl.col(col).str.split(",")).explode(col)
            .rename({"source1_entity_id": "s1", col: "s23"}).with_columns(pl.col("s23").str.strip_chars())
            .select("s1", "s23").unique())


gt = pairs(rd(f"{bench}/gt/test_ground_truth.tsv"), "matched_entity_ids")
pred = pairs(rd(f"{out}/matching_results.tsv"), "matched_entity_ids")
cand = pairs(rd(f"{out}/candidate_pairs.tsv"), "candidate_entity_ids")
t1 = rd(f"{bench}/dataset/test/test_source1.tsv").select(pl.col("entity_id").alias("s1"), "country")
t23 = pl.concat([rd(f"{bench}/dataset/test/test_source{k}.tsv") for k in (2, 3)])
NATIVE = r"[\x{0600}-\x{06FF}\x{0900}-\x{0DFF}]"
seg = t23.select(pl.col("entity_id").alias("s23"),
                 (pl.col("business_address").fill_null("").str.strip_chars().str.to_lowercase()
                  .is_in(["", "none", "null", "nan", "n/a", "<null>"])).alias("addr_empty"),
                 pl.col("business_name").fill_null("").str.contains(NATIVE).alias("native"))
owner = gt.rename({"s1": "owner"})
N = t1.height

tp = pred.join(gt, on=["s1", "s23"], how="semi")
fp = pred.join(gt, on=["s1", "s23"], how="anti").join(owner, on="s23", how="left")
fn = gt.join(pred, on=["s1", "s23"], how="anti").join(cand.with_columns(pl.lit(True).alias("in_cand")),
                                                      on=["s1", "s23"], how="left").with_columns(pl.col("in_cand").fill_null(False))
cnt = lambda df, name: df.group_by("s1").len(name)  # noqa: E731
E = (t1.join(cnt(gt, "nt"), on="s1", how="left").join(cnt(pred, "np"), on="s1", how="left").join(cnt(tp, "tp"), on="s1", how="left")
     .join(cnt(fp.filter(pl.col("owner").is_not_null()), "fp_owned"), on="s1", how="left")
     .join(cnt(fp.filter(pl.col("owner").is_null()), "fp_orphan"), on="s1", how="left")
     .join(cnt(fn.filter(~pl.col("in_cand")), "fn_block"), on="s1", how="left")
     .join(cnt(fn.filter(pl.col("in_cand")), "fn_model"), on="s1", how="left").fill_null(0))


def f05(tp_, np_, nt_):
    tp_, np_, nt_ = (np.asarray(x, float) for x in (tp_, np_, nt_))
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(np_ > 0, tp_ / np_, 0.0)
        r = np.where(nt_ > 0, tp_ / nt_, 0.0)
        f = np.where(p + r > 0, 1.25 * p * r / (0.25 * p + r), 0.0)
    return np.where((np_ == 0) & (nt_ == 0), 1.0, f)


c = {k: E[k].to_numpy() for k in E.columns if k not in ("s1", "country")}
base = f05(c["tp"], c["np"], c["nt"])
gains = {
    "blocking_miss": f05(c["tp"] + c["fn_block"], c["np"] + c["fn_block"], c["nt"]) - base,
    "model_reject": f05(c["tp"] + c["fn_model"], c["np"] + c["fn_model"], c["nt"]) - base,
    "wrong_owned": f05(c["tp"], c["np"] - c["fp_owned"], c["nt"]) - base,
    "wrong_orphan": f05(c["tp"], c["np"] - c["fp_orphan"], c["nt"]) - base,
}
print(f"S1 {N:,}  macro F0.5 {base.mean():.5f}  total loss {1 - base.mean():.5f}")
print(f"singletons {int((c['nt'] == 0).sum()):,}, predicted non-empty {int(((c['nt'] == 0) & (c['np'] > 0)).sum()):,} "
      f"(loss {(((c['nt'] == 0) & (c['np'] > 0)).sum()) / N:.5f})")
print("recoverable F0.5 by cause (fixing only that cause):")
for k, g in sorted(gains.items(), key=lambda kv: -kv[1].sum()):
    print(f"  {k:14s} +{g.sum() / N:.5f}   entities affected {int((g > 0).sum()):,}")
ctry = E["country"].to_numpy()
for cc in sorted(set(ctry)):
    m = ctry == cc
    print(f"  country {cc:6s}: F0.5 {base[m].mean():.5f} (n={int(m.sum()):,})  " +
          "  ".join(f"{k} +{g[m].sum() / N:.5f}" for k, g in gains.items()))
# pair-level segments of the errors
for name, df in (("missed true pairs (blocking)", fn.filter(~pl.col("in_cand"))),
                 ("missed true pairs (model)", fn.filter(pl.col("in_cand"))),
                 ("wrong predictions", fp), ("all true pairs", gt)):
    j = df.join(seg, on="s23", how="left")
    print(f"{name:30s} n={j.height:>8,}  addr-empty {j['addr_empty'].mean():.1%}  native {j['native'].mean():.1%}"
          + (f"  owned-by-other-S1 {j['owner'].is_not_null().mean():.1%}" if "owner" in j.columns else ""))
