"""Scores a run's output against a benchmark's held-out labels with the friend's evaluate.macro_f05.
    python bench/score.py <output dir> <bench dir> [validator.py]  -> JSON"""
import json
import os
import subprocess
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import macro_f05  # noqa: E402


def read(p):
    return pl.read_csv(p, separator="\t", infer_schema=False, quote_char=None)


def pairs(df, col):
    return (df.filter(pl.col(col).fill_null("") != "").with_columns(pl.col(col).str.split(",")).explode(col)
            .rename({"source1_entity_id": "s1", col: "s23"}).with_columns(pl.col("s23").str.strip_chars())
            .select("s1", "s23").unique())


out, bench = sys.argv[1], sys.argv[2]
gt = pairs(read(f"{bench}/gt/test_ground_truth.tsv"), "matched_entity_ids")
s1_ids = read(f"{bench}/dataset/test/test_source1.tsv")["entity_id"]
pred = pairs(read(f"{out}/matching_results.tsv"), "matched_entity_ids")
cand = pairs(read(f"{out}/candidate_pairs.tsv"), "candidate_entity_ids")
r = macro_f05(pred, gt, s1_ids)
r.update(candidate_pairs=cand.height, cand_per_s1=cand.height / len(s1_ids),
         candidate_recall=cand.join(gt, on=["s1", "s23"], how="semi").height / max(gt.height, 1))
if len(sys.argv) > 3:
    v = subprocess.run([sys.executable, sys.argv[3], "--matching", f"{out}/matching_results.tsv", "--candidate",
                        f"{out}/candidate_pairs.tsv", "--test-dir", f"{bench}/dataset/test"], capture_output=True, text=True)
    r["validator"] = "PASS" if v.returncode == 0 else "FAIL"
print(json.dumps(r))
