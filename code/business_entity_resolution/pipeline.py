#!/usr/bin/env python3
"""
Business Entity Resolution pipeline - end-to-end CLI (same logic as the Kaggle notebook).

  data -> learned native-script map + multilingual normalisation (+ static lexicon)
       -> forward + reverse multi-view TF-IDF blocking -> GBDT prefilter
       -> stage-1 GBDT (pair features) -> stage-2 GBDT (group context)
       -> two-threshold, globally one-to-one selection -> output/*.tsv

Usage:
  python pipeline.py --train-dir ../../dataset/train --test-dir ../../dataset/test --output-dir ../../output
  python pipeline.py ... --dev          # quick sanity run on the first rows of the test files
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.safe_io import read_tsv  # noqa: E402
import src.features as FE  # noqa: E402
from src.features import prepare_side  # noqa: E402
from src.extra_feats import build_vocab  # noqa: E402
from src.translit import native_map_from_frames  # noqa: E402
from src.metrics import parse_gt, gt_diagnostics  # noqa: E402
from src.sampling import region_sample, DEFAULT_REGIONS  # noqa: E402
from src.two_stage import block_candidates, fit_pipeline, predict_chunked, id_lists  # noqa: E402


def main(a):
    t0 = time.time()
    log = lambda *x: print(*x, flush=True)  # noqa: E731
    train_dir, test_dir, out_dir = Path(a.train_dir), Path(a.test_dir), Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log("[1/5] Training sample (whole regions, density preserved)...")
    s1, s2, s3 = (read_tsv(train_dir / f"train_source{k}.tsv") for k in (1, 2, 3))
    gt = read_tsv(train_dir / "train_ground_truth.tsv")
    FE.NATIVE_MAP = native_map_from_frames(s1, pd.concat([s2, s3], ignore_index=True), gt)
    FE.NAME_VOCAB = build_vocab(re.sub(r"[^a-z0-9]+", " ", str(x).lower()) for x in s1["business_name"])
    log(f"  native-script map: {len(FE.NATIVE_MAP['name']):,} words; name vocabulary {len(FE.NAME_VOCAB):,}")
    s1, s23, gt = region_sample(s1, s2, s3, gt, a.regions)
    del s2, s3
    gt_dict = parse_gt(gt)
    log(f"  S1={len(s1):,} S23={len(s23):,} true pairs={sum(map(len, gt_dict.values())):,}")
    log(f"  singleton share: {gt_diagnostics(gt_dict, s1, s23)['singleton_share']:.4f}")

    log("[2/5] Normalisation, blocking...")
    s1p, s23p = prepare_side(s1), prepare_side(s23)
    cands = block_candidates(s1p, s23p, log=log)
    s1_ids, s23_ids = s1p["entity_id"].to_numpy(), s23p["entity_id"].to_numpy()
    y = np.array([s23_ids[d] in gt_dict.get(s1_ids[q], ()) for q, d in zip(cands.qi, cands.di)], np.int32)
    n_true = np.array([len(gt_dict.get(e, ())) for e in s1_ids])
    log(f"  candidates={len(cands):,}  blocking recall={y.sum() / max(n_true.sum(), 1):.4f}")

    log("[3/5] Prefilter, pair features, two-stage GBDT...")
    model = fit_pipeline(cands, s1p, s23p, y, n_true, log=log)
    recall = model["recall_prefilter"]
    del cands, s1p, s23p

    log("[4/5] Test inference...")
    ts1, ts2, ts3 = (read_tsv(test_dir / f"test_source{k}.tsv") for k in (1, 2, 3))
    if a.dev:
        ts1, ts2, ts3 = ts1.head(2000), ts2.head(10000), ts3.head(10000)
    ts23 = pd.concat([ts2, ts3], ignore_index=True)
    del ts2, ts3
    FE.NAME_VOCAB = build_vocab(re.sub(r"[^a-z0-9]+", " ", str(x).lower()) for x in ts1["business_name"])
    ts1p, ts23p = prepare_side(ts1), prepare_side(ts23)
    cands, sel = predict_chunked(ts1p, ts23p, model, chunk_size=a.chunk_size,
                                 cache_dir=str(out_dir.parent / "stage2_cache"), log=log)

    log("[5/5] Writing outputs...")
    ids1, ids23 = ts1p["entity_id"].to_numpy(), ts23p["entity_id"].to_numpy()
    cand_out = pd.DataFrame({"source1_entity_id": ids1,
                             "candidate_entity_ids": id_lists(len(ids1), cands.qi.to_numpy(), cands.di.to_numpy(), ids23)})
    match_out = pd.DataFrame({"source1_entity_id": ids1,
                              "matched_entity_ids": id_lists(len(ids1), sel.qi.to_numpy(), sel.di.to_numpy(), ids23)})
    assert len(match_out) == len(ts1) and match_out.source1_entity_id.is_unique
    cand_out.to_csv(out_dir / "candidate_pairs.tsv", sep="\t", index=False)
    match_out.to_csv(out_dir / "matching_results.tsv", sep="\t", index=False)
    metrics = {"f05_heldout_two_stage": round(model["f05_test"], 4),
               "f05_heldout_stage1": round(model["f05_test_stage1"], 4),
               "tau1": round(model["t1"], 2), "tau2": round(model["t2"], 2),
               "blocking_recall_train": round(float(model["recall_blocking"]), 4),
               "prefilter_recall_train": round(float(recall), 4),
               "test_s1": len(ts1), "test_candidates": len(cands), "test_matches": len(sel),
               "runtime_s": round(time.time() - t0)}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log(json.dumps(metrics, indent=2))

    val = Path(a.validator) if a.validator else HERE.parents[1] / "utils" / "validate_submission.py"
    if val.exists() and not a.dev:
        r = subprocess.run([sys.executable, str(val), "--matching", str(out_dir / "matching_results.tsv"),
                            "--candidate", str(out_dir / "candidate_pairs.tsv"), "--test-dir", str(test_dir)],
                           capture_output=True, text=True)
        log(r.stdout, r.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-dir", default="dataset/train")
    ap.add_argument("--test-dir", default="dataset/test")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--regions", nargs="+", default=list(DEFAULT_REGIONS))
    ap.add_argument("--chunk-size", type=int, default=250_000)
    ap.add_argument("--validator", default=None)
    ap.add_argument("--dev", action="store_true")
    main(ap.parse_args())
