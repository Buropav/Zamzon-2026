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
import shutil
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
from src.features import prepare_files, prepare_side  # noqa: E402
from src.extra_feats import build_vocab  # noqa: E402
from src.translit import native_map_from_frames  # noqa: E402
from src.metrics import parse_gt, gt_diagnostics  # noqa: E402
from src.sampling import region_sample_keys, context_owners, DEFAULT_REGION_KEYS  # noqa: E402
from src.two_stage import (block_candidates, add_context, fit_pipeline, id_lists, mem_info,  # noqa: E402
                           predict_by_country, release_memory, split_by_country)


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
    s1_all, gt_all = s1, gt
    s1, s23, gt = region_sample_keys(s1_all, s2, s3, gt_all, a.regions, n_jobs=FE.N_JOBS, fork_map=FE._fork_map, log=log,
                                     orphan_frac=a.orphan_frac)
    del s2, s3
    gt_dict = parse_gt(gt)
    log(f"  S1={len(s1):,} S23={len(s23):,} true pairs={sum(map(len, gt_dict.values())):,}")
    log(f"  singleton share: {gt_diagnostics(gt_dict, s1, s23)['singleton_share']:.4f}")

    log("[2/5] Normalisation, blocking...")
    s1p, s23p = prepare_side(s1), prepare_side(s23)
    cands = block_candidates(s1p, s23p, log=log)
    core = None
    if a.context:  # owners of the unknown-region Source 2/3 records the sample retrieved: competitors, never trained on
        ctx, ctx_gt, ctx_di = context_owners(cands, s1p, s23p, s1_all, gt_all)
        s1p, cands, core = add_context(s1p, s23p, cands, prepare_side(ctx), relevant_di=ctx_di, log=log)
        gt_dict.update(parse_gt(ctx_gt))
        del ctx, ctx_gt
    del s1_all, gt_all
    s1_ids, s23_ids = s1p["entity_id"].to_numpy(), s23p["entity_id"].to_numpy()
    y = np.array([s23_ids[d] in gt_dict.get(s1_ids[q], ()) for q, d in zip(cands.qi, cands.di)], np.int32)
    n_true = np.array([len(gt_dict.get(e, ())) for e in s1_ids])
    log(f"  candidates={len(cands):,} (incl. context owners)")

    log("[3/5] Prefilter, pair features, two-stage GBDT...")
    model = fit_pipeline(cands, s1p, s23p, y, n_true, log=log, core=core)
    recall = model["recall_prefilter"]
    model.pop("pairs", None), model.pop("part", None)
    del cands, s1p, s23p, s1, s23, gt, gt_dict, y, n_true
    release_memory()
    log(f"  training done ({time.time() - t0:.0f}s, {mem_info()})")

    log("[4/5] Test inference...")
    ts1, ts2, ts3 = (read_tsv(test_dir / f"test_source{k}.tsv") for k in (1, 2, 3))
    if a.dev:
        ts1, ts2, ts3 = ts1.head(2000), ts2.head(10000), ts3.head(10000)
    ts23 = pd.concat([ts2, ts3], ignore_index=True)
    del ts2, ts3
    FE.NAME_VOCAB = build_vocab(re.sub(r"[^a-z0-9]+", " ", str(x).lower()) for x in ts1["business_name"])
    ids1, ids23, n_ts1 = ts1["entity_id"].to_numpy(), ts23["entity_id"].to_numpy(), len(ts1)
    # one country at a time (candidates never cross countries): the whole test set prepared at once needs ~20 GB
    spill = out_dir.parent / "test_by_country"
    parts = split_by_country(ts1, ts23, str(spill), log=log)
    del ts1, ts23
    release_memory()
    slim = ("business_name", "business_address", "ml_addr")
    cands, sel = predict_by_country(parts, model, lambda files: prepare_files(files, drop=slim), chunk_size=a.chunk_size,
                                    cache_dir=str(out_dir.parent / "stage2_cache"),
                                    block_budget_s=a.block_budget_min * 60, log=log)
    shutil.rmtree(spill, ignore_errors=True)

    log("[5/5] Writing outputs...")
    cand_out = pd.DataFrame({"source1_entity_id": ids1,
                             "candidate_entity_ids": id_lists(len(ids1), cands.qi.to_numpy(), cands.di.to_numpy(), ids23)})
    match_out = pd.DataFrame({"source1_entity_id": ids1,
                              "matched_entity_ids": id_lists(len(ids1), sel.qi.to_numpy(), sel.di.to_numpy(), ids23)})
    assert len(match_out) == n_ts1 and match_out.source1_entity_id.is_unique
    cand_out.to_csv(out_dir / "candidate_pairs.tsv", sep="\t", index=False)
    match_out.to_csv(out_dir / "matching_results.tsv", sep="\t", index=False)
    metrics = {"f05_heldout_two_stage": round(model["f05_test"], 4),
               "f05_heldout_stage1": round(model["f05_test_stage1"], 4),
               "tau1": round(model["t1"], 2), "tau2": round(model["t2"], 2), "decision_rule": model.get("policy"),
               "f05_heldout_thresholds_rule": round(model["f05_test_thresholds"], 4),
               "f05_heldout_expected_f_rule": round(model["f05_test_expected_f"], 4),
               "orphan_frac": a.orphan_frac,
               "blocking_recall_train": round(float(model["recall_blocking"]), 4),
               "prefilter_recall_train": round(float(recall), 4),
               "test_s1": n_ts1, "test_candidates": len(cands), "test_matches": len(sel),
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
    ap.add_argument("--regions", nargs="+", default=list(DEFAULT_REGION_KEYS))
    ap.add_argument("--chunk-size", type=int, default=100_000)
    ap.add_argument("--validator", default=None)
    ap.add_argument("--context", action="store_true", help="add context owners in training (notebook: USE_CONTEXT)")
    ap.add_argument("--block-budget-min", type=float, default=180, help="test blocking time budget (minutes)")
    ap.add_argument("--orphan-frac", type=float, default=0.19,
                    help="share of training Source 1 removed (their Source 2/3 stay as distractors): test density")
    ap.add_argument("--dev", action="store_true")
    main(ap.parse_args())
