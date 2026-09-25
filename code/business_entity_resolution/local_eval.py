#!/usr/bin/env python3
"""
local_eval.py - honest offline estimate of Macro F0.5 on a TRAIN region slice (no test data).

Why a region slice: sampling random records thins out look-alike neighbours (chains, same-street
businesses) and overstates precision. Taking EVERY record of a few regions (Oregon, Kerala) keeps the
local density of the real data. All ground-truth matches of the chosen Source 1 records are added
even when their address has no region tag, so recall is not flattered either.

Split by Source 1 entity: 60% train the LightGBM, 20% tune (tau1, tau2), 20% report.
  python local_eval.py                     # current pipeline (two-stage, LightGBM)
  python local_eval.py --backend xgb       # same with XGBoost (GPU if available)
  python local_eval.py --mode baseline     # old normalisation, old feature set
"""
import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import src.features as FE  # noqa: E402
from src import er_multilingual as ML  # noqa: E402
from src.metrics import parse_gt, macro_f05, select_matches  # noqa: E402

REGIONS = {"US": re.compile(r",\s*(OR|Oregon)\s*$", re.I),
           "India": re.compile(r"kerala|,\s*KL\b|കേരളം", re.I)}


def load_slice(train_dir):
    def in_region(parts):
        rx = REGIONS.get(parts[3])
        return bool(rx and rx.search(parts[2]))

    def read(path, keep):
        rows = []
        with open(path, encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 4 and keep(p):
                    rows.append(p[:4])
        return pd.DataFrame(rows, columns=header[:4])

    s1 = read(train_dir / "train_source1.tsv", in_region)
    ids = set(s1.entity_id)
    gt_rows = []
    with open(train_dir / "train_ground_truth.tsv", encoding="utf-8") as f:
        f.readline()
        for line in f:
            a, _, b = line.rstrip("\n").partition("\t")
            if a in ids:
                gt_rows.append((a, b))
    gt = pd.DataFrame(gt_rows, columns=["source1_entity_id", "matched_entity_ids"])
    need = {x.strip() for m in gt.matched_entity_ids for x in m.split(",") if x.strip()}
    s23 = pd.concat([read(train_dir / f"train_source{k}.tsv", lambda p: in_region(p) or p[0] in need)
                     for k in (2, 3)], ignore_index=True)
    return s1, s23, gt


def set_mode(mode):
    if mode.startswith("baseline"):
        FE.prepare_ml = ML.prepare_ml
        FE._LEX = {"lex": {"*": {"tok": {"name": {}, "addr": {}}, "phr": {"name": [], "addr": []}}},
                   "token_class": {}, "conflicts": set()}


def run(mode, train_dir, backend="lgbm", seed=42):
    """Same code path as the notebook: block_candidates -> fit_pipeline (prefilter, pair features, two stages)."""
    import src.two_stage as T
    from src.extra_feats import build_vocab
    T.BACKEND = backend
    t0 = time.time()
    set_mode(mode)
    s1, s23, gt = load_slice(train_dir)
    gt_dict = parse_gt(gt)
    FE.NAME_VOCAB = build_vocab(re.sub(r"[^a-z0-9]+", " ", str(x).lower()) for x in s1["business_name"])
    s1p, s23p = FE.prepare_side(s1), FE.prepare_side(s23)
    cands = T.block_candidates(s1p, s23p)
    s1_ids, s23_ids = s1p.entity_id.to_numpy(), s23p.entity_id.to_numpy()
    y = np.array([s23_ids[d] in gt_dict.get(s1_ids[q], ()) for q, d in zip(cands.qi, cands.di)], np.int32)
    n_true = np.array([len(gt_dict.get(e, ())) for e in s1_ids])
    print(f"[{mode}/{backend}] S1={len(s1p):,} candidates={len(cands):,} recall={y.sum() / n_true.sum():.4f} "
          f"({time.time() - t0:.0f}s)", flush=True)
    t1 = time.time()
    m = T.fit_pipeline(cands, s1p, s23p, y, n_true, seed=seed)
    ctry = s1p.country.to_numpy()
    part, c = m["part"], m["pairs"]
    sel = select_matches(c[part[c.qi.to_numpy()] == 2], "score", m["t1"], m["t2"], one_to_one=True)
    res = {"mode": mode, "backend": backend, "f05_test": round(m["f05_test"], 4),
           "f05_test_stage1": round(m["f05_test_stage1"], 4), "tau1": m["t1"], "tau2": m["t2"],
           "train_seconds": round(time.time() - t1), "total_seconds": round(time.time() - t0)}
    for cc in np.unique(ctry):
        ids_c = np.flatnonzero((part == 2) & (ctry == cc))
        res[f"f05_{cc}"] = round(macro_f05(sel[np.isin(sel.qi, ids_c)], n_true, ids_c), 4)
    print(f"[{mode}/{backend}] RESULT {res}", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["lexicon", "baseline"], default="lexicon")
    ap.add_argument("--backend", choices=["lgbm", "xgb"], default="lgbm")
    ap.add_argument("--train-dir", default=str(HERE.parents[1] / "dataset" / "train"))
    a = ap.parse_args()
    run(a.mode, Path(a.train_dir), backend=a.backend)
