"""Builds erk_sub3a3.ipynb from erk_sub2_1.ipynb (friend's known-good notebook, never edited).

One change: train_s1_frac 0.3 -> 0.5, with a memory guard that decides from the EXTRA memory 0.5 needs on top of
the Kaggle-proven 0.3 (erk_sub3a's guard compared absolute estimates and would have fallen back to 0.3 on Kaggle).

Why 0.5: KY+KL benchmark, 3 runs each: friend 0.98276 (run-to-run range 0.0014), train_s1_frac 0.5 alone 0.98473
(+0.0020, every run above every baseline run). It adds only +6.9% feature-matrix rows (440,628 -> 471,173) because
most rows already exist at 0.3 as competitor pairs; it adds +30% training labels. orphan_frac stays 0.15
(0.19 measured +0.0005, within noise, and lowered singleton accuracy 0.984 -> 0.970).

    python tools/make_sub3a3.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC, DST = ROOT / "erk_sub2_1.ipynb", ROOT / "erk_sub3a3.ipynb"

GUARD = '''

# ---------------------------------------------------------------------------
# [sub3a3] training fraction: 0.5 when its EXTRA memory over the Kaggle-proven 0.3 fits
# ---------------------------------------------------------------------------
# Training holds X (float32, pairs x columns) plus copies of the fit and holdout-A rows (X[tr], X[va] per fold in
# stage 1; X2[fit], X2[holdout A] in stage 2). Estimated peak = (all pairs + fit/holdout-A pairs) x columns x 4 B.
# 0.3 is known to fit on Kaggle, so 0.5 is used only if its extra estimated peak uses at most half of the RAM left
# after 0.3's estimated peak and a reserve (polars frames, XGBoost buffers). Otherwise 0.3, exactly as before.
FEATURE_COLS = 88      # stage-1 + stage-2 columns of X on the friend's feature set
RESERVE_GB = 3.0


def _roles(cfg, u, frac):
    h = cfg["holdout_frac"]
    return np.where(u < h / 2, 1,                       # holdout A
           np.where(u < h, 2,                           # holdout B
           np.where(u < h + frac * (1 - h), 0, 3)))


def _estimate(role, s1_row, s23_row):
    r = role[s1_row]
    used = np.unique(s23_row[r < 3])
    kept = (r < 3) | np.isin(s23_row, used)             # same rule as the pairs filter in train()
    n = int(kept.sum())
    n_copied = int(((r == 0) | (r == 1))[kept].sum())
    return n, (n + n_copied) * FEATURE_COLS * 4


def choose_train_frac(cfg, u, s1_row, s23_row):
    """-> (frac, role). Tries the options above the floor (largest first); the floor (smallest option, proven on
    Kaggle) is used when none passes. Every estimate is logged so the next version can be tuned from real numbers."""
    options = sorted(set(cfg.get("train_s1_frac_options") or [cfg["train_s1_frac"]]), reverse=True)
    floor = options[-1]
    try:
        import psutil
        free = psutil.virtual_memory().available
    except Exception:
        free = None
    role0 = _roles(cfg, u, floor)
    n0, peak0 = _estimate(role0, s1_row, s23_row)
    headroom = (free - peak0 - RESERVE_GB * 1e9) if free is not None else -1
    log(f"train_s1_frac {floor} (proven): {n0:,} pairs, estimated peak {peak0 / 1e9:.1f} GB; free RAM "
        f"{(free or 0) / 1e9:.1f} GB -> headroom {headroom / 1e9:.1f} GB after a {RESERVE_GB:.0f} GB reserve")
    for frac in options[:-1]:
        role = _roles(cfg, u, frac)
        n, peak = _estimate(role, s1_row, s23_row)
        extra = peak - peak0
        fits = headroom > 0 and extra <= 0.5 * headroom
        log(f"train_s1_frac {frac}: {n:,} pairs (+{(n - n0) / max(n0, 1):.1%}), estimated extra peak "
            f"{extra / 1e9:.1f} GB vs half the headroom {max(headroom, 0) / 2e9:.1f} GB -> {'fits' if fits else 'too big'}")
        if fits:
            log(f"using train_s1_frac {frac}")
            return frac, role
    log(f"using train_s1_frac {floor} (the proven setting)")
    return floor, role0


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def train(cfg):'''

ROLE_OLD = '''    rng = np.random.default_rng(cfg["seed"])
    u = rng.random(len(s1))
    role = np.where(u < cfg["holdout_frac"] / 2, 1,                       # holdout A
            np.where(u < cfg["holdout_frac"], 2,                          # holdout B
            np.where(u < cfg["holdout_frac"] + cfg["train_s1_frac"] * (1 - cfg["holdout_frac"]), 0, 3)))
'''
ROLE_NEW = '''    rng = np.random.default_rng(cfg["seed"])
    u = rng.random(len(s1))
    frac, role = choose_train_frac(cfg, u, cands["s1_row"].to_numpy(), cands["s23_row"].to_numpy())
    cfg["train_s1_frac"] = frac   # the report records the fraction actually used
'''

EDITS = [
    (0, "# Amazon ML Challenge 2026 — Business Entity Resolution, submission 2 (Kaggle, 2× T4)",
        "# Amazon ML Challenge 2026 — Business Entity Resolution, submission 3a3 (Kaggle, 2× T4)\n\n"
        "**Submission 3a3 = submission 2 (`erk_sub2_1`) + one change:** `train_s1_frac` 0.5 instead of 0.3 when the "
        "extra memory it needs fits (checked just before the feature matrix is built; otherwise 0.3, the setting "
        "proven on Kaggle). Look for the `train_s1_frac` lines in the training log. Benchmark (KY+KL, 3 runs): "
        "+0.0020 macro F0.5 over submission 2."),
    (3, '    "train_s1_frac": 0.3,\n',
        '    "train_s1_frac": 0.3,\n'
        '    # [sub3a3] 0.5 is used when its extra memory over 0.3 fits (see choose_train_frac in pipeline.py)\n'
        '    "train_s1_frac_options": [0.5, 0.3],\n'),
    (14, "\n\n# ---------------------------------------------------------------------------\n# training\n"
         "# ---------------------------------------------------------------------------\ndef train(cfg):", GUARD),
    (14, ROLE_OLD, ROLE_NEW),
    (24, "  models. This is the RAM/time knob; 0.6 fits in 30 GB.",
         "  models. This is the RAM/time knob; the notebook uses 0.5 when its extra memory over 0.3 fits."),
]


def main():
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    for cell, old, new in EDITS:
        src = "".join(nb["cells"][cell]["source"])
        n = src.count(old)
        assert n == 1, f"cell {cell}: expected 1 match, found {n}: {old[:80]!r}"
        nb["cells"][cell]["source"] = src.replace(old, new)
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    DST.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {DST.name}: {len(EDITS)} edits")


if __name__ == "__main__":
    main()
