"""Builds erk_sub3a.ipynb from erk_sub2_1.ipynb (the friend's known-good notebook, never edited).

Two changes, nothing else:
1. orphan_frac 0.15 -> 0.19: test S2/S3 records are ~39.8% unmatched (row counts: 5.754 S2/S3 per test S1 vs
   3.461 true matches per S1 in train); 0.19 gives training the same density (measured 40.4% in the old pipeline).
2. train_s1_frac chosen at run time from [0.5, 0.4, 0.3]: the largest value whose estimated training peak fits in
   the RAM free at that moment. 0.3 (the value proven on Kaggle) is the floor and is always allowed.
Every text replacement must match exactly once, otherwise the build fails.

    python tools/make_sub3a.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC, DST = ROOT / "erk_sub2_1.ipynb", ROOT / "erk_sub3a.ipynb"

GUARD = '''

# ---------------------------------------------------------------------------
# [sub3a] training fraction sized to the RAM free at run time
# ---------------------------------------------------------------------------
# Training peak ~ feature matrix X (float32, pairs x columns) + one fold's boolean-index copies
# (X[tr], X[va]) + XGBoost's quantised copy; PEAK_FACTOR is a conservative multiple of X for that.
FEATURE_COLS = 88      # stage-1 + stage-2 columns of X on the friend's feature set
PEAK_FACTOR = 2.2
RAM_SHARE = 0.70       # use at most this share of the RAM that is free when the choice is made


def _roles(cfg, u, frac):
    h = cfg["holdout_frac"]
    return np.where(u < h / 2, 1,                       # holdout A
           np.where(u < h, 2,                           # holdout B
           np.where(u < h + frac * (1 - h), 0, 3)))


def choose_train_frac(cfg, u, s1_row, s23_row):
    """Largest option whose estimated training peak fits in RAM_SHARE of the free RAM; the smallest option
    (the Kaggle-proven 0.3) is used when none fits. Logs the estimate of every option it considers."""
    options = sorted(set(cfg.get("train_s1_frac_options") or [cfg["train_s1_frac"]]), reverse=True)
    try:
        import psutil
        free = psutil.virtual_memory().available
    except Exception:
        free = None
    for frac in options:
        role = _roles(cfg, u, frac)
        r = role[s1_row]
        used = np.unique(s23_row[r < 3])
        n_pairs = int(((r < 3) | np.isin(s23_row, used)).sum())
        est = n_pairs * FEATURE_COLS * 4 * PEAK_FACTOR
        fits = free is not None and est <= RAM_SHARE * free
        log(f"train_s1_frac {frac}: {n_pairs:,} training pairs, estimated peak {est / 1e9:.1f} GB vs budget "
            f"{(RAM_SHARE * free / 1e9) if free else float('nan'):.1f} GB ({RAM_SHARE:.0%} of free) -> "
            f"{'fits' if fits else 'too big'}")
        if fits or frac == options[-1]:
            log(f"using train_s1_frac {frac}" + ("" if fits else " (floor: proven on Kaggle)"))
            return frac, role


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
        "# Amazon ML Challenge 2026 — Business Entity Resolution, submission 3a (Kaggle, 2× T4)\n\n"
        "**Submission 3a = submission 2 (`erk_sub2_1`) + two changes:** `orphan_frac` 0.15 → 0.19 (test S2/S3 are "
        "~40% unmatched), and `train_s1_frac` chosen at run time from 0.5 / 0.4 / 0.3 by the RAM free before the "
        "feature matrix is built (0.3, the value proven on Kaggle, is the floor). Look for the `train_s1_frac` lines "
        "in the training log."),
    (3, '    "train_s1_frac": 0.3,\n',
        '    "train_s1_frac": 0.3,\n'
        '    # [sub3a] tried largest first; the first whose estimated peak fits in 70% of the free RAM is used,\n'
        '    # 0.3 (proven on Kaggle) when none fits. See choose_train_frac in pipeline.py.\n'
        '    "train_s1_frac_options": [0.5, 0.4, 0.3],\n'),
    (3, '    "orphan_frac": 0.15,\n', '    "orphan_frac": 0.19,   # [sub3a] was 0.15; test S2/S3 are ~40% unmatched\n'),
    (14, "\n\n# ---------------------------------------------------------------------------\n# training\n"
         "# ---------------------------------------------------------------------------\ndef train(cfg):", GUARD),
    (14, ROLE_OLD, ROLE_NEW),
    (24, "removing 15% of train S1 entities", "removing 19% of train S1 entities"),
    (24, "Orphaning 15% of train S1 entities", "Orphaning 19% of train S1 entities"),
    (24, "Training removes 15% of train S1 entities", "Training removes 19% of train S1 entities"),
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
