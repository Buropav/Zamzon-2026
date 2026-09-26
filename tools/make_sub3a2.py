"""Builds erk_sub3a2.ipynb: erk_sub3a without the orphan_frac change (it stays 0.15).

Benchmark (KY+KL, 3 runs each): training fraction 0.5 alone +0.0020 macro F0.5 over the friend's baseline;
orphan_frac 0.19 alone +0.0005 (within noise) and lowers singleton accuracy 0.984 -> 0.970.
So 3a2 keeps only the run-time training fraction (0.5/0.4/0.3 by free RAM, floor 0.3).

    python tools/make_sub3a2.py
"""
import json
import make_sub3a as base

HEADER = ("# Amazon ML Challenge 2026 — Business Entity Resolution, submission 3a2 (Kaggle, 2× T4)\n\n"
          "**Submission 3a2 = submission 2 (`erk_sub2_1`) + one change:** `train_s1_frac` chosen at run time from "
          "0.5 / 0.4 / 0.3 by the RAM free before the feature matrix is built (0.3, the value proven on Kaggle, is "
          "the floor). Look for the `train_s1_frac` lines in the training log. `orphan_frac` stays 0.15.")


def main():
    edits = []
    for cell, old, new in base.EDITS:
        if "orphan_frac" in old or cell == 24:
            continue                       # orphan_frac change and its doc-text edits are left out
        if cell == 0:
            new = HEADER
        edits.append((cell, old, new))
    nb = json.loads(base.SRC.read_text(encoding="utf-8"))
    for cell, old, new in edits:
        src = "".join(nb["cells"][cell]["source"])
        n = src.count(old)
        assert n == 1, f"cell {cell}: expected 1 match, found {n}: {old[:80]!r}"
        nb["cells"][cell]["source"] = src.replace(old, new)
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    dst = base.ROOT / "erk_sub3a2.ipynb"
    dst.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {dst.name}: {len(edits)} edits")


if __name__ == "__main__":
    main()
