#!/usr/bin/env python3
"""
make_submission_zip.py - build <team>_submission.zip in the structure the organisers require:

  <team>_submission.zip
  ├── output/
  │   ├── matching_results.tsv
  │   └── candidate_pairs.tsv
  ├── code/
  │   └── business_entity_resolution/
  │       ├── src/                 (all source + resources/ + lexicon_build/)
  │       ├── pipeline.py, local_eval.py
  │       ├── README.md
  │       └── requirements.txt
  └── Documentation_template.md    (filled in; numbers taken from the run's metrics.json)

Usage (after downloading the Kaggle outputs into one folder):
  python tools/make_submission_zip.py --outputs ~/Downloads/kaggle_output --test-dir dataset/test
"""
import argparse
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "code" / "business_entity_resolution"
SKIP = {"__pycache__", ".pytest_cache"}


def fill_doc(team, metrics):
    doc = (ROOT / "docs" / "Documentation_filled.md").read_text(encoding="utf-8")
    f = lambda k, fmt="{:.4f}": fmt.format(metrics[k]) if k in metrics else "[see metrics.json]"  # noqa: E731
    backend = {"xgb": "XGBoost", "lgbm": "LightGBM"}.get(metrics.get("model_backend"), "XGBoost / LightGBM")
    values = {
        "TEAM": team, "DATE": time.strftime("%Y-%m-%d"),
        "HELDOUT_F05": f("heldout_macro_f05_two_stage"), "STAGE1_F05": f("heldout_macro_f05_stage1_only"),
        "TAU1": f("tau1", "{:.2f}"), "TAU2": f("tau2", "{:.2f}"),
        "TRAIN_RECALL": f("train_blocking_recall"),
        "TEST_CANDIDATES": f("test_candidate_pairs", "{:,}"), "TEST_S1": f("test_s1_count", "{:,}"),
        "BACKEND": backend,
    }
    for k, v in values.items():
        doc = doc.replace("{{" + k + "}}", v)
    left = [line for line in doc.splitlines() if "{{" in line]
    if left:
        sys.exit(f"unfilled placeholders: {left}")
    return doc


def add_tree(zf, src, arc):
    for p in sorted(src.rglob("*")):
        if p.is_file() and not (SKIP & set(p.parts)) and p.suffix not in {".pyc", ".pkl"}:
            zf.write(p, f"{arc}/{p.relative_to(src)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", required=True, help="folder with matching_results.tsv, candidate_pairs.tsv (+ metrics.json)")
    ap.add_argument("--team", default="zamzon_ai")
    ap.add_argument("--test-dir", default=None, help="run the official validator against this test folder first")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_dir = Path(a.outputs).expanduser()
    match, cand = out_dir / "matching_results.tsv", out_dir / "candidate_pairs.tsv"
    for p in (match, cand):
        if not p.exists():
            sys.exit(f"missing {p}")
    metrics = json.loads((out_dir / "metrics.json").read_text()) if (out_dir / "metrics.json").exists() else {}
    if not metrics:
        print("note: no metrics.json next to the TSVs; run numbers in the document stay as '[see metrics.json]'")

    if a.test_dir:
        r = subprocess.run([sys.executable, str(ROOT / "utils" / "validate_submission.py"), "--matching", str(match),
                            "--candidate", str(cand), "--test-dir", str(a.test_dir)], capture_output=True, text=True)
        print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr)
        if r.returncode != 0:
            sys.exit(r.stdout + r.stderr)

    doc = fill_doc(a.team, metrics)
    zpath = Path(a.out) if a.out else ROOT / f"{a.team}_submission.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.write(match, "output/matching_results.tsv")
        zf.write(cand, "output/candidate_pairs.tsv")
        base = "code/business_entity_resolution"
        add_tree(zf, PKG / "src", f"{base}/src")
        for name in ("pipeline.py", "local_eval.py", "README.md", "requirements.txt"):
            zf.write(PKG / name, f"{base}/{name}")
        for p in sorted((ROOT / "jev").glob("*.py")) + [ROOT / "jev" / "README.md"]:
            zf.write(p, f"{base}/src/lexicon_build/{p.name}")
        zf.writestr("Documentation_template.md", doc)
    with zipfile.ZipFile(zpath) as zf:
        names = zf.namelist()
    print(f"wrote {zpath} ({zpath.stat().st_size / 1e6:.1f} MB, {len(names)} files)")
    for n in names:
        if n.count("/") <= 2 or n.endswith(("README.md", "requirements.txt", "pipeline.py")):
            print("  ", n)


if __name__ == "__main__":
    main()
