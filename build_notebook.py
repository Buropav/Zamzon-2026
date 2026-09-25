import nbformat
from pathlib import Path
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

cells = []
def md(src): cells.append(new_markdown_cell(src.strip()))
def code(src):
    cleaned = src.strip()
    py_lines = [l for l in cleaned.splitlines() if not l.strip().startswith('!') and not l.strip().startswith('%')]
    py_code = '\n'.join(py_lines)
    try:
        compile(py_code, '<cell>', 'exec')
    except SyntaxError as e:
        print(f"SYNTAX ERROR IN CELL:\n{e}\nLine {e.lineno}:")
        lines = py_code.splitlines()
        if e.lineno and e.lineno <= len(lines):
            print(lines[e.lineno-1])
        raise e
    cells.append(new_code_cell(cleaned))

src_dir = Path(__file__).resolve().parent / "code" / "business_entity_resolution" / "src"


def inline(name):
    """Module source as a notebook cell: drops <package-only> blocks and relative imports (in the
    notebook those names are globals defined by earlier cells)."""
    out, skip = [], False
    for line in (src_dir / name).read_text(encoding="utf-8").splitlines():
        if "# <package-only>" in line:
            skip = True
            continue
        if "# </package-only>" in line:
            skip = False
            continue
        if skip or line.lstrip().startswith("from ."):
            continue
        out.append(line)
    return "\n".join(out)


import sys as _sys
_sys.path.insert(0, str(src_dir.parent))
from src.er_lexicon import embed as _embed_resources  # noqa: E402

# Title
md("""
# Amazon ML Challenge 2026: Business Entity Resolution
### Multilingual Normalization + Static Lexicon, Multi-View Blocking, Two-Stage GBDT (XGBoost on GPU)

**Pipeline**:
- **Data**: read directly from the attached Kaggle dataset (`/kaggle/input/...`), no downloads.
- **Normalisation**: all major Indic scripts + Urdu romanised, French ligatures/apostrophes, legal forms, landmarks,
  PIN/ZIP parsing, plus a static per-country lexicon (Indic-script English loanwords, abbreviations, state codes,
  typos, OCR digit noise) embedded in cell 7. Built offline once; no API is called here.
- **Training data**: every record of whole regions (density preserved), split 60/20/20 by Source 1.
- **Blocking**: 3-view country-partitioned char TF-IDF; vectorisers fitted once per country.
- **Matching**: stage-1 GBDT on pair features -> stage-2 GBDT with group context (out-of-fold); XGBoost on GPU, LightGBM on CPU-only machines.
- **Selection**: two thresholds (tau1, tau2) tuned on Macro F0.5, globally one-to-one.
- **Inference**: Source 1 chunks, stage-2 inputs cached on disk (float16), all 1.73M Source 1 rows written.

Set `DEV_MODE = True` in the config cell for a quick sanity run.
""")

# Cell 1: Environment check (zero network required)
code("""
# [SETUP 1] Environment Verification (Zero Network Required, Pre-installed Packages Respected)
import sys, os, importlib.util

missing = []
for mod, pkg in [("rapidfuzz", "rapidfuzz>=3.6.0"), ("indic_transliteration", "indic-transliteration>=2.3.0")]:
    if not importlib.util.find_spec(mod):
        missing.append(pkg)
import shutil as _sh
if _sh.which("nvidia-smi") and not importlib.util.find_spec("cupy"):
    missing.append("cupy-cuda12x")

if missing:
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing, "-q", "--timeout", "5"])
    except Exception as e:
        print(f"Note: Running in offline mode without extra optional packages ({missing}). Native fallbacks will be used.")
print("Python packages ready.")
try:
    import cupy as _cp
    print(f"GPU: {_cp.cuda.runtime.getDeviceCount()} CUDA device(s) -> TF-IDF blocking (and XGBoost) run on GPU")
except Exception as _e:
    print(f"GPU not available ({type(_e).__name__}) -> blocking runs on CPU")
print(f"CPU cores: {os.cpu_count()} -> text cleaning and pair features run in parallel")
""")

# Cell 2: Imports
code("""
# [SETUP 2] Standard Imports
import os, sys, csv, time, json, zipfile, re, unicodedata, subprocess, gc
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer

from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

try:
    from indic_transliteration import sanscript as _S
    HAS_TRANSLIT = True
except Exception:
    _S = None
    HAS_TRANSLIT = False

print("Libraries imported successfully.")
""")

# Cell 3: Configuration & Path Discovery
code("""
# [CONFIG] Global Configuration & Direct Dataset Path Discovery
DEV_MODE = False  # Set to False for complete 1.73M test submission run
os.environ.setdefault("ER_USE_GPU", "1")      # GPU blocking (CPU fallback is automatic)
os.environ.setdefault("ER_BACKEND", "auto")   # auto: XGBoost on GPU if present, else LightGBM

IS_KAGGLE = Path("/kaggle/input").exists()
WORKING_DIR = Path("/kaggle/working") if IS_KAGGLE else Path.cwd()
OUTPUT_DIR = WORKING_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def find_tsv(filename, dev=False):
    roots = []
    if IS_KAGGLE:
        roots.extend([
            Path("/kaggle/input/zamzon-2026/student_resource/dataset"),
            Path("/kaggle/input/zamzon-2026/student_resource"),
            Path("/kaggle/input/zamzon-2026"),
            Path("/kaggle/input"),
        ])
    roots.extend([
        Path.cwd() / "dataset",
        Path.cwd(),
        Path.cwd().parent / "dataset",
        Path.cwd().parent,
    ])
    
    if dev:
        for r in roots:
            if r.exists():
                for p in r.rglob(filename):
                    if "dev" in p.parts and ".venv" not in p.parts:
                        return p
    for r in roots:
        if r.exists():
            for p in r.rglob(filename):
                if "dev" not in p.parts and ".venv" not in p.parts:
                    return p
    return None

train_s1_path = find_tsv("train_source1.tsv", dev=DEV_MODE)
train_s2_path = find_tsv("train_source2.tsv", dev=DEV_MODE)
train_s3_path = find_tsv("train_source3.tsv", dev=DEV_MODE)
train_gt_path = find_tsv("train_ground_truth.tsv", dev=DEV_MODE)

test_s1_path  = find_tsv("test_source1.tsv", dev=DEV_MODE)
test_s2_path  = find_tsv("test_source2.tsv", dev=DEV_MODE)
test_s3_path  = find_tsv("test_source3.tsv", dev=DEV_MODE)
val_script    = find_tsv("validate_submission.py")

print("Dataset Paths Discovered (Loaded directly from input):")
print(f"  Train S1: {train_s1_path}")
print(f"  Train S2: {train_s2_path}")
print(f"  Train S3: {train_s3_path}")
print(f"  Train GT: {train_gt_path}")
print(f"  Test S1:  {test_s1_path}")
print(f"  Test S2:  {test_s2_path}")
print(f"  Test S3:  {test_s3_path}")
print(f"  Validator:{val_script}")

# Safety assertions
for p, name in [(train_s1_path, "train_s1"), (test_s1_path, "test_s1")]:
    assert p is not None and p.exists(), f"CRITICAL: {name} path not found in mounted directories!"
print("Path verification complete.")
""")

# Cell 4: Safe TSV Reader
code(inline("safe_io.py"))

# Cell 5: Country labels, legal-form stripping, core names (text.py)
code(inline("text.py"))

# Cell 6: Multilingual Normalization & Preprocessing
code(inline("er_multilingual.py"))

# Cell 7: Static lexicon resources (built offline once; no API calls at run time)
code(inline("er_lexicon.py") + f"""


# ---- resources embedded by the generator (same content as src/resources/*.tsv) ----
_LEX_BLOB = "{_embed_resources(str(src_dir / 'resources'))}"
_LEX = load_embedded(_LEX_BLOB)
if not getattr(prepare_ml, "lexicon_wrapped", False):
    prepare_ml = country_aware(prepare_ml, globals(), _LEX)
print("Lexicon loaded:", {{c: len(v["tok"]["name"]) + len(v["tok"]["addr"]) + len(v["phr"]["addr"]) for c, v in _LEX["lex"].items()}},
      "| token classes:", len(_LEX["token_class"]), "| conflict pairs:", len(_LEX["conflicts"]))
""")

# Cell 8: Scalable Country-Partitioned Blocking
code(inline("blocking.py"))

# Cell 9: Unified Feature Engineering
code(inline("features.py"))

# Cell 10: Metric & Decision Policy
code(inline("metrics.py"))

# Cell 11: Group-context (stage-2) features
code(inline("groups.py"))

# Cell 12: Density-preserving training sample
code(inline("sampling.py"))

# Cell 13: Candidate generation, two-stage matcher, chunked inference
code(inline("two_stage.py"))

# Cell 14: Training
code("""
# [EXECUTION 1] Density-preserving training sample -> normalisation -> blocking -> two-stage GBDT (XGBoost on GPU)
T0 = time.time()
TRAIN_REGIONS = ("PUNJAB",) if DEV_MODE else DEFAULT_REGIONS
print("Loading training data...")
s1 = read_tsv(train_s1_path); s2 = read_tsv(train_s2_path); s3 = read_tsv(train_s3_path); gt = read_tsv(train_gt_path)
s1, s23, gt = region_sample(s1, s2, s3, gt, TRAIN_REGIONS)
del s2, s3; gc.collect()
gt_dict = parse_gt(gt)
print(f"Training regions {TRAIN_REGIONS}: S1={len(s1):,}, S23={len(s23):,}, true pairs={sum(map(len, gt_dict.values())):,}")
diag = gt_diagnostics(gt_dict, s1, s23)
for k in ("singleton_share", "singleton_share_by_country", "s23_claimed_by_more_than_one_s1"):
    print(f"  {k}: {diag[k]}")

s1p = prepare_side(s1); s23p = prepare_side(s23)
print(f"Prepared training tables ({time.time()-T0:.0f}s)")
cands = block_candidates(s1p, s23p, log=print)
s1_ids, s23_ids = s1p["entity_id"].to_numpy(), s23p["entity_id"].to_numpy()
labels = np.array([s23_ids[d] in gt_dict.get(s1_ids[q], ()) for q, d in zip(cands.qi, cands.di)], np.int32)
n_true = np.array([len(gt_dict.get(e, ())) for e in s1_ids])
recall = labels.sum() / max(n_true.sum(), 1)
print(f"Candidates: {len(cands):,}  blocking recall: {recall*100:.2f}%  ({time.time()-T0:.0f}s)")

X_train = pair_features(cands, s1p, s23p, workers=-1)
print(f"Feature matrix: {X_train.shape}  ({time.time()-T0:.0f}s)")
model = train_two_stage(cands, X_train, labels, n_true, s23p)
best_f05, t1, t2 = model["f05_test"], model["t1"], model["t2"]
print("Top stage-2 features:", ", ".join(model["stage2_importance"].head(10).index))
del X_train, cands, s1p, s23p, s1, s23, gt; gc.collect()
print(f"Training done in {time.time()-T0:.0f}s")
""")

# Cell 15: Test inference
code("""
# [EXECUTION 2] Test inference: pass 1 (blocking, features, stage 1) per chunk -> global group features
# -> pass 2 (stage 2) -> globally one-to-one selection
T1 = time.time()
test_s1 = read_tsv(test_s1_path); test_s2 = read_tsv(test_s2_path); test_s3 = read_tsv(test_s3_path)
if DEV_MODE:
    test_s1, test_s2, test_s3 = test_s1.head(2000), test_s2.head(20000), test_s3.head(20000)
test_s23 = pd.concat([test_s2, test_s3], ignore_index=True)
del test_s2, test_s3; gc.collect()
print(f"Test: S1={len(test_s1):,}, S23={len(test_s23):,}")
print("Countries:", test_s1["country"].value_counts().to_dict())

SLIM = ("business_name", "business_address", "ml_addr")   # raw text not needed after cleaning
test_s1p = prepare_side(test_s1, drop=SLIM)
test_s23p = prepare_side(test_s23, drop=SLIM)
del test_s23; gc.collect()
print(f"Prepared test tables ({time.time()-T1:.0f}s)")

CHUNK_SIZE = 1_000 if DEV_MODE else 250_000
test_cands, test_sel = predict_chunked(test_s1p, test_s23p, model, chunk_size=CHUNK_SIZE,
                                       cache_dir=str(WORKING_DIR / "stage2_cache"))

ts1_ids, ts23_ids = test_s1p["entity_id"].to_numpy(), test_s23p["entity_id"].to_numpy()
cand_out = pd.DataFrame({"source1_entity_id": ts1_ids,
                         "candidate_entity_ids": id_lists(len(ts1_ids), test_cands.qi.to_numpy(), test_cands.di.to_numpy(), ts23_ids)})
match_out = pd.DataFrame({"source1_entity_id": ts1_ids,
                          "matched_entity_ids": id_lists(len(ts1_ids), test_sel.qi.to_numpy(), test_sel.di.to_numpy(), ts23_ids)})
cand_file = OUTPUT_DIR / "candidate_pairs.tsv"
match_file = OUTPUT_DIR / "matching_results.tsv"
cand_out.to_csv(cand_file, sep="\\t", index=False)
match_out.to_csv(match_file, sep="\\t", index=False)

assert len(match_out) == len(test_s1) and match_out["source1_entity_id"].is_unique
assert len(cand_out) == len(test_s1)
n_matched = (match_out["matched_entity_ids"] != "").sum()
print(f"Saved {match_file.name}: {len(match_out):,} rows, {n_matched:,} non-empty")
print(f"Saved {cand_file.name}: {len(cand_out):,} rows, {len(test_cands):,} candidate pairs")
print("Non-empty share by country:", (match_out["matched_entity_ids"] != "").groupby(test_s1p["country"].to_numpy()).mean().round(3).to_dict())
print(f"Inference done in {time.time()-T1:.0f}s")
""")

# Cell 16: Metrics, methodology document & packaging
code("""
# [PACKAGING] Metrics, methodology document & submission zips
metrics_data = {
    "heldout_macro_f05_two_stage": round(float(model["f05_test"]), 4),
    "heldout_macro_f05_stage1_only": round(float(model["f05_test_stage1"]), 4),
    "tau1": round(float(t1), 2), "tau2": round(float(t2), 2),
    "train_blocking_recall": round(float(recall), 4),
    "train_regions": list(TRAIN_REGIONS),
    "model_backend": resolve_backend(), "gpu_blocking": _gpu() is not None,
    "test_s1_count": len(test_s1), "test_candidate_pairs": int(len(test_cands)), "test_matches": int(len(test_sel)),
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
}
with open(WORKING_DIR / "metrics.json", "w") as f:
    json.dump(metrics_data, f, indent=2)
print(json.dumps(metrics_data, indent=2))

_backend_name = {"xgb": "XGBoost (GPU)", "lgbm": "LightGBM"}.get(resolve_backend(), resolve_backend())
methodology_content = f\"\"\"# Business Entity Resolution - Methodology
### Team: zamzon_ai

## Summary
Multilingual normalisation with an offline-built static lexicon, 3-view country-partitioned TF-IDF
blocking, a two-stage {_backend_name} matcher (pair features, then group context) and a two-threshold,
globally one-to-one selection. Held-out Macro F0.5 on whole training regions: **{model['f05_test']:.4f}**
(stage 1 alone: {model['f05_test_stage1']:.4f}); training blocking recall {recall*100:.2f}%.

## 1. Normalisation
- Every Indic script + Urdu romanised (no names wiped), French ligatures/apostrophes handled, legal forms,
  street types, landmarks (near/opp/ke paas/en face de), PIN/ZIP parsing.
- Static lexicon (src/resources/*.tsv), applied per country (unknown countries use the script-level part):
  English words written in Indic scripts (kansaltents->consultants), abbreviations (r->rue, mh->maharashtra,
  state codes), state/phrase variants (tamil nadu->tn), frequent typos, OCR digit noise (hea1th->health),
  junk address tokens (null, na, pmb).
- The lexicon was built once, offline, from word statistics: candidates from training ground-truth swaps and
  vocabulary counts, judged word-by-word by an evaluation model (TypeSafe Jev). It only saw single words or
  short phrases, never a record, and never decided whether two businesses match. It is not called at run
  time; the pipeline reads static TSV files. Every rule was checked against training labels where they exist.

## 2. Blocking
Country-partitioned char 3-4gram TF-IDF (max_df=0.01, sparse products on the GPU via CuPy when available) on name (top 25), core name (top 15) and
name+address (top 20); union. Vectorisers are fitted once per country on Source 2/3.
Test: {len(test_cands):,} candidate pairs for {len(test_s1):,} Source 1 entities.

## 3. Matching model
- Stage 1: {_backend_name} on ~57 pair features (fuzzy name/core/address ratios, IDF cosines, postcode / house /
  unit agreement, phonetic keys, DBA and landmark handling, distinctive-word similarity, look-alike name
  conflict flag, blocker similarities).
- Stage 2: {_backend_name} on the stage-1 score plus group context: rank and margin inside the Source 1 group and
  among all Source 1 entities competing for the same Source 2/3 record, similarity to the best other
  candidate, same-address support. Stage-1 scores for stage-2 training are out-of-fold.
- Training data: every record of whole regions ({', '.join(TRAIN_REGIONS)}) to keep test-like density.
  Split by Source 1: 60% fit / 20% early stopping + threshold tuning / 20% held-out report.

## 4. Decision policy
tau1 = {t1:.2f} (best candidate), tau2 = {t2:.2f} (additional candidates); each Source 2/3 record is assigned
to at most one Source 1 entity (global one-to-one, matching the ground-truth structure).
\"\"\"
doc_path = WORKING_DIR / "methodology_document.md"
doc_path.write_text(methodology_content, encoding="utf-8")

sub_zip = WORKING_DIR / "submission_outputs.zip"
with zipfile.ZipFile(sub_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(match_file, arcname="output/matching_results.tsv")
    zf.write(cand_file, arcname="output/candidate_pairs.tsv")
    zf.write(doc_path, arcname="methodology_document.md")
print(f"Ready: {sub_zip.name} ({sub_zip.stat().st_size:,} bytes)")
""")

# Cell 17: Official Submission Validation
code("""
# [VALIDATION] Official Submission Validation
if val_script and Path(val_script).exists():
    res = subprocess.run([sys.executable, str(val_script), "--matching", str(match_file),
                          "--candidate", str(cand_file), "--test-dir", str(test_s1_path.parent)],
                         capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print("VALIDATION ERRORS:")
        print(res.stderr)
else:
    print(f"Validator not found; row-count assertions passed ({len(match_out):,} rows).")
""")

# Write notebook
nb = new_notebook(cells=cells)
for i, c in enumerate(nb.cells):  # stable ids -> reproducible notebook file
    c["id"] = f"cell-{i:02d}"
out_path = Path(__file__).resolve().parent / "entity_resolution.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Successfully generated clean notebook at {out_path.name} ({len(cells)} cells)!")
