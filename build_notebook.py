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
### Learned transliteration, forward + reverse + exact-key blocking, GBDT prefilter, two-stage GBDT (XGBoost on GPU)

**Pipeline**:
- **Data**: read directly from the attached Kaggle dataset (`/kaggle/input/...`), no downloads.
- **Normalisation**: native-script names mapped word-for-word to Latin with a dictionary learned from the training
  ground truth; all Indic scripts + Urdu romanised otherwise; French ligatures/apostrophes, legal forms, landmarks,
  PIN/ZIP parsing, static per-country lexicon (cell 7). No API is called here.
- **Training data**: 10 whole regions (blocker region keys) + every unknown-region Source 2/3 record of the country,
  i.e. the same pools the test set is blocked in; split 60/20/20 by Source 1.
- **Blocking** per (country, state/region): char TF-IDF top-k forward (name, core name, name+address, address),
  reverse (each Source 2/3 record -> its best Source 1 records), and exact keys (house number + name).
- **Prefilter**: small GBDT on blocking similarities + fast features drops ~90% of candidates, keeps ~99.98% of true pairs.
- **Matching**: stage-1 GBDT on pair features (incl. address-number alignment that separates the generator's decoys
  from true copies) -> stage-2 GBDT with group context (out-of-fold); XGBoost on the GPU (CUDA).
- **Selection**: two thresholds (tau1, tau2) tuned on Macro F0.5, globally one-to-one.

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
try:
    import numpy as _np, xgboost as _xgb
    _xgb.XGBClassifier(n_estimators=1, device="cuda", tree_method="hist").fit(_np.array([[0.0], [1.0]]), _np.array([0, 1]))
    print(f"XGBoost {_xgb.__version__}: CUDA OK -> prefilter and both GBDT stages train on the GPU")
except Exception as _e:
    print(f"WARNING: XGBoost cannot use the GPU ({type(_e).__name__}: {_e}) -> it trains on CPU (slow). "
          "Set Accelerator = GPU T4 x2.")
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
import xgboost as xgb
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
os.environ.setdefault("ER_BACKEND", "xgb")    # XGBoost for the prefilter and both stages (CUDA on the GPU)
USE_CONTEXT = False  # training: add owners of unknown-region Source 2/3 records as competitors (see sampling.py)

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

# Cell 8: Scalable Country-Partitioned Blocking; native-script map learned from training pairs; generator-aware
# pair features (address-number alignment, compact names, token differences)
code(inline("blocking.py") + "\n\n\n" + inline("translit.py") + "\n\n\n" + inline("extra_feats.py"))

# Cell 9: Region keys (state / region) for regional blocking, then unified feature engineering
code(inline("geo.py") + "\n\n\n" + inline("features.py"))

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
TRAIN_REGIONS = ("PB",) if DEV_MODE else DEFAULT_REGION_KEYS
print("Loading training data...")
s1 = read_tsv(train_s1_path); s2 = read_tsv(train_s2_path); s3 = read_tsv(train_s3_path); gt = read_tsv(train_gt_path)
# word-for-word native-script -> Latin map, learned from ALL training ground-truth pairs (training data only)
NATIVE_MAP = native_map_from_frames(s1, pd.concat([s2, s3], ignore_index=True), gt)
print(f"Native-script map: {len(NATIVE_MAP['name']):,} name words, {len(NATIVE_MAP['addr'])} address components ({time.time()-T0:.0f}s)")
# Source 1 name vocabulary (full table, like the full test Source 1 at inference)
NAME_VOCAB = build_vocab(re.sub(r"[^a-z0-9]+", " ", str(x).lower()) for x in s1["business_name"])
# whole regions by the blocker's region key + every unknown-region Source 2/3 record of the country (test-like pools)
if DEV_MODE:  # smoke test: 10% of the records keeps every code path but runs in minutes on CPU
    s2, s3 = s2.sample(frac=0.1, random_state=0), s3.sample(frac=0.1, random_state=0)
S1_ALL, GT_ALL = s1, gt
s1, s23, gt = region_sample_keys(S1_ALL, s2, s3, GT_ALL, TRAIN_REGIONS, n_jobs=N_JOBS, fork_map=_fork_map)
del s2, s3; gc.collect()
gt_dict = parse_gt(gt)
print(f"Training regions {TRAIN_REGIONS}: S1={len(s1):,}, S23={len(s23):,}, true pairs={sum(map(len, gt_dict.values())):,}")
diag = gt_diagnostics(gt_dict, s1, s23)
for k in ("singleton_share", "singleton_share_by_country", "s23_claimed_by_more_than_one_s1"):
    print(f"  {k}: {diag[k]}")

s1p = prepare_side(s1); s23p = prepare_side(s23)
print(f"Prepared training tables ({time.time()-T0:.0f}s)")
cands = block_candidates(s1p, s23p, log=print)
core = None
if USE_CONTEXT:
    # owners (outside the sample) of the unknown-region Source 2/3 records the sample retrieved: scored as
    # competitors like at test time (where every Source 1 is present), never trained on or evaluated
    ctx, ctx_gt, ctx_di = context_owners(cands, s1p, s23p, S1_ALL, GT_ALL)
    s1p, cands, core = add_context(s1p, s23p, cands, prepare_side(ctx), relevant_di=ctx_di, log=print)
    gt_dict.update(parse_gt(ctx_gt))
    del ctx, ctx_gt
del S1_ALL, GT_ALL; gc.collect()
s1_ids, s23_ids = s1p["entity_id"].to_numpy(), s23p["entity_id"].to_numpy()
labels = np.array([s23_ids[d] in gt_dict.get(s1_ids[q], ()) for q, d in zip(cands.qi, cands.di)], np.int32)
n_true = np.array([len(gt_dict.get(e, ())) for e in s1_ids])
print(f"Blocked: {len(cands):,} pairs ({len(cands)/max(len(s1p),1):.1f} per Source 1)  ({time.time()-T0:.0f}s)")

model = fit_pipeline(cands, s1p, s23p, labels, n_true, core=core)   # prefilter -> pair features -> two-stage GBDT
recall = model["recall_prefilter"]
best_f05, t1, t2 = model["f05_test"], model["t1"], model["t2"]
print("Top stage-2 features:", ", ".join(model["stage2_importance"].head(10).index))
del cands, s1p, s23p, s1, s23, gt; gc.collect()
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

NAME_VOCAB = build_vocab(re.sub(r"[^a-z0-9]+", " ", str(x).lower()) for x in test_s1["business_name"])
SLIM = ("business_name", "business_address", "ml_addr")   # raw text not needed after cleaning
test_s1p = prepare_side(test_s1, drop=SLIM)
test_s23p = prepare_side(test_s23, drop=SLIM)
del test_s23; gc.collect()
print(f"Prepared test tables ({time.time()-T1:.0f}s)")

# blocking time projected from the speed measured on the training regions; optional views are dropped if the
# projection exceeds the budget, so the run always finishes inside the 12 h Kaggle session
TEST_BLOCK_BUDGET_S = 3 * 3600
project_blocking(test_s1p, test_s23p, budget_s=TEST_BLOCK_BUDGET_S)
print(f"Elapsed since start: {(time.time() - T0) / 60:.0f} min")
CHUNK_SIZE = 1_000 if DEV_MODE else 100_000   # Source 1 per chunk (bounds peak memory of the prefilter features)
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
    "train_blocking_recall": round(float(model["recall_blocking"]), 4),
    "train_recall_after_prefilter": round(float(model["recall_prefilter"]), 4),
    "train_pairs_after_prefilter": int(model["n_train_pairs"]),
    "train_regions": list(TRAIN_REGIONS),
    "model_backend": resolve_backend(), "xgboost_device": _xgb_device(), "gpu_blocking": _gpu() is not None,
    "test_s1_count": len(test_s1), "test_candidate_pairs": int(len(test_cands)), "test_matches": int(len(test_sel)),
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
}
with open(WORKING_DIR / "metrics.json", "w") as f:
    json.dump(metrics_data, f, indent=2)
print(json.dumps(metrics_data, indent=2))

_backend_name = f"XGBoost ({'GPU' if _xgb_device() == 'cuda' else 'CPU'})" if resolve_backend() == "xgb" else "LightGBM"
methodology_content = f\"\"\"# Business Entity Resolution - Methodology
### Team: zamzon_ai

## Summary
Learned native-script transliteration + multilingual normalisation, forward + reverse + exact-key blocking per
(country, region), a GBDT prefilter, a two-stage {_backend_name} matcher (pair features incl. address-number
alignment, then group context) and a two-threshold, globally one-to-one selection. Held-out Macro F0.5 on whole
training regions: **{model['f05_test']:.4f}** (stage 1 alone: {model['f05_test_stage1']:.4f}); training recall
after blocking {model['recall_blocking']*100:.2f}%, after the prefilter {model['recall_prefilter']*100:.2f}%.

## 1. Normalisation
- Native-script Source 2/3 names are word-for-word transliterations of the Source 1 name (same token count in
  99.99% of training pairs), so a word map learned from the training ground truth (1.3k words, 98.9% purity,
  96% token coverage of test native-script names) turns them back into the English words; state names written
  in native script map to their Latin component. Unmapped words fall back to rule-based romanisation.
- Legal forms, street types, landmarks, PIN/ZIP parsing, French ligatures/apostrophes, static per-country lexicon.

## 2. Blocking
Per (country, state/region) pool (US state read from whole address components, so OR/IN/ME/OK are recognised):
char 3-4gram TF-IDF top-k forward (name 30, core name 15, name+address 50, address 20), reverse (each Source 2/3
record -> its top Source 1 records: name+address 3, name 2, address 2) and exact keys (house number + name-word
prefix, house number + compact name). On a test-sized Indian pool (Karnataka) recall rose from 0.940 (old
3-view forward blocking) to 0.982. A small GBDT prefilter on blocking similarities and fast features then keeps
~10% of the candidates and ~99.98% of the true pairs.
Test: {len(test_cands):,} candidate pairs (after the prefilter) for {len(test_s1):,} Source 1 entities.

## 3. Matching model
- Stage 1: {_backend_name} on ~88 pair features: fuzzy name/core/address ratios, IDF cosines, phonetic keys,
  DBA/landmark handling, distinctive-word similarity, blocking similarities, and generator-aware features:
  address-number alignment (exact / small shift / one-digit typo / truncation - decoy records are near-copies of
  a Source 1 record with the house number shifted by a few units), compact-name similarity (domain / handle
  names), and word-difference classes (typo vs substituted real word vs unknown brand word).
- Stage 2: {_backend_name} on the stage-1 score plus group context: rank and margin inside the Source 1 group and
  among all Source 1 entities competing for the same Source 2/3 record, similarity to the best other candidate,
  support from other candidates at the same address / with the same house number. Out-of-fold stage-1 scores.
- Training data: every record of whole regions ({', '.join(TRAIN_REGIONS)}); split by Source 1:
  60% fit / 20% early stopping + threshold tuning / 20% held-out report.

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
