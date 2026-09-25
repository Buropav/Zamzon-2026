# Business Entity Resolution Pipeline (zamzon_ai)

data → learned native-script map + multilingual normalisation (+ static lexicon) →
forward + reverse + exact-key blocking per (country, region) → GBDT prefilter →
stage-1 GBDT (pair features) → stage-2 GBDT (group context) →
two-threshold, globally one-to-one selection → `output/matching_results.tsv`, `output/candidate_pairs.tsv`

GBDT = XGBoost on a GPU (used for the submitted run), LightGBM on CPU-only machines.

## Setup
```bash
pip install -r requirements.txt
```

## Run (end to end)
```bash
python3 pipeline.py --train-dir ../../dataset/train --test-dir ../../dataset/test --output-dir ../../output
```
`--dev` runs a quick sanity pass on the first test rows. `--regions` chooses the training regions
(default: OR KY AR MO WI KERALA PUNJAB HARYANA ORISSA RAJASTHAN). The Kaggle notebook `entity_resolution.ipynb` runs the
same code; it is generated from these modules by `build_notebook.py` (repository root).

## Layout
| File | Role |
|---|---|
| `src/safe_io.py` | Safe TSV reading |
| `src/text.py` | Country labels, legal-form stripping, core names |
| `src/er_multilingual.py` | Script romanisation, canonical tokens, address parsing, multilingual pair features |
| `src/er_lexicon.py` + `src/resources/*.tsv` | Static lexicon, word classes, look-alike conflicts (no API at run time) |
| `src/translit.py` | Native-script → Latin word map learned from training ground-truth pairs |
| `src/extra_feats.py` | Address-number alignment, compact-name and word-difference pair features |
| `src/blocking.py` | Top-k sparse similarity helpers |
| `src/features.py` | Pairwise features |
| `src/groups.py` | Stage-2 group-context features |
| `src/sampling.py` | Density-preserving (whole-region) training sample |
| `src/two_stage.py` | Blocking (forward / reverse / exact-key views), prefilter, two-stage training, chunked two-pass inference |
| `src/metrics.py` | Macro F0.5, selection policy, threshold tuning |
| `local_eval.py` | Offline A/B evaluation on whole training regions |

## How the lexicon was built
Once, offline (see `jev/README.md` in the project root). An evaluation model judged single words or
short phrases taken from word counts; it never saw a record or decided a match. The pipeline only
reads the resulting TSV files.
