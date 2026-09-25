# Business Entity Resolution Pipeline (zamzon_ai)

data → multilingual normalisation (+ static lexicon) → 3-view TF-IDF blocking →
stage-1 LightGBM (pair features) → stage-2 LightGBM (group context) →
two-threshold, globally one-to-one selection → `output/matching_results.tsv`, `output/candidate_pairs.tsv`

## Setup
```bash
pip install -r requirements.txt
```

## Run (end to end)
```bash
python3 pipeline.py --train-dir ../../dataset/train --test-dir ../../dataset/test --output-dir ../../output
```
`--dev` runs a quick sanity pass on the first test rows. `--regions` chooses the training regions
(default: OR KY AR KERALA PUNJAB HARYANA). The Kaggle notebook `entity_resolution.ipynb` runs the
same code; it is generated from these modules by `build_clean_notebook.py`.

## Layout
| File | Role |
|---|---|
| `src/safe_io.py` | Safe TSV reading |
| `src/text.py` | Country labels, legal-form stripping, core names |
| `src/er_multilingual.py` | Script romanisation, canonical tokens, address parsing, multilingual pair features |
| `src/er_lexicon.py` + `src/resources/*.tsv` | Static lexicon, word classes, look-alike conflicts (no API at run time) |
| `src/blocking.py` | Top-k sparse similarity helpers |
| `src/features.py` | Pairwise features |
| `src/groups.py` | Stage-2 group-context features |
| `src/sampling.py` | Density-preserving (whole-region) training sample |
| `src/two_stage.py` | Blocking views, two-stage training, chunked two-pass inference |
| `src/metrics.py` | Macro F0.5, selection policy, threshold tuning |
| `local_eval.py` | Offline A/B evaluation on whole training regions |

## How the lexicon was built
Once, offline (see `jev/README.md` in the project root). An evaluation model judged single words or
short phrases taken from word counts; it never saw a record or decided a match. The pipeline only
reads the resulting TSV files.
