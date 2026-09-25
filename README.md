# Business Entity Resolution (ML Challenge 2026, team zamzon_ai)

Match Source 2/3 business records to the deduplicated Source 1 records (US, India, and France,
which appears only in test). Metric: per-entity macro F0.5. Full problem text: `docs/problem_statement.md`.

## Layout
| Path | What |
|---|---|
| `entity_resolution.ipynb` | Kaggle notebook (generated; do not edit by hand) |
| `build_notebook.py` | Generates the notebook from `code/business_entity_resolution/src/` |
| `code/business_entity_resolution/` | Submission package: `pipeline.py`, `src/`, `local_eval.py`, README, requirements |
| `jev/` | Offline build of the static lexicon in `src/resources/` (see `jev/README.md`) |
| `dataset/{train,test,dev}` | Challenge data (not versioned) |
| `utils/validate_submission.py` | Official output validator |
| `docs/` | Problem statement, documentation template, methodology notes |
| `output/` | Latest `matching_results.tsv` / `candidate_pairs.tsv` |
| `archive/` | Superseded scripts, old artifacts and backups |

## Common commands
```bash
uv sync                                   # environment
uv run python build_notebook.py           # regenerate entity_resolution.ipynb after editing src/
cd code/business_entity_resolution
uv run python pipeline.py --train-dir ../../dataset/train --test-dir ../../dataset/test --output-dir ../../output
uv run python local_eval.py --mode lexicon --stage2   # offline A/B on whole training regions
```
