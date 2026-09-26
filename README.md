# Business Entity Resolution (ML Challenge 2026)

Match Source 2/3 business records to the deduplicated Source 1 records (US, India, and France,
which appears only in test). Metric: per-entity macro F0.5. Full problem text: `docs/problem_statement.md`.

## Layout
| Path | What |
|---|---|
| `erk_sub2_1.ipynb` | Friend's known-good notebook (submission 2). Never edited; the fallback submission. |
| `erk_sub3a3.ipynb` | Friend's notebook + training fraction 0.5 when its extra memory fits (built by `tools/make_sub3a3.py`) |
| `erk_sub4.ipynb` | Big-RAM build (AWS r6i): training fraction up to 1.0 + address-number features + same-house-number context + native-script word map (`tools/make_sub4.py`). Verified end to end here: KY+KL F0.5 0.99108 (friend 0.98276), validator PASS; full French test runs (6.0% empty, 3.19 matches per S1) |
| `erk_sub5.ipynb` | `erk_sub4` + learned generator filler words (France fix). Verified end to end: KY+KL F0.5 0.99119, validator PASS; France 5.5% empty, 3.21 matches per S1 |
| `bench/`, `aws/bench.sh` | Parallel benchmark kit for a big machine: KY+KL and a leaderboard-like benchmark (whole states, ~40% unmatched S2/S3) |
| `tools/` | Notebook builders (exact, asserted text edits on `erk_sub2_1.ipynb`), submission log helper |
| `aws/` | `run_notebook.sh`: run any notebook on an AWS instance with a RAM log |
| `code/business_entity_resolution/` | Retired old pipeline, reference only (see its README) |
| `docs/` | Problem statement, documentation template, notes |
| `utils/validate_submission.py` | Official output validator |
| `submissions/` | Submission log |

Rule for every new version: one change, benchmarked locally (same data, repeated runs, noise floor)
before it goes to Kaggle/AWS. No external data, APIs or LLM-built resources in any submission.
