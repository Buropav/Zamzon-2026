# Business Entity Resolution (ML Challenge 2026)

Match Source 2/3 business records to the deduplicated Source 1 records (US, India, and France,
which appears only in test). Metric: per-entity macro F0.5. Full problem text: `docs/problem_statement.md`.

## Layout
| Path | What |
|---|---|
| `erk_sub2_1.ipynb` | Friend's known-good notebook (submission 2). Never edited; the fallback submission. |
| `erk_sub3a3.ipynb` | Friend's notebook + training fraction 0.5 when its extra memory fits (built by `tools/make_sub3a3.py`) |
| `erk_sub4.ipynb`, `erk_sub5.ipynb` | NOT RECOMMENDED: include the native-script word map, which collapses India on a leaderboard-like benchmark (Ohio+Karnataka, 40% unmatched: erk_sub5 0.96648 vs friend 0.97630; India 0.94224 vs 0.96828). The map is learned from matched pairs, so matched native records are 93.6% fully mapped vs 63.8% of unmatched ones in train: the model learns "unmapped = no match" |
| `erk_sub6.ipynb` | `erk_sub5` without the native-script map: training fraction up to 1.0 + address-number features + same-house-number context + learned filler words. Ohio+Karnataka benchmark: **0.98224** vs friend 0.97630 (US 0.98590 vs 0.98272, India 0.97767 vs 0.96828) |
| `bench/`, `aws/bench.sh` | Parallel benchmark kit for a big machine: KY+KL and a leaderboard-like benchmark (whole states, ~40% unmatched S2/S3) |
| `tools/` | Notebook builders (exact, asserted text edits on `erk_sub2_1.ipynb`), submission log helper |
| `aws/` | `run_notebook.sh`: run any notebook on an AWS instance with a RAM log |
| `code/business_entity_resolution/` | Retired old pipeline, reference only (see its README) |
| `docs/` | Problem statement, documentation template, notes |
| `utils/validate_submission.py` | Official output validator |
| `submissions/` | Submission log |

Rule for every new version: one change, benchmarked locally (same data, repeated runs, noise floor)
before it goes to Kaggle/AWS. No external data, APIs or LLM-built resources in any submission.
