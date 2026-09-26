# Business Entity Resolution (ML Challenge 2026)

Match Source 2/3 business records to the deduplicated Source 1 records (US, India, and France,
which appears only in test). Metric: per-entity macro F0.5. Full problem text: `docs/problem_statement.md`.

## Layout
| Path | What |
|---|---|
| `erk_sub2_1.ipynb` | Friend's known-good notebook (submission 2). Never edited; the fallback submission. |
| `erk_sub3a3.ipynb` | Friend's notebook + training fraction 0.5 when its extra memory fits (built by `tools/make_sub3a3.py`) |
| `erk_sub4.ipynb`, `erk_sub5.ipynb` | NOT RECOMMENDED: include the native-script word map, which collapses India on a leaderboard-like benchmark (Ohio+Karnataka, 40% unmatched: erk_sub5 0.96648 vs friend 0.97630; India 0.94224 vs 0.96828). The map is learned from matched pairs, so matched native records are 93.6% fully mapped vs 63.8% of unmatched ones in train: the model learns "unmapped = no match" |
| `erk_sub6.ipynb` | `erk_sub5` without the native-script map: training fraction up to 1.0 + address-number features + same-house-number context + learned filler words. Ohio+Karnataka benchmark: **0.98224** vs friend 0.97630 (US 0.98590 vs 0.98272, India 0.97767 vs 0.96828). End-to-end on the French test: validator PASS, 5.4% empty, 3.21 matches per S1 |
| `erk_sub9.ipynb` | `erk_sub6` + the four blocking fixes below (stateparse, flatkey, translit, phonfix). Modules identical to the benchmarked ones; normalisation smoke-tested on 975k real test rows. End-to-end F0.5 not yet measured: keep `erk_sub6` as the fallback |
| `tools/patches/` | One script per change, applied to the friend's modules by both the benchmark (`bench/variant.sh`) and the notebook builder (`tools/make_sub4.py`), so a notebook runs byte-for-byte the benchmarked code |
| `bench/`, `aws/bench.sh` | Parallel benchmark kit for a big machine: leaderboard-like benchmarks (whole states, ~40% unmatched S2/S3); variants in `bench/variants.txt` |

### Blocking fixes after erk_sub6 (end-to-end F0.5 runs pending)
A true pair that blocking never proposes is lost for good. Measured on the Ohio+Karnataka benchmark (371,969 true pairs):

| Patch | What was wrong | Blocking misses |
|---|---|---|
| (erk_sub6) | | 5,867 |
| `stateparse` | State from the first state-like component: "Fl 1" (floor) read as Florida, "Carroll, Iowa" as Kerala (India lookup on US), "Washington, District of Columbia" as WA. Train pairs with conflicting S1/S2 states: US 1.44% -> 0.03%, India 0.39% -> 0.03% | 4,158 |
| `flatkey` | India: native-script names share the whole phonetic key with S1 ("prm prjkts") but each word is too common to count | 3,577 |
| `translit` | Malayalam final consonants dropped and its "t" written "r" ("limirrad"); Tamil "s"/"f" read as "ch"/"hp"; spelled-out LLP. Malayalam pairs with equal phonetic names 2.3% -> 70.6%, Tamil 27.7% -> 56.6% | 3,516 (Tamil Nadu: 3,466 -> 2,518) |
| `phonfix` | Phonetic key follows pronunciation (ventures ~ venchars, industries ~ indastrij, logistics ~ lojistiks). Native pairs with equal phonetic names 67.1% -> 85.0% | 3,192 (Tamil Nadu: 2,376) |
| `tools/` | Notebook builders (exact, asserted text edits on `erk_sub2_1.ipynb`), submission log helper |
| `aws/` | `run_notebook.sh`: run any notebook on an AWS instance with a RAM log |
| `code/business_entity_resolution/` | Retired old pipeline, reference only (see its README) |
| `docs/` | Problem statement, documentation template, notes |
| `utils/validate_submission.py` | Official output validator |
| `submissions/` | Submission log |

Rule for every new version: one change, benchmarked locally (same data, repeated runs, noise floor)
before it goes to Kaggle/AWS. No external data, APIs or LLM-built resources in any submission.
