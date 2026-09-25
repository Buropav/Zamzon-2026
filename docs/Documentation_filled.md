# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** {{TEAM}}  
**Team Members:** Anurup R Krishnan  
**Submission Date:** {{DATE}}

---

## 1. Executive Summary
A blocking + two-stage gradient-boosting pipeline. Records are normalised across scripts and languages
(all major Indic scripts, Urdu, French) with a static per-country lexicon, blocked with character n-gram
TF-IDF inside state/region pools, scored by a pair classifier and then re-scored by a second model that
sees the whole candidate group, and finally selected with two F0.5-tuned thresholds under a global
one-to-one constraint. Held-out macro F0.5 on whole training regions: **{{HELDOUT_F05}}**.

---

## 2. Methodology

### 2.1 Problem Analysis
- **Structure of the ground truth:** every Source 2/3 record belongs to at most one Source 1 entity
  (verified: 0 records claimed twice); ~5.5% of Source 1 entities are singletons; a matched entity has
  3-4 Source 2/3 records on average.
- **Scripts:** ~16-20% of Source 2/3 test records contain Indian scripts (Devanagari, Bengali, Gujarati,
  Gurmukhi, Oriya, Tamil, Telugu, Kannada, Malayalam) while Source 1 is Latin only. English business words
  are often written phonetically in those scripts ("कंसल्टेंट्स" -> "kansaltents" = consultants).
- **Names:** legal-suffix variants (Pvt/Private, Ltd/Limited, LLC, SARL/SAS/EURL), DBA/trade names, OCR-like
  digit-for-letter noise (hea1th, 6roup, lnc), word reordering, typos.
- **Addresses:** state codes vs names (MH/Maharashtra, TX/Texas), reordered components, landmarks
  ("near", "opp", "ke paas", "en face de"), placeholders ("null", "na"), French abbreviations (R = rue,
  Bd, Imp, Av), PIN codes split by spaces.
- **France** (15% of test Source 1) has no training labels; country must be treated as an open set.
- **Scale:** 1.73M Source 1 vs 9.97M Source 2/3 test records. Blocking against a whole country is both
  slow (cost per query grows with pool size) and less accurate: on India training data with the full 4.1M
  pool, top-k recall is 0.894 vs ~0.96 in region-sized pools, because look-alikes elsewhere crowd out the
  true match.

### 2.2 Solution Strategy
**Approach Type:** Blocking + two-stage classifier (pair model, then group-context model) + constrained selection  
**Core Innovation:** (1) region-partitioned blocking that matches the training setting and scales to the
test set; (2) a second-stage model using group context (rank/margin of a pair within its Source 1 group
and among all Source 1 entities competing for the same Source 2/3 record); (3) a static multilingual
lexicon that maps phonetically-transliterated English words and country-specific abbreviations to
canonical tokens.

---

## 3. Candidate Generation (Blocking)
- **Blocking keys used:** character 3-4-gram TF-IDF (char_wb, sublinear tf, max_df = 1% of the pool) on
  three views of the normalised record - name (top 20), core name without legal words (top 10) and
  name + address (top 40) - union of the three. Search is partitioned by country and by state/region read
  from the normalised address (codes, names, native-script names and major cities; Telangana and Andhra
  Pradesh merged). Records whose region is unknown are searched in the whole country; Source 2/3 records
  with unknown region are added to every regional pool. Sparse matrix products and top-k run on GPU (CuPy,
  split across all visible GPUs) with an automatic CPU fallback.
- **Candidate pairs generated:** {{TEST_CANDIDATES}} for {{TEST_S1}} Source 1 test entities.
- **How you ensured true matches were not lost:**
  - Region keys never split true pairs: on training data 100.00% (India) / 99.92% (US) of true pairs have
    the same region or an unknown region on one side.
  - Recall measured at test scale (India, 4.1M-record Source 2/3 pool): country-wide blocking 0.894 ->
    regional blocking 0.9245 -> regional with top-k 20/10/40: 0.9372. Top-k per view was chosen from this
    measurement (e.g. 15/10/30 gives 0.928 at the same cost as 25/15/20).
  - Training blocking recall (whole training regions): {{TRAIN_RECALL}}.

---

## 4. Matching Model

**Features used (~57 pair features, stage 1):**
- Name features: token-set / token-sort / partial / full / weighted ratios and Jaro-Winkler on the
  normalised name and on the core name, exact core match, IDF-weighted word cosine, abbreviation-aware
  soft token similarity (Monge-Elkan), Indic and French phonetic-key similarity, acronym match, DBA/trade
  name best match, similarity on distinctive words only (legal forms, titles and generic business words
  removed), look-alike conflict flag (names that differ by a known pair of *different* names, e.g.
  patel/patil), numbers-in-name conflict, name length ratio, core-name frequency (chain signal).
- Address features: token-set / partial / full ratios on the full and the landmark-free address,
  IDF-weighted word cosine, soft token similarity, postcode / house number / unit / Paris-Lyon-Marseille
  arrondissement agreement, number-set Jaccard, landmark presence and similarity, address lengths.
- Other: similarity from each blocking view, number of views that proposed the pair, Source 3 flag,
  script flags (non-Latin, cross-script).

**Stage 2 (group context):** stage-1 score (out-of-fold on the training part) plus rank, count and margin
of that score within the Source 1 group and among all Source 1 entities competing for the same Source 2/3
record, similarity to the best other candidate of the same Source 1 (name, address, postcode, house
number), and the best score among other candidates at the same address.

**Model type:** {{BACKEND}} gradient-boosted trees for both stages (XGBoost, Apache-2.0, on GPU; LightGBM,
MIT, on CPU-only machines - equal accuracy in an A/B: 0.9786 vs 0.9783). Training data: every record of
whole regions (Oregon, Kentucky, Arkansas, Kerala, Punjab, Haryana) to keep test-like density; split by
Source 1 entity 60% fit / 20% early stopping + threshold tuning / 20% held-out report.  
**Threshold selection method:** grid search of two thresholds on the tuning split with the exact
entity-level macro F0.5 (singletons included): tau1 = {{TAU1}} for each Source 1 entity's best candidate,
tau2 = {{TAU2}} for additional candidates, after a global one-to-one assignment (each Source 2/3 record
goes to the Source 1 entity that scores it highest).

**Preprocessing resources:** the lexicon (`src/resources/*.tsv`: ~5.3k per-country token/phrase
equivalences, word classes, 2.8k look-alike conflict pairs) was built once, offline. Candidates came from
training ground-truth token swaps and word counts; single words or short phrases (never a record, never a
match decision) were judged by an evaluation model (TypeSafe Jev), and every rule was checked against
training labels where they exist. The pipeline makes no API calls; it only reads these static files.
No external data about businesses was used.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** {{HELDOUT_F05}} held-out (two-stage; stage 1 alone {{STAGE1_F05}}). Local A/B
  on whole training regions (Oregon + Kerala): 0.9606 (baseline normalisation) -> 0.9668 (+ lexicon and new
  name features) -> 0.9786 (+ stage 2, XGBoost) -> 0.9807 (+ regional blocking, top-k 20/10/40).
- **Common false positives (wrong merges):** branches of the same chain or franchise at nearby addresses;
  different businesses in the same building or complex (same address, different unit); common or generic
  names (e.g. "Metro Management" vs "Metro Co Management") on the same street.
- **Common false negatives (missed matches):** records whose name is replaced by an unrelated trade name
  but share the exact address with the other matches (partly recovered by stage 2's same-address signal);
  heavily truncated or abbreviated names; true matches that blocking never proposes in very large regional
  pools (test-scale blocking recall ~0.94 for India).

---

## 6. Conclusion
Careful multilingual normalisation, blocking that stays accurate at test scale, and a second model that
reasons over the whole candidate group gave the largest gains; F0.5-tuned thresholds with a one-to-one
constraint keep precision high. The main remaining ceiling is blocking recall in very large regions, and
France, which has no labels, relies on country-agnostic features and the lexicon.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`:
- `pipeline.py` - end-to-end entry point: `python pipeline.py --train-dir <train> --test-dir <test> --output-dir output`
  writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` (and runs the official validator).
- `src/safe_io.py` (TSV reading), `src/text.py` (country labels, core names), `src/er_multilingual.py`
  (script romanisation, canonical tokens, address parsing, multilingual pair features), `src/er_lexicon.py`
  + `src/resources/` (static lexicon), `src/geo.py` (region keys), `src/blocking.py` + `src/two_stage.py`
  (blocking, two-stage training, chunked two-pass inference), `src/features.py` (pair features),
  `src/groups.py` (stage-2 features), `src/sampling.py` (whole-region training sample), `src/metrics.py`
  (macro F0.5, selection policy, threshold tuning).
- `src/lexicon_build/` - scripts that built `src/resources/` (reference only; not needed to reproduce the outputs).
- `local_eval.py` - offline A/B evaluation on whole training regions.
- The Kaggle notebook used for the submitted run is generated from the same modules.

### B. Additional Results
| Stage | Held-out macro F0.5 |
|---|---|
| Baseline normalisation, single model | 0.9606 |
| + static lexicon and new name features | 0.9668 |
| + stage 2 (group context) | 0.9786 |
| + regional blocking, top-k 20/10/40 | 0.9807 |
| Submitted run (6 training regions) | {{HELDOUT_F05}} |

| Blocking at test scale (India, 4.1M pool) | Recall | Candidates / Source 1 |
|---|---|---|
| Country-wide, top-k 25/15/20 | 0.894 | 44 |
| Regional, top-k 25/15/20 | 0.9245 | 41 |
| Regional, top-k 20/10/40 | 0.9372 | 55 |

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
