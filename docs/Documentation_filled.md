# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** {{TEAM}}  
**Team Members:** Anurup R Krishnan  
**Submission Date:** {{DATE}}

---

## 1. Executive Summary
A blocking + prefilter + two-stage gradient-boosting pipeline built around how the data was generated. Native-script
Source 2/3 names are mapped back to English with a word map learned from the training ground truth; candidates come
from forward, reverse and exact-key blocking inside state/region pools; a small GBDT prefilter drops ~90% of them;
a pair classifier with generator-aware features (address-number alignment that separates the planted "decoy"
records from true copies) is followed by a second model that sees the whole candidate group; matches are selected
with two F0.5-tuned thresholds under a global one-to-one constraint. Training pools are built exactly like test
pools. Held-out macro F0.5 on whole training regions: **{{HELDOUT_F05}}**.

---

## 2. Methodology

### 2.1 Problem Analysis
- **Ground-truth structure:** every Source 2/3 record belongs to at most one Source 1 entity; ~5.6% of Source 1
  entities are singletons; a matched entity has 3.5 Source 2/3 copies on average; 26% of Source 2/3 records belong
  to no Source 1 entity.
- **Decoys:** many unmatched Source 2/3 records are near-copies of a Source 1 record whose house number is shifted
  by a few units (2048 -> 2049, 4311 -> 4316, 32/2587/119 -> 32/2587/128) or whose name gets letters appended /
  one word swapped (Suryanth -> Suryantha, Card, Reed & Trejo -> Griffis, Reed & Trejo). True copies keep the number
  or damage it like a typo (digit dropped / inserted, leading zeros, 1/2 or -B suffixes, ranges). On US training
  candidates with near-identical names, pairs with the same house number are matches 99.3% of the time, pairs with
  the number shifted up by 1..13 only 3.4%.
- **Names:** legal-suffix variants, DBA / "formerly" aliases, fully replaced brand names (random pseudo-words),
  domain / handle names (almalindustries.com, @avilaliberty), titles (Mr, Dr, Shri), OCR noise (hea1th), typos.
  8.3% of Latin-script true pairs share no name word at all.
- **Scripts:** 7.3% of Source 2/3 names are written in an Indian script. They are word-for-word transliterations
  of the English Source 1 name (same token count in 99.99% of training pairs).
- **Addresses:** state code vs name, reordered components, landmarks, placeholders ("null"), empty addresses
  (2.6-3.4%). 97.7% of empty-address records are true copies of some Source 1.
- **France** (15% of test Source 1) has no training labels; generator-aware features are language-independent.
- **Scale and density:** test regional pools are large (Maharashtra: 180k Source 1 x 1.3M Source 2/3). Held-out
  scores on small training regions are optimistic: a model scoring 0.989 on a small-region held-out scored 0.975
  on the whole Karnataka pool (68.8k Source 1, never seen in training, with the true owners of all shared records present).

### 2.2 Solution Strategy
**Approach Type:** Blocking + prefilter + two-stage classifier (pair model, then group-context model) + constrained selection  
**Core Innovation:** (1) generator-aware features (address-number alignment, word-edit classes) that separate
decoys from true copies; (2) reverse and exact-key blocking that keep recall high in dense test-sized pools, made
affordable by a GBDT prefilter; (3) training pools built like test pools (whole regions by the blocker's own region
key, every unknown-region Source 2/3 record of the country, and the true owners of those records as context);
(4) a learned native-script -> English word map.

---

## 3. Candidate Generation (Blocking)
- **Blocking keys used:** per (country, state/region) pool, character 3-4-gram TF-IDF (char_wb, sublinear tf,
  max_df = 1% of the pool) top-k:
  - forward (each Source 1 -> its top Source 2/3 records): name 30, core name 15, name + address 50, address 20;
  - reverse (each Source 2/3 record -> its top Source 1 records): name + address 3, name 2, address 2. Source 1 is
    deduplicated, so a Source 2/3 record has few look-alikes there and its true owner is usually its top hit;
  - exact keys: house number + first 3 letters of a distinctive name word; house number + compact name
    (key values held by more than 30 Source 1 / 150 Source 2/3 records are skipped).
  Records whose region is unknown (mostly empty addresses) join every regional pool of their country for the
  forward views and run reverse retrieval once against every Source 1 of the country. The US state is read from
  whole address components, so the codes that are also English words (OR, IN, ME, OK, HI) are recognised.
  Sparse products and top-k run on GPU (CuPy, split across both GPUs) with a CPU fallback.
- **Prefilter:** a 300-tree GBDT on the blocking similarities, five rapidfuzz scores and address-number
  alignment keeps ~8-16% of the blocked pairs and 99.98% of the held-out true pairs (threshold chosen on held-out
  true pairs, clipped to [1e-4, 1e-3]). The prefilter output is the final candidate set (`candidate_pairs.tsv`).
- **Candidate pairs generated:** {{TEST_CANDIDATES}} for {{TEST_S1}} Source 1 test entities.
- **How you ensured true matches were not lost:**
  - Recall measured on a test-sized pool (Karnataka, 68.8k Source 1 x 577k Source 2/3): old 3-view forward
    blocking 0.940 -> + address view and reverse retrieval 0.965 -> + exact keys 0.982.
  - Region keys never split true pairs (0.006% of training true pairs have two different known regions).
  - Training recall after blocking {{TRAIN_RECALL}}.

---

## 4. Matching Model

**Features used (~100 pair features, stage 1):**
- Name features: token-set / token-sort / partial / full / weighted ratios and Jaro-Winkler on the normalised and
  core name, exact core match, IDF-weighted word cosine, abbreviation-aware soft token similarity, Indic and French
  phonetic keys, acronym match, DBA / trade-name best match, distinctive-word similarity, compact-name similarity
  (spaces and legal words removed: domain / handle names), word-difference classes (typo of a word on the other
  side / a real Source 1 vocabulary word / an unknown brand word; letters appended, anagram, one or two edits),
  look-alike conflict flag, numbers-in-name conflict, core-name frequency (per 100k records of the country).
- Address features: token-set / partial / full ratios on the full and the landmark-free address, IDF cosine, soft
  token similarity, postcode / house / unit agreement, landmark similarity, and **address-number alignment**:
  numbers of both addresses matched as exact, small shift up / down (1-20), mid shift (21-100), one-digit typo,
  prefix / suffix truncation, leftovers, plus the signed first-number delta.
- Other: similarity from each blocking view (forward, reverse, keys), number of views, Source 3 flag, script flags.

**Stage 2 (group context):** stage-1 score (out-of-fold) plus rank, count and margin of that score within the
Source 1 group and among all Source 1 entities competing for the same Source 2/3 record (computed globally at
inference), similarity to the best other candidate, and the support of other candidates at the same address and
with the same house number (true copies share the Source 1 number; a decoy's shifted number is alone).

**Model type:** {{BACKEND}} gradient-boosted trees for the prefilter and both stages (XGBoost, Apache-2.0, on GPU;
LightGBM, MIT, on CPU-only machines).  
**Training data:** 12 whole regions by the blocker's region key (US: OR, KY, AR, MO, WI, NC; India: KL, PB, HR,
OD, RJ, KA) + every unknown-region Source 2/3 record of both countries + the true owners (from other regions) of
the unknown-region records the sample retrieved, added as context: scored and used as competitors in the group
features and the one-to-one assignment, never trained on or evaluated. Split by Source 1 entity 60% fit / 20%
early stopping + threshold tuning / 20% held-out report.  
**Threshold selection method:** global one-to-one assignment (each Source 2/3 record goes to the Source 1 entity
that scores it highest), then grid search of two thresholds on the tuning split with the exact entity-level
macro F0.5 (singletons included): tau1 = {{TAU1}} for each entity's best candidate, tau2 = {{TAU2}} for the others.

**Preprocessing resources:** (1) the native-script word map is learned at run time from the training ground truth
(1.3k words, 98.9% purity, 96% token coverage of test native-script names); (2) the static lexicon in
`src/resources/*.tsv` (~5.3k per-country token/phrase equivalences, word classes, 2.8k look-alike conflict pairs)
was built once, offline: candidates came from training ground-truth token swaps and word counts; single words or
short phrases (never a record, never a match decision) were judged by an evaluation model, and every rule was
checked against training labels. The pipeline makes no API calls. No external data about businesses was used.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** {{HELDOUT_F05}} held-out (two-stage; stage 1 alone {{STAGE1_F05}}).
- **Local A/B (Oregon + Kerala whole regions, same split):** original pipeline 0.9802 -> + address-number
  features 0.9849 (stage 1: 0.9736 -> 0.9783) -> + transliteration map, reverse / address blocking, compact-name
  and word-difference features 0.9873 -> + exact keys, prefilter, house-number group features, region fix 0.9890.
- **Common false positives (wrong merges):** decoys whose only difference is a slightly shifted house number
  or a name with a letter appended / one word swapped; empty-address records with a generic name shared by several
  Source 1 entities in the country.
- **Common false negatives (missed matches):** copies whose house number was changed like a decoy's; empty-address
  copies of entities with generic names; copies with a random brand name and a truncated address.

---

## 6. Conclusion
Reverse-engineering the generator paid off most: address-number alignment separates decoys from true copies,
reverse and exact-key blocking keep recall in dense pools, and building training pools exactly like test pools
(including the competitors of empty-address records) removed a large train/test gap that small-region held-out
scores hide. France relies on the same language-independent signals.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`:
- `pipeline.py` - end-to-end entry point: `python pipeline.py --train-dir <train> --test-dir <test> --output-dir output`
  writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` (and runs the official validator).
- `src/safe_io.py` (TSV reading), `src/text.py` (country labels, core names), `src/er_multilingual.py`
  (script romanisation, canonical tokens, address parsing, multilingual pair features), `src/translit.py`
  (learned native-script word map), `src/er_lexicon.py` + `src/resources/` (static lexicon), `src/geo.py` (region
  keys), `src/extra_feats.py` (address-number alignment, compact names, word differences), `src/blocking.py` +
  `src/two_stage.py` (blocking views, prefilter, two-stage training, chunked two-pass inference), `src/features.py`
  (pair features), `src/groups.py` (stage-2 features), `src/sampling.py` (test-like training pools, context owners),
  `src/metrics.py` (macro F0.5, selection policy, threshold tuning).
- `local_eval.py` - offline A/B evaluation on whole training regions.
- The Kaggle notebook used for the submitted run is generated from the same modules (`build_notebook.py`).

### B. Additional Results
| Blocking on a test-sized pool (Karnataka, 68.8k S1 x 577k S2/S3) | Recall | Candidates / S1 |
|---|---|---|
| Old: forward name 20 / core 10 / name+address 40 | 0.940 | 56 |
| + address view 10, reverse name+address 2 | 0.957 | 67 |
| Forward 30/15/50/20 + reverse 3/2/2 | 0.965 | 106 |
| + exact keys (house number + name) | 0.982 | 132 (before prefilter) |

| Training pool (evaluated on the whole Karnataka pool with owners present) | Macro F0.5 | FP cost | FN cost |
|---|---|---|---|
| Regex-selected regions (empty-address records only when they match a sampled entity) | 0.9752 | 0.0120 | 0.0070 |
| + every unknown-region record of the country | 0.9750 | 0.0049 | 0.0145 |

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
