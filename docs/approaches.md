================================================================================
AMAZON ML CHALLENGE — BUSINESS ENTITY RESOLUTION
APPROACH DISCOVERY DOCUMENT  [UPDATED WITH REAL EDA — 2026-09-25]
================================================================================
Purpose: This document is a menu of technically distinct, implementable
approaches for the entity-resolution problem described in the challenge
brief. It does NOT select a winner and does NOT implement a final solution.
It is meant to let team members divide work, implement approaches
independently, and compare results later using a common protocol.

CHANGELOG vs ORIGINAL DOCUMENT:
  - Section 3 (Data Observations) is now fully populated with real measured
    statistics from the actual train/test files.
  - Several claims in the original document's approach descriptions have been
    corrected based on what the data actually shows.
  - Approach 1's "brute-force fallback as cheap insurance" note has been
    REMOVED — it is physically impossible at this scale.
  - Approach 11 (Siamese Network) has been substantially revised:
    the data has MANY labeled pairs, making the "too few positives" risk
    much lower, but the scale now makes naive training loops impossible.
  - Approach 13 (Graph-Based) has been updated with a precision-risk note
    that is more severe than the original given the confirmed multi-match
    structure (median 3 matches/S1, max 11).
  - All "India-only training" notes have been confirmed and strengthened:
    France is confirmed to appear only in test, and constitutes ~3-5% of
    test rows per source.
  - The "S3-only: 0%" claim in the original has been corrected to the real
    164,498 S3-only matched S1 entities (7.5% of matched S1s).

================================================================================
1. PROBLEM UNDERSTANDING
================================================================================

1. Prediction task:
   For every Source 1 (S1) entity, predict the set (possibly empty) of
   Source 2 (S2) and/or Source 3 (S3) entity_ids that refer to the same
   real-world business. This is entity resolution / record linkage across
   three independently-collected, noisy business registries, with S1 acting
   as the deduplicated reference set.

2. Input sources: three TSV files per split (train/test): source1, source2,
   source3. Ground truth is provided only for train, mapping
   source1_entity_id -> comma-separated matched_entity_ids (S2 and/or S3).

3. Fields available: entity_id, business_name, business_address, country.
   No source column — source is inferred from entity_id prefix (S1-/S2-/S3-)
   and from which file the record appears in.

4. Noise: abbreviations and legal-suffix variation in names (Corp/Corporation,
   Pvt/Private, Ltd/Limited), DBA/trade names, punctuation and word-order
   differences, transliteration, typos; address abbreviations (Rd/Road,
   St/Street), transliteration, missing components (PIN, state), landmark-
   based references, inconsistent component ordering.

5. Multiplicity: YES — an S1 entity can match multiple S2 and/or S3 records
   simultaneously. In training data, 94.4% of matched S1 entities have 2 or
   more matches (median 3, max 11). This MUST NOT be modeled as single-label
   classification.

6. Zero matches: YES — 5.58% of train S1 entities have no match in either
   S2 or S3. This is a real, non-trivial fraction and must be handled as an
   explicit "no-match" decision.

7. Ground truth: train_ground_truth.tsv, one row per S1 entity, with a
   (possibly empty) comma-separated list of true matching S2/S3 ids.

8. Evaluation metric: entity-level F_0.5 per S1 entity
   (F_0.5 = 1.25 * P * R / (0.25P + R)), macro-averaged across all S1
   entities in the evaluation set. Precision is weighted 2x recall.

9. Pairwise vs entity-level: The scored metric is ENTITY-LEVEL (per-S1 set
   comparison), macro-averaged — not a pairwise, and not a micro-averaged,
   metric. A model can have good pairwise AUC and still score poorly if
   entity-level set decisions (especially the no-match/empty-set decision)
   are miscalibrated.

10. Major constraints: no external data/APIs/lookup of any kind; TSV I/O
    with tab separators; reproducible train.py / predict.py; model
    license MIT/Apache-2.0, ≤8B parameters; deterministic and computationally
    feasible pipeline; blocking/ANN is NOT optional — it is mandatory given
    the dataset scale (22.7 TRILLION cross-product pairs).

11. Forbidden information: any externally sourced business/address/
    geocoding data; anything not derivable from the provided train/test
    files and their labels.

12. Computational constraints: The full cross-product S1×(S2∪S3) has
    ~22.7 TRILLION pairs and is COMPLETELY INFEASIBLE to enumerate even for
    debugging. Blocking with high recall is the most critical engineering
    challenge in this competition. All approaches must use blocking.

13. Final submission must contain: matching_results.tsv (final entity-level
    matches) and candidate_pairs.tsv (pre-filtering candidate sets), both
    TSV, exact column order, one row per S1 test entity, IDs restricted to
    entities that actually exist in the test files, final matches must be a
    subset of the emitted candidates.

14. Unseen categories / open-set: YES — test introduces "France" as a
    country value never seen in training. France appears in ~3.3-5.6% of
    test rows depending on source. Country must be treated as an open-set
    string feature, never as a hard filter or hardcoded allowlist.

15. Data properties that drive design:
    - Precision-sensitive metric (F_0.5) => conservative matching.
    - Multi-match mandatory: median 3 matches per S1, max 11; any top-1 or
      top-K fixed-K assumption is wrong.
    - 5.58% true no-match: not negligible — models must not always emit
      at least one match.
    - France in test (train = US+India only) => any country-specific
      hardcoding is a correctness risk.
    - ~3.2-17% address missingness in S2/S3 => address features must
      tolerate missing values gracefully; missing must not be treated as
      non-match evidence.
    - Scale makes blocking the bottleneck — candidate recall is a hard
      ceiling on every downstream approach's achievable recall.

================================================================================
2. IMPORTANT COMPETITION CONSTRAINTS (SUMMARY)
================================================================================
- No external data, APIs, search engines, geocoding, or business directories.
- No hardcoding of country handling (India-only or US-only assumptions will
  fail on the France subset of test).
- No forced 1:1 matching; no forced "always at least one match."
- No exact-string-only matching as the sole mechanism.
- No single blocking strategy relied upon exclusively.
- No naive threshold = 0.5 without validation.
- No optimizing plain accuracy/F1 in place of entity-level macro F_0.5.
- Final matches must be a subset of emitted candidates; all IDs must exist
  in the corresponding test source file; no duplicate IDs/rows.
- Model constraints: MIT/Apache-2.0, ≤8B parameters — classical ML trivially
  satisfies this; neural approaches must be verified before use.
- NO BRUTE-FORCE CROSS-JOIN even for debugging — 22.7 trillion pairs makes
  this physically infeasible.

================================================================================
3. DATA OBSERVATIONS — REAL EDA RESULTS (MEASURED FROM ACTUAL FILES)
================================================================================
These are real measured statistics, not illustrative examples.

--- ROW COUNTS ---
train_source1: 2,206,821 rows
train_source2: 5,034,616 rows
train_source3: 5,285,603 rows
train_ground_truth: 2,206,821 rows (one per S1 entity)
test_source1:  1,732,544 rows
test_source2:  4,887,273 rows
test_source3:  5,082,316 rows

IMPLICATION: The full cross-product S1×(S2∪S3) has ~22.7 TRILLION pairs.
Brute-force enumeration is physically impossible. Blocking is mandatory.

--- COLUMN-LEVEL MISSINGNESS ---
Train S1: name_null=0 (0.0%), addr_null=0 (0.0%), country_null=0
Train S2: name_null=2 (≈0%), addr_null=168,967 (3.4%), country_null=0
Train S3: name_null=13 (≈0%), addr_null=175,916 (3.3%), country_null=0
Test S1:  name_null=0 (0.0%), addr_null=0 (0.0%)
Test S2:  name_null=46 (≈0%), addr_null=129,408 (2.6%)
Test S3:  name_null=59 (≈0%), addr_null=136,098 (2.7%)

NOTE: Address missingness in S2/S3 is consistent at ~2.6-3.4% across splits.
S1 has NO missing names or addresses in either split. Address features must
handle missingness in candidates gracefully (treat as missing, not as
non-match evidence). Name missingness is effectively zero.

--- COUNTRY DISTRIBUTION ---
Train S1: US=1,323,633 (59.9%), India=883,188 (40.0%)  — no France
Train S2: US=3,016,817 (59.9%), India=2,017,799 (40.1%) — no France
Train S3: US=3,170,056 (59.9%), India=2,115,547 (40.0%) — no France
Test S1:  India=809,986 (46.7%), US=663,106 (38.3%), France=259,452 (15.0%)
Test S2:  India=2,312,565 (47.3%), US=1,871,330 (38.3%), France=703,378 (14.4%)
Test S3:  India=2,405,000 (47.3%), US=1,945,701 (38.3%), France=731,615 (14.4%)

CONFIRMED: France is absent from ALL training data. It appears only in test.
CRITICAL: France constitutes ~15% of test rows across all three sources —
this is NOT a minor edge case. Approaches that are not generalized for
unseen address conventions will fail on ~15% of the test leaderboard.
The training data is also notably LESS India-dominant than the original
document assumed — in test, India is ~47% and US is ~38%, with France at 15%.

NOTE ON TEST COUNTRY MIX SHIFT: The train S1 split is 60% US / 40% India;
the test S1 split is 38% US / 47% India / 15% France. This is a meaningful
covariate shift between train and test — models trained on train data may
be miscalibrated due to this shift, independent of the France novelty.
Validation splits should be constructed to reflect this imbalance explicitly.

--- GROUND TRUTH MATCH-COUNT DISTRIBUTION ---
Zero matches (no-match): 123,247 (5.58%)
1 match:   119,157 (5.40%)
2 matches: 375,212 (17.00%)
3 matches: 530,841 (24.05%)
4 matches: 484,115 (21.94%)
5 matches: 321,957 (14.59%)
6+ matches: 252,292 (11.43%)
Max matches per S1: 11
Mean matches: 3.46
Median matches: 3

IMPLICATIONS:
- The "no-match" case is real but minority (5.6%).
- The dominant case is MULTI-MATCH: 80%+ of S1 have 3+ matches.
- A fixed top-K approach (e.g., always emit exactly K) will be wrong for
  most entities.
- Median of 3 and max of 11 means the decision boundary is not binary —
  the model must learn to emit variable-length sets.

--- S2 vs S3 MATCH BREAKDOWN ---
S2-only matches (S1 has S2 match but NO S3 match): 143,029 entities (6.5%)
S3-only matches (S1 has S3 match but NO S2 match): 164,498 entities (7.5%)
Both S2 AND S3 matched: 1,776,047 entities (80.5%)
Neither (no-match): 123,247 entities (5.6%)

CORRECTION: The original document's "S3-only: 0%" was WRONG. There are
164,498 S3-only matched S1 entities. Source-specific modeling (Approach 8)
and separate thresholds per source (Approach 16) are both empirically
justified by this asymmetry.

--- TOTAL LABELED PAIRS ---
Total true positive (S1, S2/S3) pairs: 7,638,365
S2 positive pairs: 3,693,619
S3 positive pairs: 3,944,746

IMPLICATION FOR APPROACH 11 (Neural Siamese): The "too few labeled pairs
to train a neural model" risk from the original document is largely
mitigated — there are ~7.6M positive pairs. However, training efficiency
at this scale requires vectorized/batched pipelines, not naive per-pair
loops.

--- CROSS PRODUCT SIZE ---
|S1_train| × (|S2_train| + |S3_train|) = 2,206,821 × 10,320,219
= 22,773,016,459,799 ≈ 22.7 TRILLION pairs.
BRUTE-FORCE IS IMPOSSIBLE. Blocking is a hard architectural requirement,
not a performance optimization.

--- EDA CHECKLIST STATUS ---
  [x] Row counts: measured
  [x] Missingness: measured
  [x] Country distribution: measured (France confirmed test-only)
  [x] Match count distribution: measured (median 3, max 11, 5.6% no-match)
  [x] S2-only / S3-only / both split: measured
  [x] Cross-product size: computed (22.7T — brute-force impossible)
  [ ] Exact normalized name match rate for true pairs: STILL REQUIRED
  [ ] Exact address match rate for true pairs: STILL REQUIRED
  [ ] String length distributions: STILL REQUIRED
  [ ] Transliteration / abbreviation sample analysis: STILL REQUIRED
  [ ] Within-source near-duplicate rate: STILL REQUIRED

The remaining 5 items above should be measured using a stratified sample of
true positive pairs from the ground truth before finalizing any approach's
feature set.

================================================================================
4. SOLUTION-FAMILY OVERVIEW
================================================================================
A. Deterministic / Rule-Based Matching
B. Fuzzy String / Classical Similarity Matching
C. Information Retrieval / Nearest-Neighbor Candidate Generation + Scoring
D. Probabilistic Record Linkage (Fellegi-Sunter style)
E. Supervised Pairwise Classification (tabular ML on similarity features)
F. Learning-to-Rank / Reranking
G. Neural Representation Learning (embeddings, Siamese/contrastive)
H. Cross-Encoder Pair Classification (transformer-based, license-checked)
I. Graph-Based / Collective Entity Resolution
J. Hybrid Rule + ML Systems
K. Ensemble Systems (model and blocking ensembles)
L. Weak/Self-Supervised & Calibration-Focused Approaches

18 materially distinct approaches are detailed below. Hyperparameter
variants (different boosting libraries, n-gram sizes) are called out as
variations WITHIN an approach rather than as separate approaches.

================================================================================
5. DETAILED APPROACHES
================================================================================

--------------------------------------------------------------------------------
APPROACH 1 — Deterministic Rule Cascade (Exact/Normalized Key Matching)
--------------------------------------------------------------------------------
Core Idea:
A cascade of deterministic rules applied in decreasing order of confidence:
exact normalized name + exact PIN => match; exact normalized name + exact
house number => match; exact normalized name alone (if globally unique in
the source) => match. No learned model at all. Acts as both a candidate
generator and a self-contained baseline matcher.

Pipeline:
Data -> normalization -> build hash keys (name_norm, name_norm+pin,
name_norm+house_number) -> exact key join -> emit matches meeting the
highest-confidence rule tier -> no-match for everything else.

Key Features/Representations: normalized name string, normalized PIN/house
number extraction, composite keys built by string concatenation.

Candidate Generation: hash-bucket join on composite keys — inherently a
blocking mechanism too (same keys used for both candidate gen and decision).

Matching/Scoring: rule-tier membership (no probability), tiers ordered by
expected precision.

No-Match Handling: default — anything not hit by any rule tier is no-match.

Multiple-Match Handling: natural — hash join can return multiple S2/S3 rows
per S1 key.

DATA-GROUNDED NOTES:
- With 2.2M S1 and 10.3M S2+S3, hash-bucket joins remain O(n) and are fast.
- S1 has no missing addresses, so normalization is reliable on the anchor
  side. S2/S3 have ~3.3% missing addresses — these records simply won't
  get address-based rule matches and fall to the ML stage.
- No-match is ~5.6% of S1 — a deterministic rule that never emits a match
  for a record would be ~94.4% wrong for that record. The rule cascade
  should default to "no match" (not "emit nearest"), letting the ML
  fallback handle candidates.
- The original document's note suggesting brute-force as a "cheap debug
  fallback" has been REMOVED: 22.7T pairs makes any full cross-join
  infeasible even for debugging. A sample-based check (e.g., 1% of S1
  against all S2+S3 using TF-IDF retrieval) can serve as a sanity check
  instead.
- France in test: this approach's extraction regex must NOT hardcode
  India-specific PIN patterns (6-digit numeric) — US ZIP codes (5-digit) 
  and French postal codes (5-digit, starts with 0-9) differ; use a
  multi-format extractor or a generalized "numeric sequence" extractor.

Strengths: extremely high precision on the subset it covers; zero training
data needed; strong sanity-check baseline.

Weaknesses: low recall — cannot handle any name/address noise beyond what
normalization captures; brittle to transliteration/typos.

To Test Experimentally: precision of each rule tier; coverage per tier;
whether combining tiers with looser fuzzy tiers (Approach 2) recovers the
recall gap.

Implementation: pure pandas/python string ops; dict-based hash-join.

Expected Computational Cost: LOW — hash joins are near-linear in data size.

--------------------------------------------------------------------------------
APPROACH 2 — Fuzzy String Similarity Threshold Matching
--------------------------------------------------------------------------------
Core Idea:
Extends Approach 1 by replacing exact-key equality with classical string
similarity (Levenshtein/Jaro-Winkler on names, token Jaccard on addresses)
compared against a validated threshold. No learned classifier — just a
hand-tuned scoring rule.

Pipeline:
Data -> normalization -> blocking (token-overlap or prefix blocks) ->
compute similarity scores per candidate pair -> weighted-sum or max-rule
threshold -> emit matches above threshold.

Key Features/Representations: normalized name, normalized address,
Levenshtein/Jaro-Winkler similarity, token Jaccard, numeric-token overlap.

Candidate Generation: token/prefix blocking — needed because pairwise
string distance cannot be applied to 22.7T pairs.

Matching/Scoring: score = w1*name_sim + w2*address_sim; threshold selected
via validation sweep.

No-Match Handling: below-threshold candidates dropped; empty if all below.

Multiple-Match Handling: emit all above-threshold candidates, not argmax.

DATA-GROUNDED NOTES:
- Approach is feasible only after blocking. On the full corpus, even
  computing Jaro-Winkler for 22.7T pairs is impossible. This approach's
  role is as a SCORING layer on top of a blocked candidate set, not as
  a standalone pipeline.
- With a median of 3 true matches per S1 (and max 11), a fixed threshold
  that emits only the top-1 or top-2 candidates will systematically
  under-recall. Threshold must allow variable-length output.
- The ~3.3% address missingness in S2/S3 must be handled: missing address
  should contribute 0 weight (not penalize), not be treated as disagreement.

Strengths: interpretable; no training data required (only validation for
threshold tuning); fast to implement.

Weaknesses: fixed-weight scoring can't capture nonlinear interactions;
manual weight tuning doesn't scale to many features.

To Test Experimentally: threshold sensitivity curve; whether weighted-sum
beats max-of(name_sim, address_sim).

Implementation: rapidfuzz for string distances; manual weighting; grid
search over thresholds using validation F_0.5.

Expected Computational Cost: LOW-MODERATE after blocking.

--------------------------------------------------------------------------------
APPROACH 3 — TF-IDF Retrieval + Cosine Threshold
--------------------------------------------------------------------------------
Core Idea:
Treat candidate generation as an information-retrieval problem: build a
TF-IDF (word and/or character n-gram) index over S2+S3 names, and for every
S1 record retrieve its nearest neighbors by cosine similarity. Scalable
blocking mechanism for large datasets.

Pipeline:
Data -> normalization -> fit TF-IDF vectorizer on S2+S3 corpus ->
transform S1 corpus -> compute cosine similarity via sparse matrix
multiplication -> take top-K and/or above-threshold neighbors as candidates
-> apply cosine threshold for final match decision.

Key Features/Representations: TF-IDF vectors over normalized name (and
optionally normalized address), word-level and/or character n-gram.

Candidate Generation: top-K nearest neighbors by cosine similarity per S1
record — this IS the blocking strategy.

No-Match Handling: if max cosine similarity for an S1 record's best
candidate is below threshold, emit no-match.

Multiple-Match Handling: emit all above-threshold neighbors within top-K.

DATA-GROUNDED NOTES:
- With 10.3M S2+S3 records, a full dense cosine similarity matrix is
  impossible. Use sparse matrix multiplication (scipy.sparse) or batched
  matrix-vector products. The sklearn TfidfVectorizer + sparse NearestNeighbors
  approach can handle millions of records with proper batching.
- Given median 3 matches per S1 (max 11), K must be set ≥ 20-50 to avoid
  capping true multi-match recall. Validate K against candidate recall.
- Character n-gram TF-IDF (3-5 grams) is naturally typo-tolerant and
  handles Devanagari script (seen in S2) better than word-level alone
  because characters are indexed, not word boundaries.
- France in test: TF-IDF is script-agnostic (treats all characters equally),
  so French text is handled without modification.

Strengths: scales via sparse matrix operations; serves double duty as
blocker and scorer; character n-gram variant is typo/transliteration tolerant.

Weaknesses: captures lexical, not semantic, overlap; depends heavily on
normalization quality.

To Test Experimentally: word-level vs char-level n-gram TF-IDF; effect of
concatenating address text; top-K sensitivity vs candidate recall.

Implementation: scikit-learn TfidfVectorizer + sparse cosine similarity
or sklearn NearestNeighbors; scipy sparse matrices; batched computation.

Expected Computational Cost: MODERATE — vectorization fast; batched sparse
NN search tractable for millions of records.

--------------------------------------------------------------------------------
APPROACH 4 — Character N-gram Approximate Nearest Neighbor Blocking
--------------------------------------------------------------------------------
Core Idea:
A candidate-generation-focused approach: build an inverted index over
character n-grams (shingles) of normalized names/addresses and retrieve
candidates sharing a minimum number of shingles (MinHash/LSH-style or plain
inverted-index intersection counting). Optimized purely for high-recall,
low-cost blocking rather than scoring.

Pipeline:
Data -> normalization -> generate character n-grams (e.g., n=3) per record
-> build inverted index (shingle -> list of record ids) -> for each S1
record, count shingle overlap with candidates -> emit as candidate set.

Candidate Generation: inverted-index lookup + overlap-count threshold,
or MinHash/LSH banding for scale.

Matching/Scoring: NOT a final matcher — feeds candidates into a downstream
classifier.

DATA-GROUNDED NOTES:
- At 10.3M S2+S3 records, the inverted index memory footprint is non-trivial.
  With n=3 and average name length ~30 chars, that's ~28 n-grams/record →
  ~288M (shingle, record_id) entries in the index. This fits in RAM with
  a dict-of-lists or a sparse CSR matrix approach but requires planning.
- MinHash/LSH (via the `datasketch` library) is preferable at this scale
  for controlling memory and query time with probabilistic guarantees.
- Given the multi-match nature (median 3, max 11), the inverted index will
  naturally return multiple candidates per S1 entity, which is the correct
  behavior.
- Devanagari names in S2/S3 (India subset) will produce character n-grams
  natively without requiring any special tokenization — this is an advantage
  of character-level blocking over word-level.

Strengths: excellent typo/transliteration tolerance; cheap; UNION-friendly;
recovers different matches than TF-IDF blocking.

Weaknesses: not a standalone matcher; needs tuning of n and overlap threshold.

To Test Experimentally: n=2 vs 3 vs 4; overlap threshold sweep; candidate
recall comparison with Approach 3.

Implementation: python dict-based inverted index + MinHash/LSH via datasketch.

Expected Computational Cost: LOW-MODERATE; scales roughly linearly.

--------------------------------------------------------------------------------
APPROACH 5 — Structured Address Key Blocking (PIN / House Number / Locality)
--------------------------------------------------------------------------------
Core Idea:
Parse addresses into structured components (PIN code, house/building number,
locality tokens) and block on exact or near-exact matches of these
high-precision structured fields, independent of name similarity.

Pipeline:
Data -> address parsing/normalization -> extract PIN, house number, numeric
tokens, locality tokens -> build blocking keys -> union candidates across
all structured keys.

Matching/Scoring: blocking only — not a final matcher (or only a coarse
one paired with a light rule).

DATA-GROUNDED NOTES:
- S1 has no missing addresses (0%), so PIN/house extraction will have full
  coverage on the anchor side. S2/S3 have ~3.3% missing addresses — these
  records get no address-based blocking contribution; must rely on name
  blocking via Approaches 3/4.
- Address format varies widely by country: India uses 6-digit PINs; US uses
  5-digit ZIP codes; France uses 5-digit postal codes starting 01-95.
  The extraction regex MUST be country-agnostic (or at minimum, support
  all three formats) since France appears in test.
- A simple "any contiguous 5-6 digit sequence" extractor handles US ZIP,
  India PIN, and French postal codes without hardcoding. Validate against
  a sample to confirm false-positive rate is acceptable.
- S3-only matched entities (164K of them) are genuine — this blocker helps
  them equally, as S3 addresses are as structured as S2's.

Strengths: structured numeric fields carry high precision when present;
robust to name noise; cheap.

Weaknesses: ~3.3% address missingness in S2/S3 limits coverage; extraction
quality gates this approach's usefulness.

To Test Experimentally: extraction success rate per source and country;
candidate recall unique to this blocker vs Approaches 3/4.

Implementation: regex-based extraction supporting US ZIP, India PIN, French
postal codes; fallback graceful on missing addresses.

Expected Computational Cost: LOW.

--------------------------------------------------------------------------------
APPROACH 6 — Fellegi-Sunter Probabilistic Record Linkage
--------------------------------------------------------------------------------
Core Idea:
Classical probabilistic record linkage: for each comparison field, estimate
m-probabilities (P(agreement | match)) and u-probabilities (P(agreement |
non-match)) from training data, compute a log-likelihood match score per
candidate pair as the sum of per-field contributions, and threshold.

Pipeline:
Data -> normalization -> candidate generation (reuse Approaches 3/4/5 union)
-> per-field comparison (agree/partial/disagree bins) -> estimate m/u
probabilities from labeled pairs -> compute per-pair log-likelihood ratio
-> threshold.

Matching/Scoring: sum of field-level log(m/u) weights.

DATA-GROUNDED NOTES:
- With 7.6M true positive pairs available for supervised m-probability
  estimation, the original risk of "noisy m-probability estimates from
  few positives" is largely eliminated. Supervised estimation (directly
  from ground truth) is preferable to EM-based unsupervised estimation.
- The country field is heavily correlated with train distribution (US+India)
  and completely uninformative for distinguishing matches within a country
  (all India-India pairs have country="India"). Country agreement is a
  WEAK feature in this dataset. However, it should NOT be hardcoded to 0
  weight because France in test could make country agreement more useful
  when S1 and candidate share the same unseen country.
- Address missingness (~3.3% in S2/S3) must be handled as a dedicated
  "missing" bin in the comparison levels, NOT as "disagreement."
- The field independence assumption is the main theoretical weakness;
  in practice, name and address are correlated for matching businesses.
  Direct supervised estimation partially mitigates this by absorbing
  correlation into the per-field weights empirically.

Strengths: statistically principled field combination; handles missing fields
naturally; cheap to run; well-studied baseline.

Weaknesses: field independence assumption; binning loses information vs
continuous features (Approach 7).

To Test Experimentally: supervised vs EM-based parameter estimation;
number of agreement bins per field; country agreement field impact.

Implementation: recordlinkage Python library or custom; straightforward.

Expected Computational Cost: LOW.

--------------------------------------------------------------------------------
APPROACH 7 — Supervised Pairwise Classification with Gradient Boosting
(RECOMMENDED BASELINE PER BRIEF — still one hypothesis among many here)
--------------------------------------------------------------------------------
Core Idea:
Generate a rich set of handcrafted similarity/structured features per
(S1, candidate) pair and train a gradient-boosted tree classifier to
predict P(match). The brief's own recommended baseline family.

Pipeline:
Data -> normalization -> candidate generation (union of Approaches 3/4/5)
-> pairwise feature engineering -> train GBM on labeled pairs (positives
from ground truth, hard negatives from the same candidate pipeline) ->
validated threshold selection -> apply to test candidates.

Key Features/Representations: full feature set — name similarity
(Levenshtein/Jaro-Winkler/Jaccard/TF-IDF cosine, char/word n-gram cosine),
address similarity (same + PIN/house-number/numeric-overlap), country
agreement, interaction features (product/min/max of name_sim and address_sim).

Candidate Generation: union of Approaches 3, 4, 5.

Matching/Scoring: GBM P(match); validated threshold.

No-Match Handling: all candidates below threshold dropped.

Multiple-Match Handling: all above-threshold candidates emitted.

DATA-GROUNDED NOTES:
- 7.6M true positive pairs is a large labeled set. Negative sampling strategy
  is the critical design choice — with a median of 3 true matches per S1 and
  a 22.7T total pair space, the natural class imbalance is extreme. Hard
  negative mining from the candidate pipeline (not random negatives) is
  essential.
- GBM training on ~7.6M positives + appropriate negatives will be memory-
  intensive. Use chunked feature engineering or stratified sampling (e.g.,
  train on 20% of S1 entities but all their candidates) if memory is limited.
- The multi-match structure (median 3, max 11) makes threshold selection
  particularly sensitive: a threshold tuned to maximize pairwise F1 will
  likely be wrong for entity-level F_0.5 macro because it doesn't account
  for the per-entity averaging. Threshold must be tuned directly against
  entity-level F_0.5 on a held-out S1 entity split.
- Address missingness (~3.3%) must be represented as explicit NaN in the
  feature pipeline so GBMs can handle it via their native missing-value
  mechanisms (LightGBM, XGBoost, and CatBoost all support this natively —
  no imputation needed).
- France in test: features must not include hard-coded country-specific
  normalization. A per-country normalization template would fail; generic
  text normalization (lowercase, strip punctuation) is safer.

Strengths: can learn nonlinear interactions; directly thresholdable against
the competition metric; fast to train; license-compliant; flexible for
hard-negative mining.

Weaknesses: only as good as the feature set; risk of India/US-specific
feature patterns not generalizing to France.

To Test Experimentally: LightGBM vs XGBoost vs CatBoost vs
HistGradientBoostingClassifier; feature ablation groups; calibration on
top of this model.

Implementation: lightgbm/xgboost/catboost; pandas feature pipeline; early
stopping; threshold sweep directly on validation entity-level F_0.5.

Expected Computational Cost: MODERATE — feature computation dominant;
GBM training fast on tabular features.

--------------------------------------------------------------------------------
APPROACH 8 — Source-Specific Pairwise Classifiers (S1-S2 vs S1-S3)
--------------------------------------------------------------------------------
Core Idea:
Train two separate classifiers — one for S1-S2 candidate pairs, one for
S1-S3 candidate pairs — instead of one shared model.

DATA-GROUNDED NOTES:
- The EDA CONFIRMS a meaningful asymmetry: 6.5% of matched S1 entities have
  S2-only matches, 7.5% have S3-only matches. The sources are NOT symmetric.
  This justifies testing source-specific models rather than assuming them
  to be redundant.
- S2 positive pairs: 3,693,619; S3 positive pairs: 3,944,746. Both sources
  have roughly comparable labeled positive counts, so the "halving the
  training set" weakness from the original document applies (each model
  trains on about half the positives) but is less severe than with few
  positives total.
- Separately tuned thresholds per source are also directly motivated by the
  data: if S2 records have different noise levels than S3 (to be verified by
  EDA on matched pairs), a single global threshold will be suboptimal for
  one or both sources.
- The "S3-only: 0%" claim in the original document was incorrect. S3-only
  matched entities definitely exist (164K of them). Source-specific classifiers
  must each be validated against their source's specific true positives.

Strengths: specializes to each source's noise profile; separately validated
thresholds; directly testable vs Approach 7.

Weaknesses: halves effective positive training examples per classifier;
doubles implementation surface.

To Test Experimentally: direct F_0.5 comparison vs Approach 7 on identical
validation split; per-source feature importance comparison.

Expected Computational Cost: MODERATE — roughly 2x Approach 7.

--------------------------------------------------------------------------------
APPROACH 9 — Multi-Stage Retrieve-then-Rerank (TF-IDF Retrieval + GBM Reranker)
--------------------------------------------------------------------------------
Core Idea:
Explicitly separates candidate generation (maximize recall) from final
matching (maximize precision) into two decoupled, independently-tunable
stages.

Pipeline:
Stage 1: TF-IDF top-K retrieval tuned for high recall (large K, low floor)
Stage 2: GBM classifier on the Stage-1 candidate pool, tuned for precision.

DATA-GROUNDED NOTES:
- With 2.2M S1 and 10.3M S2+S3, even sparse TF-IDF retrieval at K=50
  produces 2.2M × 50 = 110M candidate pairs. Feature engineering for all
  110M pairs is expensive but feasible with chunked/parallelized processing.
  K should be validated: does K=50 capture 11 true matches (the max) for
  all S1 entities with 11 ground-truth matches?
- Stage 1 K must be ≥ 15 to have a chance of capturing S1 entities with
  up to 11 true matches (assuming true matches appear in the top-K by cosine
  similarity — this should be validated empirically on the training data).
- Stage 2 feature engineering at 110M pairs is memory-intensive. If memory
  is a constraint, reduce K or subsample S1 entities for Stage-2 training.
- The train/inference mismatch risk (training Stage-2 on candidates NOT from
  Stage-1's actual output) is critical. Stage-2 must always be trained on
  Stage-1's actual output, using the same K and threshold.

Strengths: decoupled objectives well-matched to F_0.5's precision-vs-recall
structure; allows parallel development of blocking and scoring.

Weaknesses: two-stage pipeline adds engineering complexity; Stage-2 training
dataset can be very large at high K.

To Test Experimentally: Stage-1 K sweep vs candidate recall; whether
Stage-1 cosine score as a Stage-2 feature improves final F_0.5.

Expected Computational Cost: MODERATE-HIGH — dominated by Stage-2 feature
computation on the (large, recall-tuned) Stage-1 pool.

--------------------------------------------------------------------------------
APPROACH 10 — Pairwise Learning-to-Rank per S1 Entity
--------------------------------------------------------------------------------
Core Idea:
Reframes matching as a ranking problem: rank candidates per S1 entity by
learned relevance, then apply a calibrated cutoff to decide which to emit.

DATA-GROUNDED NOTES:
- The multi-match structure (median 3, max 11) makes this approach more
  naturally motivated than the original document suggested. Standard LambdaMART
  with NDCG-based objective can handle multi-relevant items per query
  (S1 entity) natively.
- Group size imbalance: S1 entities with 11 candidates in ground truth vs
  those with 1 will create very different group sizes in the ranking training
  data. LightGBM's lambdarank can handle this but needs appropriate group
  boundary specification.
- The "emit or not" secondary threshold is still required: top-rank alone is
  insufficient (a no-match S1 entity still has a "top-ranked" candidate that
  should NOT be emitted). The absolute score floor must be calibrated against
  the 5.6% no-match S1 entities to avoid always-emitting.
- At 2.2M S1 entities × ~50 candidates each = 110M group-item pairs for
  LambdaRank training — large but tractable with LightGBM's efficient
  implementation.

Strengths: ranking objective naturally fits relative ordering of multiple
candidates; can exploit score-gap signals.

Weaknesses: requires separate calibrated absolute floor for no-match
decision; sensitive to group size imbalance.

To Test Experimentally: LambdaRank vs plain binary classification holding
features and candidate set identical; different emission rules
(absolute floor + gap-to-top).

Expected Computational Cost: MODERATE — similar to Approach 7.

--------------------------------------------------------------------------------
APPROACH 11 — Siamese Network Name/Address Embedding Similarity
--------------------------------------------------------------------------------
Core Idea:
Train a small neural encoder (character-level BiLSTM or lightweight
transformer trained from scratch) in a Siamese setup with contrastive or
triplet loss to map business names into a shared embedding space where true
matches are close. Uses ANN search (FAISS) for candidate generation.

DATA-GROUNDED NOTES:
- Original concern "too few labeled pairs to train a neural model" is
  SIGNIFICANTLY REDUCED by the actual data: 7.6M positive pairs is a
  substantial labeled set, enough to train a reasonably-sized Siamese encoder.
- However, scale introduces a new challenge: training on all 7.6M positive
  pairs (plus appropriate negatives) requires efficient batched training.
  A naive per-pair training loop over 7.6M pairs will be slow.
  Use mini-batch sampling with hard-negative mining from the blocking candidate
  pool (not all-negative, which would be dominated by easy negatives).
- At inference time, embedding all 10.3M S2+S3 records with a neural encoder
  takes non-trivial time; must benchmark before committing.
- FAISS index over 10.3M 64-128 dim float vectors is feasible (~5-10GB RAM
  with HNSW or IVF index) but requires memory planning.
- From-scratch training strictly required (no pretrained model downloads).
  A character-level BiLSTM with 64-dim embeddings, 2 layers, ~2-5M params
  satisfies the ≤8B parameter constraint trivially.
- Devanagari script in S2/S3 India subset: character-level encoding handles
  this naturally IF the character vocabulary is built from the training data
  (which it must be, per no-external-data constraint). Ensure the vocab
  includes both Latin and Devanagari character ranges.
- France test records: if French business names use accented characters
  (é, ê, à, etc.) not well-represented in training, the character embeddings
  may not generalize. A normalization step (NFD + strip diacritics) or
  dedicated handling of accented characters is advisable.

Strengths: can learn noise-pattern robustness end-to-end; sufficient labeled
data now available; ANN search enables scalable candidate generation.

Weaknesses: heavier to implement, tune, and debug; inference over 10.3M
records requires planning; French accented characters may not generalize.

To Test Experimentally: from-scratch Siamese encoder vs Approach 7's GBM
on identical validation split and candidate pool; character-level vs subword
tokenization; FAISS index type (HNSW vs IVF) for latency/recall tradeoff.

Implementation: PyTorch, character-level BiLSTM or 1D-CNN; FAISS for ANN;
vocab built from training data only.

Expected Computational Cost: HIGH — training loop + inference over 10.3M
records; GPU recommended but not strictly required at this scale with a
small model.

--------------------------------------------------------------------------------
APPROACH 12 — Cross-Encoder Pair Classifier (Compact Transformer, Trained From Scratch)
--------------------------------------------------------------------------------
Core Idea:
Concatenate (S1 record text, candidate record text) as a single input and
train a compact transformer to predict match/non-match with full cross-
attention between the two records' tokens.

DATA-GROUNDED NOTES:
- At 2.2M S1 × ~50 candidates per S1 = 110M candidate pairs for inference,
  a cross-encoder running per-pair inference is extremely expensive.
  Even at 1ms/pair (fast for a transformer), 110M pairs = ~30 GPU-hours.
  This approach requires careful compute budgeting.
- Training: with 7.6M positive pairs + negatives, the training set is large.
  A small from-scratch transformer (e.g., 4 layers, 256 hidden, ~5M params)
  can be trained in reasonable time on GPU, but requires careful batching.
- This approach is best suited as a RERANKER on a small top-K candidate set
  (e.g., top-20 from Approach 3/4), not as a scorer for all 110M candidate
  pairs.
- The "too few labeled pairs" concern from the original has been resolved by
  the data; the primary concern is now inference cost at 110M pairs.
- Data augmentation (field dropout, name abbreviation) is still valuable to
  combat domain shift to France's unseen text patterns.

Strengths: full cross-attention can catch subtle alignments; high theoretical
accuracy per pair.

Weaknesses: CANNOT be used for candidate generation; per-pair inference over
110M pairs is extremely expensive; adds significant implementation complexity
vs Approach 7's GBM.

REVISED RECOMMENDATION: This approach is viable ONLY if restricted to
reranking a small pre-filtered set (e.g., top-20 candidates from Approach 3).
Running it over all 110M candidate pairs is not feasible without GPU cluster
infrastructure beyond a typical hackathon setup.

To Test Experimentally: inference latency at realistic scale; whether the
cross-encoder outperforms Approach 7's GBM given the added compute cost.

Expected Computational Cost: HIGH — training feasible; INFERENCE IS THE
BOTTLENECK at 110M candidate pairs.

--------------------------------------------------------------------------------
APPROACH 13 — Graph-Based / Collective Entity Resolution
--------------------------------------------------------------------------------
Core Idea:
Build a similarity graph over all S1/S2/S3 records (nodes), use pairwise
scores (e.g., from Approach 7's GBM) as edge weights, and apply
connected-components or community detection to derive final match sets
collectively rather than per-pair independently.

DATA-GROUNDED NOTES:
- The multi-match structure (median 3, max 11, Some S1 with up to 11 matches)
  makes graph-based approaches more interesting here than in a 1:1 setting.
  A correct cluster for an S1 entity with 11 true matches will have 11
  edges in its connected component — the graph naturally handles this.
- Chain-merging risk is SEVERE given the data scale. With 22.7T potential
  pair edges (pre-blocking), even a small false-positive rate in edge weights
  creates many spurious transitive connections across 2.2M S1 nodes.
- The precision-weighting of F_0.5 makes chain-merging errors especially
  costly: each spuriously merged component creates false positives for the
  entity with the merged S1 ancestor.
- An important structural constraint: the competition requires S1-to-S2/S3
  DIRECTED matches, not arbitrary clustering. Graph-based approaches must
  respect the S1 reference role — S1 entities should NOT be merged with each
  other; only S2/S3 nodes are assigned to S1 anchor nodes.
- Handling multiple S1 nodes in one component requires a splitting policy
  (e.g., assign S2/S3 nodes to the S1 in the same component by highest edge
  weight) — this is a required engineering decision, not an afterthought.
- Memory: storing edges for 10.3M S2+S3 + 2.2M S1 nodes with ~50 candidate
  edges per S1 = ~110M edges is manageable (few GB with sparse adjacency).

Strengths: handles multi-match sets naturally; transitive evidence can
boost recall for hard cases.

Weaknesses: chain-merging is the dominant precision risk; directed S1-anchor
structure requires careful graph design; complex to implement and debug.

REVISED: Worth pursuing ONLY as an add-on to an already-validated pairwise
scorer (Approach 7/9), not as a standalone approach, due to the chain-merging
risk under F_0.5's precision weighting.

To Test Experimentally: connected-components vs max-weight-matching; direct
F_0.5 comparison vs Approach 7/9; frequency of multi-S1 components as a
diagnostic.

Expected Computational Cost: MODERATE-HIGH — graph construction at 110M
edges; connected-components is O(V+E) and fast.

--------------------------------------------------------------------------------
APPROACH 14 — Hybrid Deterministic Rules + ML Fallback Cascade
--------------------------------------------------------------------------------
Core Idea:
A staged system: apply the deterministic rule cascade (Approach 1) first
for maximum-precision, unambiguous matches; for S1 entities NOT resolved
by rules, fall through to the ML classifier (Approach 7) on remaining
candidates.

DATA-GROUNDED NOTES:
- With no missing names/addresses in S1, the deterministic rules have clean
  inputs on the anchor side. The ~3.3% address missingness only affects S2/S3
  candidates — those records simply won't match address-based rules.
- The "easy" vs "hard" partitioning implied by this approach is appropriate:
  rule-resolved S1 entities are likely those with clean, standardized names
  and addresses; the ML stage handles the remaining ~90%+ with noisy data.
- 5.6% true no-match S1 entities: the rule stage should NOT be expected to
  identify no-match entities — it can only do so by NOT matching anything,
  which any rule-cascade naturally does by default. The ML stage then decides
  between "some fuzzy match exists" and "true no-match" for the harder cases.
- Non-overlap logic must be carefully designed: Stage A finds an S2 match
  for an S1 entity → Stage B should still check S3 candidates for that
  same entity (the two sources are independent). Do NOT route Stage-A-resolved
  entities entirely out of Stage B; only route them out of S2-specific
  candidate search if a high-confidence S2 match was already found.

Strengths: rules guarantee precision on unambiguous cases; ML focuses on
harder subset; directly addresses the brief's interest in rule+ML hybrids.

Weaknesses: careful non-overlap logic required; two sets of hyperparameters.

To Test Experimentally: Stage-A precision/recall contribution; whether
Stage-B reviews Stage-A decisions improves F_0.5; sensitivity to routing
definition.

Implementation: composition of Approach 1 and Approach 7 with routing/merge.

Expected Computational Cost: LOW-MODERATE.

--------------------------------------------------------------------------------
APPROACH 15 — Consistency-Rule Post-Filter on Top of Any Scorer
--------------------------------------------------------------------------------
Core Idea:
A precision-boosting post-processing layer that applies explicit
contradiction checks after obtaining above-threshold candidates from any
upstream model (e.g., reject a high-name-similarity match if address PIN
is present on both sides AND different AND address token Jaccard is low).

DATA-GROUNDED NOTES:
- Address missingness (~3.3% in S2/S3): missing address MUST NOT trigger
  consistency rejection. A record missing an address provides no contradiction
  evidence, not negative evidence. The rule must check "both sides have an
  address AND they contradict" before rejecting.
- Given the 5.6% no-match rate, this post-filter's precision boost matters:
  correctly flipping a borderline-positive prediction to no-match for a true
  no-match S1 entity contributes a full +1.0 to F_0.5 for that entity.
- Country contradiction: if S1 is US and candidate is India (or France),
  this is a legitimate consistency check since genuine matches should share
  country. However, VALIDATE this rule — with ~40% India and ~60% US in
  training, the rule might over-reject if countries are occasionally
  misrecorded in the sources.
- This layer is CHEAP and MODULAR — apply on top of every approach and
  report the before/after delta per rule to derive an empirical justification.

Strengths: directly targets false positives (costly under F_0.5); cheap;
modular.

Weaknesses: can only hurt recall, never help — must validate precision gain
outweighs recall loss.

To Test Experimentally: per-rule ablation on Approach 7; validate each
rule's net F_0.5 impact before enabling it.

Expected Computational Cost: LOW.

--------------------------------------------------------------------------------
APPROACH 16 — Adaptive / Per-Source Calibrated Thresholding
--------------------------------------------------------------------------------
Core Idea:
After any upstream scorer, fit adaptive thresholds varying by source (S2 vs
S3), by candidate-count regime, or by a calibrated probability floor combined
with a relative-margin rule. Distinct from other approaches because the
object of experimentation is the decision policy itself, not the scorer.

DATA-GROUNDED NOTES:
- The confirmed S2/S3 asymmetry (143K S2-only vs 164K S3-only matched
  entities, different match count distributions if they exist) directly
  motivates per-source thresholds. This should be one of the first
  threshold experiments after establishing a baseline scorer.
- Candidate-count-adaptive thresholds: S1 entities with many candidates
  (returned by the blocker) need a higher precision bar to avoid false
  positives from the larger candidate pool. S1 entities with zero or one
  candidate need a lower bar (or explicit "emit if single surviving candidate
  above an absolute floor").
- With 5.6% true no-match: calibrate the threshold to correctly emit empty
  for ~5.6% of S1 entities on validation. If the upstream model is well-
  calibrated, the threshold that achieves this should also maximize F_0.5.
  Validate this assumption — don't assume calibration without checking.
- France in test: if the threshold is tuned on US+India validation data, it
  may be miscalibrated for France due to different address/name string length
  distributions. If the validation split includes France-like examples
  (possible if the team constructs an OOD validation split from properties),
  threshold generalization can be tested pre-submission.

Strengths: cheap; improves entity-level F_0.5 without touching the scorer;
directly tunable against the actual metric.

Weaknesses: entirely dependent on upstream score quality; overfitting risk
on small validation sets.

To Test Experimentally: global vs per-source vs relative-margin thresholds,
all on the SAME upstream score distribution; Platt vs isotonic calibration.

Expected Computational Cost: LOW.

--------------------------------------------------------------------------------
APPROACH 17 — Model Ensemble (Score Fusion Across Heterogeneous Scorers)
--------------------------------------------------------------------------------
Core Idea:
Combine predictions from multiple heterogeneous scoring approaches (e.g.,
Approach 6 probabilistic score, Approach 7 GBM, Approach 10 ranking) via
score averaging, weighted averaging, or stacking on a SHARED candidate pool.

DATA-GROUNDED NOTES:
- Shared candidate pool is critical: with 22.7T possible pairs and blocking
  as the only tractable way to generate candidates, all base models MUST use
  the SAME blocked candidate set to be comparable and combinable. A different
  blocker per base model makes fusion meaningless.
- With 2.2M S1 entities, training multiple base models (each requiring
  feature computation over ~110M candidate pairs) is expensive in aggregate.
  Plan compute budget before committing to this approach.
- Stacking meta-classifier: requires a nested validation split. With 2.2M S1
  entities, a 5-fold cross-stacking setup is feasible but requires careful
  implementation to avoid data leakage.
- Error correlation should be measured empirically (per the original brief)
  before assuming ensemble benefit: if Approach 6 and Approach 7 are both
  trained on the same features and make similar errors, fusion adds nothing.

Strengths: reduces variance; corrects uncorrelated errors across model
families; stacking learns relative trust.

Weaknesses: multiplicative compute cost; no guaranteed improvement if
base models are correlated; stacking requires nested validation.

To Test Experimentally: correlation analysis of base model errors first;
average vs weighted vs stacked; ablation of which models contribute most.

Expected Computational Cost: HIGH (sum of all base models' costs).

--------------------------------------------------------------------------------
APPROACH 18 — Blocking Ensemble with Candidate-Source Provenance Features
--------------------------------------------------------------------------------
Core Idea:
Run every blocking strategy independently, take their union as the candidate
pool, AND retain WHICH blocker(s) recovered each candidate as explicit
categorical/multi-hot features fed into the downstream classifier. The
hypothesis is that "how a candidate was found" is informative about its
reliability.

DATA-GROUNDED NOTES:
- At 2.2M S1 and 10.3M S2+S3, the union candidate pool size must be
  monitored. If each blocker independently returns ~50 candidates per S1,
  the naive union could be 200 candidates per S1 × 2.2M S1 = 440M pairs
  to score. Deduplicate the union carefully and cap per-S1 candidate counts
  if needed.
- Provenance multi-hot features (found_by_exact_name, found_by_pin,
  found_by_tfidf, found_by_char_ngram, n_blockers_that_found_this) add only
  4-5 columns to the feature matrix — negligible memory overhead.
- The "n_blockers_that_found_this" count feature is particularly valuable:
  a candidate found by 3 independent blocking strategies is a priori more
  reliable than one found by only the loosest character n-gram blocker.
- For France test records: if the PIN blocker and exact-name blocker do not
  fire (due to format differences), France candidates will be found only by
  the TF-IDF and char n-gram blockers. The classifier can learn that
  France candidates should rely more on text similarity features, but only
  if it sees enough France-like training examples. This is a generalization
  gap that cannot be fully addressed without France training data.
- This approach's per-blocker recall contribution analysis (Section 41 of
  the brief) is directly useful for diagnosing which blockers are earning
  their compute cost at this scale.

Strengths: cheap to add on top of a planned multi-blocker union; gives
downstream classifier a reliability signal; provides blocking ablation
analysis as a side effect.

Weaknesses: engineering overhead for tagging; provenance features could
overfit to train-specific blocker precision profiles.

To Test Experimentally: F_0.5 with vs without provenance features; per-
blocker unique-recall contribution; candidate volume from the union.

Expected Computational Cost: LOW-MODERATE overhead on top of Approach 7.

================================================================================
6. POTENTIAL COMBINATION STRATEGIES
================================================================================
These are scientifically interesting combinations to test — not
recommendations of a winner. Each entry states WHY the combination might
plausibly complement its parts; whether it actually helps must be validated.

1. High-recall blocker union (Approaches 3+4+5) + GBM classifier
   (Approach 7): Different blockers recover different true matches (lexical
   vs typo-tolerant vs structured), so their union should raise candidate
   recall. This is essentially Approach 7 as specified; worth listing
   explicitly as the "default combination" all experiments are compared to.

2. TF-IDF retrieval (Approach 3) as Stage 1 + GBM reranker (Approach 7)
   as Stage 2 (= Approach 9): complementary because Stage 1 is tuned for
   recall while Stage 2 is tuned for precision. Decoupling may outperform
   a single-pass pipeline.

3. Deterministic rules (Approach 1) + ML fallback (Approach 7) (=
   Approach 14): complementary because rules guarantee precision on
   unambiguous cases, freeing the ML model to focus threshold tuning on the
   genuinely ambiguous remainder.

4. Probabilistic linkage (Approach 6) + GBM (Approach 7), fused via
   Approach 17: complementary because Fellegi-Sunter's field-independent
   score and GBM's learned nonlinear interactions likely make different
   errors — worth testing for genuinely uncorrelated error patterns.

5. Neural embeddings (Approach 11) + classical similarity features
   (Approach 7's feature set), combined as extra input features to a single
   GBM: complementary because embeddings may capture semantic/soft similarity
   that string-edit-distance features miss, while classical features remain
   robust when the embedding model encounters unseen French names.

6. Blocking ensemble with provenance (Approach 18) + adaptive thresholding
   (Approach 16): complementary because provenance features let the classifier
   learn candidate reliability, and adaptive thresholds further exploit that
   by using different cutoffs for different provenance/source regimes.

7. Three-way classifier ensemble (Approach 17) from Approach 6
   (probabilistic), Approach 7 (GBM), Approach 10 (ranking): a three-way
   fusion spanning generative, discriminative, and ranking paradigms is
   most likely to have uncorrelated errors. Most expensive to validate.

8. Consistency-rule post-filter (Approach 15) applied as a standard final
   layer on EVERY scoring approach: since it is cheap and modular, test it
   as an add-on to each approach individually to see which it helps most.
   Given the 5.6% no-match rate and high-precision F_0.5 metric, even a
   modest precision boost from the filter could substantially improve
   entity-level F_0.5 for approaches with looser thresholds.

9. Source-specific models (Approach 8) + adaptive per-source thresholding
   (Approach 16): natural pairing since the data confirms S2 and S3 are NOT
   symmetric. Test whether source-specific models alone capture the benefit,
   or whether separate thresholds compound the gain.

================================================================================
7. COMMON VALIDATION PROTOCOL (REQUIRED FOR ALL TEAMMATES)
================================================================================
To allow fair comparison across independently implemented approaches,
every teammate MUST use:

1. The SAME train/validation split, performed at the Source 1 entity level
   (no S1 entity's pairs split across train and validation). Fix and share
   a single random seed and split file (a CSV of source1_entity_id -> split
   assignment). Suggested: 80/20 stratified by country to ensure France-like
   distribution doesn't accidentally appear only in train or only in val.
   NOTE: France is NOT in train data — stratify by country within the
   existing US/India partition; there is no France-aware stratification
   possible.

2. The SAME entity-level macro F_0.5 evaluator implementation (a single
   shared evaluate.py used by every approach). This evaluator must:
   - Compute precision, recall, F_0.5 per S1 entity.
   - Score an empty-prediction/empty-ground-truth match as F_0.5 = 1.0.
   - Score a non-empty prediction against an empty ground truth as F_0.5=0.0.
   - Macro-average across all S1 entities, including the 5.58% no-match.

3. The SAME candidate-recall measurement:
   candidate_recall = (true matching pairs in the emitted candidate set)
                      / (total true matching pairs on validation split).
   Computed per blocking strategy independently and for the union.

4. Consistent treatment of empty ground truth (5.58% of S1 entities have
   no true match). Approaches MUST handle this as a real case.

5. Consistent submission-format validation using the shared
   utils/validate_submission.py before reporting any results.

6. Every teammate logs the full metric suite into the shared Experiment
   Result Template (Section 9) so results are directly comparable.

================================================================================
8. SUGGESTED EXPERIMENT MATRIX
================================================================================
Group A — Candidate Generation / Blocking
  A1. Exact normalized name blocking (part of Approach 1)
  A2. TF-IDF word n-gram retrieval (Approach 3)
  A3. TF-IDF character n-gram retrieval (Approach 3 variant)
  A4. Character n-gram inverted-index blocking (Approach 4)
  A5. Structured address key blocking: PIN/ZIP/postal code / house number
      (Approach 5) — must support US ZIP, India PIN, France postal code
  A6. Full blocker union + provenance tagging (Approach 18)
  Priority: Group A should be run FIRST because candidate recall is the
  hard ceiling on all downstream approaches' achievable recall.
  Deliverable: candidate recall, avg/median/max candidates per S1,
  % zero-candidate S1, runtime.

Group B — Representations
  B1. Handcrafted similarity features only
  B2. TF-IDF vector features as direct classifier inputs
  B3. Learned dense embeddings (Approach 11) as classifier inputs
  B4. Provenance/blocker-origin features (Approach 18) added to B1
  Deliverable: entity-level macro F_0.5 holding classifier and candidate
  set fixed, varying only the representation.

Group C — Pairwise Feature Ablation
  C1. Name-only feature set
  C2. Address-only feature set
  C3. Name + address feature set
  C4. Name + address + country feature set
  C5. Name + address + structured numeric features
  C6. Full feature set
  Deliverable: entity-level macro F_0.5 per ablation.

Group D — Matching Models
  D1. Deterministic rule cascade (Approach 1)
  D2. Fuzzy threshold matching (Approach 2)
  D3. Fellegi-Sunter probabilistic linkage (Approach 6)
  D4. Gradient-boosted pairwise classifier (Approach 7) — try
      LightGBM/XGBoost/CatBoost/HistGradientBoosting as sub-variants
  D5. Learning-to-rank model (Approach 10)
  D6. Siamese embedding similarity (Approach 11)
  D7. Cross-encoder classifier (Approach 12) — ONLY AS RERANKER on small
      top-K pre-filtered set due to inference cost at 110M candidate pairs
  Deliverable: entity-level macro F_0.5 on the identical validation split
  and identical candidate pool.

Group E — Decision Policies
  E1. Fixed global threshold, swept on validation
  E2. Per-source adaptive threshold (Approach 16)
  E3. Relative-margin-to-best-score emission rule (Approach 16)
  E4. Consistency-rule post-filter (Approach 15), rule-by-rule ablation
  E5. Calibration (Platt vs isotonic) before thresholding
  Deliverable: entity-level macro F_0.5 holding the upstream scorer fixed.

Group F — Source-Specific Models
  F1. Single shared classifier for S1-S2 and S1-S3 (Approach 7 as-is)
  F2. Separate classifiers per source (Approach 8)
  F3. Shared classifier with source as an explicit feature
  Deliverable: entity-level macro F_0.5, decomposed by S2-only / S3-only /
  both-matched S1 subpopulations.

Group G — Hybrid Systems
  G1. Rules + ML fallback cascade (Approach 14)
  G2. Retrieve-then-rerank two-stage pipeline (Approach 9)
  G3. Graph-based collective resolution on top of the best pairwise scorer
      (Approach 13) — chain-merging risk must be monitored
  Deliverable: entity-level macro F_0.5 vs the best single-stage baseline
  from Group D.

Group H — Ensembles
  H1. Score-fusion ensemble across heterogeneous scorers (Approach 17)
  H2. Blocking ensemble with provenance features (A6/B4 cross-reference)
  Deliverable: entity-level macro F_0.5 for the fused system vs each base
  model individually, plus base-model error-correlation analysis.

================================================================================
9. EXPERIMENT RESULT TEMPLATE (fill in per implemented approach)
================================================================================
Approach:
Implementation:
Validation split:
Candidate generation:
Model:
Features:
Threshold:
Candidate Recall:
Entity-level Precision:
Entity-level Recall:
Entity-level Macro F_0.5:
False Positives:
False Negatives:
Runtime:
Memory:
Major Failure Modes:
Observations:
Potential Improvements:

================================================================================
10. FINAL NOTES FOR THE TEAM
================================================================================
- This document intentionally does not rank or select a winning approach.
  Every approach above is a testable hypothesis, not a recommendation.

- SCALE IS THE DEFINING CONSTRAINT: 2.2M S1, 5M S2, 5M S3 — 22.7 TRILLION
  cross-product pairs. Every approach must design for this scale from the
  start. No brute-force enumeration, even for debugging.

- MULTI-MATCH IS DOMINANT: 80%+ of matched S1 entities have 3+ matches
  (median 3, max 11). Any approach that emits at most 1 or 2 candidates
  will systematically under-recall. Validate candidate K and emission rules
  against the max match count.

- NO-MATCH IS REAL BUT MINORITY: 5.58% of S1 entities have no true match.
  The decision-policy layer (Approach 15/16) must be calibrated so that
  approximately this fraction gets empty predictions.

- FRANCE IS IN TEST ONLY: All training data is US and India. France
  constitutes ~3.2% of test S1 entities. Normalization, address parsing,
  and postal code extraction must all be country-agnostic.

- PRECISION SENSITIVITY IS THE PRIMARY TUNING TARGET: F_0.5 weights
  precision 2x recall. Every decision policy and threshold must be tuned
  directly against entity-level macro F_0.5, not against pairwise AUC,
  accuracy, or plain F1.

- CANDIDATE GENERATION QUALITY IS A HARD CEILING: Group A experiments
  must be run first and their results shared across the team before anyone
  implements a downstream scorer, so all approaches start from the same
  (high-recall) candidate pool.

- USE THE SHARED VALIDATION SPLIT AND SHARED EVALUATOR for every reported
  number in the Experiment Result Template.

- NO EXTERNAL DATA, APIs, OR LOOKUPS OF ANY KIND under any framing.
  This is a disqualification risk.

- THE REMAINING EDA CHECKLIST ITEMS (exact-name-match rate on true pairs,
  exact-address-match rate, string length distributions, transliteration
  sample, within-source duplicate rate) MUST be completed by whoever owns
  the shared repo setup before approach-specific implementation begins.
  These statistics directly validate or invalidate specific design choices
  in Approaches 1, 2, 5, and 11.
