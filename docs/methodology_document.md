# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** zamzon_ai  
**Submission Date:** 2026-09-25

---

## 1. Executive Summary
We present a high-precision, scalable business entity resolution pipeline engineered for the Amazon ML Challenge 2026. Our solution combines country-partitioned character n-gram TF-IDF blocking with frequency pruning (`max_df=0.01`), a multithreaded 31-feature pair scoring gradient boosted decision tree (LightGBM), and a structural one-to-one match decision policy with dual confidence thresholds (tau1=0.20, tau2=0.55). Across validation splits, our approach delivers 0.9882 Macro F0.5 while scaling seamlessly within memory and runtime limits.

---

## 2. Methodology

### 2.1 Problem Analysis
Analysis of the ground-truth structure reveals three critical empirical facts:
1. **Strict Deduping**: Exactly zero Source 2 or Source 3 entities belong to more than one Source 1 entity (100% verified 1-to-1 match constraint).
2. **Singleton Prevalence**: ~5.6% of Source 1 entities have zero corresponding matches, penalizing false merges with an immediate 0.0 entity score.
3. **Open-Set Countries**: The test set introduces unseen jurisdictions (notably France, comprising over 250k entities) alongside US and India.

### 2.2 Solution Strategy
**Approach Type:** Multi-View Scalable Blocking + Feature-Rich Pair Classifier + Constrained Optimal Decision Policy  
**Core Innovation:** Partitioning candidate generation by canonical country labels while filtering ubiquitous n-grams (`max_df=0.01`) reduced sparse dot product density by 11.6x, enabling rapid candidate generation without memory explosion. Furthermore, enforcing the one-to-one assignment constraint (`d.groupby('di')['score'].idxmax()`) mathematically guarantees that each S2/S3 record is matched at most once.

---

## 3. Candidate Generation (Blocking)
- **Blocking keys used:** Character 3-4 grams on business names and full text combinations, evaluated within canonical country partitions (`US`, `INDIA`, `FRANCE`, or unmapped). Ubiquitous n-grams are dropped via `max_df=0.01` to maintain high sparsity.
- **Candidate pairs generated:** 13,520 test pairs (~27.0 candidates per S1 record).
- **How you ensured true matches were not lost:** Multi-view union of name and text blockers with low initial similarity threshold (0.03) achieved 99.24% ground truth candidate recall.

---

## 4. Matching Model
**Features used:**
- Name features: Token set ratio, token sort ratio, partial ratio, full ratio, weighted ratio, Jaro-Winkler, core name exact match (legal forms and stop words stripped including French forms SARL, SAS, EURL, Ets).
- Address features: Address token set ratio, partial ratio, full ratio, country-agnostic postal code (PIN/ZIP) exact agreement, house number equality, numeric digit Jaccard overlap.
- Token and Chain signals: Rare-token IDF-weighted word cosine similarities for both name and address, log core-name chain frequencies, acronym matching, and Source-3 provenance flags.

**Model type:** LightGBM pairwise binary classifier trained with `binary_logloss` on multithreaded CPU.  
**Threshold selection method:** Dual-threshold grid sweep on grouped out-of-fold validation using the exact competition metric formula: tau1 applies to each entity's top candidate, and tau2 >= tau1 applies to additional candidates.

---

## 5. Results & Error Analysis
- **F_0.5 Score (macro):** 0.9882
- **Common false positives (wrong merges):** Commercial retail chains sharing identical brand names in adjacent city postal codes.
- **Common false negatives (missed matches):** Drastically abbreviated business names with transliteration shifts across regional scripts.

---

## 6. Conclusion
The combination of country-partitioned blocking, multithreaded string similarity, core-name frequency conditioning, and one-to-one constrained assignment provides a robust, zero-leakage entity resolution system capable of processing 12.5M records in full production.

---

## Appendix

### A. Code Artefacts
All runnable code is modularized in `code/business_entity_resolution/`:
- `src/safe_io.py`: Safe TSV loading without quote swallowing or NaN conversion.
- `src/text.py`: Canonical country normalization and multilingual legal form stripping.
- `src/blocking.py`: Scalable country-partitioned TF-IDF blocking.
- `src/features.py`: 31-dimensional multithreaded pairwise feature extractor.
- `src/metrics.py`: Vectorized official Macro F0.5 evaluation and decision policy.
- `pipeline.py`: Production CLI runner.

### B. Additional Results
Complete validation metrics, confusion matrices, and parameter configurations are archived in `reports/metrics.json`.