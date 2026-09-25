# Jev-built preprocessing resources

Static word lists that improve text normalisation. They were built **once, offline** with Jev
(TypeSafe AI, via Vercel AI Gateway). The competition pipeline never calls Jev or any other API:
it only reads the `.tsv` files in `resources/`.

## What Jev was used for (and not)

- Jev only saw **single words or short phrases**, plus a few neighbouring words taken from word
  counts. It never saw a business record, and it never judged whether two businesses match.
- Test data contributed **word counts only** (France has no training data). No test record was sent.
- Jev's answers were checked against evidence from the training ground truth wherever it exists.
  Rules in `build_resources.py` combine the two, and the check `validate_lexicon.py` runs on
  training pairs only.

## Files

| File | What it is |
|---|---|
| `resources/lexicon.tsv` | `country, field, variant, canonical`: word/phrase equivalences (`*` = all countries) |
| `resources/token_class.tsv` | `country, field, tok, kind, p`: legal_form / title / connector / generic_business / place / distinctive … |
| `resources/conflict_pairs.tsv` | `country, field, a, b, source`: look-alike words that are *different* names (patel/patil) |
| (loader) | `code/business_entity_resolution/src/er_lexicon.py`: OCR fix, per-country lexicon, `distinct_tokens`, `name_conflict` |
| `build_vocab.py` | Sampled word counts + TRAIN true pairs → `work/stats.pkl` |
| `mine.py` | Word/phrase swaps observed in TRAIN true pairs → `work/mined.tsv` |
| `questions.py`, `jev_jobs.py`, `jev_sprint.py` | Jev questions, jobs, and a cached, resumable client |
| `build_resources.py` | Acceptance rules → `resources/*.tsv` (+ audits in `work/`) |
| `validate_lexicon.py` | Before/after check on TRAIN pairs vs hard negatives |

## Where each list comes from

1. **Indian-script loanwords** (`kansaltents`→consultants, `praivet`→private, `vomve`→bombay): Jev picks
   the English word from candidates (TRAIN swap partners + sound-alike matches).
2. **Abbreviations** per country (`r`→rue, `imp`→impasse, `mh`→maharashtra, state codes): Jev picks the
   expansion. Names and single letters need higher confidence, and TRAIN evidence wins on conflicts.
3. **Mined swaps** from TRAIN true pairs (state names/codes, typos such as `hoouston`): Jev filters out
   noise. Half-phrases and padded phrases are rejected.
4. **Word types** for names/addresses (legal forms, titles, generic business words).
5. **Look-alikes** (`patel`/`patil`): TRAIN evidence decides for US/India, Jev for France.
6. Code rules: digit-for-letter OCR noise (`hea1th`, `6roup`, `lnc`), junk address tokens (`null`, `na`, `pmb`).

## Using it in the notebook (after the er_multilingual cell)

Already integrated: `build_notebook.py` embeds `code/business_entity_resolution/src/resources/*.tsv`
into cell 7, and `src/features.py` uses the lexicon, `distinct_tokens` and `name_conflict`.

## Rebuild

```bash
python jev/build_vocab.py          # a few minutes, sampled
python jev/mine.py
AI_GATEWAY_API_KEY=... python jev/jev_jobs.py   # cached in jev_cache.jsonl; re-runs are free
python jev/build_resources.py
python jev/validate_lexicon.py
cp jev/resources/*.tsv code/business_entity_resolution/src/resources/ && python build_notebook.py
```
