# Submissions

One folder per leaderboard submission, named `YYYY-MM-DD_HHMM_IST_<slug>`, holding:

- `entity_resolution.ipynb`: the exact notebook that was run on Kaggle
- `metrics.json`: the run's metrics (download it from the Kaggle output)
- `NOTES.md`: leaderboard score, commit, key numbers, what changed

Each entry also has a git tag (`sub-YYYYMMDD-HHMM`), so the full code is `git checkout <tag>`.
The output TSVs are not stored here (candidate_pairs.tsv exceeds GitHub's 100 MB limit).

```bash
python tools/record_submission.py --slug two-stage-xgb --commit <hash> --metrics ~/Downloads/metrics.json --note "..."
python tools/record_submission.py --dir submissions/<folder> --score 0.97xx
```

| Date (IST) | Slug | Held-out F0.5 | Leaderboard F0.5 |
|---|---|---|---|
| 2026-09-25 18:11 | [two-stage-xgb-gpu](2026-09-25_1811_IST_two-stage-xgb-gpu/NOTES.md) | 0.9736 | pending |
