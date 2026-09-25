# Submission 2026-09-25 18:11 IST - two-stage-xgb-gpu

- Commit: `cf433c1` (tag `sub-20260925-1811`)
- Leaderboard F0.5: pending

## Notes
First full GPU run (T4 x2): XGBoost two-stage, GPU blocking, 6 training regions. Training log: blocking recall 96.42%, held-out F0.5 stage 1 0.9670 -> two-stage 0.9736 (tau1=0.60, tau2=0.75), training 2207 s. Stage-1 XGBoost hit the 1500-tree cap. Add metrics.json and the leaderboard score when the run finishes.
