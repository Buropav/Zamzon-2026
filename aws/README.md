# Running a notebook on AWS

The pipelines are CPU-bound (text cleaning, blocking, pair features); only XGBoost uses the GPU.
Kaggle's limits are its 4 CPU cores and ~32 GB RAM, so pick an instance for cores + RAM:

| instance | vCPU | RAM | GPU | use |
|---|---|---|---|---|
| g5.8xlarge / g6.8xlarge | 32 | 128 GB | 1 (A10G / L4) | any notebook |
| r7i.8xlarge | 32 | 256 GB | none | friend-based notebooks (XGBoost falls back to CPU) |

Use an Ubuntu Deep Learning AMI (NVIDIA driver preinstalled) with ~100 GB disk. Stop the instance when idle.

```bash
git clone -b claude/festive-davinci-30m7g5 https://github.com/buropav/zamzon-2026.git && cd zamzon-2026
tmux new -s run                              # survives SSH disconnects
bash aws/run_notebook.sh erk_sub3a2.ipynb    # first call also installs Python 3.12 + packages and downloads the data
```

Results land in `~/zamzon/runs/<notebook name>/`: `output/matching_results.tsv` (leaderboard file),
`executed_<name>.ipynb` (all cell outputs), `run.log` and `mem.log` (RAM every 30 s; the last line of
`run.log` gives the peak). Each notebook runs in its own folder, so several can be run one after another.
