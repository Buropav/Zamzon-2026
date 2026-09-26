#!/usr/bin/env bash
# Run one competition notebook end to end on an AWS machine (full dataset), with a RAM log.
#
#   bash aws/run_notebook.sh erk_sub3a2.ipynb            # any of the repo's notebooks
#
# First call installs Python 3.12 + packages into ~/zamzon/.venv and downloads the dataset once
# (public Kaggle endpoint, ~1.1 GB zip). Each notebook runs in its own folder ~/zamzon/runs/<name>/,
# so two notebooks never overwrite each other's output/. Run it inside tmux (or with nohup) so an
# SSH disconnect does not stop it. Results:
#   ~/zamzon/runs/<name>/output/matching_results.tsv   leaderboard file
#   ~/zamzon/runs/<name>/executed_<name>.ipynb         notebook with all cell outputs
#   ~/zamzon/runs/<name>/run.log, mem.log              progress + RAM every 30 s
set -euo pipefail
NB_PATH=$(readlink -f "${1:?usage: bash aws/run_notebook.sh <notebook.ipynb>}")
NAME=$(basename "$NB_PATH" .ipynb)
ROOT=${ZAMZON_ROOT:-$HOME/zamzon}
RUN=$ROOT/runs/$NAME
mkdir -p "$ROOT" "$RUN"
cd "$ROOT"

# 1. Python environment (once)
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
if [ ! -x .venv/bin/python ]; then
  uv venv -q --python 3.12 .venv
  VIRTUAL_ENV=$ROOT/.venv uv pip install -q pip ipykernel nbconvert nbclient jupyter_client \
    "polars==1.44.2" "rapidfuzz==3.14.6" "sparse_dot_topn==1.2.0" "xgboost==3.4.1" psutil \
    pandas numpy scipy scikit-learn numba indic-transliteration
  if command -v nvidia-smi >/dev/null; then VIRTUAL_ENV=$ROOT/.venv uv pip install -q cupy-cuda12x; fi
  .venv/bin/python -m ipykernel install --sys-prefix >/dev/null
fi
export PATH="$ROOT/.venv/bin:$PATH" VIRTUAL_ENV=$ROOT/.venv

# 2. Dataset (once), then hard-linked into the run folder (no extra disk; the notebooks
#    find student_resource/ by searching their working folder when not on Kaggle)
if [ ! -f data/student_resource/dataset/test/test_source1.tsv ]; then
  mkdir -p data
  if [ -n "${ZAMZON_ZIP:-}" ]; then cp "$ZAMZON_ZIP" data/data.zip   # already-downloaded zip (optional)
  else curl -fL --retry 4 -o data/data.zip "https://www.kaggle.com/api/v1/datasets/download/anuruprkrishnan/zamzon-2026"; fi
  (cd data && unzip -q -o data.zip && rm data.zip)
fi
[ -d "$RUN/student_resource" ] || cp -al data/student_resource "$RUN/"
cp "$NB_PATH" "$RUN/$NAME.ipynb"

# 3. Machine facts + RAM log every 30 s
cd "$RUN"
{ echo "== $(date -u) $NAME"; nproc; free -g | head -2; nvidia-smi -L 2>/dev/null || echo "no GPU"; } | tee run.log
( while true; do echo "$(date +%T) $(free -g | awk '/Mem/{print "used " $3 " GB, available " $7 " GB"}')"; sleep 30; done ) > mem.log &
MEMLOG=$!
trap 'kill $MEMLOG 2>/dev/null' EXIT

# 4. Run the notebook (no cell timeout)
START=$(date +%s)
set +e
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
  --output "executed_$NAME.ipynb" "$NAME.ipynb" >> run.log 2>&1
RC=$?
set -e
echo "== exit code $RC after $(( ($(date +%s) - START) / 60 )) min; peak RAM used: $(awk '{print $3}' mem.log | sort -n | tail -1) GB" | tee -a run.log
ls -la output 2>/dev/null | tee -a run.log || echo "no output/ folder" | tee -a run.log
exit $RC
