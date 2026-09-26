#!/usr/bin/env bash
# Benchmark suite for a big machine (e.g. r6i.8xlarge): builds two labelled benchmarks from the TRAIN files and
# runs every variant in bench/variants.txt REPEATS times on each, PAR runs at a time (THREADS threads each).
#
#   bash aws/bench.sh                  # defaults: REPEATS=3 PAR=6 THREADS=5
#   REPEATS=2 PAR=8 THREADS=4 bash aws/bench.sh
#
# Benchmarks (seed 42): bench_kykl = Kentucky+Kerala entities split 70/30 (the one used so far);
# bench_lb = leaderboard-like: WHOLE Kentucky+Kerala as test with 19% of their S1 removed (~40% of test S2/S3
# unmatched, like the real test), trained on Oregon+Karnataka+North Carolina+Punjab.
# Output: ~/zamzon/bench_runs/results.md (paste it back), per-run logs next to it.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
ROOT=${ZAMZON_ROOT:-$HOME/zamzon}
REPEATS=${REPEATS:-3}; PAR=${PAR:-6}; export THREADS=${THREADS:-5}
mkdir -p "$ROOT"; cd "$ROOT"

# environment + data: same as aws/run_notebook.sh
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
if [ ! -x .venv/bin/python ]; then
  uv venv -q --python 3.12 .venv
  VIRTUAL_ENV=$ROOT/.venv uv pip install -q pip ipykernel nbconvert nbclient jupyter_client \
    "polars==1.44.2" "rapidfuzz==3.14.6" "sparse_dot_topn==1.2.0" "xgboost==3.4.1" psutil \
    pandas numpy scipy scikit-learn numba indic-transliteration
fi
export PATH="$ROOT/.venv/bin:$PATH" VIRTUAL_ENV=$ROOT/.venv
if [ ! -f data/student_resource/dataset/train/train_source1.tsv ]; then
  mkdir -p data
  curl -fL --retry 4 -o data/data.zip "https://www.kaggle.com/api/v1/datasets/download/anuruprkrishnan/zamzon-2026"
  (cd data && unzip -q -o data.zip && rm data.zip)
fi
RAW=$ROOT/data/student_resource/dataset
[ -f "$REPO/utils/validate_submission.py" ] || cp data/student_resource/utils/validate_submission.py "$REPO/utils/"

# benchmarks (built once)
[ -f bench_kykl/stats.json ] || python "$REPO/bench/build.py" --raw "$RAW" --out bench_kykl --mode split --states KY,KL > build_kykl.log
[ -f bench_lb/stats.json ] || python "$REPO/bench/build.py" --raw "$RAW" --out bench_lb --mode lb \
  --test-states KY,KL --train-states OR,KA,NC,PB --test-orphan 0.19 > build_lb.log
python - <<'PY'
import json
for b in ("bench_kykl", "bench_lb"):
    s = json.load(open(f"{b}/stats.json"))
    print(b, {sp: {k: s[sp][k] for k in ("S1", "S2", "S3", "gt_pairs", "S23_unmatched_share")} for sp in ("train", "test")})
PY

# run matrix: every variant x repeat x benchmark, PAR at a time
RUNS=$ROOT/bench_runs; mkdir -p "$RUNS"
grep -v '^#' "$REPO/bench/variants.txt" | grep -v '^\s*$' | while IFS='|' read -r name cfg patches; do
  for b in bench_kykl bench_lb; do for r in $(seq 1 "$REPEATS"); do
    printf '%s\0%s\0%s\0%s\0%s\0%s\0' "$RUNS" "$ROOT/$b" "$name" "$cfg" "$patches" "$r"
  done; done
done | xargs -0 -n 6 -P "$PAR" bash "$REPO/bench/variant.sh" | tee -a "$RUNS/progress.log"
{ echo "## Benchmark results ($(date -u)), $(nproc) CPUs, REPEATS=$REPEATS PAR=$PAR THREADS=$THREADS"; echo;
  python "$REPO/bench/table.py" "$RUNS"; } > "$RUNS/results.md"
cat "$RUNS/results.md"
