#!/usr/bin/env bash
# One benchmark run: bench/variant.sh <runs dir> <bench dir> <name> '<cfg json>' <patches,comma,separated|-> <repeat>
# Base code = the ber modules of erk_sub2_1.ipynb; patches from tools/patches/. Threads are capped so several
# runs can share a big machine (THREADS, default 4). Writes <runs>/<bench>/<name>_r<repeat>/{log.txt,score.json,time.txt}.
set -uo pipefail
RUNS=$1; BENCH=$2; NAME=$3; CFG=$4; PATCHES=$5; REP=$6
REPO=$(cd "$(dirname "$0")/.." && pwd)
T=${THREADS:-4}
V=$RUNS/$(basename "$BENCH")/${NAME}_r$REP
rm -rf "$V"; mkdir -p "$V"
python "$REPO/bench/extract_ber.py" "$REPO/erk_sub2_1.ipynb" "$V"
if [ "$PATCHES" != "-" ]; then
  for p in ${PATCHES//,/ }; do python "$REPO/tools/patches/$p.py" "$V/ber" > /dev/null || { echo "patch $p failed" > "$V/log.txt"; exit 1; }; done
fi
CFG2=$(python -c "import json,sys; d=json.loads(sys.argv[1]); d.setdefault('workers', $T); print(json.dumps(d))" "$CFG")
export OMP_NUM_THREADS=$T POLARS_MAX_THREADS=$T NUMBA_NUM_THREADS=$T RAYON_NUM_THREADS=$T
START=$(date +%s)
python "$REPO/bench/driver.py" "$V" "$BENCH/dataset" "$CFG2" > "$V/log.txt" 2>&1   # peak RSS recorded by driver.py
RC=$?
echo "exit=$RC wall_s=$(( $(date +%s) - START ))" > "$V/status.txt"
if [ $RC -eq 0 ]; then
  VAL=$(ls "$REPO"/utils/validate_submission.py 2>/dev/null | head -1)
  python "$REPO/bench/score.py" "$V/output" "$BENCH" $VAL > "$V/score.json" 2> "$V/score.err"
fi
rm -rf "$V/work"   # caches: several GB per run on the big benchmark
echo "$NAME r$REP on $(basename "$BENCH"): $(cat "$V/status.txt") $(cut -c1-120 "$V/score.json" 2>/dev/null)"
