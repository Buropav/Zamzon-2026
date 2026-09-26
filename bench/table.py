"""Results table from bench/variant.sh runs: mean / min / max F0.5 per (benchmark, variant) and deltas vs A.
    python bench/table.py <runs dir>"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

rows = defaultdict(list)
for d in sorted(glob.glob(os.path.join(sys.argv[1], "*", "*_r*"))):
    bench, run = d.split(os.sep)[-2:]
    name = re.sub(r"_r\d+$", "", run)
    st = open(f"{d}/status.txt").read() if os.path.exists(f"{d}/status.txt") else "exit=? wall_s=?"
    sc = json.load(open(f"{d}/score.json")) if os.path.exists(f"{d}/score.json") and os.path.getsize(f"{d}/score.json") else {}
    peak = ""
    if os.path.exists(f"{d}/driver_report.json"):
        peak = f"{json.load(open(f'{d}/driver_report.json')).get('peak_rss_gb', 0):.1f}"
    rows[(bench, name)].append((sc, st, peak))
print("| benchmark | variant | runs ok | F0.5 mean | min | max | delta vs A | precision | recall | singleton acc | cand. recall | cand/S1 | wall s | peak RSS GB |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
base = {}
for (bench, name), rs in sorted(rows.items()):
    f = [r[0]["macro_f05"] for r in rs if "macro_f05" in r[0]]
    if name == "A" and f:
        base[bench] = sum(f) / len(f)
for (bench, name), rs in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1] != "A", kv[0][1])):
    ok = [r for r in rs if "macro_f05" in r[0]]
    if not ok:
        print(f"| {bench} | {name} | 0/{len(rs)} | FAILED: {rs[0][1]} |||||||||||")
        continue
    g = lambda k: sum(r[0][k] for r in ok) / len(ok)  # noqa: E731
    f = [r[0]["macro_f05"] for r in ok]
    walls = [int(re.search(r"wall_s=(\d+)", r[1]).group(1)) for r in ok if re.search(r"wall_s=(\d+)", r[1])]
    d = f"{g('macro_f05') - base[bench]:+.4f}" if bench in base else "n/a"
    print(f"| {bench} | {name} | {len(ok)}/{len(rs)} | {g('macro_f05'):.5f} | {min(f):.5f} | {max(f):.5f} | {d} | "
          f"{g('precision'):.4f} | {g('recall'):.4f} | {g('singleton_acc'):.4f} | {g('candidate_recall'):.4f} | "
          f"{g('cand_per_s1'):.1f} | {sum(walls) // max(len(walls), 1)} | {max((r[2] for r in ok), default='')} |")
