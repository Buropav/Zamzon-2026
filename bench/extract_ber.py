"""Writes the ber/*.py modules of a notebook's %%writefile cells to <out>/ber/ (verbatim, as %%writefile would).
    python bench/extract_ber.py erk_sub2_1.ipynb <out>"""
import json
import os
import re
import sys

nb, out = sys.argv[1], sys.argv[2]
os.makedirs(os.path.join(out, "ber"), exist_ok=True)
n = 0
for c in json.load(open(nb, encoding="utf-8"))["cells"]:
    s = "".join(c["source"])
    m = re.match(r'%%writefile "\{SRC_DIR\}/ber/(\w+\.py)"\n', s)
    if m:
        open(os.path.join(out, "ber", m.group(1)), "w", encoding="utf-8").write(s[m.end():].rstrip("\n") + "\n")
        n += 1
assert n >= 10, f"only {n} modules found in {nb}"
