"""Builds erk_sub4.ipynb (AWS / big-RAM run) from erk_sub2_1.ipynb (friend's known-good notebook, never edited).

erk_sub4 = erk_sub3a3 (training fraction chosen by the extra-memory guard) with the options widened for a big
machine, plus the ported parts that passed the benchmark. Each part is applied with the SAME patch script that
was benchmarked (tools/patches/<part>.py) to the notebook's %%writefile module sources, so the notebook runs
byte-for-byte the benchmarked code. New modules become new %%writefile cells.

    python tools/make_sub4.py --parts numfeat,hnum_ctx,native_map --fracs 1.0,0.5,0.3
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_sub3a3 as s3  # noqa: E402

WF = re.compile(r'^%%writefile "\{SRC_DIR\}/ber/(\w+\.py)"\n')
ADDED_COLS = {"numfeat": 15, "hnum_ctx": 2, "native_map": 0}   # feature-matrix columns each part adds
DESCR = {
    "numfeat": "address-number alignment features (exact / small shift / one-digit typo / truncation, from the old pipeline)",
    "hnum_ctx": "same-house-number context in stage 2 (other candidates of the S1 sharing the house number)",
    "native_map": "native-script -> Latin word map learned from the TRAIN ground truth only",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="", help="comma-separated patch names from tools/patches/")
    ap.add_argument("--fracs", default="1.0,0.5,0.3")
    ap.add_argument("--out", default="erk_sub4.ipynb")
    a = ap.parse_args()
    parts = [p for p in a.parts.split(",") if p]
    fracs = [float(x) for x in a.fracs.split(",")]
    nb = json.loads(s3.SRC.read_text(encoding="utf-8"))

    # 1. erk_sub3a3's edits (training-fraction guard), with the options and column count for this build
    n_cols = 88 + sum(ADDED_COLS[p] for p in parts)
    header = ("# Amazon ML Challenge 2026 — Business Entity Resolution, submission 4 (big-RAM machine, e.g. AWS r6i.8xlarge)\n\n"
              "**Submission 4 = submission 2 (`erk_sub2_1`) + benchmarked changes:**\n"
              f"- training fraction: largest of {fracs} whose extra memory over 0.3 fits (see the `train_s1_frac` log lines)\n"
              + "".join(f"- {DESCR[p]}\n" for p in parts))
    for cell, old, new in s3.EDITS:
        if cell == 0:
            new = header
        elif cell == 3 and "train_s1_frac_options" in new:
            new = new.replace("[0.5, 0.3]", str(fracs))
        elif cell == 14 and "FEATURE_COLS = 88" in new:
            new = new.replace("FEATURE_COLS = 88      # stage-1 + stage-2 columns of X on the friend's feature set",
                              f"FEATURE_COLS = {n_cols}      # stage-1 + stage-2 columns of X (friend's 88 + ported parts)")
        src = "".join(nb["cells"][cell]["source"])
        assert src.count(old) == 1, (cell, old[:80])
        nb["cells"][cell]["source"] = src.replace(old, new)

    # 2. ported parts: run the benchmarked patch scripts on the notebook's module sources
    with tempfile.TemporaryDirectory() as tmp:
        ber = Path(tmp) / "ber"
        ber.mkdir()
        cell_of = {}
        for i, c in enumerate(nb["cells"]):
            m = WF.match("".join(c["source"]))
            if m:
                cell_of[m.group(1)] = i
                body = "".join(c["source"])[m.end():].rstrip("\n") + "\n"   # %%writefile output ends with a newline
                (ber / m.group(1)).write_text(body, encoding="utf-8")
        for p in parts:
            r = subprocess.run([sys.executable, str(ROOT / "tools" / "patches" / f"{p}.py"), str(ber)],
                               capture_output=True, text=True)
            assert r.returncode == 0, f"patch {p} failed:\n{r.stdout}\n{r.stderr}"
        new_cells = []
        for f in sorted(ber.glob("*.py")):
            body = f.read_text(encoding="utf-8")
            src = f'%%writefile "{{SRC_DIR}}/ber/{f.name}"\n' + body.rstrip("\n")
            if f.name in cell_of:
                nb["cells"][cell_of[f.name]]["source"] = src
            else:
                new_cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                                  "source": src})
        last = max(cell_of.values())
        nb["cells"][last + 1:last + 1] = new_cells   # new modules right after the existing ber modules

    # 3. packaging: numba is needed by numfeat
    if "numfeat" in parts:
        for c in nb["cells"]:
            s = "".join(c["source"])
            old = "xgboost==3.4.1           # Apache-2.0, the only learned model (GPU hist)\\n"
            if old in s:
                c["source"] = s.replace(old, old + "numba                    # BSD-2, JIT for the address-number features\\n")
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    out = ROOT / a.out
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out.name}: parts={parts}, fracs={fracs}, FEATURE_COLS={n_cols}, +{len(new_cells)} module cells")


if __name__ == "__main__":
    main()
