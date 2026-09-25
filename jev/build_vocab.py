"""
build_vocab.py - light, sampled word statistics for the Jev lexicon jobs (runs in a few minutes).

Reads ~every Nth line of each source file (never the full pipeline) and writes jev/work/stats.pkl:
  vocab[(split, country, field)]      Counter of raw tokens (er_multilingual cleaning, _CANON OFF)
  translit[(country, field)]          Counter of tokens that came from Indic-script words (train only)
  translit_src[token]                 a few original-script spellings of that token (train only)
  neighbors[(country, field, tok)]    Counter of words next to short tokens (context for job 3)
  true_pairs                          DataFrame of TRAIN ground-truth pairs (raw tokens) for calibration

Test files contribute word COUNTS only (needed for France, which has no training data).
Usage: python jev/build_vocab.py [--per-file 300000]
"""
import argparse
import os
import pickle
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code", "business_entity_resolution", "src"))  # shared modules
import er_multilingual as M  # noqa: E402

DATA = os.path.join(ROOT, "dataset")
WORK = os.path.join(ROOT, "jev", "work")
FILES = {("train", s): f"train/train_source{s}.tsv" for s in (1, 2, 3)}
FILES.update({("test", s): f"test/test_source{s}.tsv" for s in (1, 2, 3)})
_INDIC = re.compile(r"[ऀ-෿؀-ۿ]")
SHORT = 4  # neighbours are kept for alphabetic tokens up to this length

M._CANON.clear()  # raw tokens: the lexicon keys must be the words BEFORE canonicalisation


def raw(text, kind):
    return M.normalize_ml(text, kind)


def country_of(c):
    s = re.sub(r"[^a-z]", "", str(c).lower())
    return {"us": "US", "usa": "US", "unitedstates": "US", "in": "INDIA", "india": "INDIA",
            "fr": "FRANCE", "france": "FRANCE"}.get(s, s.upper())


def sample_lines(path, n, seed=0):
    with open(path, encoding="utf-8") as f:
        header = f.readline()
        total = sum(1 for _ in f)
    step = max(1, total // n)
    off = random.Random(seed).randrange(step)
    out = []
    with open(path, encoding="utf-8") as f:
        f.readline()
        for i, line in enumerate(f):
            if i % step == off:
                out.append(line.rstrip("\n").split("\t"))
    return header.rstrip("\n").split("\t"), out


def process(args):
    (split, src), rel, n = args
    _, rows = sample_lines(os.path.join(DATA, rel), n, seed=src)
    vocab, translit, neighbors = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    translit_src = defaultdict(Counter)
    for r in rows:
        if len(r) < 4:
            continue
        ctry = country_of(r[3])
        for col, field in ((1, "name"), (2, "addr")):
            text = r[col]
            toks = raw(text, field).split()
            vocab[(split, ctry, field)].update(toks)
            for i, t in enumerate(toks):
                if t.isalpha() and len(t) <= SHORT:
                    if i > 0:
                        neighbors[(ctry, field, t)]["L:" + toks[i - 1]] += 1
                    if i + 1 < len(toks):
                        neighbors[(ctry, field, t)]["R:" + toks[i + 1]] += 1
            if split == "train" and _INDIC.search(text):
                for w in text.split():
                    if _INDIC.search(w):
                        for t in raw(w, field).split():
                            if t.isalpha():
                                translit[(ctry, field)][t] += 1
                                translit_src[t][w.strip(",.()")] += 1
    return (split, src), rows if split == "train" and src == 1 else None, vocab, translit, translit_src, neighbors


def true_pairs(s1_rows, max_s1):
    """Ground-truth pairs for a sample of TRAIN Source 1 ids (streams GT and S2/S3 once)."""
    s1 = {r[0]: r for r in s1_rows[:max_s1] if len(r) >= 4}
    gt = {}
    with open(os.path.join(DATA, "train/train_ground_truth.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            a, _, b = line.rstrip("\n").partition("\t")
            if a in s1 and b:
                gt[a] = [x.strip() for x in b.split(",") if x.strip()]
    need = {x for ms in gt.values() for x in ms}
    s23 = {}
    for s in (2, 3):
        with open(os.path.join(DATA, f"train/train_source{s}.tsv"), encoding="utf-8") as f:
            f.readline()
            for line in f:
                eid = line[:line.index("\t")]
                if eid in need:
                    s23[eid] = line.rstrip("\n").split("\t")
    out = []
    for a, ms in gt.items():
        ra = s1[a]
        na, aa = raw(ra[1], "name"), raw(ra[2], "addr")
        for b in ms:
            rb = s23.get(b)
            if rb and len(rb) >= 4:
                out.append((country_of(ra[3]), a, b, na, raw(rb[1], "name"), aa, raw(rb[2], "addr"),
                            bool(_INDIC.search(rb[1]))))
    import pandas as pd
    return pd.DataFrame(out, columns=["country", "s1", "s23", "name_a", "name_b", "addr_a", "addr_b", "indic_b"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-file", type=int, default=300_000)
    ap.add_argument("--gt-s1", type=int, default=150_000)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    vocab, translit, neighbors = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    translit_src = defaultdict(Counter)
    s1_rows = None
    with ProcessPoolExecutor(a.workers) as ex:
        for key, rows, v, t, ts, nb in ex.map(process, [(k, rel, a.per_file) for k, rel in FILES.items()]):
            print("done", key, flush=True)
            s1_rows = rows if rows is not None else s1_rows
            for d, src in ((vocab, v), (translit, t), (neighbors, nb), (translit_src, ts)):
                for k, c in src.items():
                    d[k].update(c)
    print("mining ground-truth pairs...", flush=True)
    tp = true_pairs(s1_rows, a.gt_s1)
    stats = {"vocab": dict(vocab), "translit": dict(translit),
             "translit_src": {k: [w for w, _ in c.most_common(3)] for k, c in translit_src.items()},
             "neighbors": {k: c for k, c in neighbors.items() if sum(c.values()) >= 20},
             "true_pairs": tp}
    with open(os.path.join(WORK, "stats.pkl"), "wb") as f:
        pickle.dump(stats, f)
    print({k: len(v) for k, v in stats["vocab"].items()})
    print("true pairs:", len(tp), tp.country.value_counts().to_dict())


if __name__ == "__main__":
    main()
