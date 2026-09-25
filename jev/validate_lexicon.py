"""
validate_lexicon.py - cheap offline check that the Jev resources help (TRAIN data only, no API).

Positives: sampled TRAIN ground-truth pairs. Hard negatives: for the same Source 1 record, the most
similar NON-matching Source 2/3 record from the sampled pool (char TF-IDF on the raw name+address).
For each normaliser (baseline normalize_ml vs wrapped) we report, per country:
  AUC of name/address token_set_ratio and token Jaccard  (higher = better separation)
  share of positives whose cleaned names are identical     (recall-type signal)
  share of hard negatives whose cleaned names are identical (false-merge risk)
Usage: python jev/validate_lexicon.py [--n 20000]
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code", "business_entity_resolution", "src"))  # shared modules
import er_multilingual as M  # noqa: E402
import er_lexicon as L  # noqa: E402

DATA = os.path.join(ROOT, "dataset", "train")


def load_raw(ids_by_src):
    out = {}
    for src, ids in ids_by_src.items():
        with open(os.path.join(DATA, f"train_source{src}.tsv"), encoding="utf-8") as f:
            f.readline()
            for line in f:
                eid = line[:line.index("\t")]
                if eid in ids:
                    p = line.rstrip("\n").split("\t")
                    out[eid] = (p[1], p[2])
    return out


def build_pairs(n, seed=0):
    tp = pickle.load(open(os.path.join(HERE, "work", "stats.pkl"), "rb"))["true_pairs"]
    rng = np.random.default_rng(seed)
    tp = tp.iloc[rng.permutation(len(tp))[:n]]
    ids = {1: set(tp.s1), 2: {x for x in tp.s23 if x.startswith("S2-")}, 3: {x for x in tp.s23 if x.startswith("S3-")}}
    raw = load_raw(ids)
    tp = tp[tp.s1.isin(raw) & tp.s23.isin(raw)].reset_index(drop=True)
    rows = [(r.country, r.s1, r.s23, 1) for r in tp.itertuples()]
    for ctry, d in tp.groupby("country"):
        s1u, s23u = d.s1.unique(), d.s23.unique()
        txt = lambda e: " ".join(raw[e])
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2).fit([txt(e) for e in s23u])
        Q, D = vec.transform([txt(e) for e in s1u]), vec.transform([txt(e) for e in s23u])
        truth = d.groupby("s1").s23.apply(set).to_dict()
        for s in range(0, len(s1u), 2000):
            S = (Q[s:s + 2000] @ D.T).toarray()
            for i, row in enumerate(S):
                a = s1u[s + i]
                for j in np.argsort(-row)[:5]:
                    if s23u[j] not in truth[a]:
                        rows.append((ctry, a, s23u[j], 0))
                        break
    return pd.DataFrame(rows, columns=["country", "a", "b", "y"]), raw


LEX_FOR_DISTINCT = None


def evaluate(pairs, raw, norm_for, label):
    ctry_of = dict(zip(pairs.a, pairs.country)) | dict(zip(pairs.b, pairs.country))
    ents = set(pairs.a) | set(pairs.b)
    na = {e: norm_for(ctry_of[e])(raw[e][0], "name") for e in ents}
    aa = {e: norm_for(ctry_of[e])(raw[e][1], "addr") for e in ents}
    jac = lambda x, y: len(set(x.split()) & set(y.split())) / max(1, len(set(x.split()) | set(y.split())))
    out = []
    for ctry, d in pairs.groupby("country"):
        f = {
            "name_tset": [fuzz.token_set_ratio(na[a], na[b]) for a, b in zip(d.a, d.b)],
            "name_jacc": [jac(na[a], na[b]) for a, b in zip(d.a, d.b)],
            "addr_tset": [fuzz.token_set_ratio(aa[a], aa[b]) for a, b in zip(d.a, d.b)],
            "addr_jacc": [jac(aa[a], aa[b]) for a, b in zip(d.a, d.b)],
        }
        if LEX_FOR_DISTINCT is not None:
            dt = {e: " ".join(L.distinct_tokens(na[e], LEX_FOR_DISTINCT, ctry)) for e in set(d.a) | set(d.b)}
            f["distinct_jacc"] = [jac(dt[a], dt[b]) for a, b in zip(d.a, d.b)]
        if LEX_FOR_DISTINCT is not None and LEX_FOR_DISTINCT["conflicts"]:
            cf = np.array([L.name_conflict(na[a].split(), na[b].split(), LEX_FOR_DISTINCT, ctry) for a, b in zip(d.a, d.b)])
        else:
            cf = np.zeros(len(d))
        eq = np.array([na[a] == na[b] for a, b in zip(d.a, d.b)])
        rec = {"norm": label, "country": ctry, "n_pos": int(d.y.sum()), "n_neg": int((1 - d.y).sum())}
        rec.update({f"auc_{k}": roc_auc_score(d.y, v) for k, v in f.items()})
        rec["name_equal_pos"] = eq[d.y.to_numpy() == 1].mean()
        rec["name_equal_neg"] = eq[d.y.to_numpy() == 0].mean()
        rec["conflict_pos"] = cf[d.y.to_numpy() == 1].mean()
        rec["conflict_neg"] = cf[d.y.to_numpy() == 0].mean()
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--resources", default=os.path.join(HERE, "resources"))
    a = ap.parse_args()
    pairs, raw = build_pairs(a.n)
    base = M.normalize_ml
    LEX = L.load(a.resources)
    wrapped = {c: L.wrap(base, LEX, c) for c in ("US", "INDIA", "FRANCE", "*")}
    ocr = L.wrap(base, {"lex": {"*": {"tok": {"name": {}, "addr": {}}, "phr": {"name": [], "addr": []}}}})
    global LEX_FOR_DISTINCT
    LEX_FOR_DISTINCT = LEX if (LEX["token_class"] or LEX["conflicts"]) else None
    res = (evaluate(pairs, raw, lambda c: base, "baseline") + evaluate(pairs, raw, lambda c: ocr, "ocr_fix_only") +
           evaluate(pairs, raw, lambda c: wrapped.get(c, wrapped["*"]), "jev_resources"))
    df = pd.DataFrame(res).sort_values(["country", "norm"])
    pd.set_option("display.width", 220)
    print(df.round(4).to_string(index=False))
    df.to_csv(os.path.join(HERE, "work", "validation.tsv"), sep="\t", index=False)


if __name__ == "__main__":
    main()
