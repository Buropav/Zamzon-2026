"""
mine.py - data-driven candidate equivalences from TRAIN ground-truth pairs (no API).

For every true pair, tokens present on only one side are "swapped" for tokens only on the other
side. Unigram and bigram swaps are counted per country x field. The rate (swaps / occurrences of
the rarer side) separates systematic variants (st/street, kansaltents/consultants) from noise.
Writes jev/work/mined.tsv.
"""
import os
import pickle
from collections import Counter

import pandas as pd

WORK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "work")


def grams(toks):
    return set(toks) | {" ".join(p) for p in zip(toks, toks[1:])}


def mine(tp, min_n=15, min_rate=0.05, max_side=3):
    rows = []
    for (ctry, field), d in [((c, f), tp[tp.country == c]) for c in tp.country.unique() for f in ("name", "addr")]:
        sub, occ = Counter(), Counter()
        for a, b in zip(d[field + "_a"], d[field + "_b"]):
            ta, tb = a.split(), b.split()
            A, B = set(ta), set(tb)
            oa, ob = A - B, B - A
            ga, gb = grams(ta), grams(tb)
            occ.update(ga)
            occ.update(gb)
            if not oa or not ob or len(oa) > max_side or len(ob) > max_side:
                continue
            # unigram swaps
            for x in oa:
                for y in ob:
                    sub[(x, y)] += 1
            # bigram <-> unigram swaps made only of one-sided tokens (tamil nadu <-> tn)
            for s_side, o_side, src in ((ta, ob, oa), (tb, oa, ob)):
                for p in zip(s_side, s_side[1:]):
                    if p[0] in src and p[1] in src:
                        for y in o_side:
                            sub[(" ".join(p), y)] += 1
        for (x, y), n in sub.items():
            if n < min_n:
                continue
            k = tuple(sorted((x, y)))
            rate = n / max(1, min(occ[x], occ[y]))
            if rate >= min_rate:
                rows.append((ctry, field, k[0], k[1], n, rate, occ[k[0]], occ[k[1]]))
    df = pd.DataFrame(rows, columns=["country", "field", "a", "b", "n", "rate", "occ_a", "occ_b"])
    df = df.groupby(["country", "field", "a", "b"], as_index=False).agg(
        n=("n", "sum"), rate=("rate", "max"), occ_a=("occ_a", "max"), occ_b=("occ_b", "max"))
    return df.sort_values("n", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    st = pickle.load(open(os.path.join(WORK, "stats.pkl"), "rb"))
    df = mine(st["true_pairs"])
    df.to_csv(os.path.join(WORK, "mined.tsv"), sep="\t", index=False)
    print(len(df), df.groupby(["country", "field"]).size().to_dict())
