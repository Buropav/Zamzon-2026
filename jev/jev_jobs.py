"""
jev_jobs.py - builds the Jev requests for the six lexicon jobs, runs them (cached/resumable) and
saves raw judgements to jev/work/judged_<job>.pkl. Deciding what to accept is build_resources.py.

  1 translit   Indic-script loanwords -> English word          (choice among key-similar words)
  2 abbrev     short tokens -> which long token they stand for  (choice, per country x field)
  3 ambiguous  st/ste/dr/co/... with neighbour context          (choice)
  4 wordclass  frequent tokens -> legal form / title / generic / place / distinctive
  5 lookalike  similar-looking frequent tokens: same word or different names?
  6 mined      data-mined swaps from TRAIN true pairs (incl. places, state codes, phrases)

Only single words / short phrases (plus word-count context) are sent. Never a record.
Usage:  python jev/jev_jobs.py translit abbrev ...   (no args = all jobs)
"""
import os
import pickle
import re
import sys
from collections import Counter

import pandas as pd
from rapidfuzz import fuzz, process

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code", "business_entity_resolution", "src"))  # shared modules
import er_multilingual as M  # noqa: E402
import jev_sprint as J  # noqa: E402
import questions as Q  # noqa: E402

WORK = os.path.join(HERE, "work")
ST = pickle.load(open(os.path.join(WORK, "stats.pkl"), "rb"))
V = ST["vocab"]
COUNTRIES = {"US": "train", "INDIA": "train", "FRANCE": "test"}  # where each country's counts come from


def vocab(country, field):
    c = Counter(V.get(("train", country, field), Counter()))
    c.update(V.get(("test", country, field), Counter()))
    return c


# ---------------------------------------------------------------------------------------------
# 1. transliterated loanwords
# ---------------------------------------------------------------------------------------------
_LK = [("ph", "f"), ("bh", "b"), ("dh", "d"), ("th", "t"), ("kh", "k"), ("gh", "g"), ("sh", "s"),
       ("ch", "c"), ("ck", "k"), ("q", "k"), ("x", "ks"), ("w", "v"), ("z", "s"), ("j", "s"),
       ("c", "k"), ("b", "p"), ("d", "t"), ("g", "k"), ("v", "p"), ("f", "p")]


def loan_key(w):
    for a, b in _LK:
        w = w.replace(a, b)
    w = w[0] + re.sub(r"[aeiouyh]", "", w[1:]) if w else w
    return re.sub(r"(.)\1+", r"\1", w)


def _mined_partners(country="INDIA", field="name", min_rate=0.03):
    d = pd.read_csv(os.path.join(WORK, "mined.tsv"), sep="\t", keep_default_na=False)
    d = d[(d.country == country) & (d.field == field) & (d.rate >= min_rate)]
    part = {}
    for a, b, n in zip(d.a, d.b, d.n):
        if " " in a or " " in b:
            continue
        part.setdefault(a, []).append((n, b))
        part.setdefault(b, []).append((n, a))
    return {k: [w for _, w in sorted(v, reverse=True)] for k, v in part.items()}


def _noisy(w):
    return bool(re.search(r"\d", w)) or bool(re.match(r"l[nm]", w))


def job_translit(k_key=6, k_raw=3, k_mined=4, min_count=3):
    tl = Counter()
    for (ctry, field), c in ST["translit"].items():
        tl.update(c)
    eng = Counter()
    for ctry in ("US", "INDIA"):
        eng.update({w: n for w, n in V[("train", ctry, "name")].items() if w.isalpha() and n >= 20})
        eng.update({w: n for w, n in V[("train", ctry, "addr")].items() if w.isalpha() and n >= 20})
    eng_words = [w for w in eng if tl.get(w, 0) < eng[w] * 0.5 and len(w) >= 2 and not _noisy(w)]
    partners = _mined_partners()
    eng_set = set(eng_words)
    eng_keys = [loan_key(w) for w in eng_words]
    items, reqs = [], []
    for t, n in tl.most_common():
        if n < min_count or len(t) < 2:
            continue
        cands = [w for w in partners.get(t, []) if w in eng_set][:k_mined]
        cands += [eng_words[i] for _, _, i in process.extract(loan_key(t), eng_keys, scorer=fuzz.ratio, limit=k_key)]
        cands += [w for w, _, _ in process.extract(t, eng_words, scorer=fuzz.ratio, limit=k_raw)]
        cands = [c for c in dict.fromkeys(cands) if c != t][:12]
        crit = {c: f"the English word '{c}'" for c in cands}
        crit["none"] = "none of these: an Indian word or name, or a different English word"
        q = {"english": {"type": "choice",
                         "instructions": ("word is an English word written in an Indian script and then romanised "
                                          "letter by letter (e.g. kansaltents = consultants, praivet = private, "
                                          "phuds = foods). Which English word is it?"),
                         "criteria": crit}}
        items.append({"tok": t, "count": n, "cands": cands})
        reqs.append(({"word": t, "original_script": ST["translit_src"].get(t, [])}, q))
    return items, reqs


# ---------------------------------------------------------------------------------------------
# 2. abbreviations (per country x field)
# ---------------------------------------------------------------------------------------------
def job_abbrev(min_short=60, min_long=20, max_short=4, k=10):
    items, reqs = [], []
    for ctry in COUNTRIES:
        for field in ("name", "addr"):
            v = vocab(ctry, field)
            longs = [(n, w) for w, n in v.items() if n >= min_long and w.isalpha() and len(w) >= 3]
            longs.sort(reverse=True)
            for s, n in v.most_common():
                if n < min_short:
                    break
                if not s.isalpha() or len(s) > max_short:
                    continue
                cands = [w for _, w in longs if M._is_abbrev(s, w)][:k]
                if not cands:
                    continue
                crit = {c: f"'{s}' stands for '{c}'" for c in cands}
                crit["none"] = f"'{s}' is a word or name of its own, or stands for something else"
                q = {"expansion": {"type": "choice",
                                   "instructions": f"In {Q.where(ctry, field)}, what does the short form '{s}' usually stand for?",
                                   "criteria": crit}}
                items.append({"country": ctry, "field": field, "tok": s, "count": n, "cands": cands})
                reqs.append(({"short_form": s, "found_in": Q.where(ctry, field)}, q))
    return items, reqs


# ---------------------------------------------------------------------------------------------
# 3. ambiguous short forms with neighbour context
# ---------------------------------------------------------------------------------------------
AMBIG = {"st": ["saint", "street", "state"], "ste": ["sainte", "suite", "societe"],
         "dr": ["drive", "doctor"], "co": ["company", "colorado", "county", "colony"],
         "ct": ["court", "connecticut", "circuit"], "pl": ["place", "plot"],
         "r": ["rue", "road", "route"], "s": ["south", "saint", "sons"], "n": ["north", "number"],
         "e": ["east"], "w": ["west"], "mt": ["mount", "montana"], "ft": ["fort", "feet"],
         "pt": ["point", "port"], "sa": ["societe anonyme", "south australia"], "ch": ["chemin", "church"],
         "b": ["block", "bis", "building"], "t": ["terrace", "ter"], "all": ["allee", "all"],
         "cc": ["centre commercial", "cubic centimetre"], "res": ["residence", "reservation"]}


def job_ambiguous(top_ctx=6, min_ctx=20):
    items, reqs = [], []
    for (ctry, field, tok), c in ST["neighbors"].items():
        if tok not in AMBIG:
            continue
        rights = [(w[2:], n) for w, n in c.most_common() if w.startswith("R:") and n >= min_ctx][:top_ctx]
        lefts = [w[2:] for w, n in c.most_common() if w.startswith("L:")][:8]
        for nxt, n in rights + [("", sum(c.values()))]:
            phrase = f"{tok} {nxt}".strip()
            items.append({"country": ctry, "field": field, "tok": tok, "next": nxt, "count": n})
            reqs.append(({"phrase": phrase, "short_form": tok, "found_in": Q.where(ctry, field),
                          "words_often_before_it": lefts}, Q.meaning_question(AMBIG[tok])))
    return items, reqs


# ---------------------------------------------------------------------------------------------
# 4. word classes
# ---------------------------------------------------------------------------------------------
def job_wordclass(top_name=2500, top_addr=1500):
    items, reqs = [], []
    for ctry in COUNTRIES:
        for field, top, qs in (("name", top_name, Q.WORD_CLASS), ("addr", top_addr, Q.ADDR_CLASS)):
            kept = 0
            for w, n in vocab(ctry, field).most_common():
                if kept >= top:
                    break
                if not w.isalpha() or len(w) < 2:
                    continue
                kept += 1
                items.append({"country": ctry, "field": field, "tok": w, "count": n})
                reqs.append(({"word": w, "found_in": Q.where(ctry, field)}, qs))
    return items, reqs


# ---------------------------------------------------------------------------------------------
# 5. look-alike frequent tokens (+ empirical swap evidence from TRAIN true pairs)
# ---------------------------------------------------------------------------------------------
def _swap_counts(pairs_needed):
    tp = ST["true_pairs"]
    need = {(c, f): set() for c, f, _, _ in pairs_needed}
    for c, f, a, b in pairs_needed:
        need[(c, f)].update((a, b))
    sub, occ = Counter(), Counter()
    for (c, f), toks in need.items():
        d = tp[tp.country == c]
        for x, y in zip(d[f + "_a"], d[f + "_b"]):
            A, B = set(x.split()) & toks, set(y.split()) & toks
            occ.update((c, f, t) for t in A | B)
            for p in A - B:
                for q in B - A:
                    sub[(c, f) + tuple(sorted((p, q)))] += 1
    return sub, occ


def job_lookalike(min_count=60, score=88, per_tok=3):
    raw = []
    for ctry in COUNTRIES:
        for field in ("name", "addr"):
            v = vocab(ctry, field)
            words = [w for w, n in v.items() if n >= min_count and w.isalpha() and len(w) >= 4]
            for w in words:
                for o, s, _ in process.extract(w, words, scorer=fuzz.ratio, limit=per_tok + 1, score_cutoff=score):
                    if o != w and w < o:
                        raw.append((ctry, field, w, o))
            ki = {}
            for w in words:  # same Indic phonetic key (shree/sri, lakshmi/laxmi)
                ki.setdefault(M.phonetic_key(w, "indic"), []).append(w)
            for g in ki.values():
                g = sorted(g, key=lambda x: -v[x])[:5]
                raw += [(ctry, field, min(a, b), max(a, b)) for i, a in enumerate(g) for b in g[i + 1:]]
    raw = list(dict.fromkeys(raw))
    sub, occ = _swap_counts(raw)
    items, reqs = [], []
    for c, f, a, b in raw:
        items.append({"country": c, "field": f, "a": a, "b": b, "swaps": sub.get((c, f, a, b), 0),
                      "occ_a": occ.get((c, f, a), 0), "occ_b": occ.get((c, f, b), 0)})
        reqs.append((Q.same_state(a, b, c, f), Q.SAME))
    return items, reqs


# ---------------------------------------------------------------------------------------------
# 6. data-mined swaps
# ---------------------------------------------------------------------------------------------
def job_mined(min_rate=0.1):
    d = pd.read_csv(os.path.join(WORK, "mined.tsv"), sep="\t", keep_default_na=False)
    d = d[(d.rate >= min_rate) & (d.a.str.len() > 0) & (d.b.str.len() > 0)]
    items = d.to_dict("records")
    reqs = [(Q.same_state(r["a"], r["b"], r["country"], r["field"]), Q.SAME) for r in items]
    return items, reqs


JOBS = {"translit2": job_translit, "translit": job_translit, "mined": job_mined, "abbrev": job_abbrev, "ambiguous": job_ambiguous,
        "lookalike": job_lookalike, "wordclass": job_wordclass}


def run(names, dry=False):
    jev = J.Jev(cache_path=os.path.join(HERE, "jev_cache.jsonl"), workers=3)
    for name in names:
        items, reqs = JOBS[name]()
        print(f"[{name}] {len(reqs)} requests", flush=True)
        if dry:
            continue
        size = {"mined": 25, "abbrev": 16, "ambiguous": 16, "translit": 8}.get(name, 30)  # choice questions carry more text
        answers = jev.map_batched(reqs, size=size, desc=name, every=25)
        with open(os.path.join(WORK, f"judged_{name}.pkl"), "wb") as f:
            pickle.dump({"items": items, "requests": reqs, "answers": answers}, f)
        print(f"[{name}] saved; missing={sum(a is None for a in answers)} stats={dict(jev.stats)}", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run(args or list(JOBS), dry="--dry" in sys.argv)
