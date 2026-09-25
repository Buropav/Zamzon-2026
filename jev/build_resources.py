"""
build_resources.py - turns Jev judgements (jev/work/judged_*.pkl) + TRAIN evidence into static
resource files in jev/resources/. Prints an audit of every decision rule against train evidence.

  lexicon.tsv         field  variant  canonical  source   (token or phrase equivalences)
  token_class.tsv     country field tok kind p
  conflict_pairs.tsv  country field a b source           (look-alikes that are different names)
  review_*.tsv        borderline items for a human look
"""
import os
import pickle
import sys
from collections import Counter, defaultdict

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code", "business_entity_resolution", "src"))  # shared modules
import er_multilingual as M  # noqa: E402

WORK, OUT = os.path.join(HERE, "work"), os.path.join(HERE, "resources")
ST = pickle.load(open(os.path.join(WORK, "stats.pkl"), "rb"))
FREQ = Counter()
for c in ST["vocab"].values():
    FREQ.update(c)


def judged(name):
    p = os.path.join(WORK, f"judged_{name}.pkl")
    if not os.path.exists(p):
        print(f"  (no {name} judgements yet)")
        return None
    d = pickle.load(open(p, "rb"))
    return list(zip(d["items"], d["answers"]))


def p_bool(a, q):
    return (a or {}).get(q, {}).get("probability")


def choice(a, q):
    x = (a or {}).get(q) or {}
    c = x.get("choice")
    return c, (x.get("probabilities") or {}).get(c, 0.0)


def canon_of(tok):
    """_CANON target; a close typo of a _CANON word (appartment ~ apartment) gets that word's target."""
    if tok in M._CANON:
        return M._CANON[tok]
    if len(tok) >= 6 and " " not in tok:
        from rapidfuzz import process, fuzz
        hit = process.extractOne(tok, [k for k in M._CANON if len(k) >= 6], scorer=fuzz.ratio, score_cutoff=90)
        if hit:
            return M._CANON[hit[0]]
    return tok


# ---------------------------------------------------------------------------------------------
def from_translit(rows, p_min=0.6):
    out, review = [], []
    for it, a in rows:
        c, p = choice(a, "english")
        rec = {"variant": it["tok"], "canonical": c, "p": round(p, 3), "count": it["count"], "cands": ",".join(it["cands"])}
        if c and c != "none" and p >= p_min:
            out.append(("*", "both", it["tok"], c, "translit"))
        elif c and c != "none":
            review.append(rec)
    return out, review


def same_ok(a, p_same=0.6, p_diff=0.5):
    ps, pd_ = p_bool(a, "same_word"), p_bool(a, "different_names")
    rel, prel = choice(a, "relation")
    return ps is not None and ps >= p_same and (pd_ or 0) < p_diff and not (rel == "different" and prel >= 0.5)


def _is_partial(it, phrase_pairs):
    """nc/north when nc/'north carolina' exists: the unigram pair is half of a phrase swap."""
    a, b = it["a"], it["b"]
    for x, y in ((a, b), (b, a)):
        for other, n in phrase_pairs.get((it["country"], it["field"], x), []):
            if y in other.split() and n >= 0.5 * it["n"]:
                return True
    return False


def _padded_phrase(it):
    """'ave arizona' vs 'aenue': the phrase is a variant of the unigram plus an extra word."""
    from rapidfuzz import fuzz
    a, b = it["a"], it["b"]
    if (" " in a) == (" " in b):
        return False
    uni, phr = (a, b) if " " in b else (b, a)
    if phr.replace(" ", "") == uni:  # andhra pradesh / andhrapradesh
        return False
    cu = canon_of(uni)
    return any(fuzz.ratio(uni, t) >= 70 or canon_of(t) == cu or M._is_abbrev(uni, t) or M._is_abbrev(t, uni)
               for t in phr.split())


def from_mined(rows, min_rate=0.3, min_n=30):
    phrase_pairs = defaultdict(list)
    for it, _ in rows:
        for x, y in ((it["a"], it["b"]), (it["b"], it["a"])):
            if " " in y and " " not in x:
                phrase_pairs[(it["country"], it["field"], x)].append((y, it["n"]))
    out, rej = [], []
    for it, a in rows:
        rel, prel = choice(a, "relation")
        strong = (rel in ("abbreviation", "spelling_variant", "renamed_place", "translation") and prel >= 0.6
                  and it["rate"] >= min_rate and it["n"] >= min_n and (p_bool(a, "different_names") or 0) < 0.3)
        name_phrase = it["field"] == "name" and (" " in it["a"] or " " in it["b"])  # noisy in names
        ok = ((same_ok(a) or strong) and not _is_partial(it, phrase_pairs) and not _padded_phrase(it)
              and not name_phrase)
        (out if ok else rej).append((it, a))
    lex = [(it["country"], it["field"], it["a"], it["b"], "mined") for it, _ in out]
    audit = pd.DataFrame([{**it, "p_same": p_bool(a, "same_word"), "p_diffname": p_bool(a, "different_names"),
                           "rel": choice(a, "relation")[0], "kept": k}
                          for k, grp in ((1, out), (0, rej)) for it, a in grp])
    return lex, audit


ABBREV_BLOCK = {"mart", "sci", "main", "ps", "com"}  # checked by hand: wrong or too ambiguous
LETTER_OK = {("FRANCE", "addr", "r")}  # single letters are usually initials / block letters


def from_abbrev(rows, p_min=0.7, p_name=0.8, p_short=0.9, p_letter=0.8):
    """Stricter for names (initials identify people: rk/aj/vj) and single letters; compound targets
    (ring -> ringroad) and hand-blocked forms are rejected."""
    lex, review = [], []
    for it, a in rows:
        c, p = choice(a, "expansion")
        s, f = it["tok"], it["field"]
        if not c or c == "none":
            continue
        need = p_min
        if f == "name":
            need = max(need, p_name if len(s) >= 3 else p_short)
        if len(s) == 1:
            need = 1.1 if f == "name" else max(need, p_letter)
        rest = c[len(s):] if c.startswith(s) else ""
        compound = len(rest) >= 4 and FREQ.get(rest, 0) >= 500
        if len(s) == 1 and (it["country"], f, s) not in LETTER_OK:
            continue
        rec = (it["country"], f, s, c, "abbrev", p)
        (lex if p >= need and not compound and s not in ABBREV_BLOCK else review).append(rec)
    return [x[:5] for x in lex], review


def from_lookalike(rows):
    """TRAIN evidence decides for US/INDIA (Jev audited against it); Jev decides for FRANCE."""
    lex, conf, audit = [], [], []
    for it, a in rows:
        ps, pdn = p_bool(a, "same_word"), p_bool(a, "different_names")
        occ = min(it["occ_a"], it["occ_b"])
        emp = None
        if it["country"] != "FRANCE":
            if it["swaps"] >= 5 and it["swaps"] / max(occ, 1) >= 0.05:
                emp = "same"
            elif it["swaps"] == 0 and occ >= 50:
                emp = "diff"
        jev = "same" if (ps or 0) >= 0.7 and (pdn or 1) < 0.3 else "diff" if (pdn or 0) >= 0.6 and (ps or 1) < 0.4 else None
        audit.append({**it, "p_same": ps, "p_diffname": pdn, "emp": emp, "jev": jev})
        # Jev "same" agreed with TRAIN evidence only ~35% on look-alikes, "diff" ~99%: without labels
        # (France) Jev may only flag conflicts, never merge.
        final = emp if it["country"] != "FRANCE" else ("diff" if jev == "diff" else None)
        if it["country"] != "FRANCE" and emp == "same" and jev == "diff":
            final = None  # disagreement -> leave it to the model
        if final == "same":
            lex.append((it["country"], it["field"], it["a"], it["b"], "lookalike"))
        elif final == "diff" and (emp == "diff" or it["country"] == "FRANCE"):
            conf.append((it["country"], it["field"], it["a"], it["b"], "train" if emp else "jev"))
    return lex, conf, pd.DataFrame(audit)


def from_wordclass(rows, p_min=0.6):
    out = []
    for it, a in rows:
        c, p = choice(a, "kind")
        if c and p >= p_min:
            out.append((it["country"], it["field"], it["tok"], c, round(p, 3)))
    df = pd.DataFrame(out, columns=["country", "field", "tok", "kind", "p"])
    # also key the class by the canonical form used after normalisation (services -> svc)
    canon = df.assign(tok=df.tok.map(canon_of))
    df = pd.concat([df, canon[canon.tok != df.tok]]).sort_values("p", ascending=False)
    return df.drop_duplicates(["country", "field", "tok"])


# ---------------------------------------------------------------------------------------------
DIRECTED = ("translit", "abbrev")


def unify(pairs, max_group=12):
    """Per (country, field). Directed pairs (translit, abbrev: variant -> target) map straight to the
    target, so star-shaped groups (9 spellings of 'international') are fine; a token that also has
    TRAIN-mined evidence keeps the mined meaning. Undirected pairs (mined, lookalike) go through
    union-find; canonical = existing _CANON target, else a directed target, else a single token over a
    phrase, else the most frequent spelling. Oversized undirected groups are dropped (chaining).
    country "*" holds script-level mappings (translit) that apply everywhere."""
    rows, dropped = [], []
    countries = sorted({p[0] for p in pairs} | {"*"})
    for ctry in countries:
        for field in ("name", "addr"):
            P = [(a, b, src) for c, f, a, b, src in pairs
                 if c in (ctry, "*") and f in (field, "both") and a != b]
            targets = {b for a, b, src in P if src in DIRECTED}
            undirected = {t for a, b, src in P if src not in DIRECTED for t in (a, b)}
            direct = {}
            for a, b, src in P:
                if src in DIRECTED and a not in undirected:
                    direct.setdefault(a, b)
            parent = {}

            def find(x):
                parent.setdefault(x, x)
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for a, b, src in P:
                if src not in DIRECTED:
                    parent[find(direct.get(a, a))] = find(direct.get(b, b))
            groups = defaultdict(list)
            for t in list(parent):
                groups[find(t)].append(t)
            root_canon = {}
            for g in groups.values():
                if len(g) > max_group:
                    dropped.append((ctry, field, sorted(g)))
                    continue
                known = [t for t in g if t in M._CANON]
                tg = [t for t in g if t in targets]
                canon = (canon_of(known[0]) if known else tg[0] if tg else
                         max(g, key=lambda t: (" " not in t, FREQ.get(t, 0), -len(t))))
                for t in g:
                    root_canon[t] = canon
                    if t != canon and canon_of(t) != canon:
                        rows.append((ctry, field, t, canon))
            for a, b in direct.items():
                c = root_canon.get(b, canon_of(b))
                if canon_of(a) != c:
                    rows.append((ctry, field, a, c))
    df = pd.DataFrame(rows, columns=["country", "field", "variant", "canonical"])
    return df.drop_duplicates(["country", "field", "variant"]), dropped


# Hand-checked additions found by the Jev ambiguity job (country, field, variant, canonical, source)
MANUAL = [("US", "addr", "county", "co", "abbrev"), ("US", "addr", "cnty", "co", "abbrev")]


def main():
    os.makedirs(OUT, exist_ok=True)
    pairs, reviews = [m for m in MANUAL], {}
    r = judged("translit2") or judged("translit")
    if r:
        lex, rev = from_translit(r)
        pairs += lex
        reviews["translit"] = pd.DataFrame(rev)
        print(f"translit: accepted {len(lex)}/{len(r)}")
    r = judged("mined")
    if r:
        lex, audit = from_mined(r)
        pairs += lex
        audit.to_csv(os.path.join(WORK, "audit_mined.tsv"), sep="\t", index=False)
        print(f"mined: accepted {len(lex)}/{len(r)}")
    r = judged("abbrev")
    if r:
        lex, rev = from_abbrev(r)
        pairs += lex
        reviews["abbrev"] = pd.DataFrame(rev, columns=["country", "field", "tok", "expansion", "src", "p"])
        print(f"abbrev: accepted {len(lex)}/{len(r)}")
    conf = []
    r = judged("lookalike")
    if r:
        lex, conf, audit = from_lookalike(r)
        pairs += lex
        audit.to_csv(os.path.join(WORK, "audit_lookalike.tsv"), sep="\t", index=False)
        lab = audit[audit.emp.notna()]
        for lbl in ("same", "diff"):
            sub = lab[lab.jev.notna() & (lab.jev == lbl)]
            print(f"lookalike: Jev says {lbl}: {len(sub)}, agrees with train evidence {(sub.emp == lbl).mean():.1%}")
        print(f"lookalike: lexicon {len(lex)}, conflicts {len(conf)}")
    r = judged("wordclass")
    if r:
        wc = from_wordclass(r)
        wc.to_csv(os.path.join(OUT, "token_class.tsv"), sep="\t", index=False)
        print("wordclass:", wc.groupby(["country", "kind"]).size().unstack(fill_value=0).to_string())

    lexdf, dropped = unify(pairs)
    lexdf.to_csv(os.path.join(OUT, "lexicon.tsv"), sep="\t", index=False)
    pd.DataFrame(conf, columns=["country", "field", "a", "b", "source"]).to_csv(
        os.path.join(OUT, "conflict_pairs.tsv"), sep="\t", index=False)
    for k, v in reviews.items():
        v.to_csv(os.path.join(WORK, f"review_{k}.tsv"), sep="\t", index=False)
    with open(os.path.join(WORK, "dropped_groups.txt"), "w") as f:
        f.writelines(f"{c}\t{fl}\t{' | '.join(g)}\n" for c, fl, g in dropped)
    print(f"lexicon.tsv: {len(lexdf)} entries ({lexdf.groupby(['country', 'field']).size().to_dict()}); "
          f"dropped {len(dropped)} oversized groups -> work/dropped_groups.txt")


if __name__ == "__main__":
    main()
