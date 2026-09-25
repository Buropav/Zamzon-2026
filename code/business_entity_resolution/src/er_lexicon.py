"""
er_lexicon.py - static preprocessing resources (built once offline; see jev/README.md).

No API is called here. The resources are plain TSV files in src/resources/ (or, in the Kaggle
notebook, the same files embedded as a compressed string):
  lexicon.tsv         country, field, variant, canonical   word/phrase equivalences per country
  token_class.tsv     country, field, tok, kind, p         legal_form / title / generic_business ...
  conflict_pairs.tsv  country, field, a, b, source          look-alike words that are different names

What it adds on top of er_multilingual.normalize_ml:
  1. OCR-noise repair in names (hea1th -> health, 6roup -> group, lnc -> inc)
  2. per-country phrase rules and token map (tamil nadu -> tn, r -> rue, kansaltents -> consultants)
  3. junk address tokens dropped (null, na, pmb)
and two pair signals: distinct_tokens (name without legal/generic words) and name_conflict.
"""
import base64
import gzip
import io
import json
import os
import re

import pandas as pd

ADDR_DROP = {"null", "na", "nan", "none", "pmb"}  # literal junk tokens seen in US/India addresses
NON_DISTINCT = {"legal_form", "title", "connector", "generic_business"}
_OCR = str.maketrans({"0": "o", "1": "l", "5": "s", "6": "g", "8": "b"})
_OCR_TOKEN = re.compile(r"[A-Za-z]*[01568][A-Za-z01568]*")
_L_FOR_I = re.compile(r"\bl(?=[nm][a-z]+)")
_EMPTY = {"tok": {"name": {}, "addr": {}}, "phr": {"name": [], "addr": []}}


def fix_ocr(text):
    """Digit-for-letter noise inside words. Only tokens with <=2 digits, all from {0,1,5,6,8}, and
    >=3 letters (or 2 letters around one inner digit: c0m) change, so house numbers, PINs and codes
    (12b, a18a, 4x4, A1) are untouched."""
    def rep(m):
        w = m.group(0)
        digits = sum(c.isdigit() for c in w)
        letters = len(w) - digits
        inner = len(w) >= 3 and w[0].isalpha() and w[-1].isalpha()
        if digits == 0 or digits > 2 or not (letters >= 3 or (letters == 2 and digits == 1 and inner)):
            return w
        return w.translate(_OCR)
    t = _OCR_TOKEN.sub(rep, text)
    return _L_FOR_I.sub("i", t.lower()) if t else t


def country_key(c):
    """Unknown countries fall back to the '*' (script-level) lexicon, so an unseen country works."""
    s = re.sub(r"[^a-z]", "", str(c).lower())
    return {"us": "US", "usa": "US", "unitedstates": "US", "unitedstatesofamerica": "US",
            "in": "INDIA", "ind": "INDIA", "india": "INDIA", "bharat": "INDIA",
            "fr": "FRANCE", "fra": "FRANCE", "france": "FRANCE"}.get(s, "*")


def _from_frames(lexicon=None, token_class=None, conflict_pairs=None):
    lex = {}
    if lexicon is not None:
        for c, f, v, t in zip(lexicon.country, lexicon.field, lexicon.variant, lexicon.canonical):
            L = lex.setdefault(c, {"tok": {"name": {}, "addr": {}}, "phr": {"name": [], "addr": []}})
            for ff in (("name", "addr") if f == "both" else (f,)):
                (L["phr"][ff].append((v, t)) if " " in v else L["tok"][ff].__setitem__(v, t))
    lex.setdefault("*", {"tok": {"name": {}, "addr": {}}, "phr": {"name": [], "addr": []}})
    tc = {} if token_class is None else {(c, f, t): k for c, f, t, k in zip(
        token_class.country, token_class.field, token_class.tok, token_class.kind)}
    cf = set() if conflict_pairs is None else {(c, f, a, b) for c, f, a, b in zip(
        conflict_pairs.country, conflict_pairs.field, conflict_pairs.a, conflict_pairs.b)}
    return {"lex": lex, "token_class": tc, "conflicts": cf}


_FILES = ("lexicon", "token_class", "conflict_pairs")


def _read_tsv_text(txt):
    return pd.read_csv(io.StringIO(txt), sep="\t", dtype=str, keep_default_na=False)


def load(folder):
    frames = {}
    for n in _FILES:
        p = os.path.join(folder, f"{n}.tsv")
        if os.path.exists(p):
            frames[n] = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)
    return _from_frames(**frames)


def load_default():
    return load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources"))


def embed(folder):
    """-> compact ASCII string holding the resource files (used by the notebook generator)."""
    payload = {n: open(os.path.join(folder, f"{n}.tsv"), encoding="utf-8").read()
               for n in _FILES if os.path.exists(os.path.join(folder, f"{n}.tsv"))}
    return base64.b64encode(gzip.compress(json.dumps(payload).encode("utf-8"), 9, mtime=0)).decode("ascii")


def load_embedded(blob):
    payload = json.loads(gzip.decompress(base64.b64decode(blob)).decode("utf-8"))
    return _from_frames(**{n: _read_tsv_text(t) for n, t in payload.items()})


def wrap(normalize_ml, LEX, country="*"):
    """normalize_ml'(text, kind) for one country. Hand-written _CANON rules always win: a lexicon
    entry whose variant normalize_ml already rewrites is ignored. Phrase keys and targets are
    normalised with the ORIGINAL normalize_ml so they live in its canonical space."""
    L = LEX["lex"].get(country) or LEX["lex"].get("*") or _EMPTY
    compiled = {}
    for kind in ("name", "addr"):
        rules = []
        for v, c in L["phr"][kind]:
            key = normalize_ml(v, kind)
            if key and " " in key:
                rules.append((re.compile(rf"(?<!\S){re.escape(key)}(?!\S)"), normalize_ml(c, kind) or c))
        rules.sort(key=lambda r: -len(r[0].pattern))  # longest phrase first
        tmap = {}
        for v, c in L["tok"][kind].items():
            tv, tc = normalize_ml(v, kind), normalize_ml(c, kind) or c
            if tv != v:
                continue
            if tv and " " not in tv and tv != tc:
                tmap[tv] = tc
        compiled[kind] = (rules, tmap)

    def normalize_ml2(text, kind="name"):
        if kind == "name" and isinstance(text, str):
            text = fix_ocr(text)
        t = normalize_ml(text, kind)
        if not t:
            return t
        rules, tmap = compiled.get(kind, ([], {}))
        for rx, rep in rules:
            t = rx.sub(rep, t)
        toks = (tmap.get(w, w) for w in t.split())
        if kind == "addr":
            toks = (w for w in toks if w not in ADDR_DROP)
        return " ".join(toks)

    normalize_ml2.country = country
    return normalize_ml2


def country_aware(prepare_ml, namespace, LEX, country_col="country"):
    """Wraps prepare_ml(df, ...) so each country's rows are normalised with that country's lexicon.
    `namespace` is the dict where prepare_ml looks up normalize_ml (notebook: globals();
    package: vars(er_multilingual)). Row order and index are preserved; the original
    normalize_ml is always restored."""
    base = namespace["normalize_ml"]
    norms = {c: wrap(base, LEX, c) for c in LEX["lex"]}

    def prepare_ml2(df, *args, **kwargs):
        if country_col not in df:
            return prepare_ml(df, *args, **kwargs)
        keys = df[country_col].map(country_key)
        parts = []
        try:
            for c, idx in keys.groupby(keys).groups.items():
                namespace["normalize_ml"] = norms.get(c, norms["*"])
                parts.append(prepare_ml(df.loc[idx], *args, **kwargs))
        finally:
            namespace["normalize_ml"] = base
        return pd.concat(parts).loc[df.index] if parts else prepare_ml(df, *args, **kwargs)

    prepare_ml2.lexicon_wrapped = True
    return prepare_ml2


def distinct_tokens(norm_name, LEX, country="*"):
    """Tokens that identify the business: drops legal forms, titles, connectors and generic business
    words ('services', 'groupe', 'traders'). Falls back to all tokens."""
    tc = LEX["token_class"]
    toks = str(norm_name).split()
    keep = [t for t in toks if tc.get((country, "name", t)) not in NON_DISTINCT]
    return keep or toks


def name_conflict(a_tokens, b_tokens, LEX, country="*", field="name"):
    """1.0 if the names differ by a known look-alike pair of different names (patel vs patil)."""
    A, B = set(a_tokens) - set(b_tokens), set(b_tokens) - set(a_tokens)
    if not A or not B:
        return 0.0
    C = LEX["conflicts"]
    return float(any((country, field, min(x, y), max(x, y)) in C for x in A for y in B))
