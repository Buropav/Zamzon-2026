# er_multilingual.py - multilingual / country-specific text handling for the ER pipeline.
"""
er_multilingual.py - multilingual / country-specific text handling for the ER pipeline.

Replaces the notebook's normalize_text, which (verified):
  * turns Tamil, Telugu, Bengali, Gujarati, Gurmukhi, Malayalam, Odia and Urdu text into ""
    -> two unrelated wiped names then score 1.0 on Jaro-Winkler, Levenshtein and Jaccard
  * romanises Devanagari/Kannada with Harvard-Kyoto: "शर्मा ट्रेडर्स" -> "zarma tredarsa"
  * drops the French ligature: "Cœur" -> "cur"
  * treats curly and straight apostrophes differently: "L’Atelier" -> "latelier", "L'Atelier" -> "l atelier"
  * maps every "st" to "street", so "Pharmacie St-Denis" -> "pharmacie street denis"
  * splits Indian PINs written "560 001" into two tokens

Design rule: every mapping here is symmetric and language-agnostic (applied to all rows
regardless of the country label), so it works unchanged for an unseen country.

Requires: indic-transliteration (MIT), rapidfuzz.
"""
import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

try:
    from indic_transliteration import sanscript as _S
except Exception:  # pragma: no cover
    _S = None

# --------------------------------------------------------------------------------------
# Script detection and audit
# --------------------------------------------------------------------------------------
_SCRIPTS = [  # (name, lo, hi, sanscript scheme, schwa-deleting language?)
    ("devanagari", 0x0900, 0x097F, "DEVANAGARI", True),
    ("bengali", 0x0980, 0x09FF, "BENGALI", True),
    ("gurmukhi", 0x0A00, 0x0A7F, "GURMUKHI", True),
    ("gujarati", 0x0A80, 0x0AFF, "GUJARATI", True),
    ("oriya", 0x0B00, 0x0B7F, "ORIYA", True),
    ("tamil", 0x0B80, 0x0BFF, "TAMIL", False),
    ("telugu", 0x0C00, 0x0C7F, "TELUGU", False),
    ("kannada", 0x0C80, 0x0CFF, "KANNADA", False),
    ("malayalam", 0x0D00, 0x0D7F, "MALAYALAM", False),
    ("arabic", 0x0600, 0x06FF, None, False),
]


def script_of(ch):
    o = ord(ch)
    if o < 0x80:
        return "ascii"
    for name, lo, hi, _, _ in _SCRIPTS:
        if lo <= o <= hi:
            return name
    if 0x00C0 <= o <= 0x024F:
        return "latin_accented"
    return "other"


def script_audit(frames, cols=("business_name", "business_address")):
    """frames: {"train_s1": df, ...} with a country column. Share of records containing each
    script, per source x country x column. Run on train AND test before building anything."""
    rows = []
    for src, df in frames.items():
        for col in cols:
            present = df[col].fillna("").map(lambda s: frozenset(script_of(c) for c in s if c.isalpha()))
            for ctry, grp in present.groupby(df["country"].fillna("")):
                cnt = Counter(s for fs in grp for s in fs)
                for s, n in cnt.items():
                    if s != "ascii":
                        rows.append((src, ctry, col, s, n / len(grp)))
    return (pd.DataFrame(rows, columns=["source", "country", "column", "script", "share"])
            .sort_values("share", ascending=False).reset_index(drop=True))


def vocab_shift(texts_a, texts_b, top=25, min_count=20):
    """Tokens over-represented in A vs B (log ratio). Run per country between sources, e.g.
    France S1 names vs France S3 names: English business words on one side and French on the
    other means a translation-type variation exists and needs handling."""
    ca = Counter(t for s in texts_a for t in str(s).split())
    cb = Counter(t for s in texts_b for t in str(s).split())
    na, nb = sum(ca.values()) or 1, sum(cb.values()) or 1
    toks = [t for t in set(ca) | set(cb) if ca[t] + cb[t] >= min_count]
    lr = pd.Series({t: np.log((ca[t] + 1) / na) - np.log((cb[t] + 1) / nb) for t in toks})
    return lr.nlargest(top).rename("more_in_A"), lr.nsmallest(top).rename("more_in_B")


# --------------------------------------------------------------------------------------
# Romanisation
# --------------------------------------------------------------------------------------
_CHARMAP = str.maketrans({
    "œ": "oe", "Œ": "OE", "æ": "ae", "Æ": "AE", "ß": "ss", "ø": "o", "Ø": "O", "ł": "l", "Ł": "L",
    "đ": "d", "Đ": "D", "ð": "d", "þ": "th", "ı": "i",
    "’": "'", "‘": "'", "‛": "'", "`": "'", "´": "'", "ʼ": "'",
    "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-", "№": " no ", "°": " ", "º": "", "ª": "",
})
# Urdu/Arabic: consonant skeleton only (the script omits short vowels, so keys below still work)
_ARABIC = str.maketrans({
    "ا": "a", "آ": "a", "ب": "b", "پ": "p", "ت": "t", "ٹ": "t", "ث": "s", "ج": "j", "چ": "ch", "ح": "h",
    "خ": "kh", "د": "d", "ڈ": "d", "ذ": "z", "r": "r", "ڑ": "r", "ز": "z", "ژ": "zh", "س": "s", "ش": "sh",
    "ص": "s", "ض": "z", "ط": "t", "ظ": "z", "ع": "", "غ": "gh", "ف": "f", "ق": "q", "ک": "k", "ك": "k",
    "گ": "g", "ل": "l", "م": "m", "ن": "n", "ں": "n", "و": "o", "ہ": "h", "ه": "h", "ھ": "h", "ء": "",
    "ی": "i", "ي": "i", "ے": "e", "ئ": "i", "ؤ": "o", "ة": "a",
})
_UNIT_RX = r"(?:[kgcCjJTDtdpbsS]h|~n|[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ])"  # ITRANS consonant units


def _itrans_cleanup(tok, schwa):
    t = tok.replace("RRi", "ri").replace("R^i", "ri").replace("j~n", "gy").replace("~n", "n")
    t = re.sub(r"M(?=[pbmPB])", "m", t).replace("M", "n").replace(".N", "n").replace(".n", "n")
    t = re.sub(r"[\^~.]", "", t)
    if schwa:  # word-final inherent 'a' is silent in Hindi/Gujarati/Bengali/Punjabi: rAma -> ram,
        # but kept after a consonant cluster (kRShNa -> krishna, gupta, mishra) unless it ends in s (TreDarsa -> treDars)
        m = re.search(rf"((?:{_UNIT_RX})+)a$", t)
        if m and len(re.findall(r"[aeiouAEIOU]", t)) > 1:
            units = re.findall(_UNIT_RX, m.group(1))
            if len(units) == 1 or units[-1] == "s":
                t = t[:-1]
    return t.replace("x", "ksh").lower()


def _unicode_name_fallback(ch):
    try:
        nm = unicodedata.name(ch)
    except ValueError:
        return ""
    if "LETTER" in nm or "VOWEL SIGN" in nm:
        w = nm.split()[-1].lower()
        w = w[:-1] if len(w) > 1 and w.endswith("a") and "VOWEL" not in nm else w
        return re.sub(r"(.)\1+", r"\1", w)
    return ""


def to_latin(text):
    """Any script -> lowercase ASCII-ish Latin, never silently dropping letters."""
    if not isinstance(text, str) or not text.strip():
        return ""
    t = unicodedata.normalize("NFKC", text).translate(_CHARMAP)
    if any(ord(c) > 0x5FF for c in t):
        out = []
        for tok in t.split():
            scripts = {script_of(c) for c in tok if c.isalpha()}
            for name, _, _, scheme, schwa in _SCRIPTS:
                if name in scripts and scheme and _S is not None:
                    tok = _itrans_cleanup(_S.transliterate(tok, getattr(_S, scheme), _S.ITRANS), schwa)
            if "arabic" in scripts:
                tok = tok.translate(_ARABIC)
            out.append(tok)
        t = " ".join(out)
    t = "".join(c for c in unicodedata.normalize("NFKD", t) if not unicodedata.combining(c))
    if any(ord(c) > 127 for c in t):  # leftover letters (e.g. Tamil final consonants)
        t = "".join(c if ord(c) < 128 else _unicode_name_fallback(c) for c in t)
    return t.lower()


# --------------------------------------------------------------------------------------
# Canonical tokens (symmetric: both spellings map to one token; collisions are harmless)
# --------------------------------------------------------------------------------------
_PHRASES = [  # multi-word first
    (r"\bprivate limited\b", "pvt ltd"), (r"\bdoing business as\b|\bd b a\b", "dba"),
    (r"\btrading as\b|\bt a\b", "dba"), (r"\ba k a\b|\balso known as\b", "aka"),
    (r"\bzone industrielle\b", "zi"), (r"\bzone d activites?\b|\bzone artisanale\b", "za"),
    (r"\bcentre commercial\b|\bshopping (?:centre|center|mall)\b", "cc"),
    (r"\blieu dit\b", "ld"), (r"\bin front of\b|\ben face de\b|\bface a\b", "opp"),
    (r"\bnext to\b|\ba cote de\b|\bclose to\b|\bpres de\b", "near"),
    (r"\bunited states(?: of america)?\b", "usa"),
    (r"\b(?:h|house|door|d|plot|shop|flat|bldg) no\b", "no"),
]
_CANON = {}
for canon, variants in {
    # legal / company words (EN, IN, FR)
    "co": "co company compagnie cie", "corp": "corp corporation", "inc": "inc incorporated",
    "ltd": "ltd limited", "pvt": "pvt private", "llc": "llc", "and": "and et &",
    "bros": "bros brothers freres", "intl": "intl international", "mfg": "mfg manufacturing",
    "svc": "svc svcs services service", "ent": "ent enterprises enterprise entreprise",
    "ind": "ind inds industries industry", "tech": "tech technologies technology",
    "assoc": "assoc associates associes", "mgmt": "mgmt management", "engg": "engg engineering",
    "ets": "ets etablissements etablissement", "ste": "ste suite sainte societe",
    "st": "st saint street str", "mkt": "mkt market bazaar bazar",
    # street types
    "rd": "rd road raod", "ave": "ave av avenue", "blvd": "blvd bd boulevard boul",
    "dr": "dr drive", "ln": "ln lane gali gully", "ct": "ct court", "pl": "pl place",
    "hwy": "hwy highway", "pkwy": "pkwy parkway", "sq": "sq square", "rte": "rte route",
    "imp": "imp impasse", "chem": "chem chemin", "fbg": "fbg fg faubourg", "res": "res residence",
    "bldg": "bldg building batiment bat", "apt": "apt apts apartment apartments appt",
    "fl": "fl floor etage", "sec": "sec sector", "ph": "ph phase", "extn": "extn ext extension",
    "nagar": "nagar ngr nager", "layout": "layout lyt", "colony": "colony col clny",
    "dist": "dist district dt", "vill": "vill village vil", "taluk": "taluk taluka tq",
    "opp": "opp opposite", "near": "near nr", "behind": "behind bhd",
    # directions
    "n": "n north", "s": "s south", "e": "e east", "w": "w west",
}.items():
    for v in variants.split():
        _CANON[v] = canon
_NUMWORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}
_NUMWORDS.update({w: str(i) for i, w in enumerate(
    "zeroth first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth".split())})
_NUMWORDS.update({"premier": "1", "premiere": "1", "deuxieme": "2", "troisieme": "3", "quatrieme": "4",
                  "cinquieme": "5", "sixieme": "6", "septieme": "7", "huitieme": "8", "neuvieme": "9",
                  "dixieme": "10", "bis": "b", "ter": "t"})
_DROP = {"no", "num", "number", "cedex", "bp", "usa", "india", "bharat", "france"}
_PREFIX = re.compile(r"^\s*(?:m/s\.?|messrs\.?|m\.s\.)\s*", re.I)


def normalize_ml(text, kind="name"):
    """kind: 'name' or 'addr'. Returns canonical lowercase tokens joined by spaces."""
    t = to_latin(text)
    if not t:
        return ""
    if kind == "name":
        t = _PREFIX.sub("", t)
    t = re.sub(r"'s\b", "", t)                          # joe's -> joe
    t = re.sub(r"(\d)\s*[-/]\s*(?=\d)", r"\1-", t)      # 12 - 3 / 456 -> 12-3-456
    t = re.sub(r"(\d{5})-\d{4}\b", r"\1", t)            # ZIP+4 -> ZIP
    t = re.sub(r"[^a-z0-9\-&]+", " ", t.replace("&", " & "))
    t = re.sub(r"(?<![0-9])-|-(?![0-9])", " ", t)       # keep hyphens only between digits
    if kind == "addr":
        t = re.sub(r"\b([1-9]\d{2})\s+(\d{3})\b", r"\1\2", t)   # PIN "560 001" -> "560001"
        t = re.sub(r"\bcedex\s*\d*\b", " ", t)
    t = re.sub(r"\b(\d+)(?:st|nd|rd|th|er|ere|e|eme|ieme)\b", r"\1", t)  # ordinals: 5th / 2eme -> 5 / 2
    t = re.sub(r"\b(\d+)\s*(bis|ter)\b", lambda m: m.group(1) + m.group(2)[0], t)  # 12 bis -> 12b
    t = re.sub(r"\b([a-z])\s+(?=[a-z]\b)", r"\1", t)     # dotted initials: "m g rd" -> "mg rd", "a b c" -> "abc"
    for pat, rep in _PHRASES:
        t = re.sub(pat, rep, t)
    toks = []
    for w in t.split():
        w = _NUMWORDS.get(w, w)
        w = _CANON.get(w, w)
        if kind == "addr" and w in _DROP:
            continue
        toks.append(w)
    return " ".join(toks)


# --------------------------------------------------------------------------------------
# Phonetic keys (feature + optional extra blocking key)
# --------------------------------------------------------------------------------------
_IND_RULES = [("ksh", "x"), ("ks", "x"), ("chh", "c"), ("ch", "c"), ("sh", "s"), ("th", "t"),
              ("dh", "d"), ("bh", "b"), ("ph", "f"), ("kh", "k"), ("gh", "g"), ("jh", "j"),
              ("ow", "o"), ("au", "o"), ("ou", "o"), ("ee", "i"), ("oo", "u"), ("aa", "a"),
              ("w", "v"), ("z", "j"), ("q", "k"), ("ck", "k"), ("y", "i"),
              ("g", "k"), ("d", "t"), ("b", "p")]          # Tamil-style voicing merges
_FR_RULES = [("eaux", "o"), ("eau", "o"), ("aux", "o"), ("ault", "o"), ("au", "o"), ("ph", "f"),
             ("qu", "k"), ("ch", "s"), ("gn", "n"), ("th", "t"), ("h", ""), ("y", "i"),
             ("ai", "e"), ("ei", "e"), ("ez", "e"), ("er", "e"), ("oi", "va"), ("w", "v"),
             ("ce", "se"), ("ci", "si"), ("cy", "si"), ("ge", "je"), ("gi", "ji"), ("c", "k")]


def _key(word, rules, strip_final=""):
    w = word
    for a, b in rules:
        w = w.replace(a, b)
    w = re.sub(r"(.)\1+", r"\1", w)
    if strip_final:
        w = re.sub(f"[{strip_final}]+$", "", w) or w
    return (w[0] + re.sub(r"[aeiou]", "", w[1:])) if w else ""


def phonetic_key(norm_text, style="indic"):
    """style="indic": sri/shri/shree/sree -> sr, lakshmi/laxmi -> lxm, chaudhary/chowdhury -> ctr.
    style="french": dupont/dupond -> dpn, thibault/thibaut -> tp. Words with digits pass through."""
    rules, strip = (_IND_RULES, "") if style == "indic" else (_FR_RULES, "stdxzpe")
    return " ".join(w if any(c.isdigit() for c in w) else _key(w, rules, strip) for w in norm_text.split())


# --------------------------------------------------------------------------------------
# Name / address structure
# --------------------------------------------------------------------------------------
_DBA = re.compile(r"\b(?:d\s*/?\s*b\s*/?\s*a|doing business as|t\s*/\s*a|trading as|a\s*/?\s*k\s*/?\s*a|"
                  r"formerly|f\s*/?\s*k\s*/?\s*a)\b\.?|[()\[\]]", re.I)
_LANDMARK = re.compile(r"^\s*(?:near|nr\.?|opp\.?|opposite|behind|beside|besides|next to|adj\.?|adjacent to|"
                       r"in front of|above|below|close to|off|pres de|près de|a cote de|à côté de|"
                       r"en face de|face a|face à|derriere|derrière)\b", re.I)
_HI_LANDMARK = re.compile(r"(?:\b\w+\s+){1,3}\bke\s+(?:paas|pass|saamne|samne|peeche|piche|bagal)\b", re.I)
_LANDMARK_INLINE = re.compile(r"\b(?:near|nr|opp|opposite|behind|beside|next to|in front of)\b(?:\s+[a-z]+){1,3}", re.I)


def split_dba(raw_name):
    """'ABC Holdings LLC dba Joe's Pizza' -> ['ABC Holdings LLC', "Joe's Pizza"]."""
    parts = [p.strip(" ,.-") for p in _DBA.split(raw_name or "")]
    return [p for p in parts if p] or [raw_name or ""]


def split_landmarks(raw_addr):
    """Returns (core_address, landmark_text) from the RAW address (commas still present)."""
    raw = raw_addr or ""
    marks = [m.group(0) for m in _HI_LANDMARK.finditer(raw)]
    raw = _HI_LANDMARK.sub(" ", raw)
    core = []
    for seg in re.split(r"[,;]", raw):
        (marks if _LANDMARK.match(seg) else core).append(seg)
    core_txt = ", ".join(core)
    if not marks:  # no comma around the landmark: remove keyword + up to 3 words
        marks = [m.group(0) for m in _LANDMARK_INLINE.finditer(core_txt)]
        core_txt = _LANDMARK_INLINE.sub(" ", core_txt)
    return core_txt, " ".join(marks)


_UNIT = re.compile(r"\b(?:ste|apt|unit|fl|shop|office|room|bldg|bureau|flat|wing|block|blk)\s*(?:no\s*)?#?\s*([a-z]?\d+[a-z]?)\b")
_PIN = re.compile(r"\b[1-9]\d{5}\b")
_ZIP = re.compile(r"\b\d{5}\b")
_HOUSE = re.compile(r"\b\d+[a-z]?(?:-\d+[a-z]?)*\b")
_ARR = {"75": "paris", "69": "lyon", "13": "marseille"}


def address_parts(norm_addr):
    """Structured pieces of a normalize_ml(kind="addr") string."""
    pc = (_PIN.findall(norm_addr) or _ZIP.findall(norm_addr) or [""])[-1]
    unit = _UNIT.search(norm_addr)
    no_unit = _UNIT.sub(" ", norm_addr)
    house = [h for h in _HOUSE.findall(no_unit) if h != pc]
    arr = ""
    if len(pc) == 5 and pc[:2] in _ARR and pc[2] == "0" and pc[3:] != "00":
        arr = f"{_ARR[pc[:2]]}{int(pc[3:])}"
    else:
        m = re.search(r"\b(paris|lyon|marseille) (\d{1,2})\b", norm_addr)
        arr = f"{m.group(1)}{int(m.group(2))}" if m else ""
    return {"pc": pc, "unit": unit.group(1) if unit else "", "house": house[0] if house else "", "arr": arr}


# --------------------------------------------------------------------------------------
# Abbreviation-aware soft token similarity (generalises to unseen languages)
# --------------------------------------------------------------------------------------
def _is_abbrev(short, long):
    """'mfg'<-'manufacturing', 'svcs'<-'services', 'bd'<-'boulevard', 'ste'<-'societe', 'r'<-'rajesh'."""
    if not short or short[0] != long[0] or len(short) >= len(long):
        return False
    it = iter(long)
    return all(c in it for c in short)


def _tok_sim(a, b):
    if a == b:
        return 1.0
    s, l = (a, b) if len(a) <= len(b) else (b, a)
    if not s.isdigit() and _is_abbrev(s, l):
        return 0.6 if len(s) == 1 else 0.85
    return JaroWinkler.normalized_similarity(a, b)


def soft_token_sim(x, y):
    """Monge-Elkan over tokens with abbreviation credit, both directions -> (min, max). NaN if a side is empty."""
    tx, ty = x.split(), y.split()
    if not tx or not ty:
        return np.nan, np.nan
    fwd = np.mean([max(_tok_sim(a, b) for b in ty) for a in tx])
    bwd = np.mean([max(_tok_sim(a, b) for a in tx) for b in ty])
    return float(min(fwd, bwd)), float(max(fwd, bwd))


# --------------------------------------------------------------------------------------
# Data-driven variant mining (train positives, or high-confidence address-anchored pairs)
# --------------------------------------------------------------------------------------
def mine_token_variants(pairs, min_count=10, max_side=3):
    """pairs: iterable of (norm_text_a, norm_text_b) for TRUE matches. Returns candidate
    token equivalences (tok_a, tok_b, n, conf) - review the top rows and add the good ones to
    _CANON. Finds transliteration variants (shree/sri), city renames, local abbreviations and
    cross-language words that actually occur in this dataset."""
    pair_n, a_n, b_n = Counter(), Counter(), Counter()
    for a, b in pairs:
        ta, tb = set(a.split()), set(b.split())
        oa, ob = ta - tb, tb - ta
        if not oa or not ob or len(oa) > max_side or len(ob) > max_side:
            continue
        a_n.update(oa)
        b_n.update(ob)
        pair_n.update((x, y) for x in oa for y in ob)
    rows = [(x, y, n, n / min(a_n[x], b_n[y])) for (x, y), n in pair_n.items() if n >= min_count]
    return (pd.DataFrame(rows, columns=["tok_a", "tok_b", "n", "conf"])
            .sort_values(["conf", "n"], ascending=False).reset_index(drop=True))


# --------------------------------------------------------------------------------------
# Pair features built on the above: normalise each RECORD once, then compare pairs
# --------------------------------------------------------------------------------------
def prepare_ml(df, name_col="business_name", addr_col="business_address"):
    """Per-record multilingual columns (run once per table; ~30 us per string)."""
    out = pd.DataFrame(index=df.index)
    names = df[name_col].fillna("").astype(str)
    addrs = df[addr_col].fillna("").astype(str)
    out["ml_name"] = names.map(lambda x: normalize_ml(x, "name"))
    out["ml_addr"] = addrs.map(lambda x: normalize_ml(x, "addr"))
    split = addrs.map(split_landmarks)
    out["ml_addr_core"] = split.map(lambda x: normalize_ml(x[0], "addr"))
    out["ml_landmark"] = split.map(lambda x: normalize_ml(x[1], "addr"))
    out["ml_dba"] = names.map(lambda x: tuple(normalize_ml(p, "name") for p in split_dba(x)))
    out["key_indic"] = out.ml_name.map(lambda x: phonetic_key(x, "indic"))
    out["key_french"] = out.ml_name.map(lambda x: phonetic_key(x, "french"))
    parts = out.ml_addr.map(address_parts)
    for k in ("pc", "unit", "house", "arr"):
        out[f"ml_{k}"] = parts.map(lambda d, k=k: d[k])
    out["ml_name_nums"] = out.ml_name.map(lambda x: frozenset(re.findall(r"\d+", x)))
    out["nonlatin"] = names.map(lambda x: any(script_of(c) not in ("ascii", "latin_accented") for c in x if c.isalpha()))
    return out


def ml_pair_features(qi, di, A, B):
    """qi/di: integer row positions into prepared tables A (Source 1) and B (Source 2/3)."""
    g = lambda T, c, idx: T[c].to_numpy()[idx]
    an, bn = g(A, "ml_name", qi), g(B, "ml_name", di)
    ac, bc = g(A, "ml_addr_core", qi), g(B, "ml_addr_core", di)
    F = pd.DataFrame(index=range(len(qi)))
    soft_n = [soft_token_sim(x, y) for x, y in zip(an, bn)]
    soft_a = [soft_token_sim(x, y) for x, y in zip(ac, bc)]
    F["name_soft_min"], F["name_soft_max"] = zip(*soft_n) if len(qi) else ([], [])
    F["addr_soft_min"], F["addr_soft_max"] = zip(*soft_a) if len(qi) else ([], [])
    ts = lambda xs, ys: [np.nan if not x or not y else fuzz.token_set_ratio(x, y) / 100 for x, y in zip(xs, ys)]
    F["name_tset_ml"] = ts(an, bn)
    F["addr_core_tset"] = ts(ac, bc)
    F["name_key_indic"] = ts(g(A, "key_indic", qi), g(B, "key_indic", di))
    F["name_key_french"] = ts(g(A, "key_french", qi), g(B, "key_french", di))
    la, lb = g(A, "ml_landmark", qi), g(B, "ml_landmark", di)
    F["landmark_a"], F["landmark_b"] = (la != "").astype(float), (lb != "").astype(float)
    F["landmark_sim"] = ts(la, lb)
    da, db = g(A, "ml_dba", qi), g(B, "ml_dba", di)
    F["has_dba"] = [float(len(x) > 1 or len(y) > 1) for x, y in zip(da, db)]
    F["name_dba_best"] = [max((fuzz.token_set_ratio(p, q) / 100 for p in x for q in y if p and q), default=np.nan)
                          if (len(x) > 1 or len(y) > 1) else np.nan for x, y in zip(da, db)]
    na, nb = g(A, "ml_name_nums", qi), g(B, "ml_name_nums", di)
    F["name_num_conflict"] = [float(bool(x and y and x != y)) for x, y in zip(na, nb)]
    for k in ("pc", "unit", "house", "arr"):
        x, y = g(A, f"ml_{k}", qi), g(B, f"ml_{k}", di)
        F[f"{k}_eq"] = np.where((x != "") & (y != ""), (x == y).astype(float), np.nan)
    xa, xb = g(A, "nonlatin", qi).astype(float), g(B, "nonlatin", di).astype(float)
    F["nonlatin_a"], F["nonlatin_b"], F["cross_script"] = xa, xb, (xa != xb).astype(float)
    return F
