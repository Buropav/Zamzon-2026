# translit.py - native-script -> Latin word map learned from training pairs
"""Native-script -> Latin word map learned from TRAIN true pairs only.
Names: a native-script Source 2/3 name is a word-for-word transliteration of its Source 1 name (same token
count in 99.99% of pairs), so tokens align by position. Addresses: only whole components (state names) are
written in native script; each native component maps to the Source 1 component that co-occurs with it most often."""
import re
from collections import Counter, defaultdict

_NATIVE = re.compile(r"[؀-ۿऀ-෿]")
_LAT = re.compile(r"[^a-z0-9&]+")


def _has_native(s):
    return bool(_NATIVE.search(s))


def _lat(t):
    return _LAT.sub("", t.lower())


def _pick(counter_by_tok, min_count, min_purity):
    out = {}
    for tok, c in counter_by_tok.items():
        (best, n), tot = c.most_common(1)[0], sum(c.values())
        if n >= min_count and n / tot >= min_purity and best:
            out[tok] = best
    return out


def build_native_map(pairs_names, pairs_addrs, min_count=3, min_purity=0.6):
    """pairs_*: iterables of (source1_text, source23_text) for TRUE pairs. Returns {"name": {...}, "addr": {...}}."""
    nm = defaultdict(Counter)
    for a, b in pairs_names:
        if not _has_native(b):
            continue
        ta, tb = a.split(), b.split()
        if len(ta) != len(tb):
            continue
        for x, y in zip(tb, ta):
            if _has_native(x):
                nm[x.strip('",.;:()[]')][_lat(y)] += 1
    comp, seen = defaultdict(Counter), Counter()
    for a, b in pairs_addrs:
        if not _has_native(b):
            continue
        ca = [c.strip() for c in a.split(",") if c.strip()]
        for cb in (c.strip().strip('"') for c in b.split(",")):
            if not _has_native(cb):
                continue
            seen[cb] += 1
            for c in set(ca):
                comp[cb][" ".join(_lat(w) for w in c.split())] += 1
    am = {}
    for cb, cnt in comp.items():
        best, n = cnt.most_common(1)[0]
        if n >= min_count and n / seen[cb] >= 0.5:
            am[cb] = best
    return {"name": _pick(nm, min_count, min_purity), "addr": am}


def apply_native_addr(text, mapping):
    """Replace whole native-script address components (comma separated) by their Latin phrase."""
    if not isinstance(text, str) or not _has_native(text):
        return text
    parts = text.split(",")
    for i, p in enumerate(parts):
        core = p.strip().strip('"')
        if core in mapping:
            parts[i] = p.replace(core, mapping[core])
    return ",".join(parts)


def apply_native(text, mapping):
    """Replace mapped native-script tokens by their Latin word; everything else unchanged."""
    if not isinstance(text, str) or not _has_native(text):
        return text
    out = []
    for t in text.split():
        if _has_native(t):
            core = t.strip('",.;:()[]')
            if core in mapping:
                t = t.replace(core, mapping[core])
        out.append(t)
    return " ".join(out)


def native_map_from_frames(s1, s23, gt):
    """Map from the training tables: s1/s23 with entity_id, business_name, business_address; gt as read."""
    import pandas as pd
    g = gt.iloc[:, 1].fillna("").astype(str).str.split(",")
    pairs = pd.DataFrame({"a": gt.iloc[:, 0].to_numpy(), "b": g}).explode("b")
    pairs = pairs[pairs["b"].notna() & (pairs["b"] != "")]
    A = s1.set_index("entity_id").reindex(pairs["a"].to_numpy())
    B = s23.set_index("entity_id").reindex(pairs["b"].to_numpy())
    ok = A["business_name"].notna().to_numpy() & B["business_name"].notna().to_numpy()
    an, bn = A["business_name"].to_numpy()[ok], B["business_name"].to_numpy()[ok]
    aa, ba = A["business_address"].fillna("").to_numpy()[ok], B["business_address"].fillna("").to_numpy()[ok]
    nat = [bool(_NATIVE.search(x or "")) or bool(_NATIVE.search(y or "")) for x, y in zip(bn, ba)]
    idx = [i for i, v in enumerate(nat) if v]
    return build_native_map(((an[i], bn[i]) for i in idx), ((aa[i], ba[i]) for i in idx))


def apply_frame(df, M):
    """Returns a copy of df with native-script name tokens / address components mapped to Latin."""
    if not M:
        return df
    df = df.copy()
    df["business_name"] = [apply_native(x, M["name"]) for x in df["business_name"].fillna("").astype(str)]
    df["business_address"] = [apply_native_addr(x, M["addr"]) for x in df["business_address"].fillna("").astype(str)]
    return df
