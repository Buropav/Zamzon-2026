"""Experiment 'native_map': the old pipeline's native-script -> Latin word map (translit.py), learned from the
TRAIN ground truth only, applied to the raw names/addresses of every split before the friend's normalisation.
A native-script S2/S3 name is a word-for-word transliteration of its S1 name (same token count), so aligned
words give e.g. 'गैलेक्सी' -> 'galaxy'; whole native-script address components map to their S1 component.
Unmapped words still go through the friend's rule-based romanisation. The map is built once from
<data_dir>/train/*.tsv and cached in the work dir; cached normalised tables get a new file name.
usage: python native_map.py <ber dir>"""
import os, sys
d = sys.argv[1]

MODULE = r'''"""Native-script -> Latin word map learned from TRAIN true pairs only (ported from the old translit.py)."""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

_NATIVE = re.compile(r"[؀-ۿऀ-෿]")
_LAT = re.compile(r"[^a-z0-9&]+")
_NATIVE_PL = r"[\x{0600}-\x{06FF}\x{0900}-\x{0DFF}]"


def _has_native(s):
    return bool(_NATIVE.search(s or ""))


def _lat(t):
    return _LAT.sub("", t.lower())


def build_native_map(pairs_names, pairs_addrs, min_count=3, min_purity=0.6):
    nm = defaultdict(Counter)
    for a, b in pairs_names:
        ta, tb = (a or "").split(), (b or "").split()
        if len(ta) != len(tb):
            continue
        for x, y in zip(tb, ta):
            if _has_native(x):
                nm[x.strip('",.;:()[]')][_lat(y)] += 1
    names = {}
    for tok, c in nm.items():
        (best, n), tot = c.most_common(1)[0], sum(c.values())
        if n >= min_count and n / tot >= min_purity and best:
            names[tok] = best
    comp, seen = defaultdict(Counter), Counter()
    for a, b in pairs_addrs:
        ca = [c.strip() for c in (a or "").split(",") if c.strip()]
        for cb in (c.strip().strip('"') for c in (b or "").split(",")):
            if not _has_native(cb):
                continue
            seen[cb] += 1
            for c in set(ca):
                comp[cb][" ".join(_lat(w) for w in c.split())] += 1
    addrs = {}
    for cb, cnt in comp.items():
        best, n = cnt.most_common(1)[0]
        if n >= min_count and n / seen[cb] >= 0.5:
            addrs[cb] = best
    return {"name": names, "addr": addrs}


def build_from_train(data_dir, read):
    """read: the pipeline's TSV reader. Uses train_source1/2/3 + train_ground_truth only."""
    t = Path(data_dir) / "train"
    s1 = read(t / "train_source1.tsv").select("entity_id", "business_name", "business_address")
    s23 = pl.concat([read(t / f"train_source{k}.tsv").select("entity_id", "business_name", "business_address")
                     for k in (2, 3)])
    nat = s23.filter(pl.col("business_name").fill_null("").str.contains(_NATIVE_PL)
                     | pl.col("business_address").fill_null("").str.contains(_NATIVE_PL))
    gt = (read(t / "train_ground_truth.tsv").rename({"source1_entity_id": "a", "matched_entity_ids": "b"})
          .filter(pl.col("b").fill_null("") != "").with_columns(pl.col("b").str.split(",")).explode("b")
          .with_columns(pl.col("b").str.strip_chars()))
    p = (gt.join(nat.rename({"entity_id": "b", "business_name": "bn", "business_address": "ba"}), on="b")
           .join(s1.rename({"entity_id": "a", "business_name": "an", "business_address": "aa"}), on="a"))
    return build_native_map(zip(p["an"].to_list(), p["bn"].to_list()), zip(p["aa"].to_list(), p["ba"].to_list()))


_CACHE = {}


def load_or_build(data_dir, work_dir, read):
    path = Path(work_dir) / "native_map.json"
    if str(path) not in _CACHE:
        if path.exists():
            M = json.loads(path.read_text(encoding="utf-8"))
        else:
            M = build_from_train(data_dir, read)
            path.write_text(json.dumps(M, ensure_ascii=False), encoding="utf-8")
            print(f"  native-script map from train pairs: {len(M['name']):,} name words, {len(M['addr']):,} address components")
        _CACHE[str(path)] = M
    return _CACHE[str(path)]


def _apply_name(text, mapping):
    if not text or not _has_native(text):
        return text
    out = []
    for t in text.split():
        if _has_native(t):
            core = t.strip('",.;:()[]')
            if core in mapping:
                t = t.replace(core, mapping[core])
        out.append(t)
    return " ".join(out)


def _apply_addr(text, mapping):
    if not text or not _has_native(text):
        return text
    parts = text.split(",")
    for i, p in enumerate(parts):
        core = p.strip().strip('"')
        if core in mapping:
            parts[i] = p.replace(core, mapping[core])
    return ",".join(parts)


def apply_map(df, M):
    """Maps native-script name words / address components of a raw source frame (only rows containing them)."""
    nat = (df["business_name"].fill_null("").str.contains(_NATIVE_PL)
           | df["business_address"].fill_null("").str.contains(_NATIVE_PL))
    if not nat.any():
        return df
    idx = nat.arg_true()
    names = [_apply_name(x, M["name"]) for x in df["business_name"].gather(idx).to_list()]
    addrs = [_apply_addr(x, M["addr"]) for x in df["business_address"].gather(idx).to_list()]
    bn = df["business_name"].scatter(idx, pl.Series(names, dtype=pl.Utf8))
    ba = df["business_address"].scatter(idx, pl.Series(addrs, dtype=pl.Utf8))
    return df.with_columns(bn.alias("business_name"), ba.alias("business_address"))
'''

def edit(fn, old, new):
    p = os.path.join(d, fn); s = open(p).read()
    assert s.count(old) == 1, (fn, old[:60], s.count(old))
    open(p, "w").write(s.replace(old, new))

assert not os.path.exists(os.path.join(d, "native_map.py")), "native_map already applied"
open(os.path.join(d, "native_map.py"), "w").write(MODULE)
edit("prep.py", "from .normalize import RECORD_COLUMNS, normalize_record\n",
     "from .normalize import RECORD_COLUMNS, normalize_record\nfrom .native_map import apply_map, load_or_build\n")
edit("prep.py", 'dst = work_dir / f"{split}_{src}_norm_v{NORM_VERSION}.parquet"',
     'dst = work_dir / f"{split}_{src}_norm_v{NORM_VERSION}nm.parquet"   # [native_map] mapped tables')
edit("prep.py", '            raw = read_source_tsv(Path(data_dir) / split / f"{split}_{src}.tsv")\n',
     '            raw = read_source_tsv(Path(data_dir) / split / f"{split}_{src}.tsv")\n'
     '            raw = apply_map(raw, load_or_build(data_dir, work_dir, read_source_tsv))   # [native_map]\n')
print("native_map applied to", d)
