"""Experiment 'fillers': generator filler words learned from the data, removed from the core name (France fix #1).

Evidence (real test data, no labels): French S2/S3 names carry words that S1 names never contain -
'developpement' 44,064 in S2/S3 vs 0 in S1, 'participations' 34,280 vs 1, 'holding' 33,976 vs 17,
'distribution' 34,182 vs 64, 'associes' 10,477 vs 0; US S2/S3 have 'southside'/'eastgate'/'northside'
(~19k each vs 0). The friend's NAME_FILLER is a short hand-written English/Indian list. A word that is (per
country, per record) >= 50x more frequent in S2/S3 names than in S1 names and occurs >= 200 times cannot help
match an S1 record, so it is dropped (Latin-script S2/S3 names only, >= 0.2% of records, no digits,
not one edit away from a frequent S1 word) from `core` (and `core_phon`) on both sides; `name` keeps everything.
Learned per split and per country from that split's own records (no labels), so train and test get the same rule.
Also: French 'et' -> 'and' (S1 writes '&'), 'cie'/'compagnie' -> legal form 'co' like English 'company'.
usage: python fillers.py <ber dir>"""
import os, sys
d = sys.argv[1]

MODULE = r'''"Generator filler words learned per country from S2/S3 vs S1 name-token frequencies (no labels)."
import polars as pl
from rapidfuzz import distance, process

from .normalize import phonetic_key

MIN_RATE = 0.002     # share of the country's Latin-script S2/S3 names containing the word
MIN_RATIO = 50.0     # per-record rate in S2/S3 names / per-record rate in S1 names (S1 count + 1)
TYPO_VOCAB_MIN = 50  # S1 words this frequent protect their one-edit variants (typos are not fillers)
VERSION = "fill3"    # part of the candidate cache key


def _token_counts(df):
    return (df.select("country", pl.col("name").str.split(" ").list.unique().alias("t")).explode("t")
              .filter(pl.col("t").is_not_null() & (pl.col("t").str.len_chars() >= 2)
                      & ~pl.col("t").str.contains(r"\d"))
              .group_by("country", "t").len("c"))


def learn_fillers(s1, s23):
    """-> {country: set(tokens)}. S2/S3 counts use Latin-script names only (name_indic == 0): romanised Indic
    words are real name words that S1 writes in English, not fillers."""
    lat = s23.filter(pl.col("name_indic") == 0)
    n1 = dict(s1.group_by("country").len().rows())
    n23 = dict(lat.group_by("country").len().rows())
    c1 = _token_counts(s1).rename({"c": "c1"})
    j = _token_counts(lat).rename({"c": "c23"}).join(c1, on=["country", "t"], how="left").with_columns(
        pl.col("c1").fill_null(0))
    out = {}
    for ctry in sorted(n23):
        if ctry not in n1:
            continue
        vocab = [t for t, c in c1.filter((pl.col("country") == ctry) & (pl.col("c1") >= TYPO_VOCAB_MIN))
                 .select("t", "c1").rows()]
        for t, a, b in j.filter(pl.col("country") == ctry).select("t", "c23", "c1").rows():
            if a < MIN_RATE * n23[ctry] or (a / n23[ctry]) < MIN_RATIO * ((b + 1) / n1[ctry]):
                continue
            near = process.extract(t, vocab, scorer=distance.Levenshtein.distance, score_cutoff=1, limit=3)
            if any(v != t and dist == 1 for v, dist, _ in near):
                continue   # one edit away from a common S1 word: a typo of it, not a filler
            out.setdefault(ctry, set()).add(t)
    return out


def strip_fillers(df, fillers):
    """Removes each row's country fillers from core / core_phon (kept when nothing would remain)."""
    if not fillers:
        return df, 0
    core, ctry = df["core"].to_list(), df["country"].to_list()
    phon = df["core_phon"].to_list()
    changed = 0
    for i, (c, k) in enumerate(zip(core, ctry)):
        f = fillers.get(k)
        if not f or not c:
            continue
        toks = c.split()
        kept = [t for t in toks if t not in f]
        if kept and len(kept) < len(toks):
            core[i] = " ".join(kept)
            phon[i] = " ".join(phonetic_key(t) for t in kept)
            changed += 1
    return df.with_columns(pl.Series("core", core, dtype=pl.Utf8), pl.Series("core_phon", phon, dtype=pl.Utf8)), changed
'''

def edit(fn, old, new):
    p = os.path.join(d, fn); s = open(p).read()
    assert s.count(old) == 1, (fn, old[:60], s.count(old))
    open(p, "w").write(s.replace(old, new))

assert not os.path.exists(os.path.join(d, "fillers.py")), "fillers already applied"
open(os.path.join(d, "fillers.py"), "w").write(MODULE)
# canonical French connectors / company word (normalisation changes -> new cached table names)
edit("normalize.py", '    "ets": "ets", "etablissements": "ets", "etablissement": "ets", "selarl": "selarl", "scp": "scp",\n',
     '    "ets": "ets", "etablissements": "ets", "etablissement": "ets", "selarl": "selarl", "scp": "scp",\n'
     '    "compagnie": "co", "cie": "co",   # [fillers] French "company", like English co\n')
edit("normalize.py", "    toks = [_fix_leet(t) for t in s.split()]\n",
     "    toks = [_fix_leet(t) for t in s.split()]\n"
     "    toks = [\"and\" if t == \"et\" else t for t in toks]   # [fillers] French 'et' = S1's '&'\n")
edit("prep.py", "NORM_VERSION = 2\n", 'NORM_VERSION = "2f"   # [fillers] normalisation changed\n')
# learned fillers, applied in load_split for both splits
edit("pipeline.py", "from . import block, decide, geo, model\n", "from . import block, decide, fillers, geo, model\n")
edit("pipeline.py", "    return s1, s23\n\n\ndef cached_candidates",
     "    # [fillers] generator filler words (no labels), removed from core names. Countries seen in training use\n"
     "    # the list learned on the TRAIN split for both splits (same rule, no train/test skew); countries only in\n"
     "    # test (France) use the list learned on the test split.\n"
     "    fl = fillers.learn_fillers(s1, s23)\n"
     "    fpath = Path(cfg[\"work_dir\"]) / f\"fillers_train_{fillers.VERSION}.json\"\n"
     "    if split == \"train\":\n"
     "        fpath.write_text(json.dumps({\"countries\": sorted(set(s1[\"country\"].unique().to_list())),\n"
     "                                     \"fillers\": {k: sorted(v) for k, v in fl.items()}}))\n"
     "    elif fpath.exists():\n"
     "        tr = json.loads(fpath.read_text())\n"
     "        fl = {k: v for k, v in fl.items() if k not in tr[\"countries\"]}\n"
     "        fl.update({k: set(v) for k, v in tr[\"fillers\"].items()})\n"
     "    s1, n1 = fillers.strip_fillers(s1, fl)\n"
     "    s23, n23 = fillers.strip_fillers(s23, fl)\n"
     "    log(f\"{split}: learned filler words {({k: sorted(v) for k, v in sorted(fl.items())})}; core names changed: \"\n"
     "        f\"S1 {n1:,}, S2/S3 {n23:,}\")\n"
     "    return s1, s23\n\n\ndef cached_candidates")
edit("pipeline.py", "geo.MIN_COUNT, geo.MIN_PURITY, geo.MIN_MARGIN,", "geo.MIN_COUNT, geo.MIN_PURITY, geo.MIN_MARGIN, fillers.VERSION,")
print("fillers applied to", d)
