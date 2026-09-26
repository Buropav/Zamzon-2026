"""Experiment 'flatkey': for India, the whole phonetic name without spaces is one more blocking token ('q' prefix,
name channel too).

A native-script S2/S3 name transliterates to different words with the SAME phonetic key ('ಪ್ರೈಮ್ ಪ್ರಾಜೆಕ್ಟ್ಸ್' ->
'praim prajekts' ~ S1 'prime projects', both 'prm prjkts'), but each phonetic token ('pprjkts', 'pknsltns',
'pantrprs') is too common in a big state to survive max_df, and the existing whole-name key ('d' + spelling without
spaces) never meets. bench_ohka (with stateparse), 371,969 true pairs, blocking misses:
  no key 4,158 (US 886, India 3,272); key for all countries 3,778 (US 1,088, India 2,690); key for India only
  3,577 (US 887, India 2,690), 3.4% fewer candidates.
In the US the key only pushes near-misses below min_rel (an exact-key competitor raises the best score), so it is
India only. Only names of 2+ phonetic tokens get it (a 1-token key would repeat the 'p' token).
usage: python flatkey.py <ber dir>"""
import os, sys
d = sys.argv[1]


def edit(fn, old, new):
    p = os.path.join(d, fn); s = open(p).read()
    assert s.count(old) == 1, (fn, old[:60], s.count(old))
    open(p, "w").write(s.replace(old, new))


edit("block.py", 'NAME_PREFIXES = ("n", "p", "d")\n', 'NAME_PREFIXES = ("n", "p", "d", "q")   # [flatkey] q = whole phonetic name\n')
edit("block.py", "    return pl.concat([name, phon, addr, hs, dom]).unique()\n",
     "    # [flatkey] India: whole phonetic name without spaces. Transliterated names ('praim prajekts' ~ 'prime\n"
     "    # projects') share it although their single phonetic tokens are too common to survive max_df\n"
     "    pk = df.filter(pl.col(\"core_phon\").str.contains(\" \") & (pl.col(\"country\") == \"india\")).select(\n"
     "        \"row\", (\"q\" + pl.col(\"core_phon\").str.replace_all(\" \", \"\")).alias(\"t\")).filter(pl.col(\"t\").str.len_chars() >= 5)\n"
     "    return pl.concat([name, phon, addr, hs, dom, pk]).unique()\n")
edit("block.py", 'tok1 = token_frame(s1.select("row", "core", "core_phon", "addr", "house", "domain"))',
     'tok1 = token_frame(s1.select("row", "core", "core_phon", "addr", "house", "domain", "country"))')
edit("block.py", 'tok2 = token_frame(s23.select("row", "core", "core_phon", "addr", "house", "domain"))',
     'tok2 = token_frame(s23.select("row", "core", "core_phon", "addr", "house", "domain", "country"))')
edit("pipeline.py", "geo.MIN_COUNT, geo.MIN_PURITY, geo.MIN_MARGIN,", 'geo.MIN_COUNT, geo.MIN_PURITY, geo.MIN_MARGIN, "flatkey2",')
print("flatkey applied to", d)
