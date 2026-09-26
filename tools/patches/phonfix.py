"""Experiment 'phonfix': phonetic key follows English pronunciation where Indic transliterations do.

S1 names are English spellings; native-script S2/S3 names are transliterated from pronunciation, so the phonetic
key disagrees wherever English spelling and sound differ. Most common disagreements on train native-script true
pairs (each 1-2% of Devanagari pairs):
  'tu' before r/n sounds 'chu'       ventures bntrs / venchars bnkrs, future, fortune, structure (ctu -> kchu)
  'c' before s or at the end is 'k'  logistics ljsts / lojistiks ljstks, dynamic, classic
  final/pre-consonant 'j' is 's'     industries antstrs / indastrij antstrj (English s = z = Hindi j), business/bijnes
  'gh' before t or at the end is silent   high hg / hai h, bright
  final 'ew' is 'yu'                 new nb / nyu n
Romaniser: anusvara before 'ph' (f) is 'n' ('इंफ्रा' -> inphra, was imphra); word-final anusvara after 'i' is 'n'
(Bengali '-ing': 'মার্কেটিং' -> marketin, was marketim); Gurmukhi tippi (U+0A70) is an anusvara (was dropped:
'ਇੰਟਰਨੈਸ਼ਨਲ' -> itarneshanal).
The key is one function for both sides, so matching Latin pairs cannot diverge; the cost is extra collisions.
Train (600k S1): other S1 names sharing an S1's whole key: US 9.586 -> 9.586, India 8.136 -> 8.344.
Native-script true pairs with equal whole phonetic names 67.1% -> 85.0% (Devanagari 65.8 -> 87.4, Bengali 62.4 ->
84.9, Gurmukhi 56.0 -> 87.8, Tamil 56.6 -> 63.2). Blocking misses: bench_ohka 3,516 -> 3,192 (India 2,631 -> 2,307,
US 885 -> 885); Tamil Nadu 2,518 -> 2,376. Apply after translit (uses its codepoint table).
usage: python phonfix.py <ber dir>"""
import os, re, sys
d = sys.argv[1]


def edit(fn, old, new):
    p = os.path.join(d, fn); s = open(p).read()
    assert s.count(old) == 1, (fn, old[:60], s.count(old))
    open(p, "w").write(s.replace(old, new))


edit("normalize.py", '_PHON_RULES = [\n    (re.compile(r"ction"), "kshon"),',
     '_PHON_RULES = [\n'
     '    # [phonfix] English spelling -> sound where transliterations follow the sound\n'
     '    (re.compile(r"ctu(?=[rn])"), "kchu"),               # structure ~ strakchar\n'
     '    (re.compile(r"(?<=[a-z])tu(?=[rn])"), "chu"),       # ventures ~ venchars, future, fortune\n'
     '    (re.compile(r"(?<=[a-z])gh(?=t|$)"), ""),          # high ~ hai, bright ~ brait (never the whole token)\n'
     '    (re.compile(r"ew$"), "yu"),                         # new ~ nyu\n'
     '    (re.compile(r"ction"), "kshon"),')
edit("normalize.py", '    (re.compile(r"(ck|q|c(?=[aoulrkt])|ch|kh)"), "k"),\n',
     '    (re.compile(r"(ck|q|c(?=[aoulrkts]|$)|ch|kh)"), "k"),   # [phonfix] +s/end: logistics ~ lojistiks, dynamic\n')
edit("normalize.py", '    (re.compile(r"(sh|z)"), "s"),\n',
     '    (re.compile(r"(sh|z)"), "s"),\n'
     '    (re.compile(r"j(?=$|[^aeiouy])"), "s"),   # [phonfix] English s = z = Hindi j: industries ~ indastrij\n')
# romaniser: Gurmukhi tippi, anusvara before ph and after i at the word end
# (needs translit: its codepoint table flushes the pending consonant's vowel, as a nasal sign needs)
edit("normalize.py", '_CHILLU = {0x0D7A: "n", 0x0D7B: "n", 0x0D7C: "r", 0x0D7D: "l", 0x0D7E: "l", 0x0D7F: "k"}',
     '_CHILLU = {0x0D7A: "n", 0x0D7B: "n", 0x0D7C: "r", 0x0D7D: "l", 0x0D7E: "l", 0x0D7F: "k",\n'
     '           0x0A70: "\\x02"}  # [phonfix] Gurmukhi tippi = anusvara (was dropped)\n#')
edit("normalize.py", '    s = re.sub("\\x02(?=[pbm]|$|[^a-z])", "m", s)\n',
     '    s = re.sub("(?<=i)\\x02(?=$|[^a-z])", "n", s)   # [phonfix] Bengali/Gurmukhi \'-ing\'\n'
     '    s = re.sub("\\x02(?=p(?!h)|[bm]|$|[^a-z])", "m", s)   # [phonfix] not before ph (f)\n')
p = os.path.join(d, "prep.py"); s = open(p).read()
m = re.search(r'^NORM_VERSION = ("?)(\w+)\1(.*)$', s, re.M)
assert m and s.count("NORM_VERSION = ") == 1, "NORM_VERSION line not found"
open(p, "w").write(s[:m.start()] + f'NORM_VERSION = "{m.group(2)}p"   # [phonfix] phonetic key changed' + s[m.end():])
edit("pipeline.py", "geo.MIN_COUNT, geo.MIN_PURITY, geo.MIN_MARGIN,", 'geo.MIN_COUNT, geo.MIN_PURITY, geo.MIN_MARGIN, "phonfix1",')
print("phonfix applied to", d)
