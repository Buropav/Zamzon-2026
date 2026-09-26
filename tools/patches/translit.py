"""Experiment 'translit': Malayalam and Tamil romanisation fixes, spelled-out LLP.

The friend's romaniser uses one Devanagari-layout table for every Brahmic script. Where a script differs:
Malayalam
  - chillu letters (U+0D7A-0D7F, word-final consonants without a vowel) are outside the table and were DROPPED:
    'ഡിജിറ്റൽ' -> 'dijirr', 'എൽഎൽപി' (LLP) -> 'eepi', 'ഗോൾഡ്' -> 'god'
  - 'റ്റ' (RRA virama RRA) is how Malayalam writes an English 't', 'ന്റ' is 'nt'; both came out with 'r':
    'ലിമിറ്റഡ്' -> 'limirrad', 'പ്രൈവറ്റ്' -> 'praivarr', 'എന്റർപ്രൈസസ്' -> 'enrapraisas'
Tamil
  - 'ச' is 's' in English words ('சில்வர்' silver came out 'chilvar', phonetic key 'k...'); 'ச்ச' is 'ch', 'ஞ்ச' 'nj'
  - 'ஃப' is 'f' ('ஃபுட்ஸ்' foods came out 'hputs')
Both: the spelled-out LLP ('एलएलपी' -> 'elaelapi', 'ಎಲ್ಎಲ್ಪಿ' -> 'elelpi', 5% of native-script names) is legal form 'llp'.
Train ground truth, native-script true pairs whose whole phonetic name equals S1's: all 110k 59.6% -> 67.1%,
Malayalam 2.3% -> 70.6%, Tamil 27.7% -> 56.6%. Tamil Nadu test (149k true pairs), blocking misses 3,466 -> 2,518.
usage: python translit.py <ber dir>"""
import os, re, sys
d = sys.argv[1]


def edit(fn, old, new):
    p = os.path.join(d, fn); s = open(p).read()
    assert s.count(old) == 1, (fn, old[:60], s.count(old))
    open(p, "w").write(s.replace(old, new))


# script-specific letters get private marker characters inside romanize_indic, resolved once the word is known
edit("normalize.py", '_MODIFIERS = {0x01: "n", 0x02: "\\x02", 0x03: "h"}\n', r'''_MODIFIERS = {0x01: "n", 0x02: "\x02", 0x03: "h"}
# [translit] letters whose sound depends on the script or on the neighbours: marker now, resolved at the end
_CHILLU = {0x0D7A: "n", 0x0D7B: "n", 0x0D7C: "r", 0x0D7D: "l", 0x0D7E: "l", 0x0D7F: "k"}  # Malayalam final consonants
_MARK = {0x0D31: "\x03",   # Malayalam RRA: 'റ്റ' tt, 'ന്റ' nt, else r
         0x0B9A: "\x05",   # Tamil CA: s, 'ச்ச' ch, 'ஞ்ச' j
         0x0B9E: "\x06",   # Tamil NYA: n
         0x0B83: "\x07"}   # Tamil AYTHAM: 'ஃப' f, else h
_MARK_RESOLVE = [("\x03\x03", "tt"), ("n\x03", "nt"), ("\x03", "r"),
                 ("\x06\x05", "nj"), ("\x06", "n"), ("\x05\x05", "ch"), ("\x05", "s"),
                 ("\x07p", "f"), ("\x07", "h")]
''')
edit("normalize.py", "    for ch in text:\n        off = _indic_offset(ch)\n        if off is None:\n",
     "    for ch in text:\n        off = _indic_offset(ch)\n"
     "        if ord(ch) in _CHILLU:  # [translit] was dropped\n"
     "            if pending is not None:\n"
     "                out.append(pending + \"a\")\n"
     "                pending = None\n"
     "            out.append(_CHILLU[ord(ch)])\n"
     "            continue\n"
     "        if off is None:\n")
edit("normalize.py", "                out.append(pending + \"a\")\n            pending = _CONSONANTS[off]\n",
     "                out.append(pending + \"a\")\n            pending = _MARK.get(ord(ch)) or _CONSONANTS[off]   # [translit]\n")
edit("normalize.py", "            out.append(_MODIFIERS[off])\n",
     "            out.append(_MARK.get(ord(ch)) or _MODIFIERS[off])   # [translit] Tamil aytham\n")
edit("normalize.py", "    s = re.sub(\"\\x02(?=[pbm]|$|[^a-z])\", \"m\", \"\".join(out))\n",
     "    s = \"\".join(out)\n"
     "    for a, b in _MARK_RESOLVE:  # [translit]\n"
     "        s = s.replace(a, b)\n"
     "    s = re.sub(\"\\x02(?=[pbm]|$|[^a-z])\", \"m\", s)\n")
edit("normalize.py", '    "llc": "llc", "llp": "llp", "lp": "lp", "plc": "plc", "pllc": "pllc",\n',
     '    "llc": "llc", "llp": "llp", "lp": "lp", "plc": "plc", "pllc": "pllc",\n'
     '    "elelpi": "llp", "elaelapi": "llp",   # [translit] spelled-out LLP in Indic scripts\n')
p = os.path.join(d, "prep.py"); s = open(p).read()
m = re.search(r'^NORM_VERSION = ("?)(\w+)\1(.*)$', s, re.M)
assert m and s.count("NORM_VERSION = ") == 1, "NORM_VERSION line not found"
open(p, "w").write(s[:m.start()] + f'NORM_VERSION = "{m.group(2)}t"   # [translit] romanisation changed' + s[m.end():])
edit("pipeline.py", "geo.MIN_COUNT, geo.MIN_PURITY, geo.MIN_MARGIN,", 'geo.MIN_COUNT, geo.MIN_PURITY, geo.MIN_MARGIN, "translit1",')
print("translit applied to", d)
