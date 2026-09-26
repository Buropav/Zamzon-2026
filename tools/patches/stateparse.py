"""Experiment 'stateparse': the state is read from the record's own country, strongest kind wins, last one of it.

Blocking searches only the query's own state (plus stateless S1), so an S2/S3 record whose state differs from its
S1's can never be matched. The friend's parser took the FIRST state-like component, let a code replace an earlier
full name only once, applied the India phonetic lookup to every country and read 2-letter codes inside components
with digits. Real addresses defeat that:
  'Fl 1, Beaverton, Oregon, 7436 165th Pl'           -> Florida   ('Fl 1' is a floor)
  '1038 Simon Avenue, Carroll, Iowa'                 -> Kerala    (India phonetic key of 'Carroll')
  'DC, Washington, Unit 301, 2300 Washington Place'  vs  '..., Washington, District of Columbia' -> WA vs DC
  'Rajasthan, C/O Rinku Yadav, ..., Bahror'          -> Bihar     (phonetic key of 'Bahror')
  '..., Rajajinag, Ar, Bangalore, Karnataka'          -> Arunachal ('Nag, Ar' is a split 'Nagar')
Rule: a component with a digit is never a state; only the record's own country's states count (US codes, the 16
India codes the sources use, exact names; the India phonetic lookup only for India); a 2-letter code beats an exact
name, which beats a phonetic match; within the strongest kind the LAST component wins. 'Fl N' floor components are
dropped from the address text, as the friend's parser did when it took them for Florida.
Train ground truth (200k S1, 692k pairs), pairs whose S1 and S2/S3 states differ (unreachable by blocking):
  US 1.419% -> 0.028%, India 0.387% -> 0.031%.
usage: python stateparse.py <ber dir>"""
import os, re, sys
d = sys.argv[1]


def edit(fn, old, new):
    p = os.path.join(d, fn); s = open(p).read()
    assert s.count(old) == 1, (fn, old[:60], s.count(old))
    open(p, "w").write(s.replace(old, new))


OLD = '''    # State = a whole comma-separated component ("..., TX" / "..., karnataka").
    # A 2-letter code beats a full name, so "Washington, DC" -> DC.
    ctry = (country or "").lower()
    codes = US_STATES if ctry.startswith("us") else IN_STATES if ctry.startswith("ind") else {}
    prefix = "us_" if ctry.startswith("us") else "in_"
    state, by_code, kept = "", False, []
    for comp in s.split(","):
        c = " ".join(re.sub(r"[^a-z]+", " ", comp).split())
        code = None
        if len(c) == 2 and c in codes:
            code, is_code = prefix + c, True
        elif c:
            code = _STATE_FULL.get(c) or _STATE_SKEL.get(" ".join(phonetic_key(w) for w in c.split()))
            is_code = False
            if code and re.search(r"\\d", comp):  # "12 Texas Ave" is a street, not a state
                code = None
        if code and (not state or (is_code and not by_code)):
            if state and not by_code:
                kept.append(prev_comp)  # demote the earlier full name back to address text
            state, by_code, prev_comp = code, is_code, comp
            continue
        kept.append(comp)
'''
NEW = '''    # [stateparse] State = a whole comma-separated component naming a state of the record's OWN country.
    # Sources shuffle components and write cities named like states next to the state ("DC, Washington",
    # "Oregon, OH", "Rajasthan, ..., Bahror"): a 2-letter code beats an exact name, which beats the India-only
    # phonetic match, and within the strongest kind the LAST one wins. A component with a digit is never a state
    # ("Fl 1" is a floor, "PR 623" a road); India uses only the codes its sources write ("Nag, Ar" is a split
    # "Nagar"). Train pairs whose S1 / S2-S3 states differ (a blocking miss): US 1.42% -> 0.03%, India 0.39% -> 0.03%.
    ctry = (country or "").lower()
    prefix = "us_" if ctry.startswith("us") else "in_" if ctry.startswith("ind") else "fr_" if ctry.startswith("fr") else ""
    codes = US_STATES if prefix == "us_" else IN_CODES_USED if prefix == "in_" else ()
    comps = s.split(",")
    best, best_kind, state = -1, -1, ""
    for i, comp in enumerate(comps):
        if not prefix or re.search(r"\\d", comp):
            continue
        c = " ".join(re.sub(r"[^a-z]+", " ", comp).split())
        if not c:
            continue
        if len(c) == 2 and c in codes:
            code, kind = prefix + c, 2
        else:
            code, kind = _STATE_FULL.get(c), 1
            if prefix == "in_" and not (code or "").startswith(prefix):
                code, kind = _STATE_SKEL.get(" ".join(phonetic_key(w) for w in c.split())), 0
        if code and code.startswith(prefix) and kind >= best_kind:
            best, best_kind, state = i, kind, code
    # the state component leaves the text; so do floor markers ("Fl 1"), which the friend's parser read as Florida
    kept = [comp for i, comp in enumerate(comps) if i != best and not _FLOOR_RE.match(comp)]
'''

edit("normalize.py", OLD, NEW)
edit("normalize.py", "_STATE_FULL, _STATE_SKEL = _build_state_lookup()\n",
     "_STATE_FULL, _STATE_SKEL = _build_state_lookup()\n"
     "# [stateparse] India codes the sources actually write (>= 1,000 standalone uses each in train); the others\n"
     "# (ar, an, ch, la, ga, as, uk, ...) appear almost only as fragments of split words (\"Rajajinag, Ar\").\n"
     "IN_CODES_USED = {\"ap\", \"br\", \"dl\", \"gj\", \"hr\", \"ka\", \"kl\", \"mh\", \"mp\", \"od\", \"pb\", \"rj\", \"tg\", \"tn\", \"up\", \"wb\"}\n"
     "_FLOOR_RE = re.compile(r\"^\\s*fl\\.?\\s*\\d+\\s*$\")\n")
# normalised tables are cached by NORM_VERSION; the candidate cache key gets a version string of its own
p = os.path.join(d, "prep.py"); s = open(p).read()
m = re.search(r'^NORM_VERSION = ("?)(\w+)\1(.*)$', s, re.M)
assert m and s.count("NORM_VERSION = ") == 1, "NORM_VERSION line not found"
open(p, "w").write(s[:m.start()] + f'NORM_VERSION = "{m.group(2)}s"   # [stateparse] state parsing changed' + s[m.end():])
edit("pipeline.py", "geo.MIN_COUNT, geo.MIN_PURITY, geo.MIN_MARGIN,", 'geo.MIN_COUNT, geo.MIN_PURITY, geo.MIN_MARGIN, "stateparse1",')
print("stateparse applied to", d)
