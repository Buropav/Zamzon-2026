import re
import unicodedata
import pandas as pd

try:
    from indic_transliteration.sanscript import transliterate, DEVANAGARI, HK
    HAS_TRANSLIT = True
except Exception:
    HAS_TRANSLIT = False

_STOP = set("""
pvt private ltd limited llc llp inc incorporated corp corporation co company plc
the and of
sarl sas sasu sa eurl snc sci scp selarl ets etablissements societe cie et
de du des la le les l d au aux
""".split())

def canon_country(c):
    if not c or pd.isna(c):
        return ""
    s = str(c).strip().lower()
    s = re.sub(r'[^a-z0-9]', '', s)
    if s in {"us", "usa", "unitedstates", "unitedstatesofamerica"}:
        return "US"
    if s in {"in", "ind", "india", "bharat"}:
        return "INDIA"
    if s in {"fr", "fra", "france", "french", "republiquefrancaise"}:
        return "FRANCE"
    return s.upper()

def transliterate_text(text):
    if not HAS_TRANSLIT or not isinstance(text, str):
        return text
    if any('\u0900' <= ch <= '\u097f' for ch in text):
        try:
            return transliterate(text, DEVANAGARI, HK)
        except Exception:
            return text
    return text

def normalize_text(text):
    if not text or pd.isna(text):
        return ""
    s = transliterate_text(str(text))
    s = unicodedata.normalize('NFKD', s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    # Replace & with and
    s = re.sub(r'&', ' and ', s)
    # Remove unwanted punctuation
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def core_name(norm_name):
    toks = [t for t in str(norm_name).split() if t not in _STOP]
    return " ".join(toks) if toks else str(norm_name)

def _initials(s):
    toks = str(s).split()
    return "".join(t[0] for t in toks) if len(toks) >= 2 else ""
