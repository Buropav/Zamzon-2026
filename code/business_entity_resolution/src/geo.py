"""
geo.py - region key (state / region) from a normalised address, used to split blocking into smaller pools.

Why: blocking a Source 1 record against a whole country (4-5M Source 2/3 records) is slow (cost per query
grows with the pool) AND loses true matches (measured on India training data: 89% top-k recall against the
full 4.1M pool vs ~96% against region-sized pools - look-alikes elsewhere in the country crowd out the true
match). True matches share the region, so each record is blocked only against its own region (+ records
whose region is unknown). Records whose region cannot be read, or reads as several, get key "".

Keys are matched on normalize_ml(kind="addr") tokens after the lexicon (so "MH", "Maharashtra" and the
Devanagari spelling all arrive as known tokens). Unknown countries get "" everywhere (plain country blocking).
"""
import re

_IN = {
    "MH": "maharashtra mh mumbai bombay pune poona thane nagpur nashik navi aurangabad kalyan vasai solapur kolhapur",
    "DL": "delhi dilli dl",
    "KA": "karnataka ka bangalore bengaluru mysore mysuru mangalore hubli belgaum",
    "TN": "tn tamilnadu chennai madras coimbatore madurai tiruchirappalli trichy salem tiruppur",
    "WB": "wb bengal pashchimavanga kolkata calcutta howrah durgapur siliguri",
    "UP": "up uttar lucknow noida ghaziabad kanpur agra varanasi meerut allahabad prayagraj",
    "GJ": "gujarat gj gujarata ahmedabad surat vadodara baroda rajkot gandhinagar",
    "TG_AP": "telangana tg ts andhra ap andhrapradesh hyderabad secunderabad vijayawada visakhapatnam vizag guntur warangal",
    "KL": "kerala kl keralam keralan kochi cochin ernakulam trivandrum thiruvananthapuram kozhikode calicut thrissur",
    "HR": "haryana hariyana hr gurgaon gurugram faridabad panipat",
    "RJ": "rajasthan rajasthana rj jaipur jodhpur udaipur kota ajmer",
    "BR": "bihar bihara br patna",
    "MP": "mp madhya indore bhopal gwalior jabalpur",
    "OD": "odisha orissa odaisha od bhubaneswar cuttack khordha",
    "PB": "punjab panjab panjaba pb ludhiana amritsar jalandhar mohali",
    "JH": "jharkhand jh ranchi jamshedpur dhanbad",
    "CG": "chhattisgarh chattisgarh cg raipur bilaspur",
    "UK": "uttarakhand uttaranchal dehradun haridwar",
    "HP": "himachal shimla",
    "GA": "goa panaji margao",
    "AS": "assam guwahati",
    "JK": "jammu kashmir srinagar",
    "CH": "chandigarh",
    "PY": "puducherry pondicherry",
}
_US = {
    "AL": "alabama al", "AK": "alaska ak", "AZ": "arizona az", "AR": "arkansas ar", "CA": "california ca",
    "CO": "colorado", "CT": "connecticut", "DE": "delaware de", "DC": "dc", "FL": "florida",
    "GA": "georgia ga", "HI": "hawaii hi", "ID": "idaho id", "IL": "illinois il", "IN": "indiana",
    "IA": "iowa ia", "KS": "kansas ks", "KY": "kentucky ky", "LA": "louisiana la", "ME": "maine me",
    "MD": "maryland md", "MA": "massachusetts ma", "MI": "michigan mi", "MN": "minnesota mn",
    "MS": "mississippi ms", "MO": "missouri mo", "MT": "montana mt", "NE": "nebraska", "NV": "nevada nv",
    "NH": "nh", "NJ": "nj", "NM": "nm", "NY": "ny", "NC": "nc", "ND": "nd", "OH": "ohio oh",
    "OK": "oklahoma ok", "OR": "oregon", "PA": "pennsylvania pa", "RI": "ri", "SC": "sc", "SD": "sd",
    "TN": "tennessee tn", "TX": "texas tx", "UT": "utah ut", "VT": "vermont vt", "VA": "virginia va",
    "WA": "washington wa", "WV": "wv", "WI": "wisconsin wi", "WY": "wyoming wy",
}
_US_PHRASES = {"NH": "new hampshire", "NJ": "new jersey", "NM": "new mexico", "NY": "new york",
               "NC": "north carolina|n carolina", "ND": "north dakota|n dakota", "RI": "rhode island",
               "SC": "south carolina|s carolina", "SD": "south dakota|s dakota", "WV": "west virginia|w virginia",
               "DC": "district of columbia"}
_FR = {
    "HDF": "hauts nord pas calais lille roubaix tourcoing dunkerque villeneuve ascq marcq baroeul wattrelos "
           "lambersart armentieres boulogne arras lens douai valenciennes",
    "NAQ": "nouvelle aquitaine gironde bordeaux pessac merignac talence arcachon teste buch cap ferret lege "
           "begles blanquefort cenon lormont gradignan andernos",
    "PDL": "pays loire atlantique nantes nazaire herblain pornic baule escoublac reze saint sebastien orvault "
           "vertou guerande",
}
# ambiguous French words shared across buckets are removed below; multi-word regions handled by tokens
_FR_AMBIG = {"saint", "pas", "loire", "nord", "cap"}


def _index(table):
    idx = {}
    for key, words in table.items():
        for w in words.split():
            idx.setdefault(w, set()).add(key)
    return idx


_IDX = {"INDIA": _index(_IN), "US": _index(_US), "FRANCE": _index(_FR)}
for w in _FR_AMBIG:
    _IDX["FRANCE"].pop(w, None)
_IDX["FRANCE"]["calais"] = {"HDF"}
_US_RX = [(k, re.compile(rf"(?<!\S)(?:{v})(?!\S)")) for k, v in _US_PHRASES.items()]


_US_NAMES = {}
for _k, _v in _US.items():
    for _w in _v.split():
        if len(_w) > 2:
            _US_NAMES[_w] = _k
for _k, _v in _US_PHRASES.items():
    for _w in _v.split("|"):
        _US_NAMES[_w] = _k


def _us_components(raw_addr):
    """US state from whole comma-separated components of the RAW address: 'OR', 'Oregon', 'North Carolina'.
    Two-letter codes that are also English words (OR, IN, ME, OK, HI, ...) are only trusted this way."""
    found = set()
    for comp in str(raw_addr).split(","):
        c = comp.strip()
        if len(c) == 2 and c.isalpha() and c.upper() in _US:
            found.add(c.upper())
        else:
            k = _US_NAMES.get(re.sub(r"[^a-z ]", "", c.lower()).strip())
            if k:
                found.add(k)
    return found


def region_key(country, norm_addr, raw_addr=None):
    """'' when the region is unknown or ambiguous."""
    idx = _IDX.get(country)
    if not idx or not norm_addr:
        return ""
    found = set()
    if country == "US" and raw_addr:
        found = _us_components(raw_addr)
        if len(found) == 1:
            return next(iter(found))
        found = set()
    if country == "US":
        for k, rx in _US_RX:
            if rx.search(norm_addr):
                found.add(k)
        toks = norm_addr.split()
        # US state codes/names are reliable only near the end of the address or as a whole component
        for t in toks[-3:] + toks[:2]:
            found |= idx.get(t, set())
    else:
        for t in norm_addr.split():
            found |= idx.get(t, set())
    return next(iter(found)) if len(found) == 1 else ""
