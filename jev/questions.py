"""
questions.py - Jev question sets for the six lexicon jobs. One focused judgement per question
(TypeSafe guidance); the combination rules live in code (jev_jobs.py), not in the prompt.
Every state holds WORDS only (plus word-count context), never a business record.
"""

COUNTRY_TXT = {"US": "the United States", "INDIA": "India (Indian words are romanised)", "FRANCE": "France (French)"}
FIELD_TXT = {"name": "business names", "addr": "street addresses"}


def where(country, field):
    return f"{FIELD_TXT.get(field, field)} in {COUNTRY_TXT.get(country, country)}"


# ---- jobs 1, 2, 6: are two words interchangeable? ------------------------------------------
SAME = {
    "same_word": {
        "type": "boolean",
        "instructions": ("Do word_1 and word_2 stand for the same word, so a writer could use either one in "
                         "the same place? Count abbreviations, alternative spellings, romanisations of the "
                         "same word, old and new names of the same city, and translations with the same meaning."),
        "criteria": {"true": "same word or same meaning, e.g. bd/boulevard, shree/sri, bombay/mumbai, "
                             "boulangerie/bakery, inbhestament/investment",
                     "false": "different words: different names, surnames or places, opposite or merely related "
                              "meanings, e.g. patel/patil, north/south, road/street, traders/enterprises"},
    },
    "different_names": {
        "type": "boolean",
        "instructions": ("Are word_1 and word_2 two DIFFERENT proper names (people, families, brands, towns or "
                         "streets) that only look or sound alike?"),
        "criteria": {"true": "distinct names that look alike, e.g. patel/patil, singh/sinha, sharma/verma, "
                             "paris/parie, lyon/lion",
                     "false": "the same name spelled differently, or not proper names, e.g. shree/sri, "
                              "mohammed/mohammad, bd/boulevard"},
    },
    "relation": {
        "type": "choice",
        "instructions": "How are word_1 and word_2 related?",
        "criteria": {
            "abbreviation": "one is a short form of the other (bd/boulevard, pvt/private, r/rue)",
            "spelling_variant": "same word spelled, romanised or mistyped differently (shree/sri, boulveard/boulevard)",
            "translation": "same meaning in another language (boulangerie/bakery, bhandar/store)",
            "renamed_place": "old and new names of the same place (bombay/mumbai, gurgaon/gurugram)",
            "different": "not the same word or meaning",
        },
    },
}


def same_state(a, b, country, field, extra=None):
    s = {"word_1": a, "word_2": b, "found_in": where(country, field)}
    if extra:
        s.update(extra)
    return s


# ---- job 3: what does an ambiguous short form mean? -----------------------------------------
def meaning_question(options):
    crit = {o: f"it means '{o}'" for o in options}
    crit["other"] = "none of these / cannot tell"
    return {"meaning": {"type": "choice",
                        "instructions": "In this phrase, what does the short form stand for?",
                        "criteria": crit}}


# ---- job 4: what kind of word is it? --------------------------------------------------------
WORD_CLASS = {
    "kind": {
        "type": "choice",
        "instructions": "What kind of word is this inside a business name?",
        "criteria": {
            "legal_form": "company legal form or its abbreviation (llc, inc, pvt, ltd, llp, sarl, sas, eurl, gmbh, plc)",
            "title": "honorific or professional title (dr, mr, shri, m/s, md, phd, cpa, esq)",
            "generic_business": "common word describing a business type or activity (traders, services, "
                                "consulting, boulangerie, pharmacy, enterprises, group, solutions)",
            "connector": "article, preposition or conjunction (and, of, the, de, la, du, et)",
            "place": "a town, city, region, state or country name, or a direction (mumbai, texas, lyon, north)",
            "distinctive": "a person, family or brand name, or any other word that identifies this business",
        },
    },
}

ADDR_CLASS = {
    "kind": {
        "type": "choice",
        "instructions": "What kind of word is this inside a street address?",
        "criteria": {
            "street_type": "street type or its abbreviation (road, rd, street, rue, avenue, marg, gali, lane)",
            "unit_or_building": "unit, floor, building or plot word (suite, floor, flat, shop, plot, tower, batiment)",
            "locality_word": "generic locality word (nagar, colony, sector, layout, phase, zone, quartier)",
            "city_or_town": "name of a city, town or village",
            "state_or_region": "name or code of a state, region, department or district",
            "direction_or_connector": "direction or connector word (north, east, de, la, near, opp)",
            "distinctive": "any other name (a person, landmark, street or building name)",
        },
    },
}
