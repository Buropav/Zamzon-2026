"""
jev_sprint.py - one-day, rules-conscious use of Jev (TypeSafe AI) through Vercel AI Gateway.

Jev returns typed decisions (boolean probability / choice / score). It does not generate text.
So it cannot write rules; it can JUDGE candidates we generate locally.

Two jobs, in priority order:
  A. Lexicon judge (low rules risk): is token A the same word as token B (abbreviation,
     spelling/transliteration, translation)? No business identity is judged, no records are
     sent - only single words. Approved pairs become a static file, resources/lexicon.tsv,
     loaded by er_multilingual. The final pipeline never calls any API.
  B. Error taxonomy (gray area - ask the organisers first): for TRAIN pairs whose label we
     already know, classify WHY the model got them wrong (translation? chain branch? ...).
     The label is given to Jev as a fact; Jev never decides whether two records match.
     Output is a report that tells you which feature to build next.

Never: send test records, use Jev scores as a model feature, or use Jev as a tie-breaker.
The rules forbid "APIs or services to ... resolve entities", the final model must be
MIT/Apache <= 8B, and reviewers must be able to reproduce your outputs after today.

Setup:  pip install requests ; export AI_GATEWAY_API_KEY=...
"""
import hashlib
import json
import os
import random
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import pandas as pd

URL = "https://ai-gateway.vercel.sh/v1/evaluate"
MODEL = "typesafe-ai/jev"


class _Retryable(Exception):
    def __init__(self, msg, wait=None):
        super().__init__(msg)
        self.wait = wait


# --------------------------------------------------------------------------------------
# Client: cached, resumable, order-safe, retrying, deadline-aware
# --------------------------------------------------------------------------------------
class Jev:
    """Every answer is appended to a JSONL cache immediately, so a crash or the promo ending
    mid-run loses nothing, and re-running skips everything already answered.

    deadline: ISO time (e.g. "2026-09-25T23:30:00+05:30"); no new requests start after it.
    post: injectable for offline testing (callable(payload) -> response dict).
    """

    def __init__(self, cache_path="jev_cache.jsonl", api_key=None, workers=8, max_retries=10,
                 timeout=60, deadline=None, zero_data_retention=False, post=None):
        self.key = api_key or os.environ.get("AI_GATEWAY_API_KEY", "")
        self.cache_path, self.workers, self.max_retries, self.timeout = cache_path, workers, max_retries, timeout
        self.deadline = datetime.fromisoformat(deadline) if deadline else None
        self.zdr = zero_data_retention
        self.post = post or self._http_post
        self.lock = threading.Lock()
        self.stats = Counter()
        self.cache = {}
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        self.cache[rec["h"]] = rec["answers"]
                    except Exception:
                        pass

    def _http_post(self, payload):
        import requests
        r = requests.post(URL, json=payload, timeout=self.timeout,
                          headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        if r.status_code in (408, 409, 425, 429, 500, 502, 503, 504):
            ra = r.headers.get("retry-after")
            raise _Retryable(f"HTTP {r.status_code}", float(ra) if ra and ra.replace(".", "").isdigit() else None)
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        return r.json()

    def _payload(self, state, questions):
        p = {"model": MODEL, "state": state, "questions": questions}
        if self.zdr:
            p["providerOptions"] = {"gateway": {"zeroDataRetention": True}}
        return p

    def healthcheck(self):
        """One cheap call. Falls back to no zero-data-retention if the provider rejects it."""
        q = {"ok": {"type": "boolean", "instructions": "Is the word 'road' a street type?"}}
        try:
            return self.post(self._payload("road", q))["answers"]
        except RuntimeError as e:
            if self.zdr:
                print(f"ZDR request failed ({e}); retrying without zeroDataRetention.")
                self.zdr = False
                return self.post(self._payload("road", q))["answers"]
            raise

    def ask(self, state, questions):
        payload = self._payload(state, questions)
        h = hashlib.sha1(json.dumps({"s": state, "q": questions}, sort_keys=True).encode()).hexdigest()
        if h in self.cache:
            self.stats["cached"] += 1
            return self.cache[h]
        if self.deadline and datetime.now(timezone.utc) > self.deadline.astimezone(timezone.utc):
            self.stats["skipped_deadline"] += 1
            return None
        for attempt in range(self.max_retries):
            try:
                resp = self.post(payload)
                break
            except _Retryable as e:
                self.stats["retries"] += 1
                time.sleep(e.wait or min(60, 2 ** attempt + random.random()))
            except Exception as e:  # network errors etc.
                self.stats["retries"] += 1
                if attempt == self.max_retries - 1:
                    self.stats["failed"] += 1
                    print(f"failed: {e}")
                    return None
                time.sleep(min(60, 2 ** attempt + random.random()))
        else:
            self.stats["failed"] += 1
            return None
        answers = resp.get("answers")
        cost = (resp.get("providerMetadata") or resp.get("provider_metadata") or {}).get("gateway", {}).get("cost")
        with self.lock:
            self.cache[h] = answers
            self.stats["calls"] += 1
            if cost:
                self.stats["cost_usd_x1e6"] += int(float(cost) * 1e6)
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"h": h, "state": state, "answers": answers}, ensure_ascii=False) + "\n")
        return answers

    def map_batched(self, requests_list, size=15, desc="jev", every=20):
        """Packs `size` items into one request: state = {"items": {"i0": state0, ...}} and every
        question is duplicated per item ("i0__q"), prefixed with "About items.i0:". Jev answers
        questions in parallel, so this cuts request count (and rate-limit pressure) ~size-fold.
        Returns per-item answers in the SAME order as requests_list."""
        packed, slices = [], []
        for s in range(0, len(requests_list), size):
            chunk = requests_list[s:s + size]
            state, qs = {"items": {}}, {}
            for k, (st, q) in enumerate(chunk):
                state["items"][f"i{k}"] = st
                for qn, spec in q.items():
                    spec = dict(spec)
                    spec["instructions"] = f"About items.i{k} only: " + spec["instructions"]
                    qs[f"i{k}__{qn}"] = spec
            packed.append((state, qs))
            slices.append(len(chunk))
        res = self.map(packed, desc=desc, every=every)
        out = []
        for ans, n in zip(res, slices):
            for k in range(n):
                if not ans:
                    out.append(None)
                    continue
                pre = f"i{k}__"
                out.append({key[len(pre):]: v for key, v in ans.items() if key.startswith(pre)} or None)
        return out

    def map(self, requests_list, desc="jev", every=500):
        """requests_list: [(state, questions), ...] -> answers list in the SAME order.
        (asyncio.as_completed in the attachment returns results in completion order and then
        pastes them next to the rows in input order -> labels land on the wrong pairs.)"""
        out = [None] * len(requests_list)
        t0 = time.time()
        with ThreadPoolExecutor(self.workers) as ex:
            futs = {ex.submit(self.ask, s, q): i for i, (s, q) in enumerate(requests_list)}
            for n, f in enumerate(as_completed(futs), 1):
                out[futs[f]] = f.result()
                if n % every == 0 or n == len(futs):
                    print(f"[{desc}] {n}/{len(futs)} {time.time() - t0:.0f}s {dict(self.stats)}")
        return out


# --------------------------------------------------------------------------------------
# A. Lexicon judge
# --------------------------------------------------------------------------------------
LEX_QUESTIONS = {
    "relation": {
        "type": "choice",
        "instructions": ("Token A and token B come from business names or street addresses in the US, India "
                         "or France (Indic words may be romanised). How are they related?"),
        "criteria": {
            "abbreviation": "one is a standard abbreviation or short form of the other, e.g. bd/boulevard, pvt/private, mfg/manufacturing",
            "spelling_variant": "the same word spelled or romanised differently, e.g. shree/sri, bengaluru/bangalore, colour/color",
            "translation": "the same meaning in different languages, e.g. boulangerie/bakery, bhandar/store",
            "different": "different words, or only loosely related, e.g. traders/enterprises, road/street, north/south",
        },
    },
    "safe_to_merge": {
        "type": "boolean",
        "instructions": ("If both tokens were replaced by one shared token before comparing two business records, "
                         "would that almost never make two different businesses look alike?"),
        "criteria": {"true": "safe: they are interchangeable in names/addresses",
                     "false": "unsafe: they can distinguish different businesses or places"},
    },
}


def _is_abbrev(short, long):
    if not short or short[0] != long[0] or len(short) >= len(long):
        return False
    it = iter(long)
    return all(c in it for c in short)


def abbreviation_candidates(vocab, min_count=20, max_short=5, per_short=4):
    """vocab: Counter of normalised tokens (one per country/field works best).
    For each frequent short token, propose the most frequent longer tokens it could abbreviate."""
    longs = defaultdict(list)
    for w, c in vocab.items():
        if c >= min_count and w.isalpha() and len(w) >= 3:
            longs[w[0]].append((c, w))
    for k in longs:
        longs[k].sort(reverse=True)
    pairs = []
    for s, c in vocab.items():
        if c < min_count or not s.isalpha() or not (2 <= len(s) <= max_short):
            continue
        cands = [w for _, w in longs.get(s[0], []) if _is_abbrev(s, w)][:per_short]
        pairs += [(s, w) for w in cands]
    return pairs


def phonetic_collision_candidates(vocab, key_fn, min_count=20, max_group=8):
    """Tokens that share a phonetic key but are spelled differently: shree/sri/shri (same word)
    but also patel/patil (different surnames). Jev decides which collisions are safe to merge.
    key_fn: e.g. lambda w: er_multilingual.phonetic_key(w, "indic")."""
    groups = defaultdict(list)
    for w, c in vocab.items():
        if c >= min_count and w.isalpha() and len(w) >= 3:
            groups[key_fn(w)].append((c, w))
    pairs = []
    for g in groups.values():
        g = [w for _, w in sorted(g, reverse=True)[:max_group]]
        pairs += [(g[i], g[j]) for i in range(len(g)) for j in range(i + 1, len(g))]
    return pairs


def cross_vocab_candidates(texts_a, texts_b, top=30, min_count=20):
    """Translation check without sending records: words over-represented in source A vs source B
    of the SAME country (e.g. France S1 vs France S3 names). If A says 'boulangerie' where B says
    'bakery', the pair shows up here. Needs er_multilingual.vocab_shift."""
    from er_multilingual import vocab_shift
    more_a, more_b = vocab_shift(texts_a, texts_b, top=top, min_count=min_count)
    return [(a, b) for a in more_a.index for b in more_b.index if a != b]


def lexicon_requests(pairs, field="business name or address"):
    return [({"token_a": a, "token_b": b, "field": field}, LEX_QUESTIONS) for a, b in pairs]


def lexicon_table(pairs, answers, p_safe=0.85, p_rel=0.8):
    rows = []
    for (a, b), ans in zip(pairs, answers):
        if not ans:
            continue
        rel = ans["relation"]
        probs = rel.get("probabilities", {})
        best = rel.get("choice")
        safe = ans["safe_to_merge"].get("probability", ans["safe_to_merge"].get("noul"))
        rows.append({"tok_a": a, "tok_b": b, "relation": best, "p_relation": probs.get(best, np.nan),
                     "p_different": probs.get("different", np.nan), "p_safe": safe,
                     "accept": best != "different" and probs.get(best, 0) >= p_rel and (safe or 0) >= p_safe})
    return pd.DataFrame(rows)


def build_lexicon(table, vocab, max_group=6):
    """Union-find over accepted pairs -> variant->canonical (most frequent token in the group).
    Groups larger than max_group are skipped (chaining a~b~c usually means a bad edge)."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in table[table.accept].itertuples():
        parent[find(r.tok_a)] = find(r.tok_b)
    groups = defaultdict(list)
    for t in list(parent):
        groups[find(t)].append(t)
    out, skipped = [], []
    for g in groups.values():
        if len(g) > max_group:
            skipped.append(sorted(g))
            continue
        canon = max(g, key=lambda t: (vocab.get(t, 0), len(t)))
        out += [(t, canon) for t in g if t != canon]
    return pd.DataFrame(out, columns=["variant", "canonical"]), skipped


def apply_lexicon(path, canon_dict):
    """Merge resources/lexicon.tsv into er_multilingual._CANON. Hand-written rules win on conflict,
    and chains are resolved (bd->boulevard, boulevard->blvd  =>  bd->blvd)."""
    lex = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    added = 0
    for v, c in zip(lex.variant, lex.canonical):
        if v not in canon_dict:
            canon_dict[v] = canon_dict.get(c, c)
            added += 1
    return added


# --------------------------------------------------------------------------------------
# B. Error taxonomy on TRAIN errors (labels known; Jev explains, never decides)
# --------------------------------------------------------------------------------------
FN_CAUSES = {
    "abbreviation_or_legal_form": "names differ by abbreviations or legal suffixes (Pvt Ltd, LLC, SARL)",
    "transliteration_or_spelling": "same words romanised or spelled differently, or written in another script",
    "translation": "same name or business type written in a different language",
    "typo": "small character-level typos",
    "trade_name_vs_legal_name": "one record uses a trade/DBA name, the other the legal name",
    "address_reformatted_or_partial": "address components missing, reordered or abbreviated",
    "landmark_address": "one address is described by a nearby landmark (near X, opposite Y)",
    "generic_or_missing_name": "name is missing, very short or very generic on one side",
    "looks_unrelated": "no visible reason these would be the same business (possible label noise)",
}
FP_CAUSES = {
    "same_chain_different_branch": "same brand or chain name at different locations",
    "different_business_same_address": "different businesses in the same building, mall or complex",
    "generic_or_common_name": "common or generic name shared by unrelated businesses",
    "same_person_or_family_name": "businesses named after the same person or family name",
    "parent_or_subsidiary": "related companies (parent, subsidiary, franchise owner)",
    "typo_level_near_duplicate": "they differ only like typos of each other (possible label noise)",
    "looks_unrelated": "they do not look alike at all",
}


def taxonomy_requests(df, kind):
    """df columns: s1_name, s1_addr, s23_name, s23_addr (raw text, optionally also normalised).
    kind: "missed_match" (label 1, low score or not retrieved) or "false_alarm" (label 0, high score)."""
    fact = ("These two records are KNOWN to be the SAME business, but our matcher missed the match."
            if kind == "missed_match" else
            "These two records are KNOWN to be DIFFERENT businesses, but our matcher wrongly linked them.")
    causes = FN_CAUSES if kind == "missed_match" else FP_CAUSES
    q = {"cause": {"type": "choice", "instructions": f"{fact} What is the main reason?", "criteria": causes}}
    return [({"fact": fact,
              "record_1": {"name": r.s1_name, "address": r.s1_addr},
              "record_2": {"name": r.s23_name, "address": r.s23_addr}}, q) for r in df.itertuples()]


def taxonomy_table(df, answers, kind):
    causes = list((FN_CAUSES if kind == "missed_match" else FP_CAUSES).keys())
    probs = [((a or {}).get("cause") or {}).get("probabilities", {}) for a in answers]
    P = pd.DataFrame([[p.get(c, np.nan) for c in causes] for p in probs], columns=causes, index=df.index)
    return pd.concat([df, P.add_prefix("p_")], axis=1)


def taxonomy_report(tab, by="country"):
    """Share of errors by cause (probability mass), per group. This is the prioritised to-do list."""
    pc = [c for c in tab.columns if c.startswith("p_")]
    rep = tab.groupby(by)[pc].mean().T * 100
    rep["all"] = tab[pc].mean() * 100
    return rep.round(1).sort_values("all", ascending=False)


def sample_hard(cand, score_col, label_col="y", n_each=1500, seed=0, by="country"):
    """Hard cases, stratified by country: missed matches (label 1, lowest scores) and false
    alarms (label 0, highest scores). Add blocking misses (GT pairs never retrieved) separately."""
    rng = np.random.default_rng(seed)
    out = []
    for g, d in cand.groupby(by):
        pos = d[d[label_col] == 1].nsmallest(max(n_each * 3, 1), score_col)
        neg = d[d[label_col] == 0].nlargest(max(n_each * 3, 1), score_col)
        for part, kind in ((pos, "missed_match"), (neg, "false_alarm")):
            take = part.iloc[rng.permutation(len(part))[:n_each]].assign(kind=kind)
            out.append(take)
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------------------
# RECIPE - paste into a notebook cell (train_s1/2/3, test_s1/2/3, gt_dict already loaded,
# country canonicalised). Nothing below touches test RECORDS: only word counts from test text.
# --------------------------------------------------------------------------------------
RECIPE = r'''
import pandas as pd, er_multilingual as M, jev_sprint as J
from collections import Counter

jev = J.Jev(cache_path="jev_cache.jsonl", workers=16, deadline="2026-09-25T23:00:00+05:30")
print(jev.healthcheck())

def vocab_of(df, col, kind):                      # Counter of normalised tokens
    return Counter(t for x in df[col].fillna("") for t in M.normalize_ml(x, kind).split())

pairs = set()
for name, dfs in {"train": [train_s1, train_s2, train_s3], "test": [test_s1, test_s2, test_s3]}.items():
    for ctry in pd.concat(dfs).country.unique():
        sub = pd.concat([d[d.country == ctry] for d in dfs])
        for col, kind in (("business_name", "name"), ("business_address", "addr")):
            v = vocab_of(sub, col, kind)
            pairs |= set(J.abbreviation_candidates(v, min_count=20))
            key = "french" if ctry == "FRANCE" else "indic"
            pairs |= set(J.phonetic_collision_candidates(v, lambda w: M.phonetic_key(w, key)))
    # translation check between sources of the same country (names)
    for ctry in dfs[0].country.unique():
        a = dfs[0][dfs[0].country == ctry].business_name.map(lambda x: M.normalize_ml(x, "name"))
        for other in dfs[1:]:
            b = other[other.country == ctry].business_name.map(lambda x: M.normalize_ml(x, "name"))
            pairs |= set(J.cross_vocab_candidates(a, b, top=30))

# variants that co-occur in TRUE train pairs (India/US)
s23 = pd.concat([train_s2, train_s3]).set_index("entity_id")
s1 = train_s1.set_index("entity_id")
true_pairs = [(s1.at[a, "business_name"], s23.at[b, "business_name"]) for a, ms in gt_dict.items() for b in ms if b in s23.index]
mined = M.mine_token_variants([(M.normalize_ml(x, "name"), M.normalize_ml(y, "name")) for x, y in true_pairs[:500_000]])
pairs |= set(map(tuple, mined.head(3000)[["tok_a", "tok_b"]].values))

pairs = sorted(pairs)
answers = jev.map(J.lexicon_requests(pairs), desc="lexicon")
tab = J.lexicon_table(pairs, answers); tab.to_csv("jev_lexicon_judgements.tsv", sep="\t", index=False)
vocab_all = Counter()            # frequencies for choosing canonical forms
for d in [train_s1, train_s2, train_s3, test_s1, test_s2, test_s3]:
    vocab_all += vocab_of(d, "business_name", "name")
lex, skipped = J.build_lexicon(tab, vocab_all)
lex.to_csv("resources/lexicon.tsv", sep="\t", index=False)   # static file shipped in src/
print(len(lex), "lexicon entries;", len(skipped), "groups skipped for manual review")

# later, in the pipeline (no API):  J.apply_lexicon("resources/lexicon.tsv", M._CANON)
'''
