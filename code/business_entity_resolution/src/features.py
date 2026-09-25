import os
import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist
from sklearn.feature_extraction.text import TfidfVectorizer

# <package-only>  (the notebook generator removes this block; there these names are notebook globals)
from .text import core_name, _initials, normalize_text, canon_country
from . import er_multilingual as _ML
from .er_multilingual import ml_pair_features
from .er_lexicon import load_default, country_aware, country_key, distinct_tokens, name_conflict
_LEX = load_default()
prepare_ml = country_aware(_ML.prepare_ml, vars(_ML), _LEX)
# </package-only>

_PC = re.compile(r"\d{5,6}")
_NUM = re.compile(r"\d+")

def _first(rx, s):
    m = rx.search(str(s))
    return m.group(0) if m else ""

def _rowwise_cos(A, B, ia, ib, chunk=500_000):
    out = np.empty(len(ia), dtype=np.float32)
    for s in range(0, len(ia), chunk):
        out[s:s + chunk] = np.asarray(A[ia[s:s + chunk]].multiply(B[ib[s:s + chunk]]).sum(axis=1)).ravel()
    return out

_TFIDF_CACHE = {}


def _pool_tfidf(s23, col):
    key = (id(s23), len(s23), col, str(s23["entity_id"].iat[0]) if len(s23) else "")
    if key not in _TFIDF_CACHE:
        if len(_TFIDF_CACHE) > 8:
            _TFIDF_CACHE.clear()
        v = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", sublinear_tf=True, dtype=np.float32)
        B = v.fit_transform(s23[col].to_numpy())
        _TFIDF_CACHE[key] = (v, B.tocsr())
    return _TFIDF_CACHE[key]


def _idfcos(cand, s1, s23):
    qi, di = cand.qi.to_numpy(), cand.di.to_numpy()
    out = {}
    for col, key in (("norm_name", "name"), ("norm_addr", "addr")):
        v, B = _pool_tfidf(s23, col)
        out[f"{key}_idfcos"] = _rowwise_cos(v.transform(s1[col]), B, qi, di)
    return out


def _pf_task(args):
    c, a, b = args
    return _pair_features(c, a, b, workers=1, idfcos=False)


def pair_features(cand, s1, s23, workers=-1, n_jobs=None):
    """Pairwise features. Large inputs are split across CPU cores; each worker receives only its pairs
    and the Source 1 / Source 2/3 rows they reference (re-indexed), so memory stays bounded. The IDF
    cosines need the full Source 2/3 pool and are computed in the parent."""
    n = n_jobs or N_JOBS
    if n <= 1 or len(cand) < 200_000:
        return _pair_features(cand, s1, s23, workers)
    order = list(_pair_features(cand.iloc[:200], s1, s23, workers).columns)
    keep1 = [c for c in s1.columns if c not in _DROP_FOR_WORKERS]
    keep2 = [c for c in s23.columns if c not in _DROP_FOR_WORKERS]
    n_parts = max(n * 3, int(np.ceil(len(cand) / MAX_PAIRS_PER_TASK)))  # small tasks bound worker memory
    parts = np.array_split(np.arange(len(cand)), n_parts)

    def make(i):
        c = cand.iloc[parts[i]].copy()
        qu, qinv = np.unique(c["qi"].to_numpy(), return_inverse=True)
        du, dinv = np.unique(c["di"].to_numpy(), return_inverse=True)
        c["qi"], c["di"] = qinv, dinv
        return c, s1.iloc[qu][keep1].reset_index(drop=True), s23.iloc[du][keep2].reset_index(drop=True)

    tasks = _Lazy(len(parts), make)  # built one at a time as workers ask for them
    F = pd.concat(_fork_map(_pf_task, tasks, n))
    del tasks
    for key, arr in _idfcos(cand, s1, s23).items():
        F[key] = arr
    return F[order]


def _jacc(x, y):
    a, b = set(x.split()), set(y.split())
    return len(a & b) / len(a | b) if (a or b) else np.nan


N_JOBS = int(os.environ.get("ER_N_JOBS", "0")) or (os.cpu_count() or 1)
MAX_PAIRS_PER_TASK = 100_000


class _Lazy:
    """Sequence whose items are built on demand (keeps only in-flight tasks in memory)."""

    def __init__(self, n, make):
        self.n, self.make = n, make

    def __len__(self):
        return self.n

    def __iter__(self):
        return (self.make(i) for i in range(self.n))
_DROP_FOR_WORKERS = ("business_name", "business_address", "ml_addr")


def _fork_map(fn, items, n):
    """Ordered map over n forked worker processes. `fn` is inherited through fork (works for notebook-
    defined functions); each task is PICKLED to the worker through a bounded queue, so workers only touch
    their own private copy and never the parent's objects (reading an inherited Python object writes its
    reference count, which would copy that memory page into every worker). Results come back pickled.
    Raises instead of hanging if a worker fails or is killed (e.g. out of memory)."""
    import gc as _gc
    import multiprocessing as mp
    import queue as _queue
    import threading
    import traceback
    ctx = mp.get_context("fork")
    tasks, results = ctx.Queue(maxsize=max(2, n)), ctx.Queue()

    def work():
        while True:
            job = tasks.get()
            if job is None:
                return
            i, payload = job
            del job
            try:
                results.put((i, True, fn(payload)))
            except BaseException:
                results.put((i, False, traceback.format_exc()))
                return
            del payload

    _gc.collect()
    _gc.freeze()
    procs = [ctx.Process(target=work, daemon=True) for _ in range(min(n, len(items)))]
    for p in procs:
        p.start()
    _gc.unfreeze()
    stop = threading.Event()

    def feed():
        for i, it in enumerate(items):
            while not stop.is_set():
                try:
                    tasks.put((i, it), timeout=1)
                    break
                except _queue.Full:
                    continue
        for _ in procs:
            while not stop.is_set():
                try:
                    tasks.put(None, timeout=1)
                    break
                except _queue.Full:
                    continue

    feeder = threading.Thread(target=feed, daemon=True)
    feeder.start()
    try:
        out, done = [None] * len(items), 0
        while done < len(items):
            try:
                i, ok, val = results.get(timeout=5)
            except _queue.Empty:
                dead = [p for p in procs if not p.is_alive() and p.exitcode not in (0, None)]
                if dead:
                    raise RuntimeError(f"worker died (exit code {dead[0].exitcode}; -9 usually means out of memory)")
                continue
            if not ok:
                raise RuntimeError(f"worker failed:\n{val}")
            out[i] = val
            done += 1
        return out
    finally:
        stop.set()
        for p in procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        feeder.join(timeout=5)


def _prepare_rows(df, drop=()):
    """All per-row work of prepare_side (everything except the table-level core frequency)."""
    df = df.copy()
    ml_cols = prepare_ml(df, name_col="business_name", addr_col="business_address")
    for col in ml_cols.columns:
        df[col] = ml_cols[col]
    df["norm_name"] = df["ml_name"]
    df["norm_addr"] = df["ml_addr_core"]
    df["country"] = df["country"].map(canon_country) if "country" in df.columns else ""
    df["core"] = df["norm_name"].map(core_name)
    df["pc"] = df["ml_pc"]
    df["house"] = df["ml_house"]
    df["nums"] = df["norm_addr"].map(lambda s: frozenset(_NUM.findall(str(s))))
    df["ckey"] = df["country"].map(country_key)
    df["distinct"] = [" ".join(distinct_tokens(n, _LEX, c)) for n, c in zip(df["norm_name"], df["ckey"])]
    _share_objects(df)
    return df.drop(columns=list(drop), errors="ignore")


def _share_objects(df):
    """Same values, less memory (~550 bytes/row on 10M rows): identical number-sets become one shared
    object (97% of name number-sets are empty), and core / distinct / DBA parts equal to the normalised
    name reuse that string instead of holding a copy."""
    for col in ("nums", "ml_name_nums"):
        if col in df:
            cache = {}
            df[col] = [cache.setdefault(v, v) for v in df[col]]
    name = df["norm_name"].to_numpy(object)
    for col in ("core", "distinct"):
        df[col] = [n if v == n else v for v, n in zip(df[col].to_numpy(object), name)]
    if "ml_dba" in df:
        df["ml_dba"] = [(n,) if len(t) == 1 and t[0] == n else t for t, n in zip(df["ml_dba"], name)]


def _prepare_rows_task(args):
    return _prepare_rows(*args)


def prepare_side(df, drop=(), n_jobs=None):
    """Precomputes side-level metadata, multilingual representations, and chain frequencies.
    Runs on all CPU cores for large tables. `drop` removes raw columns afterwards (saves memory)."""
    n = n_jobs or N_JOBS
    if n > 1 and len(df) >= 40_000:
        parts = np.array_split(np.arange(len(df)), n * 4)
        out = pd.concat(_fork_map(_prepare_rows_task, _Lazy(len(parts), lambda i: (df.iloc[parts[i]], tuple(drop))), n))
        for col in ("nums", "ml_name_nums"):  # share identical sets across worker parts too
            if col in out:
                cache = {}
                out[col] = [cache.setdefault(v, v) for v in out[col]]
    else:
        out = _prepare_rows(df, drop)
    out["core_freq"] = out.groupby(["country", "core"])["core"].transform("size").astype(np.float32)
    return out


def _pair_features(cand, s1, s23, workers=-1, idfcos=True):
    """Computes unified 51-dimensional pairwise features across name, address, postcodes, and multilingual representations."""
    qi, di = cand.qi.to_numpy(), cand.di.to_numpy()
    F = pd.DataFrame(index=cand.index)
    
    blk_cols = [c for c in cand.columns if c.startswith("blk_")]
    if "n_blockers" in cand:
        blk_cols.append("n_blockers")
    for c in blk_cols:
        F[c] = cand[c].to_numpy(np.float32)

    an, bn = s1.norm_name.to_numpy()[qi], s23.norm_name.to_numpy()[di]
    ac, bc = s1.core.to_numpy()[qi], s23.core.to_numpy()[di]
    aa, ba = s1.norm_addr.to_numpy()[qi], s23.norm_addr.to_numpy()[di]

    scorers = {
        "tset": fuzz.token_set_ratio,
        "tsort": fuzz.token_sort_ratio,
        "part": fuzz.partial_ratio,
        "ratio": fuzz.ratio,
        "wr": fuzz.WRatio
    }
    for nm, sc in scorers.items():
        F[f"name_{nm}"] = cpdist(an, bn, scorer=sc, workers=workers, dtype=np.float32) / 100.0
        F[f"core_{nm}"] = cpdist(ac, bc, scorer=sc, workers=workers, dtype=np.float32) / 100.0
        
    F["name_jw"] = cpdist(an, bn, scorer=JaroWinkler.normalized_similarity, workers=workers, dtype=np.float32)
    F["core_exact"] = (ac == bc).astype(np.float32)
    
    for nm in ("tset", "part", "ratio"):
        F[f"addr_{nm}"] = cpdist(aa, ba, scorer=scorers[nm], workers=workers, dtype=np.float32) / 100.0

    # Rare token IDF-weighted word cosine (vectoriser fitted once per S2/S3 pool and cached)
    if idfcos:
        for key, arr in _idfcos(cand, s1, s23).items():
            F[key] = arr

    pa, pb = s1.pc.to_numpy()[qi], s23.pc.to_numpy()[di]
    both = (pa != "") & (pb != "")
    F["pc_both"] = both.astype(np.float32)
    F["pc_eq"] = np.where(both, (pa == pb).astype(np.float32), np.nan)
    
    ha, hb = s1.house.to_numpy()[qi], s23.house.to_numpy()[di]
    F["house_eq"] = np.where((ha != "") & (hb != ""), (ha == hb).astype(np.float32), np.nan)
    
    na, nb = s1.nums.to_numpy()[qi], s23.nums.to_numpy()[di]
    F["num_jacc"] = np.array([len(x & y) / len(x | y) if (x or y) else np.nan for x, y in zip(na, nb)], np.float32)

    ia = np.array([_initials(x) for x in ac], dtype=object)
    ib = np.array([_initials(x) for x in bc], dtype=object)
    F["acronym"] = (((ia != "") & (ia == np.char.replace(bc.astype(str), " ", ""))) |
                    ((ib != "") & (ib == np.char.replace(ac.astype(str), " ", "")))).astype(np.float32)

    F["core_freq_s1"] = np.log1p(s1.core_freq.to_numpy()[qi])
    F["core_freq_s23"] = np.log1p(s23.core_freq.to_numpy()[di])
    
    la, lb = np.char.str_len(an.astype(str)), np.char.str_len(bn.astype(str))
    F["name_len_ratio"] = (np.minimum(la, lb) / np.maximum(np.maximum(la, lb), 1)).astype(np.float32)
    F["addr_len_a"] = np.char.str_len(aa.astype(str)).astype(np.float32)
    F["addr_len_b"] = np.char.str_len(ba.astype(str)).astype(np.float32)
    F["is_s3"] = np.char.startswith(s23.entity_id.to_numpy()[di].astype(str), "S3-").astype(np.float32)

    # Distinctive-word name similarity (legal forms, titles, generic business words removed) and
    # look-alike conflict (the names differ by a known pair of different names: patel / patil)
    xa, xb = s1["distinct"].to_numpy()[qi], s23["distinct"].to_numpy()[di]
    F["distinct_tset"] = cpdist(xa, xb, scorer=fuzz.token_set_ratio, workers=workers, dtype=np.float32) / 100.0
    F["distinct_jacc"] = np.array([_jacc(x, y) for x, y in zip(xa, xb)], np.float32)
    ck = s1["ckey"].to_numpy()[qi]
    F["name_conflict"] = np.array([name_conflict(x.split(), y.split(), _LEX, c) for x, y, c in zip(an, bn, ck)],
                                  np.float32)

    # Multilingual structured signals (abbreviations, landmarks, DBAs, non-Latin flags)
    F_ml = ml_pair_features(qi, di, s1, s23)
    for col in F_ml.columns:
        if col not in F.columns:
            F[col] = F_ml[col].to_numpy(np.float32)

    return F


def _side_stats(key, score):
    order = np.lexsort((-score, key))
    k_sorted, s_sorted = key[order], score[order]
    new = np.r_[True, k_sorted[1:] != k_sorted[:-1]]
    gid = np.cumsum(new) - 1
    starts = np.flatnonzero(new)
    sizes = np.diff(np.r_[starts, len(key)])
    rank = np.arange(len(key)) - starts[gid] + 1
    top1 = s_sorted[starts]
    top2 = np.where(sizes > 1, s_sorted[np.minimum(starts + 1, len(key) - 1)], np.nan)
    best_other = np.where(rank == 1, top2[gid], top1[gid])
    out_rank, out_size, out_margin = (np.empty(len(key)) for _ in range(3))
    out_rank[order], out_size[order] = rank, sizes[gid]
    out_margin[order] = s_sorted - best_other
    return out_rank, out_size, out_margin

def add_group_features(df, score_col="p1"):
    s = df[score_col].to_numpy(np.float64)
    for side, key in (("s1", df.qi.to_numpy()), ("s23", df.di.to_numpy())):
        r, n, m = _side_stats(key, s)
        df[f"{score_col}_rank_{side}"] = r
        df[f"{score_col}_n_{side}"] = n
        df[f"{score_col}_margin_{side}"] = m
    return df
