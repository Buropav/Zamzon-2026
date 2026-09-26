"""
two_stage.py - candidate generation, two-stage GBDT matcher (LightGBM or XGBoost) and chunked, memory-safe inference.

Blocking: char 3-4gram TF-IDF per (country, region) group, forward views (each Source 1 -> its top-k Source 2/3
         records by name, core name, name+address, address) and reverse views (each Source 2/3 record -> its
         top-k Source 1 records). Reverse retrieval is cheap and finds matches a crowded forward list drops
         (Source 1 is deduplicated, so a Source 2/3 record has few look-alikes there).
Prefilter: a small GBDT on blocking similarities + fast string/number features drops near-certain
         non-matches (keeps ~12% of candidates and ~99.98% of true pairs) before the expensive features.
Stage 1: pair classifier on pairwise features (features.pair_features).
Stage 2: re-scores each pair with group context (groups.py): rank/margin of the stage-1 score inside
         the Source 1 group AND among all Source 1 entities competing for the same Source 2/3 record.
Selection: two thresholds (tau1 for each Source 1's best candidate, tau2 for the others) and a GLOBAL
         one-to-one constraint (each Source 2/3 record goes to at most one Source 1).

Inference runs in Source 1 chunks. Pass 1 caches each chunk's stage-2 inputs (float16) on disk;
the Source 2/3-side group features need every chunk's stage-1 scores, so they are computed once
globally; pass 2 applies stage 2 chunk by chunk; selection is global.
"""
# <package-only>
from .blocking import _topk_rows, union_candidates
from .features import pair_features
from .extra_feats import num_matrix, num_features
from .groups import s1_side_features, s23_side_features, stage2_matrix
from .metrics import macro_f05, select_matches, tune_policy
# </package-only>
import gc
import os
import shutil
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
from sklearn.feature_extraction.text import TfidfVectorizer

LGB_PARAMS = dict(n_estimators=1500, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, min_child_samples=40, random_state=42, n_jobs=-1, verbose=-1)


BLOCKERS = (("name", 30), ("core", 15), ("text", 50), ("addr", 20))  # forward: (view, top-k per Source 1)
REVERSE = (("text", 3), ("name", 2), ("addr", 2))                    # reverse: (view, top-k per Source 2/3)
KEYS = (("kb", 30, 150), ("kb2", 30, 150))  # exact keys (view, max Source 1, max Source 2/3 per key value):
# kb = house number + first 3 letters of a distinctive name word, kb2 = house number + compact name. They find
# copies whose address was cut to "No 6, Bengaluru" and whose name is damaged, which TF-IDF top-k loses in dense
# pools. Karnataka pool (69k S1 x 577k S2/S3): old blocking 0.9399 -> TF-IDF views 0.9654 -> + keys 0.9821.
# Measured on whole training regions (Oregon + Kerala, 61k Source 1): old 20/10/40 forward only -> recall 0.9854
# (53 cands/S1); + address view + reverse text/name/addr -> 0.9920-0.9932. The prefilter keeps the cost flat.
MAX_DF_FLOOR = int(os.environ.get("ER_MAX_DF_FLOOR", "0"))
USE_GPU = os.environ.get("ER_USE_GPU", "1") != "0"


def _gpu():
    """cupy + cupyx.scipy.sparse if a CUDA device is usable, else None."""
    if not USE_GPU:
        return None
    try:
        import cupy as cp
        import cupyx.scipy.sparse as cs
        if cp.cuda.runtime.getDeviceCount() < 1:
            return None
        cp.arange(2).sum()  # compiles a kernel: fails fast if CUDA headers/driver are unusable
        return cp, cs
    except Exception as e:  # pragma: no cover
        print(f"  (GPU unavailable, blocking on CPU: {type(e).__name__}: {e})")
        return None


def _view(p, name):
    if name == "name":
        return p["norm_name"].to_numpy(object)
    if name == "core":
        return p["core"].to_numpy(object)
    if name == "addr":
        return p["norm_addr"].to_numpy(object)
    return (p["norm_name"] + " " + p["norm_addr"]).to_numpy(object)


def _topk_cpu(Q, DT, k, min_sim, batch=2000):
    rs, cs_, vs = [], [], []
    for b in range(0, Q.shape[0], batch):
        r, c, v = _topk_rows((Q[b:b + batch] @ DT).tocsr(), k, min_sim)
        rs.append(r.astype(np.int64) + b), cs_.append(c), vs.append(v)
    if not rs:
        return np.array([], np.int64), np.array([], np.int64), np.array([], np.float32)
    return np.concatenate(rs), np.concatenate(cs_), np.concatenate(vs).astype(np.float32)


def _topk_gpu(Q, DT, k, min_sim, gpu, batch=1024):
    """Same result as _topk_cpu (up to ties): sparse Q @ DT on the GPU, then per-row top-k by sorting
    (row asc, score desc). Halves the batch on GPU out-of-memory."""
    cp, cs = gpu
    DTg = cs.csr_matrix(DT)
    rs, cs_, vs = [], [], []
    b = 0
    try:
        while b < Q.shape[0]:
            try:
                Qb = cs.csr_matrix(Q[b:b + batch])
                S = (Qb @ DTg).tocoo()
                keep = S.data >= min_sim
                r, c, v = S.row[keep], S.col[keep], S.data[keep]
                del S, Qb
                order = cp.lexsort(cp.stack([-v, r.astype(cp.float32)]))
                r, c, v = r[order], c[order], v[order]
                rank = cp.arange(r.size) - cp.searchsorted(r, r, side="left")
                sel = rank < k
                rs.append(cp.asnumpy(r[sel]).astype(np.int64) + b)
                cs_.append(cp.asnumpy(c[sel]).astype(np.int64))
                vs.append(cp.asnumpy(v[sel]).astype(np.float32))
                del r, c, v, order, rank, sel
                b += batch
            except cp.cuda.memory.OutOfMemoryError:
                cp.get_default_memory_pool().free_all_blocks()
                if batch <= 16:
                    raise
                batch //= 2
    finally:
        del DTg
        cp.get_default_memory_pool().free_all_blocks()
    if not rs:
        return np.array([], np.int64), np.array([], np.int64), np.array([], np.float32)
    return np.concatenate(rs), np.concatenate(cs_), np.concatenate(vs)


def _gpu_devices(gpu):
    """Devices to use: all visible GPUs (override with ER_GPU_DEVICES="0,1")."""
    env = os.environ.get("ER_GPU_DEVICES")
    if env:
        return [int(x) for x in env.split(",") if x.strip()]
    return list(range(gpu[0].cuda.runtime.getDeviceCount()))


def _topk_multi_gpu(Q, DT, k, min_sim, gpu):
    """Split the Source 1 rows across GPUs (one thread per device; each gets its own copy of DT).
    CuPy work on one device does not block Python threads driving another, so devices run in parallel.
    Rows keep their order, so results are identical to a single-GPU run."""
    import threading
    cp = gpu[0]
    devs = _gpu_devices(gpu)
    if len(devs) < 2 or Q.shape[0] < 2 * 1024:
        with cp.cuda.Device(devs[0] if devs else 0):
            return _topk_gpu(Q, DT, k, min_sim, gpu)
    bounds = np.linspace(0, Q.shape[0], len(devs) + 1).astype(int)
    out, errs = [None] * len(devs), []

    def run(j):
        try:
            with cp.cuda.Device(devs[j]):
                r, c, v = _topk_gpu(Q[bounds[j]:bounds[j + 1]], DT, k, min_sim, gpu)
            out[j] = (r + bounds[j], c, v)
        except BaseException as e:  # re-raised below -> CPU fallback for this view
            errs.append(e)

    th = [threading.Thread(target=run, args=(j,)) for j in range(len(devs))]
    for t in th:
        t.start()
    for t in th:
        t.join()
    if errs:
        raise errs[0]
    return tuple(np.concatenate([o[i] for o in out]) for i in range(3))


def _topk_any(Q, DT, k, min_sim, gpu):
    r = c = v = None
    if gpu is not None:
        try:
            r, c, v = _topk_multi_gpu(Q, DT, k, min_sim, gpu)
        except Exception as e:  # any GPU failure -> CPU for this view
            print(f"  (GPU blocking failed: {type(e).__name__}; using CPU)")
    if r is None:
        r, c, v = _topk_cpu(Q, DT, k, min_sim)
    return r, c, v


def _key_values(p, idx, kind):
    """(row position in idx, key string) for exact-key blocking."""
    nums = p["addr_nums"].to_numpy()[idx]
    rows_i, rows_k = [], []
    if kind == "kb":
        names = p["distinct"].to_numpy()[idx]
        for i, (ns, nm) in enumerate(zip(nums, names)):
            toks = [t[:3] for t in str(nm).split() if not t.isdigit()][:2]
            for n in ns.split()[:2]:
                n = n.lstrip("0") or "0"
                for t in toks:
                    rows_i.append(i)
                    rows_k.append(f"{n}|{t}")
    else:
        names = p["compact"].to_numpy()[idx]
        for i, (ns, nm) in enumerate(zip(nums, names)):
            if not nm:
                continue
            for n in ns.split()[:3]:
                rows_i.append(i)
                rows_k.append(f"{n.lstrip('0') or '0'}|{nm}")
    return pd.DataFrame({"i": np.asarray(rows_i, np.int64), "k": rows_k})


def _key_block(qp, dp, q_idx, d_idx, kind, max_q, max_d):
    """Pairs sharing a key value, skipping values held by more than max_q Source 1 / max_d Source 2/3 records."""
    A, B = _key_values(qp, q_idx, kind), _key_values(dp, d_idx, kind)
    if A.empty or B.empty:
        return np.array([], np.int64), np.array([], np.int64), np.array([], np.float32)
    ca, cb = A["k"].value_counts(), B["k"].value_counts()
    ok = ca.index[ca <= max_q].intersection(cb.index[cb <= max_d])
    m = A[A["k"].isin(ok)].merge(B[B["k"].isin(ok)], on="k")[["i_x", "i_y"]].drop_duplicates()
    m = m.sort_values("i_x", kind="stable")
    r = m["i_x"].to_numpy(np.int64)
    return r, d_idx[m["i_y"].to_numpy(np.int64)], np.ones(len(r), np.float32)


def _block_group(qp, dp, q_idx, d_idx, gpu, min_sim, max_df, rev_rows=None):
    """Top-k for every view of one blocking group -> {label: (r local to q_idx, di global, sim)}, r sorted.
    Forward labels are the view name, reverse labels 'r' + view name. rev_rows: bool mask over d_idx of the
    Source 2/3 records that run reverse retrieval here (None = all, empty = none)."""
    views = {}
    if max_df < 1:  # relative -> absolute, with a floor so small regional pools keep informative n-grams
        max_df = max(int(np.ceil(max_df * len(d_idx))), MAX_DF_FLOOR)
    fwd, rev = dict(BLOCKERS), dict(REVERSE)
    rsel = np.arange(len(d_idx)) if rev_rows is None else np.flatnonzero(rev_rows)
    if len(rsel) == 0:
        rev = {}
    empty = (np.array([], np.int64), np.array([], np.int64), np.array([], np.float32))
    for name in list(dict.fromkeys(list(fwd) + list(rev))):
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=2, max_df=max_df,
                              sublinear_tf=True, dtype=np.float32)
        try:
            D = vec.fit_transform(_view(dp.iloc[d_idx], name)).tocsr()
        except ValueError:  # tiny pool: no n-gram survives min_df/max_df
            for lab in ([name] if name in fwd else []) + (["r" + name] if name in rev else []):
                views[lab] = empty
            continue
        Q = vec.transform(_view(qp.iloc[q_idx], name)).tocsr()
        if name in fwd:
            r, c, v = _topk_any(Q, D.T.tocsr(), min(fwd[name], len(d_idx)), min_sim, gpu)
            views[name] = (r, d_idx[c], v)
        if name in rev:
            rd, cq, v = _topk_any(D[rsel], Q.T.tocsr(), min(rev[name], len(q_idx)), min_sim, gpu)
            o = np.argsort(cq, kind="stable")
            views["r" + name] = (cq[o].astype(np.int64), d_idx[rsel[rd[o]]], v[o])
        del vec, D, Q
    for kind, max_q, max_d in KEYS:
        views[kind] = _key_block(qp, dp, q_idx, d_idx, kind, max_q, max_d)
    return views


def _reverse_unknown(qp, dp, gpu, min_sim, max_df, log=None):
    """Source 2/3 records whose region is unknown (mostly empty addresses) join every regional pool of their
    country for the forward views, but run reverse retrieval ONCE, against all Source 1 of the country (their
    owner can be anywhere in it). -> {label: (qi global, di global, sim)}."""
    qc, dc = qp["country"].to_numpy(object), dp["country"].to_numpy(object)
    dr = dp["region"].to_numpy(object) if "region" in dp else np.full(len(dp), "", object)
    out = {"r" + n: [] for n, _ in REVERSE}
    for g in pd.unique(qc):
        if g == "":
            continue
        q_idx = np.flatnonzero(qc == g)
        d_idx = np.flatnonzero((dc == g) & (dr == ""))
        if len(q_idx) == 0 or len(d_idx) == 0:
            continue
        mdf = max(int(np.ceil(max_df * len(d_idx))), MAX_DF_FLOOR) if max_df < 1 else max_df
        for name, k in REVERSE:
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=2, max_df=mdf,
                                  sublinear_tf=True, dtype=np.float32)
            try:
                D = vec.fit_transform(_view(dp.iloc[d_idx], name)).tocsr()
            except ValueError:
                continue
            Q = vec.transform(_view(qp.iloc[q_idx], name)).tocsr()
            rd, cq, v = _topk_any(D, Q.T.tocsr(), min(k, len(q_idx)), min_sim, gpu)
            out["r" + name].append((q_idx[cq], d_idx[rd], v))
            del vec, D, Q
        if log:
            log(f"  reverse retrieval of {len(d_idx):,} unknown-region S23 against {len(q_idx):,} S1 ({g})")
    res = {}
    for lab, parts in out.items():
        if parts:
            qi = np.concatenate([p[0] for p in parts]).astype(np.int64)
            o = np.argsort(qi, kind="stable")
            res[lab] = (qi[o], np.concatenate([p[1] for p in parts])[o].astype(np.int64),
                        np.concatenate([p[2] for p in parts])[o].astype(np.float32))
    return res


def _groups(qp, dp):
    """Blocking groups: (label, q_idx, d_idx). Within a country, a Source 1 record with a known region is
    searched in that region plus Source 2/3 records whose region is unknown; unknown-region Source 1 records
    are searched in the whole country. Countries without region rules (e.g. a new country) are one group."""
    qc, dc = qp["country"].to_numpy(object), dp["country"].to_numpy(object)
    qr = qp["region"].to_numpy(object) if "region" in qp else np.full(len(qp), "", object)
    dr = dp["region"].to_numpy(object) if "region" in dp else np.full(len(dp), "", object)
    for g in pd.unique(qc):
        q_c = qc == g
        d_c = np.ones(len(dp), bool) if g == "" else (dc == g) | (dc == "")
        for r in sorted(pd.unique(qr[q_c]), key=lambda x: (x == "", x)):
            q_idx = np.flatnonzero(q_c & (qr == r))
            d_idx = np.flatnonzero(d_c if r == "" else d_c & ((dr == r) | (dr == "")))
            if len(q_idx) and len(d_idx):
                yield f"{g}/{r or '*'}", q_idx, d_idx


def iter_candidate_chunks(qp, dp, chunk_size=250_000, min_sim=0.015, max_df=0.01, log=None):
    """Char 3-4gram TF-IDF blocking, forward + reverse views (BLOCKERS, REVERSE), per (country, region) group (see _groups),
    on the GPU when available. Groups are packed into chunks of about chunk_size Source 1 records.
    Yields (qidx, cands) with cands.qi LOCAL to qp.iloc[qidx] and cands.di global."""
    gpu = _gpu()
    buf_q, buf_c, n_buf = [], [], 0
    t0 = time.time()
    unk = _reverse_unknown(qp, dp, gpu, min_sim, max_df, log=log)
    d_region = dp["region"].to_numpy(object) if "region" in dp else np.full(len(dp), "", object)
    in_slice = np.zeros(len(qp), bool)

    def flush():
        qidx = np.concatenate(buf_q)
        off = np.cumsum([0] + [len(x) for x in buf_q[:-1]])
        cands = pd.concat([c.assign(qi=c["qi"].to_numpy() + o) for c, o in zip(buf_c, off)], ignore_index=True)
        return qidx, cands

    for label, q_all, d_all in _groups(qp, dp):
        region = label.split("/", 1)[1]
        # reverse retrieval inside a regional pool only for that region's records (unknown-region ones: _reverse_unknown)
        rev_rows = (d_region[d_all] == region) if region != "*" else np.zeros(len(d_all), bool)
        views = _block_group(qp, dp, q_all, d_all, gpu, min_sim, max_df, rev_rows=rev_rows)
        if log and (len(q_all) >= 20_000 or label.endswith("*")):
            log(f"  blocking {label}: {len(q_all):,} S1 x {len(d_all):,} S23 ({time.time() - t0:.0f}s, "
                f"{f'{len(_gpu_devices(gpu))} GPU' if gpu else 'CPU'})")
        for s in range(0, len(q_all), chunk_size):
            e = min(s + chunk_size, len(q_all))
            blocks = {}
            for name, (r, d, v) in views.items():
                lo, hi = np.searchsorted(r, s), np.searchsorted(r, e)  # r is sorted (row-major)
                blocks[name] = pd.DataFrame({"qi": r[lo:hi] - s, "di": d[lo:hi], "sim": v[lo:hi]})
            q_slice = q_all[s:e]
            in_slice[q_slice] = True
            local = np.full(len(qp), -1, np.int64)
            local[q_slice] = np.arange(len(q_slice))
            for name, (gq, gd, gv) in unk.items():  # country-level reverse retrieval for these Source 1
                m = in_slice[gq]
                u = pd.DataFrame({"qi": local[gq[m]], "di": gd[m], "sim": gv[m]})
                blocks[name] = pd.concat([blocks[name], u], ignore_index=True) if name in blocks else u
            in_slice[q_slice] = False
            buf_q.append(q_all[s:e])
            buf_c.append(_union_fast(blocks))
            n_buf += e - s
            if n_buf >= chunk_size:
                yield flush()
                buf_q, buf_c, n_buf = [], [], 0
        del views
        gc.collect()
    if buf_q:
        yield flush()


BLK_COLS = [f"blk_{n}" for n, _ in BLOCKERS] + [f"blk_r{n}" for n, _ in REVERSE] + [f"blk_{n}" for n, _, _ in KEYS]


def _union_fast(blocks):
    """{label: DataFrame(qi, di, sim)} -> one row per (qi, di) with blk_<label> similarity columns (NaN when
    that view did not retrieve the pair) and n_blockers. Same content as blocking.union_candidates, faster."""
    parts = [(lab, b) for lab, b in blocks.items() if len(b)]
    if not parts:
        return pd.DataFrame({"qi": np.array([], np.int64), "di": np.array([], np.int64),
                             **{c: np.array([], np.float32) for c in BLK_COLS}, "n_blockers": np.array([], np.int64)})
    qi = np.concatenate([b["qi"].to_numpy(np.int64) for _, b in parts])
    di = np.concatenate([b["di"].to_numpy(np.int64) for _, b in parts])
    key = qi * (int(di.max()) + 1) + di
    uk, inv = np.unique(key, return_inverse=True)
    first = np.full(len(uk), -1, np.int64)
    first[inv[::-1]] = np.arange(len(inv))[::-1]
    out = pd.DataFrame({"qi": qi[first], "di": di[first]})
    off = 0
    cols = {c: np.full(len(uk), np.nan, np.float32) for c in BLK_COLS}
    for lab, b in parts:
        n = len(b)
        c = f"blk_{lab}"
        if c not in cols:
            cols[c] = np.full(len(uk), np.nan, np.float32)
        np.fmax.at(cols[c], inv[off:off + n], b["sim"].to_numpy(np.float32))
        off += n
    for c, v in cols.items():
        out[c] = v
    out["n_blockers"] = np.isfinite(out[list(cols)].to_numpy()).sum(axis=1)
    return out


def block_candidates(qp, dp, log=None):
    """All candidates at once (training): cands.qi indexes qp, cands.di indexes dp."""
    out = []
    for qidx, c in iter_candidate_chunks(qp, dp, chunk_size=len(qp) or 1, log=log):
        c["qi"] = qidx[c["qi"].to_numpy()]
        out.append(c)
    if not out:
        return pd.DataFrame(columns=["qi", "di", "n_blockers"])
    c = pd.concat(out, ignore_index=True)
    blk = [x for x in c.columns if x.startswith("blk_")]
    c[blk] = c[blk].astype(np.float32)
    return c


CTX_REGION = "~ctx"


def add_context(s1p, s23p, cands, ctx_p, relevant_di=None, log=print):
    """Appends prepared context Source 1 records (see sampling.context_owners), searched only among the
    unknown-region Source 2/3 records (their own regions are not in the sample); only their pairs with the
    relevant Source 2/3 records (relevant_di) are kept. -> (s1p, cands, core mask)."""
    if len(ctx_p) == 0:
        return s1p, cands, np.ones(len(s1p), bool)
    ctx_p = ctx_p.copy()
    ctx_p["region"] = CTX_REGION
    cc = block_candidates(ctx_p, s23p, log=log)
    if relevant_di is not None:
        cc = cc[np.isin(cc["di"].to_numpy(), relevant_di)].reset_index(drop=True)
    cc["qi"] = cc["qi"].to_numpy() + len(s1p)
    core = np.r_[np.ones(len(s1p), bool), np.zeros(len(ctx_p), bool)]
    log(f"  context: {len(ctx_p):,} owners of unknown-region S23 records, {len(cc):,} candidate pairs")
    return pd.concat([s1p, ctx_p], ignore_index=True), pd.concat([cands, cc], ignore_index=True), core


# ------------------------------------------------------------------ prefilter
PF_THRESHOLD_CAP = float(os.environ.get("ER_PF_CAP", "0.001"))
PF_THRESHOLD_FLOOR = float(os.environ.get("ER_PF_FLOOR", "0.0001"))  # pairs below 1e-4 are never selected anyway


def cheap_features(cands, qp, dp, workers=-1):
    """Fast features for the prefilter: blocking similarities, 5 rapidfuzz scores, address-number alignment.
    cands.qi -> rows of qp, cands.di -> rows of dp."""
    qi, di = cands["qi"].to_numpy(), cands["di"].to_numpy()
    F = pd.DataFrame({c: (cands[c].to_numpy(np.float32) if c in cands else np.full(len(cands), np.nan, np.float32))
                      for c in BLK_COLS + ["n_blockers"]})
    sc = lambda a, b, f: cpdist(a, b, scorer=f, workers=workers, dtype=np.float32) / 100.0  # noqa: E731
    an, bn = qp["norm_name"].to_numpy()[qi], dp["norm_name"].to_numpy()[di]
    F["c_name_ratio"] = sc(an, bn, fuzz.ratio)
    F["c_name_tset"] = sc(an, bn, fuzz.token_set_ratio)
    F["c_core_tset"] = sc(qp["core"].to_numpy()[qi], dp["core"].to_numpy()[di], fuzz.token_set_ratio)
    F["c_addr_tset"] = sc(qp["norm_addr"].to_numpy()[qi], dp["norm_addr"].to_numpy()[di], fuzz.token_set_ratio)
    F["c_cmp_partial"] = sc(qp["compact"].to_numpy()[qi], dp["compact"].to_numpy()[di], fuzz.partial_ratio)
    uq, iq = np.unique(qi, return_inverse=True)
    ud, id_ = np.unique(di, return_inverse=True)
    N = num_features(num_matrix(qp["addr_nums"].to_numpy()[uq]), num_matrix(dp["addr_nums"].to_numpy()[ud]), iq, id_,
                     parallel=True)
    for c in ("n_a", "n_b", "n_exact", "n_shift_up", "n_a_left", "n_b_left", "h_delta", "h_eq"):
        F[f"c_{c}"] = N[c].to_numpy()
    return F


def train_prefilter(C, y, part_of_row, keep_recall=0.9998, log=print):
    """Small GBDT on cheap features. Fit on part 0; threshold = score below which at most (1 - keep_recall) of
    the true pairs of parts 1-2 fall, clipped to [PF_THRESHOLD_FLOOR, PF_THRESHOLD_CAP]."""
    fit = part_of_row == 0
    m = _Model(resolve_backend(), n_estimators=300, learning_rate=0.1, max_depth=6, num_leaves=63).fit(C[fit], y[fit])
    p = m.predict_proba(C)[:, 1]
    ho = ((part_of_row == 1) | (part_of_row == 2)) & (y == 1)
    th = float(np.quantile(p[ho], 1 - keep_recall)) if ho.any() else 0.0
    th = min(max(th, PF_THRESHOLD_FLOOR), PF_THRESHOLD_CAP)
    keep = p >= th
    log(f"  prefilter: threshold {th:.5f}, keeps {keep.mean():.3f} of candidates, "
        f"{(keep & ho).sum() / max(ho.sum(), 1):.5f} of held-out true pairs")
    return {"model": m, "threshold": th}, keep


def apply_prefilter(cands, qp, dp, pf, step=4_000_000):
    """-> boolean keep mask for cands."""
    if pf is None or len(cands) == 0:
        return np.ones(len(cands), bool)
    return np.concatenate([pf["model"].predict_proba(cheap_features(cands.iloc[i:i + step], qp, dp))[:, 1] >= pf["threshold"]
                           for i in range(0, len(cands), step)])


# ------------------------------------------------------------------ model backends
BACKEND = os.environ.get("ER_BACKEND", "auto")  # "auto" (xgb on GPU, else lgbm), "lgbm" or "xgb"
XGB_PARAMS = dict(n_estimators=3000, learning_rate=0.05, max_depth=8, min_child_weight=5, subsample=0.8,
                  colsample_bytree=0.8, tree_method="hist", max_bin=256, eval_metric="logloss",
                  random_state=42, n_jobs=-1)


def _xgb_device():
    try:
        import cupy as cp
        return "cuda" if USE_GPU and cp.cuda.runtime.getDeviceCount() > 0 else "cpu"
    except Exception:
        return "cpu"


class _Model:
    """Uniform wrapper: fit / predict_proba / best_iteration / feature_importance / cols."""

    def __init__(self, backend, n_estimators=None, **over):
        self.backend = backend
        if backend == "xgb":
            import xgboost as xgb
            p = {**XGB_PARAMS, **{k: v for k, v in over.items() if k in XGB_PARAMS}}
            if n_estimators:
                p["n_estimators"] = n_estimators
            self.m = xgb.XGBClassifier(device=_xgb_device(), **p)
        else:
            p = {**LGB_PARAMS, **over}
            if n_estimators:
                p["n_estimators"] = n_estimators
            self.m = lgb.LGBMClassifier(**p)

    def fit(self, X, y, Xv=None, yv=None):
        self.cols = list(X.columns)
        if self.backend == "xgb":
            if Xv is not None:
                self.m.set_params(early_stopping_rounds=50)
                self.m.fit(X, y, eval_set=[(Xv, yv)], verbose=False)
            else:
                self.m.fit(X, y, verbose=False)
            self.best_iteration_ = (self.m.best_iteration + 1) if Xv is not None else self.m.n_estimators
        else:
            if Xv is not None:
                self.m.fit(X, y, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(50, verbose=False)])
            else:
                self.m.fit(X, y)
            self.best_iteration_ = self.m.best_iteration_ or self.m.n_estimators
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X[self.cols])

    def importance(self):
        if self.backend == "xgb":
            g = self.m.get_booster().get_score(importance_type="total_gain")
            return pd.Series({c: g.get(c, 0.0) for c in self.cols}).sort_values(ascending=False)
        return pd.Series(self.m.booster_.feature_importance("gain"), index=self.cols).sort_values(ascending=False)


def resolve_backend(backend=None):
    """A/B on whole training regions: XGBoost 0.9786 vs LightGBM 0.9783 held-out F0.5 (equal within noise);
    XGBoost is much faster on a GPU, LightGBM on CPU-only machines."""
    b = backend or BACKEND
    if b == "auto":
        try:
            import xgboost  # noqa: F401
            b = "xgb" if _xgb_device() == "cuda" else "lgbm"
        except Exception:
            b = "lgbm"
    return b


def _fit(X, y, Xv=None, yv=None, n_estimators=None, backend=None, **over):
    return _Model(resolve_backend(backend), n_estimators=n_estimators, **over).fit(X, y, Xv, yv)


def train_two_stage(cands, X, y, n_true, s23p, seed=42, folds=5, log=print, core=None):
    """cands: qi, di (+ blocker cols). X: stage-1 features. y: labels. n_true: true match count per S1.
    Split by Source 1: 60% fit, 20% early stopping + threshold tuning, 20% held-out report.
    core: optional bool per Source 1; False = context record (scored and used as a competitor in the group
    features and the one-to-one assignment, never trained on or evaluated)."""
    qi = cands["qi"].to_numpy()
    part = split_parts(len(n_true), seed, core)
    pq = part[qi]
    tr, va = pq == 0, pq == 1

    t0 = time.time()
    clf1 = _fit(X[tr], y[tr], X[va], y[va])
    p1 = clf1.predict_proba(X)[:, 1].astype(np.float32)
    n_iter = clf1.best_iteration_
    fold = (qi.astype(np.int64) * 2654435761) % folds
    for k in range(folds):  # out-of-fold stage-1 scores on the fit part (stage 2 must not see leaked scores)
        a, b = tr & (fold != k), tr & (fold == k)
        p1[b] = _fit(X[a], y[a], n_estimators=n_iter).predict_proba(X[b])[:, 1]
    log(f"  stage 1 [{resolve_backend()}]: {n_iter} trees, OOF done ({time.time() - t0:.0f}s)")

    c = cands[["qi", "di"]].copy()
    c["p1"] = p1
    X2 = stage2_matrix(X, p1, s1_side_features(c, "p1", s23p), s23_side_features(c, "p1"))
    clf2 = _fit(X2[tr], y[tr], X2[va], y[va], num_leaves=31)
    score = clf2.predict_proba(X2)[:, 1]
    log(f"  stage 2: {clf2.best_iteration_} trees ({time.time() - t0:.0f}s)")

    c["y"], c["score"] = y, score
    val_ids, test_ids = np.flatnonzero(part == 1), np.flatnonzero(part == 2)
    # one-to-one over ALL Source 1 (every part and context competes for a Source 2/3 record, as at test time),
    # then thresholds tuned / reported on the validation / held-out Source 1
    g = c.loc[c.groupby("di")["score"].idxmax()]
    gq = part[g["qi"].to_numpy()]
    f_val, t1, t2 = tune_policy(g[gq == 1], "score", n_true, val_ids, one_to_one=False)
    sel = select_matches(g[gq == 2], "score", t1, t2, one_to_one=False)
    f_test = macro_f05(sel, n_true, test_ids)
    # stage 1 alone, for reference
    c["s1score"] = clf1.predict_proba(X)[:, 1]
    g1 = c.loc[c.groupby("di")["s1score"].idxmax()]
    g1q = part[g1["qi"].to_numpy()]
    f1v, a1, a2 = tune_policy(g1[g1q == 1], "s1score", n_true, val_ids, one_to_one=False)
    f1t = macro_f05(select_matches(g1[g1q == 2], "s1score", a1, a2, one_to_one=False), n_true, test_ids)
    log(f"  held-out Macro F0.5: stage 1 = {f1t:.4f}, two-stage = {f_test:.4f} (tau1={t1:.2f}, tau2={t2:.2f})")
    imp = clf2.importance()
    return {"clf1": clf1, "clf2": clf2, "t1": t1, "t2": t2, "f05_val": f_val, "f05_test": f_test,
            "f05_test_stage1": f1t, "stage2_cols": list(X2.columns), "stage2_importance": imp,
            "part": part, "pairs": c}


def split_parts(n_s1, seed=42, core=None):
    """Source 1 split used everywhere: 0 = fit (60%), 1 = early stopping + tuning (20%), 2 = held-out (20%),
    3 = context record (core == False)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_s1)
    part = np.empty(n_s1, np.int8)
    part[perm[: int(0.6 * n_s1)]] = 0
    part[perm[int(0.6 * n_s1): int(0.8 * n_s1)]] = 1
    part[perm[int(0.8 * n_s1):]] = 2
    if core is not None:
        part[~np.asarray(core, bool)] = 3
    return part


def fit_pipeline(cands_raw, s1p, s23p, y_raw, n_true, seed=42, log=print, core=None):
    """Training entry point: prefilter (fit on the 60% part) -> pair features on kept candidates -> two stages.
    core: optional bool per Source 1 (False = context record, see train_two_stage).
    Returns the model dict (with the prefilter) plus recall figures."""
    t0 = time.time()
    part = split_parts(len(s1p), seed, core)
    step = 4_000_000  # bounded peak memory: the string temporaries of the fast features are built per slice
    C = pd.concat([cheap_features(cands_raw.iloc[i:i + step], s1p, s23p) for i in range(0, len(cands_raw), step)],
                  ignore_index=True)
    pf, keep = train_prefilter(C, y_raw, part[cands_raw["qi"].to_numpy()], log=log)
    del C
    cands = cands_raw[keep].reset_index(drop=True).reindex(columns=["qi", "di"] + BLK_COLS + ["n_blockers"])
    y = np.asarray(y_raw)[keep].astype(np.int32)
    is_core = part < 3
    cq_raw, cq = is_core[cands_raw["qi"].to_numpy()], is_core[cands["qi"].to_numpy()]
    n_core = max(n_true[is_core].sum(), 1)
    rec_raw, rec = np.asarray(y_raw)[cq_raw].sum() / n_core, y[cq].sum() / n_core
    log(f"  recall: blocking {rec_raw:.4f}, after prefilter {rec:.4f}; {len(cands):,} pairs ({time.time() - t0:.0f}s)")
    X = pair_features(cands, s1p, s23p, workers=-1)
    log(f"  pair features {X.shape} ({time.time() - t0:.0f}s)")
    model = train_two_stage(cands, X, y, n_true, s23p, seed=seed, log=log, core=core)
    model.update(prefilter=pf, recall_blocking=float(rec_raw), recall_prefilter=float(rec), n_train_pairs=len(cands))
    return model


def predict_chunked(test_s1p, test_s23p, model, chunk_size=250_000, cache_dir="stage2_cache", log=print,
                    workers=-1):
    """Returns (cands, selected): DataFrames with global qi (into test_s1p) and di (into test_s23p)."""
    os.makedirs(cache_dir, exist_ok=True)
    clf1, clf2 = model["clf1"], model["clf2"]
    cols = model["stage2_cols"]
    g23_cols = [c for c in cols if c.endswith("_s23") and c.startswith("g_")]
    chunks = []
    t0 = time.time()
    for ci, (qidx, cc) in enumerate(iter_candidate_chunks(test_s1p, test_s23p, chunk_size=chunk_size, log=log)):
        if len(cc) == 0:
            continue
        cs = test_s1p.iloc[qidx].reset_index(drop=True)
        n_raw = len(cc)
        cc = cc[apply_prefilter(cc, cs, test_s23p, model.get("prefilter"))].reset_index(drop=True)
        if len(cc) == 0:
            continue
        cc = cc.reindex(columns=["qi", "di"] + BLK_COLS + ["n_blockers"])
        X = pair_features(cc, cs, test_s23p, workers=workers)
        p1 = clf1.predict_proba(X)[:, 1].astype(np.float32)
        cc = cc[["qi", "di"]].copy()
        cc["p1"] = p1
        G1 = s1_side_features(cc, "p1", test_s23p)
        part = stage2_matrix(X, p1, G1, pd.DataFrame(index=cc.index))
        part = part.reindex(columns=[c for c in cols if c not in g23_cols])
        path = os.path.join(cache_dir, f"chunk{ci}.npy")
        np.save(path, part.to_numpy(np.float32))
        chunks.append((path, list(part.columns), qidx[cc["qi"].to_numpy()].astype(np.int32),
                       cc["di"].to_numpy().astype(np.int32), p1))
        log(f"  [pass 1] chunk {ci + 1}: {len(qidx):,} S1 ({cs['country'].iat[0]}), {n_raw:,} blocked -> "
            f"{len(cc):,} after prefilter ({time.time() - t0:.0f}s)")
        del cs, cc, X, G1, part
        gc.collect()

    if not chunks:
        empty = pd.DataFrame({"qi": np.array([], np.int32), "di": np.array([], np.int32)})
        return empty, empty
    qi = np.concatenate([c[2] for c in chunks])
    di = np.concatenate([c[3] for c in chunks])
    p1 = np.concatenate([c[4] for c in chunks])
    G23 = s23_side_features(pd.DataFrame({"di": di, "p1": p1}), "p1")
    score = np.empty(len(qi), np.float32)
    off = 0
    for path, pcols, cqi, _, _ in chunks:
        n = len(cqi)
        M = pd.DataFrame(np.load(path), columns=pcols)
        for c in g23_cols:
            M[c] = G23[c].to_numpy()[off:off + n]
        score[off:off + n] = clf2.predict_proba(M[cols])[:, 1]
        off += n
        os.remove(path)
    shutil.rmtree(cache_dir, ignore_errors=True)
    log(f"  [pass 2] stage 2 scored {len(qi):,} pairs ({time.time() - t0:.0f}s)")
    cands = pd.DataFrame({"qi": qi, "di": di, "score": score})
    selected = select_matches(cands, "score", model["t1"], model["t2"], one_to_one=True)
    return cands, selected


def id_lists(n_s1, qi, di, s23_ids):
    """One comma-joined, de-duplicated, sorted ID string per Source 1 row (empty when none)."""
    out = np.full(n_s1, "", dtype=object)
    if len(qi) == 0:
        return out
    ids = np.asarray(s23_ids, dtype=object)[di]
    order = np.argsort(qi, kind="stable")
    q_sorted, ids_sorted = qi[order], ids[order]
    starts = np.flatnonzero(np.r_[True, q_sorted[1:] != q_sorted[:-1]])
    ends = np.r_[starts[1:], len(q_sorted)]
    for s, e in zip(starts, ends):
        out[q_sorted[s]] = ",".join(sorted(set(ids_sorted[s:e])))
    return out
