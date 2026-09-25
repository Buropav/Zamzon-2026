"""
two_stage.py - candidate generation, two-stage GBDT matcher (LightGBM or XGBoost) and chunked, memory-safe inference.

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
from sklearn.feature_extraction.text import TfidfVectorizer

LGB_PARAMS = dict(n_estimators=1500, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, min_child_samples=40, random_state=42, n_jobs=-1, verbose=-1)


BLOCKERS = (("name", 25), ("core", 15), ("text", 20))  # (view, top-k per Source 1)
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


def iter_candidate_chunks(qp, dp, chunk_size=250_000, min_sim=0.015, max_df=0.01, log=None):
    """Country-partitioned char 3-4gram TF-IDF blocking, 3 views, top-k each. Per country and view the
    vectoriser is fitted ONCE on the Source 2/3 side and all Source 1 records of that country are scored
    (on the GPU when available). Yields (qidx, cands) per Source 1 chunk, with cands.qi LOCAL to
    qp.iloc[qidx] and cands.di global."""
    gpu = _gpu()
    qc, dc = qp["country"].to_numpy(object), dp["country"].to_numpy(object)
    for g in pd.unique(qc):
        q_all = np.flatnonzero(qc == g)
        d_all = np.arange(len(dp)) if g == "" else np.flatnonzero((dc == g) | (dc == ""))
        if len(q_all) == 0 or len(d_all) == 0:
            continue
        t0 = time.time()
        views = {}
        for name, k in BLOCKERS:
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=2, max_df=max_df,
                                  sublinear_tf=True, dtype=np.float32)
            DT = vec.fit_transform(_view(dp.iloc[d_all], name)).T.tocsr()
            Q = vec.transform(_view(qp.iloc[q_all], name)).tocsr()
            kk = min(k, len(d_all))
            r, c, v = None, None, None
            if gpu is not None:
                try:
                    r, c, v = _topk_multi_gpu(Q, DT, kk, min_sim, gpu)
                except Exception as e:  # any GPU failure -> CPU for this view
                    print(f"  (GPU blocking failed for {g}/{name}: {type(e).__name__}; using CPU)")
            if r is None:
                r, c, v = _topk_cpu(Q, DT, kk, min_sim)
            views[name] = (r, d_all[c], v)
            del vec, DT, Q
            gc.collect()
        if log:
            log(f"  blocking {g}: {len(q_all):,} S1 x {len(d_all):,} S23 in {time.time() - t0:.0f}s "
                f"({f'{len(_gpu_devices(gpu))} GPU' if gpu else 'CPU'})")
        for s in range(0, len(q_all), chunk_size):
            e = min(s + chunk_size, len(q_all))
            blocks = {}
            for name, (r, d, v) in views.items():
                lo, hi = np.searchsorted(r, s), np.searchsorted(r, e)  # r is sorted (row-major)
                blocks[name] = pd.DataFrame({"qi": r[lo:hi] - s, "di": d[lo:hi], "sim": v[lo:hi]})
            yield q_all[s:e], union_candidates(blocks)
        del views
        gc.collect()


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


# ------------------------------------------------------------------ model backends
BACKEND = os.environ.get("ER_BACKEND", "auto")  # "auto" (xgb on GPU, else lgbm), "lgbm" or "xgb"
XGB_PARAMS = dict(n_estimators=1500, learning_rate=0.05, max_depth=8, min_child_weight=5, subsample=0.8,
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


def train_two_stage(cands, X, y, n_true, s23p, seed=42, folds=5, log=print):
    """cands: qi, di (+ blocker cols). X: stage-1 features. y: labels. n_true: true match count per S1.
    Split by Source 1: 60% fit, 20% early stopping + threshold tuning, 20% held-out report."""
    qi = cands["qi"].to_numpy()
    n_s1 = len(n_true)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_s1)
    part = np.empty(n_s1, np.int8)
    part[perm[: int(0.6 * n_s1)]] = 0
    part[perm[int(0.6 * n_s1): int(0.8 * n_s1)]] = 1
    part[perm[int(0.8 * n_s1):]] = 2
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
    f_val, t1, t2 = tune_policy(c[va], "score", n_true, val_ids, one_to_one=True)
    sel = select_matches(c[pq == 2], "score", t1, t2, one_to_one=True)
    f_test = macro_f05(sel, n_true, test_ids)
    # stage 1 alone, for reference
    c["s1score"] = clf1.predict_proba(X)[:, 1]
    f1v, a1, a2 = tune_policy(c[va], "s1score", n_true, val_ids, one_to_one=True)
    f1t = macro_f05(select_matches(c[pq == 2], "s1score", a1, a2, one_to_one=True), n_true, test_ids)
    log(f"  held-out Macro F0.5: stage 1 = {f1t:.4f}, two-stage = {f_test:.4f} (tau1={t1:.2f}, tau2={t2:.2f})")
    imp = clf2.importance()
    return {"clf1": clf1, "clf2": clf2, "t1": t1, "t2": t2, "f05_val": f_val, "f05_test": f_test,
            "f05_test_stage1": f1t, "stage2_cols": list(X2.columns), "stage2_importance": imp,
            "part": part, "pairs": c}


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
        X = pair_features(cc, cs, test_s23p, workers=workers)
        p1 = clf1.predict_proba(X)[:, 1].astype(np.float32)
        cc = cc[["qi", "di"]].copy()
        cc["p1"] = p1
        G1 = s1_side_features(cc, "p1", test_s23p)
        part = stage2_matrix(X, p1, G1, pd.DataFrame(index=cc.index))
        part = part.reindex(columns=[c for c in cols if c not in g23_cols])
        path = os.path.join(cache_dir, f"chunk{ci}.npy")
        np.save(path, part.to_numpy(np.float16))
        chunks.append((path, list(part.columns), qidx[cc["qi"].to_numpy()].astype(np.int32),
                       cc["di"].to_numpy().astype(np.int32), p1))
        log(f"  [pass 1] chunk {ci + 1}: {len(qidx):,} S1 ({cs['country'].iat[0]}), {len(cc):,} candidates "
            f"({time.time() - t0:.0f}s)")
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
        M = pd.DataFrame(np.load(path).astype(np.float32), columns=pcols)
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
