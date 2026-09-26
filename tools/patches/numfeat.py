"""Experiment 'numfeat': the old pipeline's address-number alignment features (extra_feats.py) as extra stage-1
pair features in the friend's pipeline. Adds ber/numfeat.py and one call in features.pair_features.
Numbers come from the friend's normalised `addr` (digit runs in order, first 8, last 12 digits).
Duplicates of existing friend features are left out: h_eq (= house_eq), h_lev (= house_lev).
Threads only (numba prange), no processes.   usage: python numfeat.py <ber dir>"""
import os, sys
d = sys.argv[1]

MODULE = r'''"""Address-number alignment between the two records of a pair (ported from the old pipeline's extra_feats.py).

The data generator makes near-copies of S1 records that are NOT matches by shifting the house number a little
(2048 -> 2049); true copies keep the number or damage it like a typo (digit dropped / inserted / replaced,
truncation). Numbers of both addresses are aligned: exact, small shift up/down, one-digit typo, prefix/suffix
truncation, medium shifts, leftovers; plus the signed smallest shift and the first-number difference."""
import re

import numpy as np

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover - numba is preinstalled on Kaggle; without it this is slow but correct
    print("WARNING: numba not installed -> address-number features run in pure Python (slow)")
    prange = range

    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f

MAXN = 8
_DIGITS = re.compile(r"\d+")
NUM_COLS = ["n_a", "n_b", "n_exact", "n_shift_up", "n_shift_down", "n_typo", "n_prefix", "n_a_left", "n_b_left",
            "h_delta", "h_lev", "h_eq", "h_prefix", "h_in_b", "n_minshift", "n_shift_up_mid", "n_shift_down_mid"]
KEEP = ["n_a", "n_b", "n_exact", "n_shift_up", "n_shift_down", "n_typo", "n_prefix", "n_a_left", "n_b_left",
        "h_delta", "h_prefix", "h_in_b", "n_minshift", "n_shift_up_mid", "n_shift_down_mid"]


def num_matrix(addrs):
    """addr strings -> int64 matrix (rows x MAXN) of the first MAXN digit runs (last 12 digits), -1 = empty."""
    M = np.full((len(addrs), MAXN), -1, np.int64)
    for i, s in enumerate(addrs):
        if s:
            for k, t in enumerate(_DIGITS.findall(s)[:MAXN]):
                M[i, k] = int(t[-12:])
    return M


@njit(cache=True)
def _digits(x, out):
    n = 1
    y = x
    while y >= 10:
        y //= 10
        n += 1
    for i in range(n - 1, -1, -1):
        out[i] = x % 10
        x //= 10
    return n


@njit(cache=True)
def _lev_digits(a, b):
    da = np.empty(13, np.int64)
    db = np.empty(13, np.int64)
    na = _digits(a, da)
    nb = _digits(b, db)
    prev = np.arange(nb + 1)
    cur = np.empty(nb + 1, np.int64)
    for i in range(1, na + 1):
        cur[0] = i
        for j in range(1, nb + 1):
            v = prev[j - 1] + (0 if da[i - 1] == db[j - 1] else 1)
            if prev[j] + 1 < v:
                v = prev[j] + 1
            if cur[j - 1] + 1 < v:
                v = cur[j - 1] + 1
            cur[j] = v
        for j in range(nb + 1):
            prev[j] = cur[j]
    return prev[nb]


@njit(cache=True)
def _is_prefix(a, b):
    """Digits of the shorter number are a prefix or a suffix of the longer one (truncation)."""
    da = np.empty(13, np.int64)
    db = np.empty(13, np.int64)
    na = _digits(a, da)
    nb = _digits(b, db)
    if na == nb:
        return False
    if na > nb:
        da, db, na, nb = db, da, nb, na
    pre = True
    for i in range(na):
        if da[i] != db[i]:
            pre = False
            break
    suf = True
    for i in range(na):
        if da[i] != db[nb - na + i]:
            suf = False
            break
    return pre or suf


@njit(parallel=True, cache=True)
def _num_feats(A, B, qi, di, out):
    for r in prange(len(qi)):
        a = A[qi[r]]
        b = B[di[r]]
        na = 0
        nb = 0
        for k in range(MAXN):
            if a[k] >= 0:
                na += 1
            if b[k] >= 0:
                nb += 1
        usedb = np.zeros(MAXN, np.bool_)
        useda = np.zeros(MAXN, np.bool_)
        exact = 0
        for i in range(na):
            for j in range(nb):
                if not usedb[j] and a[i] == b[j]:
                    usedb[j] = True
                    useda[i] = True
                    exact += 1
                    break
        up = 0
        down = 0
        typo = 0
        pref = 0
        up_mid = 0
        down_mid = 0
        minshift = 1000000
        for i in range(na):
            if useda[i]:
                continue
            for j in range(nb):
                if usedb[j]:
                    continue
                d = b[j] - a[i]
                kind = 0
                if 1 <= d <= 20:
                    kind = 1
                elif -20 <= d <= -1:
                    kind = 2
                elif _is_prefix(a[i], b[j]):
                    kind = 4
                elif _lev_digits(a[i], b[j]) == 1:
                    kind = 3
                elif 21 <= d <= 100:
                    kind = 5
                elif -100 <= d <= -21:
                    kind = 6
                if kind > 0:
                    usedb[j] = True
                    useda[i] = True
                    if abs(d) < abs(minshift):
                        minshift = d
                    if kind == 1:
                        up += 1
                    elif kind == 2:
                        down += 1
                    elif kind == 3:
                        typo += 1
                    elif kind == 4:
                        pref += 1
                    elif kind == 5:
                        up_mid += 1
                    else:
                        down_mid += 1
                    break
        a_left = 0
        for i in range(na):
            if not useda[i]:
                a_left += 1
        b_left = 0
        for j in range(nb):
            if not usedb[j]:
                b_left += 1
        out[r, 0] = na
        out[r, 1] = nb
        out[r, 2] = exact
        out[r, 3] = up
        out[r, 4] = down
        out[r, 5] = typo
        out[r, 6] = pref
        out[r, 7] = a_left
        out[r, 8] = b_left
        if na > 0 and nb > 0:
            d = b[0] - a[0]
            out[r, 9] = max(-1000, min(1000, d))
            out[r, 10] = _lev_digits(a[0], b[0])
            out[r, 11] = 1.0 if a[0] == b[0] else 0.0
            out[r, 12] = 1.0 if _is_prefix(a[0], b[0]) else 0.0
            f = 0.0
            for j in range(nb):
                if b[j] == a[0]:
                    f = 1.0
            out[r, 13] = f
        else:
            for c in range(9, 14):
                out[r, c] = np.nan
        out[r, 14] = minshift if minshift != 1000000 else np.nan
        out[r, 15] = up_mid
        out[r, 16] = down_mid


def num_feature_dict(s1_rows, s23_rows, s1_addr, s23_addr):
    """-> {name: float32 array} for the pairs (s1_rows[i], s23_rows[i]); number matrices built once per unique row."""
    uq, iq = np.unique(s1_rows, return_inverse=True)
    ud, id_ = np.unique(s23_rows, return_inverse=True)
    A = num_matrix(s1_addr.gather(uq).to_list())
    B = num_matrix(s23_addr.gather(ud).to_list())
    out = np.zeros((len(iq), len(NUM_COLS)), np.float32)
    _num_feats(A, B, iq.astype(np.int64), id_.astype(np.int64), out)
    return {c: np.ascontiguousarray(out[:, NUM_COLS.index(c)]) for c in KEEP}
'''

def edit(fn, old, new):
    p = os.path.join(d, fn); s = open(p).read()
    assert s.count(old) == 1, (fn, old[:60], s.count(old))
    open(p, "w").write(s.replace(old, new))

assert not os.path.exists(os.path.join(d, "numfeat.py")), "numfeat already applied"
open(os.path.join(d, "numfeat.py"), "w").write(MODULE)
edit("features.py", "from rapidfuzz.distance import JaroWinkler, Levenshtein\n",
     "from rapidfuzz.distance import JaroWinkler, Levenshtein\n\nfrom .numfeat import num_feature_dict\n")
edit("features.py", "    # combined evidence\n",
     "    # [numfeat] address-number alignment (exact / small shift / typo / truncation / leftovers)\n"
     "    f.update(num_feature_dict(pairs[\"s1_row\"].to_numpy(), pairs[\"s23_row\"].to_numpy(), s1[\"addr\"], s23[\"addr\"]))\n"
     "    # combined evidence\n")
print("numfeat applied to", d)
