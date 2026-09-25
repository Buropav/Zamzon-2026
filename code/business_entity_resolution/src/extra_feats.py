"""
extra_feats.py - pair features aimed at how the data was generated (measured on training pairs):

* Address numbers. Source 2/3 contain "decoys": near-copies of a Source 1 record that are NOT matches, whose
  house number is shifted by a small amount (2048 -> 2049, 4311 -> 4316, 32/2587/119 -> 32/2587/128).
  True copies keep the number or damage it like a typo (digit dropped / inserted / replaced, leading zeros,
  1/2 or -B suffixes, ranges 1537-1539). US pairs with the same house number are 99.3% matches; pairs with
  the number shifted up by 1..13 only 3.4%. Numbers of both addresses are aligned: exact, small shift up,
  small shift down, one-digit typo, prefix/suffix truncation, leftovers.
* Compact names: domain / handle style names (almalindustries.com, @avilaliberty, #bellmodern) compared
  with the name without spaces and legal words.
* Token differences: words only on one side, classified as typo of a word on the other side, a real word
  (known Source 1 vocabulary: a substituted word, "Island" for "Vista") or an unknown word (generated brand).
"""
import re

import numpy as np
import pandas as pd
from numba import njit, prange
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler as _JW
from rapidfuzz.distance import Levenshtein as _Lev
from rapidfuzz.process import cpdist

MAXN = 8
_DIGITS = re.compile(r"\d+")


def addr_numbers(norm_addr):
    """'2048-b huckleberry ave 12' -> '2048 12' (digit runs in order, as a compact string)."""
    return " ".join(_DIGITS.findall(norm_addr or "")[:MAXN])


def num_matrix(num_strings):
    M = np.full((len(num_strings), MAXN), -1, np.int64)
    for i, s in enumerate(num_strings):
        if s:
            for k, t in enumerate(s.split()[:MAXN]):
                M[i, k] = int(t[-12:])
    return M


@njit
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


@njit
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


@njit
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


def _num_feats_impl(A, B, qi, di, out):
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
                    else:
                        pref += 1
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


_num_feats = njit(_num_feats_impl)                    # serial: used inside forked feature workers
_num_feats_par = njit(parallel=True)(_num_feats_impl)  # all cores: used in the parent (prefilter features)

NUM_COLS = ["n_a", "n_b", "n_exact", "n_shift_up", "n_shift_down", "n_typo", "n_prefix", "n_a_left", "n_b_left",
            "h_delta", "h_lev", "h_eq", "h_prefix", "h_in_b", "n_minshift"]


def num_features(NA, NB, qi, di, parallel=False):
    out = np.zeros((len(qi), len(NUM_COLS)), np.float32)
    (_num_feats_par if parallel else _num_feats)(NA, NB, np.asarray(qi, np.int64), np.asarray(di, np.int64), out)
    return pd.DataFrame(out, columns=NUM_COLS)


# compile once in the parent process (forked workers inherit the compiled code)
num_features(num_matrix(["1 2"]), num_matrix(["3"]), [0], [0])

_LEGAL = set("pvt private ltd limited llc llp inc incorporated corp corporation co company plc the and of "
             "sarl sas sasu sa eurl snc sci ets cie com www dba".split())


def compact(norm_name):
    """Name without spaces and legal / web words: 'almal industries pvt ltd' -> 'almalindustries'."""
    return "".join(t for t in str(norm_name).split() if t not in _LEGAL) or str(norm_name).replace(" ", "")


CMP_COLS = ["cmp_ratio", "cmp_partial", "cmp_contain", "cmp_len_ratio"]


def compact_features(ca, cb, workers=1):
    F = pd.DataFrame(index=range(len(ca)))
    F["cmp_ratio"] = cpdist(ca, cb, scorer=fuzz.ratio, workers=workers, dtype=np.float32) / 100
    F["cmp_partial"] = cpdist(ca, cb, scorer=fuzz.partial_ratio, workers=workers, dtype=np.float32) / 100
    F["cmp_contain"] = np.fromiter(((len(x) >= 4 and x in y) or (len(y) >= 4 and y in x) for x, y in zip(ca, cb)),
                                   np.float32, len(ca))
    la = np.fromiter((len(x) for x in ca), np.float32, len(ca))
    lb = np.fromiter((len(x) for x in cb), np.float32, len(cb))
    F["cmp_len_ratio"] = np.minimum(la, lb) / np.maximum(np.maximum(la, lb), 1)
    return F


def build_vocab(norm_names, min_count=3):
    """Words used in Source 1 names (at least min_count records)."""
    from collections import Counter
    c = Counter(t for n in norm_names for t in set(str(n).split()))
    return frozenset(t for t, k in c.items() if k >= min_count)


def _similar(a, b):
    if a == b:
        return True
    if abs(len(a) - len(b)) > 3:
        return False
    return (_Lev.distance(a, b) <= (1 if min(len(a), len(b)) <= 4 else 2)
            or _JW.normalized_similarity(a, b) >= 0.9)


TOK_COLS = ["tk_a_n", "tk_b_n", "tk_common", "tk_b_typo", "tk_b_word", "tk_b_oov", "tk_a_typo", "tk_a_word",
            "tk_a_oov", "tk_subst", "tk_b_oov_share"]


def token_features(ca, cb, vocab):
    """ca, cb: distinctive-word names (legal / generic words removed) per pair."""
    vocab = vocab or frozenset()
    out = np.zeros((len(ca), len(TOK_COLS)), np.float32)
    for r, (x, y) in enumerate(zip(ca, cb)):
        A, B = set(str(x).split()), set(str(y).split())
        com = A & B
        a_only, b_only = A - com, B - com
        bt = bw = bo = 0
        for t in b_only:
            if any(_similar(t, s) for s in a_only):
                bt += 1
            elif t in vocab:
                bw += 1
            else:
                bo += 1
        at = aw = ao = 0
        for t in a_only:
            if any(_similar(t, s) for s in b_only):
                at += 1
            elif t in vocab:
                aw += 1
            else:
                ao += 1
        out[r] = (len(A), len(B), len(com), bt, bw, bo, at, aw, ao, float(bw > 0 and aw > 0),
                  sum(t not in vocab for t in B) / max(len(B), 1))
    return pd.DataFrame(out, columns=TOK_COLS)
