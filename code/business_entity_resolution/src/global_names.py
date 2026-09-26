"""
global_names.py - country-wide name competition for Source 2/3 records whose region is unknown.

Unknown-region records (mostly EMPTY addresses; 97.7% of those are true copies of some Source 1) sit in every
regional pool of their country, so their owner can be any Source 1 of the country and the only evidence is the
name. For each such record the top-k Source 1 by name among ALL Source 1 of the country are retrieved once;
a pair (Source 1 q, record d) then gets: q's name similarity, the best similarity of any OTHER Source 1 of the
country, the margin, q's rank and the number of near-ties. In training this uses the full training Source 1 table
(not just the sampled regions), so the competition a pair faces is the same as at test time even when the true
owner of the record is outside the training sample.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

GN_COLS = ["gn_unknown", "gn_sim", "gn_best_other", "gn_margin", "gn_rank", "gn_n_close"]


class GlobalNames:
    def __init__(self, s1_names, s1_country, s23p, topk_fn, k=8, max_df=0.01, min_sim=0.05, log=print):
        """s1_names / s1_country: normalised names and countries of EVERY Source 1 of the split (training: the
        full training table). s23p: prepared Source 2/3 table (region, country, norm_name).
        topk_fn(Q, DT, k, min_sim) -> (row, col, sim) sorted by row (GPU or CPU)."""
        self.k = k
        s1_names = np.asarray(s1_names, dtype=object)
        self.s1_country = np.asarray(s1_country, dtype=object)
        n1 = len(self.s1_country)
        unk = np.flatnonzero(s23p["region"].to_numpy(object) == "")
        self.pos = np.full(len(s23p), -1, np.int64)
        self.pos[unk] = np.arange(len(unk))
        self.top_idx = np.full((len(unk), k), -1, np.int64)
        self.top_sim = np.zeros((len(unk), k), np.float32)
        self.loc1 = np.full(n1, -1, np.int64)
        self.locu = np.full(len(unk), -1, np.int64)
        self.unk_country = s23p["country"].to_numpy(object)[unk]
        names23 = s23p["norm_name"].to_numpy(object)[unk]
        self.S1M, self.RM = {}, {}
        for g in pd.unique(self.unk_country):
            if g == "":
                continue
            rows1 = np.flatnonzero(self.s1_country == g)
            rows_u = np.flatnonzero(self.unk_country == g)
            if len(rows1) < 2 or len(rows_u) == 0:
                continue
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=2,
                                  max_df=max(int(max_df * len(rows1)), 2), sublinear_tf=True, dtype=np.float32)
            try:
                S1M = vec.fit_transform(s1_names[rows1]).tocsr()
            except ValueError:
                continue
            RM = vec.transform(names23[rows_u]).tocsr()
            r, c, v = topk_fn(RM, S1M.T.tocsr(), min(k, len(rows1)), min_sim)
            start = np.searchsorted(r, r, side="left")
            rank = np.arange(len(r)) - start
            self.top_idx[rows_u[r], rank] = rows1[c]
            self.top_sim[rows_u[r], rank] = v
            self.loc1[rows1] = np.arange(len(rows1))
            self.locu[rows_u] = np.arange(len(rows_u))
            self.S1M[g], self.RM[g] = S1M, RM
            if log:
                log(f"  global names {g}: {len(rows_u):,} unknown-region S23 vs {len(rows1):,} S1")

    def features(self, q_full, d):
        """q_full: index of each pair's Source 1 in the table given to __init__; d: Source 2/3 row of each pair."""
        q_full, d = np.asarray(q_full, np.int64), np.asarray(d, np.int64)
        out = np.full((len(d), len(GN_COLS)), np.nan, np.float32)
        pu = self.pos[d]
        out[:, 0] = (pu >= 0).astype(np.float32)
        for g in self.S1M:
            m = np.flatnonzero((pu >= 0) & (self.unk_country[np.maximum(pu, 0)] == g) & (q_full >= 0))
            if len(m) == 0:
                continue
            m = m[self.s1_country[q_full[m]] == g]
            if len(m) == 0:
                continue
            S1M, RM = self.S1M[g], self.RM[g]
            sim = np.empty(len(m), np.float32)
            for s in range(0, len(m), 500_000):
                mm = m[s:s + 500_000]
                sim[s:s + 500_000] = np.asarray(S1M[self.loc1[q_full[mm]]].multiply(RM[self.locu[pu[mm]]]).sum(axis=1)).ravel()
            tops, sims = self.top_idx[pu[m]], self.top_sim[pu[m]]
            other = np.where((tops == q_full[m][:, None]) | (tops < 0), -1.0, sims)
            best = other.max(axis=1)
            out[m, 1] = sim
            out[m, 2] = np.where(best >= 0, best, np.nan)
            out[m, 3] = np.where(best >= 0, sim - best, sim)
            out[m, 4] = 1 + (other > sim[:, None]).sum(axis=1)
            out[m, 5] = (other >= sim[:, None] - 0.02).sum(axis=1)
        return pd.DataFrame(out, columns=GN_COLS)
