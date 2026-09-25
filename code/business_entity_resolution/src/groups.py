"""
groups.py - second-stage (group-context) features for the pair classifier.

A Source 1 entity usually has 3-4 true matches that also resemble EACH OTHER. Stage 1 scores every
pair alone, so a match whose name was replaced by a trade name ("Kororbikor" at the exact same
address as three confirmed matches) scores low. Stage 2 sees:
  * rank / count / margin of the stage-1 score within the Source 1 group and within the Source 2/3
    competitors (one-to-one structure),
  * how many strong matches the Source 1 entity already has,
  * similarity of the candidate to the best OTHER candidate of the same Source 1 (name, address,
    exact address, postcode/house agreement).
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist


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
    r, n, m, bo = (np.empty(len(key)) for _ in range(4))
    r[order], n[order] = rank, sizes[gid]
    m[order] = s_sorted - best_other
    bo[order] = best_other
    return r, n, m, bo


def s1_side_features(cands, p_col, s23, strong=(0.5, 0.9)):
    """Features that only need the candidates of each Source 1 (safe to compute per S1 chunk).
    cands: qi, di, p_col. s23: prepared Source 2/3 table (norm_name, norm_addr, pc, house)."""
    qi, di = cands["qi"].to_numpy(), cands["di"].to_numpy()
    p = cands[p_col].to_numpy(np.float64)
    G = pd.DataFrame(index=cands.index)
    r, n, m, bo = _side_stats(qi, p)
    G["g_rank_s1"], G["g_n_s1"], G["g_margin_s1"], G["g_best_other_s1"] = r, n, m, bo
    for t in strong:
        cnt = pd.Series(p >= t).groupby(qi).transform("sum").to_numpy()
        G[f"g_n_ge{int(t * 100)}_s1"] = cnt - (p >= t)  # strong OTHER candidates
    # best OTHER candidate of the same Source 1: top-1, or top-2 for the top-1 row itself
    order = np.lexsort((-p, qi))
    q_sorted = qi[order]
    first = np.r_[True, q_sorted[1:] != q_sorted[:-1]]
    top1_pos = order[first]
    grp_start = np.cumsum(first) - 1
    top1_of = np.empty(len(qi), np.int64)
    top1_of[order] = top1_pos[grp_start]
    second = np.full(len(top1_pos), -1)
    idx_second = np.flatnonzero(first) + 1
    ok = (idx_second < len(order)) & ~np.r_[first, True][idx_second]
    second[ok] = order[idx_second[ok]]
    second_of = np.empty(len(qi), np.int64)
    second_of[order] = second[grp_start]
    other = np.where(np.arange(len(qi)) == top1_of, second_of, top1_of)
    has_other = other >= 0
    o_di = np.where(has_other, di[np.maximum(other, 0)], di)
    nn, na = s23["norm_name"].to_numpy(), s23["norm_addr"].to_numpy()
    G["g_top_name_tset"] = np.where(has_other, cpdist(nn[di], nn[o_di], scorer=fuzz.token_set_ratio,
                                                      workers=-1, dtype=np.float32) / 100.0, np.nan)
    G["g_top_addr_tset"] = np.where(has_other, cpdist(na[di], na[o_di], scorer=fuzz.token_set_ratio,
                                                      workers=-1, dtype=np.float32) / 100.0, np.nan)
    G["g_top_addr_eq"] = np.where(has_other, (na[di] == na[o_di]) & (na[di] != ""), np.nan)
    for col in ("pc", "house"):
        if col in s23:
            v = s23[col].to_numpy()
            a, b = v[di], v[o_di]
            G[f"g_top_{col}_eq"] = np.where(has_other & (a != "") & (b != ""), a == b, np.nan)
    G["g_top_p"] = np.where(has_other, p[np.maximum(other, 0)], np.nan)
    # strongest stage-1 score among OTHER candidates with the same normalised address
    addr = na[di]
    codes = pd.factorize(pd.Series(qi).astype(str) + "\x00" + pd.Series(addr))[0]
    _, n_a, _, bo_a = _side_stats(codes, p)
    G["g_same_addr_n"] = np.where(addr != "", n_a - 1, 0)
    G["g_same_addr_max"] = np.where((addr != "") & (n_a > 1), bo_a, np.nan)
    # same for the house number: true copies share the Source 1 number, a decoy's shifted number is alone
    if "addr_nums" in s23:
        hn = np.array([x.split(" ", 1)[0] for x in s23["addr_nums"].to_numpy()[di]], dtype=object)
        codes = pd.factorize(pd.Series(qi).astype(str) + "\x00" + pd.Series(hn))[0]
        _, n_h, _, bo_h = _side_stats(codes, p)
        G["g_same_hnum_n"] = np.where(hn != "", n_h - 1, np.nan)
        G["g_same_hnum_max"] = np.where((hn != "") & (n_h > 1), bo_h, np.nan)
    return G.astype(np.float32)


def s23_side_features(cands, p_col):
    """Competition for the same Source 2/3 record. Needs ALL candidates of that record, so in chunked
    inference compute it after every chunk has been scored."""
    r, n, m, bo = _side_stats(cands["di"].to_numpy(), cands[p_col].to_numpy(np.float64))
    return pd.DataFrame({"g_rank_s23": r, "g_n_s23": n, "g_margin_s23": m, "g_best_other_s23": bo},
                        index=cands.index).astype(np.float32)


STAGE2_BASE = ["name_tset", "name_wr", "core_tset", "core_exact", "addr_tset", "addr_core_tset", "pc_eq",
               "house_eq", "num_jacc", "name_idfcos", "addr_idfcos", "blk_text", "blk_name", "n_blockers",
               "name_soft_min", "addr_soft_max", "distinct_jacc", "name_conflict", "is_s3",
               # address numbers, compact names, token differences, extra blocking views (extra_feats.py)
               "n_a", "n_b", "n_exact", "n_shift_up", "n_shift_down", "n_typo", "n_prefix", "n_a_left", "n_b_left",
               "h_delta", "h_lev", "h_eq", "h_prefix", "h_in_b", "n_minshift", "cmp_partial", "cmp_contain",
               "tk_b_word", "tk_a_word", "tk_subst", "tk_b_oov_share", "blk_addr", "blk_rtext", "blk_rname"]


def stage2_matrix(X1, p1, G1, G23):
    """Stage-2 design matrix: stage-1 score + key raw features + both group blocks."""
    base = X1[[c for c in STAGE2_BASE if c in X1.columns]].reset_index(drop=True)
    out = pd.concat([base, G1.reset_index(drop=True), G23.reset_index(drop=True)], axis=1)
    out.insert(0, "p1", np.asarray(p1, np.float32))
    return out
