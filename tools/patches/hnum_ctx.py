"""Experiment 'hnum_ctx': the old pipeline's g_same_hnum_n / g_same_hnum_max (groups.py) as two extra stage-2
context columns in the friend's write_context. Per S1 entity, candidates are grouped by the S2/S3 record's house
number; true copies share the S1's number, a decoy with a shifted number stands alone.
  same_hnum_n   = other candidates of the same S1 with the same house number (NaN when the house is empty)
  same_hnum_max = best stage-1 score among those OTHER candidates (NaN when house empty or no other)
Uses only p1 (out-of-fold in training) and record fields. Numpy only.   usage: python hnum_ctx.py <ber dir>"""
import os, sys
d = sys.argv[1]

def edit(fn, old, new):
    p = os.path.join(d, fn); s = open(p).read()
    assert s.count(old) == 1, (fn, old[:60], s.count(old))
    open(p, "w").write(s.replace(old, new))

edit("features.py",
     '    "sib1_p", "sib1_name", "sib1_addr", "sib1_house_eq", "sib2_p", "sib2_name", "sib2_addr",\n]',
     '    "sib1_p", "sib1_name", "sib1_addr", "sib1_house_eq", "sib2_p", "sib2_name", "sib2_addr",\n'
     '    "same_hnum_n", "same_hnum_max",   # [hnum_ctx]\n]')
edit("features.py", "    return list(CONTEXT_NAMES)\n",
'''    # [hnum_ctx] support from other candidates of the same S1 that share this S2/S3 record's house number
    house = s23["house"]
    hcode = house.cast(pl.Categorical).to_physical().to_numpy().astype(np.int64)[s23r]
    has_h = (house.to_numpy() != "")[s23r]
    key = s1r * (int(hcode.max()) + 1 if n else 1) + hcode
    order = np.lexsort((-p1, key))
    k_sorted, p_sorted = key[order], p1[order]
    new = np.r_[True, k_sorted[1:] != k_sorted[:-1]]
    gid = np.cumsum(new) - 1
    starts = np.flatnonzero(new)
    sizes = np.diff(np.r_[starts, n])
    rank = np.arange(n) - starts[gid] + 1
    top1 = p_sorted[starts]
    top2 = np.where(sizes > 1, p_sorted[np.minimum(starts + 1, n - 1)], np.nan)
    best_other = np.empty(n, np.float32)
    best_other[order] = np.where(rank == 1, top2[gid], top1[gid])
    size = np.empty(n, np.int64)
    size[order] = sizes[gid]
    del order, k_sorted, p_sorted, new, gid, key, hcode
    X[:, col0 + CONTEXT_NAMES.index("same_hnum_n")] = np.where(has_h, size - 1, np.nan)
    X[:, col0 + CONTEXT_NAMES.index("same_hnum_max")] = np.where(has_h & (size > 1), best_other, np.nan)
    return list(CONTEXT_NAMES)
''')
print("hnum_ctx applied to", d)
