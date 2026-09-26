"""Experiment 'lowmem': score the test set in row chunks (two passes) instead of one (pairs x columns) matrix.

predict_test built one float32 matrix for ALL test candidates: ~35M pairs x 105 columns = ~15 GB on the full test
(France alone: 6.5M pairs), the largest single allocation of the run and the usual OOM point on a 30 GB Kaggle
machine. The stage-2 context (write_context) only WRITES its own columns from (pairs, p1, s23) and never reads the
base features, so the test can be scored exactly in two passes over 2M-row chunks:
  pass 1: base features of a chunk -> stage-1 score, chunk discarded
  context: stage-2 context for all pairs into a (pairs x N_CONTEXT) array (~21 columns)
  pass 2: base features of the chunk again + its context rows -> stage-2 score
Same rows, same features (pair_features is row-wise; the per-pair group context is computed once on all pairs, as
before), same models: predictions are identical. Cost: test features computed twice.
VERIFIED: Tamil Nadu test split, 831,558 candidates in 9 chunks, same models: stage-2 predictions bit-identical
(numpy array_equal), matching_results.tsv and candidate_pairs.tsv byte-identical.
usage: python lowmem.py <ber dir>"""
import os, sys
d = sys.argv[1]


def edit(fn, old, new):
    p = os.path.join(d, fn); s = open(p).read()
    assert s.count(old) == 1, (fn, old[:60], s.count(old))
    open(p, "w").write(s.replace(old, new))


edit("pipeline.py", "\n\ndef with_context(X, names, pairs, p1, s23):", '''

def iter_base_features(pairs, s1, s23, chunk):
    """[lowmem] Same rows and columns as base_features, one chunk at a time: yields (start, X_chunk, names) with
    X_chunk shaped (chunk rows, base columns + model.N_CONTEXT), context columns left empty."""
    s1, s23, idf = record_stats(s1, s23)
    ctx = group_features(pairs.select("s1_row", "s23_row", "blk_full"), "blk_full").drop(
        "s1_row", "s23_row", "blk_full")
    blk = pairs.select(BLOCK_COLS)
    for i in range(0, len(pairs), chunk):
        f = pl.concat([pair_features(pairs.slice(i, chunk), s1, s23, idf),
                       blk.slice(i, chunk), ctx.slice(i, chunk)], how="horizontal")
        X = np.empty((len(f), f.width + model.N_CONTEXT), dtype=np.float32)
        X[:, :f.width] = f.to_numpy().astype(np.float32, copy=False)
        yield i, X, f.columns


def with_context(X, names, pairs, p1, s23):''')

edit("pipeline.py", '''    log(f"test candidates {len(cands):,} ({len(cands) / len(s1):.1f}/S1)")
    X, names = base_features(cands, s1, s23, cfg["feature_chunk"])
    s1 = s1.select("entity_id", "country")
    s23 = s23.select("entity_id", "core", "addr", "house", "is_s3")
    gc.collect()
    p1 = model.stage1_predict(m1, X[:, :len(names)])
    log("test stage 1 scored")
    X2, _ = with_context(X, names, cands, p1, s23)
    gc.collect()
    p = m2.predict(X2)
    del X, X2
    gc.collect()
    log("test stage 2 scored, feature matrix released")
''', '''    log(f"test candidates {len(cands):,} ({len(cands) / len(s1):.1f}/S1)")
    # [lowmem] two passes over row chunks: never one (pairs x columns) matrix (~15 GB on the full test)
    n, chunk = len(cands), cfg["feature_chunk"]
    p1 = np.empty(n, dtype=np.float32)
    for i, X, names in iter_base_features(cands, s1, s23, chunk):
        p1[i:i + len(X)] = model.stage1_predict(m1, X[:, :len(names)])
        del X
        log(f"  test stage 1 {min(i + chunk, n):,}/{n:,}")
    gc.collect()
    log("test stage 1 scored")
    C = np.empty((n, model.N_CONTEXT), dtype=np.float32)
    write_context(C, 0, cands, p1, s23.select("entity_id", "core", "addr", "house", "is_s3"))
    gc.collect()
    log("stage-2 context features written")
    p = np.empty(n, dtype=np.float32)
    for i, X, names in iter_base_features(cands, s1, s23, chunk):
        X[:, len(names):] = C[i:i + len(X)]
        p[i:i + len(X)] = m2.predict(X)
        del X
        log(f"  test stage 2 {min(i + chunk, n):,}/{n:,}")
    del C
    s1 = s1.select("entity_id", "country")
    s23 = s23.select("entity_id", "core", "addr", "house", "is_s3")
    gc.collect()
    log("test stage 2 scored, feature matrix released")
''')
print("lowmem applied to", d)
