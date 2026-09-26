"""Experiment 'memtrain': train XGBoost from row indices, never from copied row subsets.

Kaggle (2x T4, ~30 GB) killed erk_sub9 right after 'stage-2 context features written': train_stage2 received
X2[fit] and X2[holdout A], boolean-index COPIES of the 27.2M x 105 float32 matrix (11.4 GB), on top of the 16 GB
already in use; stage 1 made the same copies per fold (X[tr], X[va]). Now the QuantileDMatrix is fed by an
xgboost.DataIter over 1M-row chunks of the original matrix, so only one chunk copy exists at a time.
Measured (4M x 105 matrix, 56% of rows in fit + holdout, CPU): extra peak RAM over the matrix 1.99 GB (copy) ->
1.09 GB (chunks); same best iteration and same predictions. At Kaggle scale (11.4 GB matrix, 16.1 GB in use):
~29.6 GB -> ~23.5 GB of the 32.2 GB limit.
usage: python memtrain.py <ber dir>"""
import os, sys
d = sys.argv[1]


def edit(fn, old, new):
    p = os.path.join(d, fn); s = open(p).read()
    assert s.count(old) == 1, (fn, old[:60], s.count(old))
    open(p, "w").write(s.replace(old, new))


edit("model.py", "\n\ndef stage1_oof(", '''

ROW_CHUNK = 1_000_000   # [memtrain] rows copied at a time while building a QuantileDMatrix


class _Rows(xgb.DataIter):
    """[memtrain] X[rows] in chunks, so the training/validation subsets are never materialised."""

    def __init__(self, X, y, rows, names):
        self.X, self.y, self.rows, self.names, self.i = X, y, rows, names, 0
        super().__init__()

    def next(self, input_data):
        if self.i >= len(self.rows):
            return False
        r = self.rows[self.i:self.i + ROW_CHUNK]
        self.i += ROW_CHUNK
        input_data(data=np.ascontiguousarray(self.X[r]), label=self.y[r], feature_names=self.names)
        return True

    def reset(self):
        self.i = 0


def train_rows(X, y, tr, va, rounds, names, device=None, params=None):
    """[memtrain] train_one on X[tr] with early stopping on X[va], from row indices."""
    device = device or _device(0)
    p = {**XGB_PARAMS, **(params or {}), "device": device}
    dtr = xgb.QuantileDMatrix(_Rows(X, y, tr, names), max_bin=p["max_bin"])
    dva = xgb.QuantileDMatrix(_Rows(X, y, va, names), ref=dtr)
    b = xgb.train(p, dtr, num_boost_round=rounds, evals=[(dva, "valid")],
                  early_stopping_rounds=50, verbose_eval=200)
    del dtr, dva
    gc.collect()
    return Model(b, names)


def stage1_oof(''')
edit("model.py", '''        Xtr, Xva = X[tr], X[va]  # copies (boolean index); freed at the end of this iteration
        m = train_one(Xtr, y[tr], Xva, y[va], rounds, names, device=_device(f))
        del Xtr, Xva
''', '''        m = train_rows(X, y, np.flatnonzero(tr), np.flatnonzero(va), rounds, names, device=_device(f))  # [memtrain]
''')
edit("model.py", "\n\ndef train_stage2(X, y, Xv, yv, rounds, names):", '''

def train_stage2_rows(X, y, tr, va, rounds, names):
    """[memtrain] train_stage2 on X[tr] / X[va] from row indices."""
    models = []
    for i in range(len(STAGE2_VARIANTS)):
        m = train_rows(X, y, tr, va, rounds, names, device=_device(i), params=STAGE2_VARIANTS[i])
        print(f"    stage2 variant {i}: best_iter {m.best_iteration}{_ram()}", flush=True)
        models.append(m)
        gc.collect()
    return Ensemble(models)


def train_stage2(X, y, Xv, yv, rounds, names):''')
edit("pipeline.py", '''    m2 = model.train_stage2(X2[fit], y[fit], X2[r == 1], y[r == 1], cfg["stage2_rounds"], names2)''',
     '''    m2 = model.train_stage2_rows(X2, y, np.flatnonzero(fit), np.flatnonzero(r == 1), cfg["stage2_rounds"], names2)  # [memtrain]''')
print("memtrain applied to", d)
