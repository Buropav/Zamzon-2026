import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def _topk_rows(S, k, min_sim):
    """Extract top-k column indices and similarity scores per CSR row."""
    rows, cols, vals = [], [], []
    ip, ix, dv = S.indptr, S.indices, S.data
    for r in range(S.shape[0]):
        a, b = ip[r], ip[r + 1]
        if a == b:
            continue
        seg = dv[a:b]
        if b - a > k:
            sel = np.argpartition(-seg, k)[:k]
        else:
            sel = np.arange(b - a)
        sel = sel[seg[sel] >= min_sim]
        rows.append(np.full(len(sel), r, dtype=np.int32))
        cols.append(ix[a:b][sel])
        vals.append(seg[sel])
    if not rows:
        return np.array([], np.int32), np.array([], np.int32), np.array([], np.float32)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)

def block_topk(q_text, d_text, q_country, d_country, k=25, min_sim=0.01,
               max_df=0.01, ngram_range=(3, 4), batch=2000, fit_on="both"):
    """Scalable character n-gram TF-IDF blocking partitioned by country.
    
    1. Splits search space by country (US, INDIA, FRANCE) to eliminate cross-country waste.
    2. Drops frequent n-grams with max_df=0.01 to drop sparse matrix density from ~90% to ~5-7%.
    3. Transposes target matrix once.
    4. Returns DataFrame with integer row indices (qi, di) and similarity.
    """
    q_text, d_text = np.asarray(q_text, dtype=object), np.asarray(d_text, dtype=object)
    q_country, d_country = np.asarray(q_country, dtype=object), np.asarray(d_country, dtype=object)
    out = []
    
    for g in pd.unique(q_country):
        qi = np.flatnonzero(q_country == g)
        di = np.arange(len(d_text)) if g == "" else np.flatnonzero((d_country == g) | (d_country == ""))
        if len(qi) == 0 or len(di) == 0:
            continue
            
        vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=ngram_range,
            min_df=2,
            max_df=max_df,
            sublinear_tf=True,
            dtype=np.float32
        )
        corpus = np.concatenate([q_text[qi], d_text[di]]) if fit_on == "both" else d_text[di]
        vec.fit(corpus)
        
        Q = vec.transform(q_text[qi])
        DT = vec.transform(d_text[di]).T.tocsr()
        kk = min(k, len(di))
        
        for s in range(0, len(qi), batch):
            S = (Q[s:s + batch] @ DT).tocsr()
            r, c, v = _topk_rows(S, kk, min_sim)
            out.append(pd.DataFrame({"qi": qi[s + r], "di": di[c], "sim": v}))
            
    if not out:
        return pd.DataFrame({"qi": [], "di": [], "sim": []})
    return pd.concat(out, ignore_index=True)

def union_candidates(named_blocks):
    """Combines multiple blocker outputs (e.g. name, text combo) on [qi, di]."""
    merged = None
    for nm, df in named_blocks.items():
        if df.empty:
            continue
        df_named = df.rename(columns={"sim": f"blk_{nm}"})
        merged = df_named if merged is None else merged.merge(df_named, on=["qi", "di"], how="outer")
        
    if merged is None or merged.empty:
        return pd.DataFrame(columns=["qi", "di", "n_blockers"])
        
    blk_cols = [c for c in merged.columns if c.startswith("blk_")]
    merged["n_blockers"] = merged[blk_cols].notna().sum(axis=1)
    return merged
