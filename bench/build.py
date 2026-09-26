"""Builds a labelled benchmark from the TRAIN files (whole states, seed 42), in the challenge's TSV layout:
<out>/dataset/{train,test}/*.tsv, <out>/gt/test_ground_truth.tsv (held out), <out>/stats.json.

State of a record = last comma-separated address component, upper-cased, matched against ALIASES.
  --mode split : the chosen states' S1 entities split 70/30 train/test; each S2/S3 follows its S1 owner;
                 unowned in-state S2/S3 and a proportional sample of the country's stateless unowned S2/S3
                 are split 70/30 as distractors (the original KY+KL benchmark).
  --mode lb    : leaderboard-like. Test = WHOLE --test-states (full-size pools); --test-orphan of their S1 are
                 removed while their S2/S3 stay, so ~40% of test S2/S3 have no S1 (the real test: ~39.8%).
                 Train = WHOLE --train-states (disjoint). Distractors as above.

  python bench/build.py --raw <student_resource/dataset> --out bench_kykl --mode split --states KY,KL
  python bench/build.py --raw <...> --out bench_lb --mode lb --test-states KY,KL --train-states OR,KA,NC,PB
"""
import argparse
import hashlib
import json
import os

import numpy as np
import polars as pl

ALIASES = {  # state -> (country, last-address-component spellings seen in the train files)
    "KY": ("US", {"KY", "KENTUCKY"}), "OR": ("US", {"OR", "OREGON"}), "NC": ("US", {"NC", "NORTH CAROLINA"}),
    "TX": ("US", {"TX", "TEXAS"}), "OH": ("US", {"OH", "OHIO"}), "AR": ("US", {"AR", "ARKANSAS"}),
    "KL": ("India", {"KERALA", "KERALAM", "കേരളം", "KL"}),
    "KA": ("India", {"KARNATAKA", "ಕರ್ನಾಟಕ", "KA"}),
    "PB": ("India", {"PUNJAB", "PB", "ਪੰਜਾਬ"}),
    "HR": ("India", {"HARYANA", "HR", "हरियाणा"}),
    "UP": ("India", {"UTTAR PRADESH", "UP", "उत्तर प्रदेश"}),
    "MP": ("India", {"MADHYA PRADESH", "MP", "मध्य प्रदेश"}),
    "RJ": ("India", {"RAJASTHAN", "RJ", "राजस्थान"}),
    "BR": ("India", {"BIHAR", "BR", "बिहार"}),
}
STATELESS = {"", "NULL", "N/A", "<NULL>", "NONE", "NA", "-"}
SEED = 42


def read(path):
    return pl.read_csv(path, separator="\t", quote_char='"', infer_schema=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="folder with train/train_source1.tsv ...")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["split", "lb"], default="split")
    ap.add_argument("--states", default="KY,KL")
    ap.add_argument("--test-states", default="KY,KL")
    ap.add_argument("--train-states", default="OR,KA,NC,PB")
    ap.add_argument("--test-orphan", type=float, default=0.19)
    a = ap.parse_args()
    t = os.path.join(a.raw, "train")
    last = (pl.col("business_address").fill_null("").str.split(",").list.last().str.strip_chars()
            .str.to_uppercase().alias("last"))
    s1 = read(f"{t}/train_source1.tsv").with_columns(last)
    s23 = pl.concat([read(f"{t}/train_source2.tsv").with_columns(pl.lit(2).alias("src")),
                     read(f"{t}/train_source3.tsv").with_columns(pl.lit(3).alias("src"))]).with_columns(last)
    gt = read(f"{t}/train_ground_truth.tsv")
    pairs = (gt.filter(pl.col("matched_entity_ids").fill_null("") != "")
             .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
             .rename({"source1_entity_id": "s1", "matched_entity_ids": "s23"})
             .with_columns(pl.col("s23").str.strip_chars()))
    owned_all = set(pairs["s23"].to_list())

    def in_states(states):
        e = pl.lit(False)
        for st in states:
            c, al = ALIASES[st]
            e = e | ((pl.col("country") == c) & pl.col("last").is_in(sorted(al)))
        return e

    rng = np.random.default_rng(SEED)
    if a.mode == "split":
        states = a.states.split(",")
        sel1 = s1.filter(in_states(states))
        ids = np.sort(sel1["entity_id"].to_numpy())
        split1 = pl.DataFrame({"entity_id": ids, "split": np.where(rng.random(len(ids)) < 0.30, "test", "train")})
        keep1 = split1
        dist_states = {"train": states, "test": states}
    else:
        te, tr = a.test_states.split(","), a.train_states.split(",")
        assert not set(te) & set(tr), "test and train states must differ"
        te1 = np.sort(s1.filter(in_states(te))["entity_id"].to_numpy())
        tr1 = np.sort(s1.filter(in_states(tr))["entity_id"].to_numpy())
        split1 = pl.DataFrame({"entity_id": np.r_[te1, tr1], "split": ["test"] * len(te1) + ["train"] * len(tr1)})
        orphan = rng.random(len(te1)) < a.test_orphan            # removed from test S1, their S2/S3 stay
        keep1 = pl.DataFrame({"entity_id": np.r_[te1[~orphan], tr1],
                              "split": ["test"] * int((~orphan).sum()) + ["train"] * len(tr1)})
        dist_states = {"train": tr, "test": te}
    sel1 = s1.join(keep1, on="entity_id")
    owner = pairs.join(split1.rename({"entity_id": "s1"}), on="s1").select(pl.col("s23").alias("entity_id"), "split")
    owned = s23.join(owner, on="entity_id")                    # includes S2/S3 of removed (orphaned) test S1
    unowned = s23.filter(~pl.col("entity_id").is_in(sorted(owned_all)))
    parts = []
    for sp in ("train", "test") if a.mode == "lb" else ("both",):
        sts = dist_states["test" if sp == "both" else sp]
        d = unowned.filter(in_states(sts))
        lab = np.where(rng.random(len(d)) < 0.30, "test", "train") if sp == "both" else np.full(len(d), sp)
        parts.append(d.with_columns(pl.Series("split", lab)))
        for ctry in dict.fromkeys(ALIASES[x][0] for x in sts):   # order of --states (reproducible draws)
            n_c = s1.filter(pl.col("country") == ctry).height
            n_sel = s1.filter(in_states([x for x in sts if ALIASES[x][0] == ctry])).height
            pool = unowned.filter((pl.col("country") == ctry) & pl.col("last").is_in(sorted(STATELESS))).sort("entity_id")
            k = int(round(len(pool) * n_sel / n_c))
            take = pool[np.sort(rng.choice(len(pool), k, replace=False))]
            lab = np.where(rng.random(k) < 0.30, "test", "train") if sp == "both" else np.full(k, sp)
            parts.append(take.with_columns(pl.Series("split", lab)))
    distract = pl.concat(parts).unique("entity_id", keep="first", maintain_order=True)
    all23 = pl.concat([owned.select(distract.columns), distract.filter(~pl.col("entity_id").is_in(owned["entity_id"].implode()))])
    assert all23["entity_id"].n_unique() == all23.height
    cols = ["entity_id", "business_name", "business_address", "country"]
    w = lambda df, p: df.select(cols).write_csv(p, separator="\t", quote_style="necessary")
    stats = {"mode": a.mode, "args": vars(a)}
    for sp in ("train", "test"):
        os.makedirs(f"{a.out}/dataset/{sp}", exist_ok=True)
        s1s = sel1.filter(pl.col("split") == sp).sort("entity_id")
        x = all23.filter(pl.col("split") == sp)
        w(s1s, f"{a.out}/dataset/{sp}/{sp}_source1.tsv")
        w(x.filter(pl.col("src") == 2).sort("entity_id"), f"{a.out}/dataset/{sp}/{sp}_source2.tsv")
        w(x.filter(pl.col("src") == 3).sort("entity_id"), f"{a.out}/dataset/{sp}/{sp}_source3.tsv")
        g = gt.filter(pl.col("source1_entity_id").is_in(s1s["entity_id"].implode())).sort("source1_entity_id")
        gp = pairs.filter(pl.col("s1").is_in(s1s["entity_id"].implode()))
        assert gp["s23"].is_in(x["entity_id"].implode()).all(), "an owned S2/S3 record is missing"
        if sp == "train":
            g.write_csv(f"{a.out}/dataset/train/train_ground_truth.tsv", separator="\t", quote_style="necessary")
        else:
            os.makedirs(f"{a.out}/gt", exist_ok=True)
            g.write_csv(f"{a.out}/gt/test_ground_truth.tsv", separator="\t", quote_style="necessary")
        matched = x.filter(pl.col("entity_id").is_in(gp["s23"].implode())).height
        stats[sp] = {"S1": s1s.height, "S1_by_country": dict(sorted(s1s["country"].value_counts().rows())),
                     "S2": x.filter(pl.col("src") == 2).height, "S3": x.filter(pl.col("src") == 3).height,
                     "gt_pairs": gp.height, "S23_unmatched_share": round(1 - matched / max(x.height, 1), 4),
                     "singleton_S1": s1s.height - gp["s1"].n_unique()}
    tr_ids = set(sel1.filter(pl.col("split") == "train")["entity_id"].to_list())
    assert not tr_ids & set(sel1.filter(pl.col("split") == "test")["entity_id"].to_list())
    stats["sha256"] = {}
    for root, _, files in os.walk(a.out):
        for f in sorted(files):
            if f.endswith(".tsv"):
                p = os.path.join(root, f)
                stats["sha256"][os.path.relpath(p, a.out)] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    json.dump(stats, open(f"{a.out}/stats.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k != "sha256"}, indent=1))


if __name__ == "__main__":
    main()
