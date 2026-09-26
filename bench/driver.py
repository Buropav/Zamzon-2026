"""Runs the friend's pipeline exactly as erk_sub2_1 cells 18 + 20 do: cfg = {**DEFAULT_CFG, **CFG, **override};
pipeline.train; pipeline.predict_test.   usage: driver.py <variant dir with ber/> <dataset dir> '<override json>'"""
import json
import sys
import time

vdir, data, over = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
sys.path.insert(0, vdir)
from ber import pipeline  # noqa: E402

CFG = {  # erk_sub2_1.ipynb cell 3 values (paths redirected)
    "data_dir": data, "work_dir": f"{vdir}/work", "output_dir": f"{vdir}/output", "workers": None,
    "dev_states": None, "train_s1_frac": 0.3, "orphan_frac": 0.15, "stage1_rounds": 3000, "stage2_rounds": 3000,
    "xgb_params": {"learning_rate": 0.05},
}
CFG.update(over)
cfg = {**pipeline.DEFAULT_CFG, **CFG}
t = time.time()
m1, m2, params, report = pipeline.train(cfg)
pipeline.predict_test(cfg, m1, m2, params)
import resource  # noqa: E402
peak_gb = max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) / 1e6
json.dump({"cfg": cfg, "holdout_b": report["holdout_b"], "decision": report["decision"], "seconds": time.time() - t,
           "peak_rss_gb": peak_gb},
          open(f"{vdir}/driver_report.json", "w"), indent=1, default=str)
