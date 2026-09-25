#!/usr/bin/env python3
"""
record_submission.py - log one leaderboard submission: submissions/<date>_<time>_IST_<slug>/ with the
exact notebook that was run, its metrics.json (downloaded from the Kaggle output), a NOTES.md, and a
git tag on the commit the notebook came from.

  python tools/record_submission.py --slug two-stage-xgb --commit cf433c1 \
      --metrics ~/Downloads/metrics.json --note "first GPU run"
  python tools/record_submission.py --dir submissions/<folder> --score 0.9712   # add the LB score later
"""
import argparse
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "submissions"
IST = timezone(timedelta(hours=5, minutes=30))


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def new_entry(a):
    now = datetime.now(IST)
    d = SUB / f"{now:%Y-%m-%d_%H%M}_IST_{a.slug}"
    d.mkdir(parents=True, exist_ok=False)
    commit = git("rev-parse", "--short", a.commit)
    (d / "entity_resolution.ipynb").write_text(git("show", f"{commit}:entity_resolution.ipynb") + "\n")
    metrics = {}
    if a.metrics:
        metrics = json.loads(Path(a.metrics).expanduser().read_text())
        (d / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    tag = f"sub-{now:%Y%m%d-%H%M}"
    lines = [f"# Submission {now:%Y-%m-%d %H:%M} IST - {a.slug}", "",
             f"- Commit: `{commit}` (tag `{tag}`)",
             f"- Leaderboard F0.5: {a.score if a.score is not None else 'pending'}"]
    for k in ("heldout_macro_f05_two_stage", "heldout_macro_f05_stage1_only", "tau1", "tau2",
              "train_blocking_recall", "model_backend", "gpu_blocking", "test_candidate_pairs", "test_matches"):
        if k in metrics:
            lines.append(f"- {k}: {metrics[k]}")
    lines += ["", "## Notes", a.note or "-", ""]
    (d / "NOTES.md").write_text("\n".join(lines))
    git("tag", "-a", tag, commit, "-m", f"Submission {a.slug} ({now:%Y-%m-%d %H:%M} IST)")
    print(f"created {d.relative_to(ROOT)} and tag {tag} -> {commit}")


def set_score(a):
    notes = Path(a.dir) / "NOTES.md"
    t = notes.read_text()
    old = next(line for line in t.splitlines() if line.startswith("- Leaderboard F0.5:"))
    notes.write_text(t.replace(old, f"- Leaderboard F0.5: {a.score}"))
    print(f"updated {notes}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slug")
    ap.add_argument("--commit", default="HEAD")
    ap.add_argument("--metrics")
    ap.add_argument("--score")
    ap.add_argument("--note")
    ap.add_argument("--dir", help="existing entry: only update its leaderboard score")
    a = ap.parse_args()
    set_score(a) if a.dir else new_entry(a)
