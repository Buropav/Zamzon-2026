import os
import csv
from pathlib import Path
import pandas as pd

def read_tsv(path):
    """Read a challenge TSV safely (one record per line).
    
    Prevents unclosed quote character swallowing and 'NA' string nullification.
    Asserts exact line count and entity_id uniqueness.
    """
    path = Path(path)
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        quoting=csv.QUOTE_NONE,
        keep_default_na=False,
        na_values=[],
        encoding="utf-8"
    )
    with open(path, "r", encoding="utf-8") as f:
        next(f, None)
        n_lines = sum(1 for line in f if line.strip())
    assert len(df) == n_lines, f"{path.name}: parsed {len(df)} rows, file has {n_lines} data lines"
    assert df.iloc[:, 0].is_unique, f"{path.name}: duplicate IDs in first column"
    return df

def find_file(filename, dev_mode=False):
    """Locate dataset TSV files across Kaggle, student_resource, and repo folders."""
    roots = []
    if Path("/kaggle/input").exists():
        roots.extend([
            Path("/kaggle/input/zamzon-2026/student_resource/dataset"),
            Path("/kaggle/input/zamzon-2026/student_resource"),
            Path("/kaggle/input/zamzon-2026/dataset"),
            Path("/kaggle/input/zamzon-2026"),
            Path("/tmp/dataset"),
            Path("/kaggle/input"),
        ])
    
    roots.extend([
        Path.cwd() / "dataset",
        Path.cwd(),
        Path.cwd().parent / "dataset",
        Path.cwd().parent,
    ])
    
    if dev_mode:
        for r in roots:
            if r.exists():
                for p in r.rglob(filename):
                    if "dev" in p.parts and ".venv" not in p.parts:
                        return p
                        
    for r in roots:
        if r.exists():
            for p in r.rglob(filename):
                if "dev" not in p.parts and ".venv" not in p.parts:
                    return p
    return None
