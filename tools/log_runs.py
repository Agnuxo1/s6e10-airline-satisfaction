"""Append the finished runs of a kernel to experiments/runs.csv (one row per experiment and seed).

usage: python tools/log_runs.py <kernel-slug> <hypothesis> [--decision TEXT]
"""
import csv
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
slug, hyp = sys.argv[1], sys.argv[2]
decision = sys.argv[sys.argv.index("--decision") + 1] if "--decision" in sys.argv else ""
res = json.loads((ROOT / "runs" / slug / "results.json").read_text())
commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
path = ROOT / "experiments" / "runs.csv"
new = not path.exists()
with path.open("a", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    if new:
        w.writerow(["id", "fecha_utc", "hipotesis", "config", "kernel", "commit", "semilla", "oof_auc", "fold_mean",
                    "fold_std", "segundos", "decision"])
    for r in res:
        cfg = {k: r[k] for k in ("model", "features", "cats", "use_orig", "params", "gpu")}
        w.writerow([r["name"], dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M"), hyp, json.dumps(cfg), slug, commit,
                    r["seed"], f"{r['oof_auc']:.6f}", f"{r['fold_mean']:.6f}", f"{r['fold_std']:.6f}", r["sec"], decision])
print(f"{len(res)} rows -> {path}")
