"""OOF blend for S6E10: Caruana hill climbing on logits and logistic stacking on logits.

Weights are learned from out-of-fold predictions only. The honest estimate is leave-one-fold-out (LOFO): weights are
fit on 4 of the 5 folds and scored on the 5th; the final weights are then refit on all rows.
usage: python tools/blend.py --runs runs/<dir> [runs/<dir> ...] --out blends/<name> [--include REGEX] [--exclude REGEX]
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def logit(p):
    p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def load(run_dirs, include, exclude):
    oof, test, folds = {}, {}, None
    for d in run_dirs:
        d = Path(d)
        f = np.load(d / "folds.npy")
        if folds is None:
            folds = f
        assert (f == folds).all(), f"fold mismatch in {d}"
        for p in sorted(d.glob("oof_*.npy")):
            tag = p.stem[4:]
            if (include and not re.search(include, tag)) or (exclude and re.search(exclude, tag)):
                continue
            oof[tag] = logit(np.load(p))
            test[tag] = logit(np.load(d / f"test_{tag}.npy"))
    return oof, test, folds


def hill_climb(Z, y, rows, steps=60, tol=1e-7):
    """Greedy forward selection with replacement on the mean of logits; returns integer counts per model."""
    counts = np.zeros(Z.shape[1], dtype=int)
    cur, best = np.zeros(rows.sum()), -1.0
    Zr, yr = Z[rows], y[rows]
    for _ in range(steps):
        n = counts.sum()
        cand = [roc_auc_score(yr, (cur * n + Zr[:, j]) / (n + 1)) for j in range(Z.shape[1])]
        j = int(np.argmax(cand))
        if cand[j] <= best + tol:
            break
        best, counts[j] = cand[j], counts[j] + 1
        cur = (cur * n + Zr[:, j]) / (n + 1)
    return counts / counts.sum(), best


def stack(Z, y, rows, C=1.0):
    m = LogisticRegression(C=C, max_iter=2000).fit(Z[rows], y[rows])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--include")
    ap.add_argument("--exclude")
    a = ap.parse_args()
    oof, test, folds = load(a.runs, a.include, a.exclude)
    y = pd.read_csv(ROOT / "data" / "train.csv", usecols=["satisfaction"])["satisfaction"].astype(int).to_numpy()
    tags = list(oof)
    Z, T = np.column_stack([oof[t] for t in tags]), np.column_stack([test[t] for t in tags])
    single = {t: roc_auc_score(y, oof[t]) for t in tags}
    rep = dict(models=tags, single_auc=single, best_single=max(single.values()))
    lofo_hc, lofo_st, oof_hc, oof_st = [], [], np.zeros(len(y)), np.zeros(len(y))
    for k in range(int(folds.max()) + 1):
        tr, va = folds != k, folds == k
        w, _ = hill_climb(Z, y, tr)
        oof_hc[va] = Z[va] @ w
        lofo_hc.append(roc_auc_score(y[va], oof_hc[va]))
        m = stack(Z, y, tr)
        oof_st[va] = m.decision_function(Z[va])
        lofo_st.append(roc_auc_score(y[va], oof_st[va]))
    w, fit_auc = hill_climb(Z, y, np.ones(len(y), bool))
    m = stack(Z, y, np.ones(len(y), bool))
    rep.update(hill_climb=dict(weights={t: float(v) for t, v in zip(tags, w) if v > 0}, fit_auc=fit_auc,
                               lofo_auc=roc_auc_score(y, oof_hc), lofo_fold=lofo_hc,
                               lofo_fold_mean=float(np.mean(lofo_hc)), lofo_fold_std=float(np.std(lofo_hc))),
               stacking=dict(coef=dict(zip(tags, m.coef_[0].tolist())), intercept=float(m.intercept_[0]),
                             lofo_auc=roc_auc_score(y, oof_st), lofo_fold=lofo_st,
                             lofo_fold_mean=float(np.mean(lofo_st)), lofo_fold_std=float(np.std(lofo_st))))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ids = pd.read_csv(ROOT / "data" / "test.csv", usecols=["id"])["id"]
    sig = lambda z: 1 / (1 + np.exp(-z))
    np.save(out / "oof_hc_lofo.npy", sig(oof_hc).astype(np.float32))
    np.save(out / "oof_stack_lofo.npy", sig(oof_st).astype(np.float32))
    np.save(out / "folds.npy", folds)
    pd.DataFrame({"id": ids, "satisfaction": sig(T @ w)}).to_csv(out / "submission_hc.csv", index=False)
    pd.DataFrame({"id": ids, "satisfaction": sig(m.decision_function(T))}).to_csv(out / "submission_stack.csv",
                                                                                    index=False)
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({"best_single": rep["best_single"], "hc_lofo": rep["hill_climb"]["lofo_auc"],
                      "hc_weights": rep["hill_climb"]["weights"], "stack_lofo": rep["stacking"]["lofo_auc"]}, indent=1))


if __name__ == "__main__":
    main()
