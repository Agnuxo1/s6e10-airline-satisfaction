# --- S6E10 stack kernel body (CONFIG injected above) ---
# Members: our own fold-safe runs (kernel outputs) and, optionally, the public cdeotte 124-member OOF library.
# Level-2 models are scored with nested CV: StratifiedKFold(5, seed 42) -- the same folds as our level-1 runs --
# so every level-2 prediction for a row comes from a stacker that never saw that row's label.
import glob
import json
import os
import re
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

OUT = os.environ.get("S6E10_OUT", "/kaggle/working")
IN = os.environ.get("S6E10_IN", "/kaggle/input")
t0 = time.time()


def log(*a):
    print(f"{time.time() - t0:7.0f}s", *a, flush=True)


def find(name, must=None):
    hits = sorted(h for h in glob.glob(f"{IN}/**/{name}", recursive=True) if not must or must in h)
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


train = pd.read_csv(find("train.csv", "s6e10"), usecols=["id", "satisfaction"])
test = pd.read_csv(find("test.csv", "s6e10"), usecols=["id"])
y = train["satisfaction"].astype(int).to_numpy()
N, M = len(y), len(test)
folds = np.zeros(N, dtype=np.int8)
for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(N), y)):
    folds[va] = k

OOF, TEST, SRC = {}, {}, {}


def add(name, o, t, src):
    o, t = np.asarray(o, np.float64).ravel(), np.asarray(t, np.float64).ravel()
    if o.shape != (N,) or t.shape != (M,) or not (np.isfinite(o).all() and np.isfinite(t).all()):
        log("skip", name, o.shape, t.shape)
        return
    OOF[name], TEST[name], SRC[name] = o, t, src


# own runs: oof_<tag>.npy / test_<tag>.npy in each attached kernel output
for p in sorted(glob.glob(f"{IN}/**/oof_E*.npy", recursive=True)):
    tag = os.path.basename(p)[4:-4]
    if CONFIG.get("own_exclude") and re.search(CONFIG["own_exclude"], tag):
        continue
    add(tag, np.load(p), np.load(p.replace("/oof_", "/test_")), "own")
if CONFIG.get("public"):
    for p in sorted(glob.glob(f"{IN}/**/s6e10-124-public-oof-and-test-predictions/**/oof_*.npy", recursive=True)):
        add("pub_" + os.path.basename(p)[4:-4], np.load(p), np.load(p.replace("/oof_", "/test_")), "cdeotte124")
names = list(OOF)
single = pd.Series({n: roc_auc_score(y, OOF[n]) for n in names}).sort_values(ascending=False)
log("members", len(names), "own", sum(SRC[n] == "own" for n in names))
print(single.head(25).to_string(), flush=True)

Z = np.column_stack([logit(OOF[n]) for n in names]).astype(np.float32)
ZT = np.column_stack([logit(TEST[n]) for n in names]).astype(np.float32)
report = dict(members=names, single=single.to_dict(), runs={})


def nested_lr(cols, C, tag):
    o = np.zeros(N)
    for k in range(5):
        tr, va = folds != k, folds == k
        o[va] = LogisticRegression(C=C, max_iter=3000).fit(Z[tr][:, cols], y[tr]).decision_function(Z[va][:, cols])
    m = LogisticRegression(C=C, max_iter=3000).fit(Z[:, cols], y)
    auc = roc_auc_score(y, o)
    fa = [roc_auc_score(y[folds == k], o[folds == k]) for k in range(5)]
    report["runs"][tag] = dict(kind="lr", C=C, n=len(cols), nested_auc=auc, fold_auc=fa,
                               coef=dict(zip([names[c] for c in cols], m.coef_[0].round(5).tolist())))
    log(f"{tag:28s} nested {auc:.6f}  folds {' '.join(f'{v:.5f}' for v in fa)}")
    np.save(f"{OUT}/l2oof_{tag}.npy", o.astype(np.float32))
    np.save(f"{OUT}/l2test_{tag}.npy", m.decision_function(ZT[:, cols]).astype(np.float32))
    return auc


own = [i for i, n in enumerate(names) if SRC[n] == "own"]
allm = list(range(len(names)))
for C in CONFIG.get("C_grid", [0.003, 0.03, 1.0]):
    if own:
        nested_lr(own, C, f"lr_own_C{C}")
    if len(allm) > len(own):
        nested_lr(allm, C, f"lr_all_C{C}")
json.dump(report, open(f"{OUT}/stack_report.json", "w"), indent=1)
best = max(report["runs"], key=lambda t: report["runs"][t]["nested_auc"])
log("best", best, report["runs"][best]["nested_auc"])
for tag in report["runs"]:
    p = 1 / (1 + np.exp(-np.load(f"{OUT}/l2test_{tag}.npy").astype(np.float64)))
    pd.DataFrame({"id": test["id"], "satisfaction": p}).to_csv(f"{OUT}/submission_{tag}.csv", index=False)
log("ALL DONE")
