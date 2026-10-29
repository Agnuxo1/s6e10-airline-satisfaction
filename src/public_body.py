# --- S6E10 public notebook body (CONFIG injected above; darwin_core.py inlined above this body) ---
# 1) Level-1 members: our own fold-safe GBDT runs (public dataset franciscoangulo/s6e10-own-oof-library) and the
#    public cdeotte 124-member OOF library. All OOF predictions use StratifiedKFold(5, shuffle=True, random_state=42).
# 2) Level 2: logistic regression on logits, scored with nested CV on the same 5 folds.
# 3) Darwin-Cage: evolutionary search of grouping programs on the residual of the level-2 OOF, rewarded only by
#    structure confirmed on a reserved fold; confirmed programs are applied fold-safely and kept only if they raise
#    the 5-fold OOF AUC. The ablation (stack vs stack + Darwin-Cage) is printed at the end.
import glob
import json
import os
import re
import time

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

OUT = os.environ.get("S6E10_OUT", "/kaggle/working")
IN = os.environ.get("S6E10_IN", "/kaggle/input")
t0 = time.time()
CATS = ["Gender", "Customer_Type", "Type_of_Travel", "Class", "Inflight_wifi_service",
        "Departure_Arrival_time_convenient", "Ease_of_Online_booking", "Gate_location", "Food_and_drink",
        "Online_boarding", "Seat_comfort", "Inflight_entertainment", "On_board_service", "Leg_room_service",
        "Baggage_handling", "Checkin_service", "Cleanliness"]
NUM_SCALE = {"Age": 1, "Flight_Distance": 1, "Departure_Delay_in_Minutes": 1, "Arrival_Delay_in_Minutes": 1}


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


def clean(df):
    df.columns = [re.sub(r"[^0-9a-zA-Z]+", "_", c).strip("_") for c in df.columns]
    return df


train = clean(pd.read_csv(find("train.csv", "s6e10")))
test = clean(pd.read_csv(find("test.csv", "s6e10")))
y = train["satisfaction"].astype(str).str.lower().map({"true": 1, "false": 0}).to_numpy()
N, M = len(y), len(test)
folds = np.zeros(N, dtype=np.int64)
for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(N), y)):
    folds[va] = k

# ---------------- level 1 ----------------
OOF, TEST = {}, {}
for p in sorted(glob.glob(f"{IN}/**/s6e10-own-oof-library/**/oof_*.npy", recursive=True)):
    OOF["own_" + os.path.basename(p)[4:-4]], TEST["own_" + os.path.basename(p)[4:-4]] = \
        np.load(p), np.load(p.replace("/oof_", "/test_"))
for p in sorted(glob.glob(f"{IN}/**/s6e10-124-public-oof-and-test-predictions/**/oof_*.npy", recursive=True)):
    OOF["pub_" + os.path.basename(p)[4:-4]], TEST["pub_" + os.path.basename(p)[4:-4]] = \
        np.load(p), np.load(p.replace("/oof_", "/test_"))
names = [n for n in OOF if np.asarray(OOF[n]).ravel().shape == (N,) and np.asarray(TEST[n]).ravel().shape == (M,)]
Z = np.column_stack([logit(np.ravel(OOF[n])) for n in names]).astype(np.float32)
ZT = np.column_stack([logit(np.ravel(TEST[n])) for n in names]).astype(np.float32)
own = [i for i, n in enumerate(names) if n.startswith("own_")]
log("members", len(names), "own", len(own))

# ---------------- level 2 ----------------


def nested_lr(cols, C):
    o = np.zeros(N)
    for k in range(5):
        tr, va = folds != k, folds == k
        o[va] = LogisticRegression(C=C, max_iter=3000).fit(Z[tr][:, cols], y[tr]).decision_function(Z[va][:, cols])
    t = LogisticRegression(C=C, max_iter=3000).fit(Z[:, cols], y).decision_function(ZT[:, cols])
    return o, t


C = CONFIG.get("C", 0.003)
oof_own, test_own = nested_lr(own, C)
oof_all, test_all = nested_lr(list(range(len(names))), C)
report = dict(members=len(names), own=len(own), C=C, auc_own_stack=roc_auc_score(y, oof_own),
              auc_all_stack=roc_auc_score(y, oof_all))
log("own-only stack nested AUC %.6f | own+public stack nested AUC %.6f" % (report["auc_own_stack"],
                                                                           report["auc_all_stack"]))

# ---------------- Darwin-Cage on the level-2 residual ----------------
s, st = oof_all, test_all
p_search = np.zeros(N)
rng = np.random.default_rng(0)
idx = rng.permutation(N)
h = N // 2
for a, b in ((idx[:h], idx[h:]), (idx[h:], idx[:h])):
    p_search[b] = IsotonicRegression(out_of_bounds="clip").fit(s[a], y[a]).predict(s[b])
r = y - np.clip(p_search, 1e-4, 1 - 1e-4)
f10 = np.zeros(N, dtype=np.int64)
for k, (_, va) in enumerate(StratifiedKFold(10, shuffle=True, random_state=42).split(np.zeros(N), y)):
    f10[va] = k
D_all = {}
both = pd.concat([train.drop(columns=["satisfaction"]), test], ignore_index=True)
for c in CATS:
    D_all[c] = pd.factorize(both[c].astype(str))[0].astype(np.int64)
for c, sc in NUM_SCALE.items():
    D_all[c] = np.round(both[c].fillna(-1).to_numpy(np.float64) * sc).astype(np.int64)
D = {c: v[:N] for c, v in D_all.items()}
search = DarwinSearch(D, r, f10, z=CONFIG.get("z", 3.0))
res = search.run(seconds=CONFIG.get("minutes", 35) * 60, seed=CONFIG.get("seed", 2026))
log("darwin evaluated", res["evaluated"], "confirmed", len(res["confirmed"]))

po, pt = np.zeros(N), IsotonicRegression(out_of_bounds="clip").fit(s, y).predict(st)
for k in range(5):
    tr, va = folds != k, folds == k
    po[va] = IsotonicRegression(out_of_bounds="clip").fit(s[tr], y[tr]).predict(s[va])
po, pt = np.clip(po, 1e-4, 1 - 1e-4), np.clip(pt, 1e-4, 1 - 1e-4)
steps, seen = [dict(step=0, oof_auc=float(roc_auc_score(y, po)))], set()
cur_auc = steps[0]["oof_auc"]
for hh in sorted(res["confirmed"] or [], key=lambda q: -q["confirm"]):
    if sum(st_.get("kept", False) for st_ in steps) >= CONFIG.get("apply_k", 3):
        break
    key = program_key([tuple(a) for a in hh["program"]], D_all)
    sig = (int(key.max()), round(float(hh["confirm"]), 9))
    if sig in seen:
        continue
    seen.add(sig)
    ktr, kte, n = key[:N], key[N:], int(key.max()) + 1
    new_o = po.copy()
    for k in range(5):
        tr, va = folds != k, folds == k
        sm, c = np.bincount(ktr[tr], weights=(y - po)[tr], minlength=n), np.bincount(ktr[tr], minlength=n)
        new_o[va] = po[va] + (sm / (c + search.alpha))[ktr[va]]
    sm, c = np.bincount(ktr, weights=y - po, minlength=n), np.bincount(ktr, minlength=n)
    new_t = pt + (sm / (c + search.alpha))[kte]
    auc_new = float(roc_auc_score(y, new_o))
    keep = auc_new > cur_auc
    steps.append(dict(step=len(steps), program=hh["program"], oof_auc=auc_new, kept=bool(keep)))
    if keep:
        po, pt, cur_auc = np.clip(new_o, 1e-4, 1 - 1e-4), np.clip(new_t, 1e-4, 1 - 1e-4), auc_new
report["darwin_steps"] = steps
report["auc_stack_plus_darwin"] = float(roc_auc_score(y, po))
log("ABLATION  own-only stack %.6f | own+public stack %.6f | + Darwin-Cage %.6f" % (
    report["auc_own_stack"], report["auc_all_stack"], report["auc_stack_plus_darwin"]))
json.dump(report, open(f"{OUT}/report.json", "w"), indent=1, default=str)
use_darwin = report["auc_stack_plus_darwin"] > report["auc_all_stack"] and CONFIG.get("submit_darwin", True)
final = pt if use_darwin else 1 / (1 + np.exp(-test_all))
pd.DataFrame({"id": test["id"], "satisfaction": final}).to_csv(f"{OUT}/submission.csv", index=False)
log("submission.csv written (Darwin-Cage layer %s)" % ("ON" if use_darwin else "OFF"))
