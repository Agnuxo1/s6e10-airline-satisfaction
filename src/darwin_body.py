# --- S6E10 Darwin-Cage kernel body (CONFIG injected above; darwin_core.py inlined above this body) ---
# Search alphabet adapted to S6E10: the 4 categoricals and 13 ratings are categorical atoms; Age, Flight Distance
# and both delays are integer atoms (NaN arrival delay -> -1). Folds: StratifiedKFold(10, seed 42), fold 9 reserved.
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

OUT = os.environ.get("S6E10_OUT", "/kaggle/working")
IN = os.environ.get("S6E10_IN", "/kaggle/input")
CATS = ["Gender", "Customer_Type", "Type_of_Travel", "Class", "Inflight_wifi_service",
        "Departure_Arrival_time_convenient", "Ease_of_Online_booking", "Gate_location", "Food_and_drink",
        "Online_boarding", "Seat_comfort", "Inflight_entertainment", "On_board_service", "Leg_room_service",
        "Baggage_handling", "Checkin_service", "Cleanliness"]
NUM_SCALE = {"Age": 1, "Flight_Distance": 1, "Departure_Delay_in_Minutes": 1, "Arrival_Delay_in_Minutes": 1}
# These rebind the module globals that random_atom() of the inlined core reads.


def find(name, must=None):
    hits = sorted(h for h in glob.glob(f"{IN}/**/{name}", recursive=True) if not must or must in h)
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


def integer_columns_s6e10(df):
    D = {c: pd.factorize(df[c].astype(str))[0].astype(np.int64) for c in CATS}
    for c, s in NUM_SCALE.items():
        D[c] = np.round(df[c].fillna(-1).to_numpy(np.float64) * s).astype(np.int64)
    return D


def calibrated_probability(scores, y, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    h = len(y) // 2
    p = np.zeros(len(y))
    for a, b in ((idx[:h], idx[h:]), (idx[h:], idx[:h])):
        p[b] = IsotonicRegression(out_of_bounds="clip").fit(scores[a], y[a]).predict(scores[b])
    return np.clip(p, 1e-4, 1 - 1e-4)


def correction_gain(key, r, p, y, fold, confirm_fold, alpha):
    """AUC on the reserved fold of p + shrunk residual group mean fit on the search folds."""
    fit, ho = fold != confirm_fold, fold == confirm_fold
    n = int(key.max()) + 1
    s = np.bincount(key[fit], weights=r[fit], minlength=n)
    c = np.bincount(key[fit], minlength=n)
    adj = (s / (c + alpha))[key[ho]]
    # no tuned weight: the shrunk residual mean is added to the calibrated probability as it is
    return dict(base=roc_auc_score(y[ho], p[ho]), corrected=roc_auc_score(y[ho], p[ho] + adj))


train = pd.read_csv(find("train.csv", "s6e10"))
train.columns = [re.sub(r"[^0-9a-zA-Z]+", "_", c).strip("_") for c in train.columns]
y = train["satisfaction"].astype(str).str.lower().map({"true": 1, "false": 0}).to_numpy()
scores = np.load(find(CONFIG["scores_file"]))
assert len(scores) == len(y)
p = calibrated_probability(scores.astype(np.float64), y)
r = (y - p).astype(np.float64)
fold = np.zeros(len(y), dtype=np.int64)
for k, (_, va) in enumerate(StratifiedKFold(10, shuffle=True, random_state=42).split(np.zeros(len(y)), y)):
    fold[va] = k
D = integer_columns_s6e10(train)
search = DarwinSearch(D, r, fold, z=CONFIG.get("z", 3.0))
null = search.null()
print("scores AUC %.6f  calibrated AUC %.6f  threshold %.5f  null %.5f+-%.5f"
      % (roc_auc_score(y, scores), roc_auc_score(y, p), search.threshold, *null), flush=True)
res = search.run(seconds=CONFIG["minutes"] * 60, seed=CONFIG.get("seed", 2026), log_path=f"{OUT}/darwin_log.jsonl")
res["null_mean"], res["null_std"] = null
res["scores_file"], res["scores_auc"] = CONFIG["scores_file"], float(roc_auc_score(y, scores))
res["gain_on_reserved_fold"] = []
for h in (res["confirmed"] or res["hall"][:5]):
    key = program_key([tuple(a) for a in h["program"]], D)
    res["gain_on_reserved_fold"].append(dict(program=h["program"], confirm=h["confirm"],
                                             **correction_gain(key, r, p, y, fold, 9, search.alpha)))
json.dump(res, open(f"{OUT}/darwin_result.json", "w"), indent=1, default=str)
print(json.dumps({k: res[k] for k in ("evaluated", "generations", "seconds", "threshold", "n_hall")}
                 | {"confirmed": len(res["confirmed"]), "gain": res["gain_on_reserved_fold"][:3]}, default=str))

# --- apply: turn the confirmed programs into fold-safe OOF and test predictions (for the blend ablation) ---
# Same 5 folds as the level-1 runs (StratifiedKFold(5, seed 42)). For each fold the isotonic calibration and the
# shrunk residual group means are fit on the other 4 folds only; test uses all train rows. Programs are applied
# one after another on the remaining residual, deduplicated by the key partition they induce.
if CONFIG.get("apply"):
    test_df = pd.read_csv(find("test.csv", "s6e10"))
    test_df.columns = [re.sub(r"[^0-9a-zA-Z]+", "_", c).strip("_") for c in test_df.columns]
    st = np.load(find(CONFIG["scores_test_file"])).astype(np.float64)
    N = len(y)
    DA = integer_columns_s6e10(pd.concat([train.drop(columns=["satisfaction"]), test_df], ignore_index=True))
    f5 = np.zeros(N, dtype=np.int64)
    for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(N), y)):
        f5[va] = k
    s = scores.astype(np.float64)
    po, pt = np.zeros(N), IsotonicRegression(out_of_bounds="clip").fit(s, y).predict(st)
    for k in range(5):
        tr, va = f5 != k, f5 == k
        po[va] = IsotonicRegression(out_of_bounds="clip").fit(s[tr], y[tr]).predict(s[va])
    po, pt = np.clip(po, 1e-4, 1 - 1e-4), np.clip(pt, 1e-4, 1 - 1e-4)
    applied, seen, steps = [], set(), [dict(step=0, oof_auc=float(roc_auc_score(y, po)))]
    cur_auc = steps[0]["oof_auc"]
    pool = sorted(res["confirmed"] or [], key=lambda h: -h["confirm"])
    for h in pool:
        if len(applied) >= CONFIG.get("apply_k", 3):
            break
        key = program_key([tuple(a) for a in h["program"]], DA)
        sig = (int(key.max()), round(float(h["confirm"]), 9))
        if sig in seen:
            continue
        seen.add(sig)
        ktr, kte = key[:N], key[N:]
        n = int(key.max()) + 1
        new_o = po.copy()
        for k in range(5):
            tr, va = f5 != k, f5 == k
            sm = np.bincount(ktr[tr], weights=(y - po)[tr], minlength=n)
            c = np.bincount(ktr[tr], minlength=n)
            new_o[va] = po[va] + (sm / (c + search.alpha))[ktr[va]]
        sm = np.bincount(ktr, weights=y - po, minlength=n)
        c = np.bincount(ktr, minlength=n)
        new_t = pt + (sm / (c + search.alpha))[kte]
        auc_new = float(roc_auc_score(y, new_o))
        keep = auc_new > cur_auc
        steps.append(dict(step=len(steps), program=h["program"], confirm=h["confirm"], oof_auc=auc_new, kept=keep))
        if keep:
            po, pt, cur_auc = np.clip(new_o, 1e-4, 1 - 1e-4), np.clip(new_t, 1e-4, 1 - 1e-4), auc_new
            applied.append(h["program"])
    tag = CONFIG.get("apply_tag", "D_darwin")
    np.save(f"{OUT}/oof_{tag}.npy", po.astype(np.float32))
    np.save(f"{OUT}/test_{tag}.npy", pt.astype(np.float32))
    np.save(f"{OUT}/folds.npy", f5.astype(np.int8))
    json.dump(dict(base_auc=float(roc_auc_score(y, scores)), steps=steps, applied=applied),
              open(f"{OUT}/darwin_apply.json", "w"), indent=1, default=str)
    pd.DataFrame({"id": test_df["id"], "satisfaction": pt}).to_csv(f"{OUT}/submission_{tag}.csv", index=False)
    print("APPLY", json.dumps(steps, default=str), flush=True)
