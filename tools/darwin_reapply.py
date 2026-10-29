"""Re-apply the confirmed Darwin-Cage programs of a finished search with a rank-preserving (Platt) calibration.

The kernel's apply step calibrates with isotonic regression, whose ties cost ~0.0001 AUC before any correction.
Here the base is mapped with a logistic fit on the logit of the scores (strictly monotone, AUC unchanged), then the
shrunk residual group means of each confirmed program are added fold-safely (same 5 folds as level 1).
usage: python tools/darwin_reapply.py <run_dir> <oof_scores.npy> <test_scores.npy> [--k 3] [--logit]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from darwin_core import program_key  # noqa: E402

CATS = ["Gender", "Customer_Type", "Type_of_Travel", "Class", "Inflight_wifi_service",
        "Departure_Arrival_time_convenient", "Ease_of_Online_booking", "Gate_location", "Food_and_drink",
        "Online_boarding", "Seat_comfort", "Inflight_entertainment", "On_board_service", "Leg_room_service",
        "Baggage_handling", "Checkin_service", "Cleanliness"]
NUMS = ["Age", "Flight_Distance", "Departure_Delay_in_Minutes", "Arrival_Delay_in_Minutes"]
ALPHA = 5.0


def clean(df):
    df.columns = [re.sub(r"[^0-9a-zA-Z]+", "_", c).strip("_") for c in df.columns]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("oof")
    ap.add_argument("test")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--logit", action="store_true", help="scores are already logits")
    a = ap.parse_args()
    run = Path(a.run_dir)
    tr_df = clean(pd.read_csv(ROOT / "data" / "train.csv"))
    te_df = clean(pd.read_csv(ROOT / "data" / "test.csv"))
    y = tr_df["satisfaction"].astype(str).str.lower().map({"true": 1, "false": 0}).to_numpy()
    N = len(y)
    both = pd.concat([tr_df.drop(columns=["satisfaction"]), te_df], ignore_index=True)
    D = {c: pd.factorize(both[c].astype(str))[0].astype(np.int64) for c in CATS}
    D |= {c: np.round(both[c].fillna(-1).to_numpy(np.float64)).astype(np.int64) for c in NUMS}
    folds = np.load(ROOT / "runs" / "s6e10-e001-cat-base" / "folds.npy").astype(np.int64)
    so, st = np.load(a.oof).astype(np.float64), np.load(a.test).astype(np.float64)
    if not a.logit:
        so, st = (np.log(np.clip(v, 1e-7, 1 - 1e-7) / (1 - np.clip(v, 1e-7, 1 - 1e-7))) for v in (so, st))
    po = np.zeros(N)
    for k in range(5):
        tr, va = folds != k, folds == k
        po[va] = LogisticRegression().fit(so[tr, None], y[tr]).predict_proba(so[va, None])[:, 1]
    pt = LogisticRegression().fit(so[:, None], y).predict_proba(st[:, None])[:, 1]
    res = json.loads((run / "darwin_result.json").read_text())
    base = float(roc_auc_score(y, so))
    steps, seen, cur = [dict(step=0, oof_auc=float(roc_auc_score(y, po)))], set(), None
    cur = steps[0]["oof_auc"]
    for h in sorted(res.get("confirmed") or [], key=lambda q: -q["confirm"]):
        if sum(s.get("kept", False) for s in steps) >= a.k:
            break
        key = program_key([tuple(x) for x in h["program"]], D)
        sig = (int(key.max()), round(float(h["confirm"]), 9))
        if sig in seen:
            continue
        seen.add(sig)
        ktr, kte, n = key[:N], key[N:], int(key.max()) + 1
        new_o = po.copy()
        for k in range(5):
            tr, va = folds != k, folds == k
            sm, c = np.bincount(ktr[tr], weights=(y - po)[tr], minlength=n), np.bincount(ktr[tr], minlength=n)
            new_o[va] = po[va] + (sm / (c + ALPHA))[ktr[va]]
        sm, c = np.bincount(ktr, weights=y - po, minlength=n), np.bincount(ktr, minlength=n)
        new_t = pt + (sm / (c + ALPHA))[kte]
        auc = float(roc_auc_score(y, new_o))
        keep = auc > cur
        fold_gain = [float(roc_auc_score(y[folds == k], new_o[folds == k]) - roc_auc_score(y[folds == k], po[folds == k]))
                     for k in range(5)]
        steps.append(dict(step=len(steps), program=h["program"], oof_auc=auc, fold_gain=fold_gain, kept=keep))
        if keep:
            po, pt, cur = new_o, new_t, auc
    out = dict(run=str(run), base_auc=base, calib="platt", steps=steps, final_auc=cur, gain=cur - base)
    (run / "darwin_reapply_platt.json").write_text(json.dumps(out, indent=1, default=str))
    np.save(run / "oof_platt_darwin.npy", np.clip(po, 1e-6, 1 - 1e-6).astype(np.float32))
    np.save(run / "test_platt_darwin.npy", np.clip(pt, 1e-6, 1 - 1e-6).astype(np.float32))
    print(json.dumps(dict(base_auc=base, final_auc=cur, gain=cur - base,
                          steps=[(s["step"], round(s["oof_auc"], 6), s.get("kept")) for s in steps])))


if __name__ == "__main__":
    main()
