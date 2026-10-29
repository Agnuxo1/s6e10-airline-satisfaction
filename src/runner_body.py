# --- S6E10 runner body (CONFIG is injected above by tools/make_kernel.py) ---
import glob
import json
import os
import re
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

OUT = os.environ.get("S6E10_OUT", "/kaggle/working")
IN = os.environ.get("S6E10_IN", "/kaggle/input")
SMOKE = int(os.environ.get("S6E10_SMOKE", CONFIG.get("smoke", 0)))
TARGET = "satisfaction"
CATS = ["Gender", "Customer_Type", "Type_of_Travel", "Class"]
MAPS = {
    "Gender": {"Female": 0, "Male": 1},
    "Customer_Type": {"Loyal Customer": 0, "disloyal Customer": 1},
    "Type_of_Travel": {"Personal Travel": 0, "Business travel": 1},
    "Class": {"Eco": 0, "Eco Plus": 1, "Business": 2},
}
RATINGS = ["Inflight_wifi_service", "Departure_Arrival_time_convenient", "Ease_of_Online_booking", "Gate_location",
           "Food_and_drink", "Online_boarding", "Seat_comfort", "Inflight_entertainment", "On_board_service",
           "Leg_room_service", "Baggage_handling", "Checkin_service", "Cleanliness"]
NFOLDS, FOLD_SEED = 5, 42


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def find(name, must=None):
    hits = sorted(glob.glob(f"{IN}/**/{name}", recursive=True))
    if must:
        hits = [h for h in hits if must in h]
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


def clean_cols(df):
    df.columns = [re.sub(r"[^0-9a-zA-Z]+", "_", c).strip("_") for c in df.columns]
    return df


def prep(df):
    df = clean_cols(df.copy())
    for c in CATS:
        df[c] = df[c].map(MAPS[c]).astype("int8")
    if TARGET in df:
        df[TARGET] = df[TARGET].astype(str).str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0}).astype("int8")
    return df


train = prep(pd.read_csv(find("train.csv", "s6e10")))
test = prep(pd.read_csv(find("test.csv", "s6e10")))
try:
    orig = prep(pd.read_csv(find("data.csv", "aviation")))
except FileNotFoundError:
    orig = None
if SMOKE:
    train, test = train.sample(SMOKE, random_state=0).reset_index(drop=True), test.head(SMOKE // 4)
    orig = None if orig is None else orig.sample(SMOKE // 4, random_state=0).reset_index(drop=True)
FEATS = [c for c in test.columns if c != "id"]
y = train[TARGET].to_numpy()
folds = np.zeros(len(train), dtype=np.int8)
for k, (_, va) in enumerate(StratifiedKFold(NFOLDS, shuffle=True, random_state=FOLD_SEED).split(train, y)):
    folds[va] = k
np.save(f"{OUT}/folds.npy", folds)
log("train", train.shape, "test", test.shape, "orig", None if orig is None else orig.shape, "feats", len(FEATS))


FD = "Flight_Distance"
_CACHE = {}


def route_tables():
    """Label-free route statistics over train+test (counts, route profile) and original-table statistics
    (count and smoothed satisfaction rate per Flight Distance value; original labels only)."""
    if "route" not in _CACHE:
        allx = pd.concat([train[FEATS], test[FEATS]], ignore_index=True)
        g = allx.groupby(FD)
        prof = pd.DataFrame({"fd_cnt": g.size(), "fd_business": g["Class"].apply(lambda s: (s == 2).mean()),
                             "fd_btravel": g["Type_of_Travel"].mean(), "fd_loyal": 1 - g["Customer_Type"].mean(),
                             "fd_age": g["Age"].mean(), "fd_rating": g[RATINGS].mean().mean(1)})
        if orig is not None:
            go = orig.groupby(FD)[TARGET]
            m, prior = 20.0, orig[TARGET].mean()
            prof["fd_og_cnt"] = go.size()
            prof["fd_og_rate"] = (go.sum() + m * prior) / (go.size() + m)
            prof[["fd_og_cnt"]] = prof[["fd_og_cnt"]].fillna(0)
            prof["fd_og_rate"] = prof["fd_og_rate"].fillna(prior)
        _CACHE["route"] = prof.astype("float32")
    return _CACHE["route"]


def og_model_pred():
    """Logit of a LightGBM trained only on the original table, for competition train and test rows."""
    if "og" not in _CACHE:
        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=63, min_child_samples=20,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.7, random_state=0, n_jobs=4,
                               verbose=-1).fit(orig[FEATS], orig[TARGET])
        def lo(X):
            q = np.clip(m.predict_proba(X[FEATS])[:, 1], 1e-6, 1 - 1e-6)
            return np.log(q / (1 - q))
        _CACHE["og"] = (lo(train).astype("float32"), lo(test).astype("float32"))
    return _CACHE["og"]


def add_features(df, fs, which=None):
    """Named feature sets. Everything here is label-free with respect to competition labels: 'route' uses
    train+test counts/profiles and original-table labels; 'og' uses a model trained on the original only.
    Fold-safe target encodings of competition labels are added inside the fold loop (exp 'te')."""
    X = df[FEATS].copy()
    if "eng" in fs:
        r = X[RATINGS].astype("float32")
        X["rating_mean"] = r.mean(1)
        X["rating_min"] = r.min(1)
        X["rating_zeros"] = (r == 0).sum(1)
        X["delay_diff"] = X["Arrival_Delay_in_Minutes"] - X["Departure_Delay_in_Minutes"]
        X["delay_total"] = X["Arrival_Delay_in_Minutes"].fillna(X["Departure_Delay_in_Minutes"]) +             X["Departure_Delay_in_Minutes"]
    if "route" in fs:
        X = X.join(route_tables(), on=FD)
    if "og" in fs:
        assert which in ("train", "test"), "og features are defined for competition rows only"
        X["og_pred"] = og_model_pred()[0 if which == "train" else 1]
    return X


def te_fold(col, Xtr, ytr, Xapply, m=10.0, inner=5, seed=0):
    """Target encoding of one column: inner cross-fit for the training rows, full-fit for the applied rows."""
    prior = ytr.mean()
    def fit_apply(keys_fit, y_fit, keys_app):
        d = pd.DataFrame({"k": keys_fit, "y": y_fit}).groupby("k")["y"].agg(["sum", "count"])
        enc = (d["sum"] + m * prior) / (d["count"] + m)
        return pd.Series(keys_app).map(enc).fillna(prior).to_numpy("float32")
    ktr = Xtr[col].to_numpy()
    tr_enc = np.zeros(len(ktr), dtype="float32")
    for a, b in StratifiedKFold(inner, shuffle=True, random_state=seed).split(ktr, ytr):
        tr_enc[b] = fit_apply(ktr[a], ytr[a], ktr[b])
    return tr_enc, [fit_apply(ktr, ytr, X[col].to_numpy()) for X in Xapply]


def cat_list(X, spec):
    if spec == "none":
        return []
    if spec == "base":
        return list(CATS)
    if spec == "ratings":
        return list(CATS) + RATINGS
    if spec == "all":
        return [c for c in X.columns if c in FEATS]
    raise ValueError(spec)


def fit_predict(model, params, Xtr, ytr, Xva, yva, Xte, cats, seed, use_gpu):
    p = dict(params)
    if model == "cat":
        from catboost import CatBoostClassifier, Pool
        cast = lambda X: X.assign(**{c: X[c].fillna(-1).astype("int64") for c in cats})
        Xtr, Xva, Xte = cast(Xtr), cast(Xva), cast(Xte)
        m = CatBoostClassifier(iterations=p.pop("iterations", 6000), learning_rate=p.pop("learning_rate", 0.06),
                               depth=p.pop("depth", 7), random_seed=seed, eval_metric="AUC" if not use_gpu else "Logloss",
                               od_type="Iter", od_wait=p.pop("od_wait", 300), verbose=0,
                               task_type="GPU" if use_gpu else "CPU", thread_count=4, **p)
        m.fit(Pool(Xtr, ytr, cat_features=cats), eval_set=Pool(Xva, yva, cat_features=cats), use_best_model=True)
        return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1], int(m.get_best_iteration())
    if model == "xgb":
        import xgboost as xgb
        cast = lambda X: X.assign(**{c: X[c].astype("category") for c in cats})
        Xtr, Xva, Xte = cast(Xtr), cast(Xva), cast(Xte)
        m = xgb.XGBClassifier(n_estimators=p.pop("n_estimators", 8000), learning_rate=p.pop("learning_rate", 0.04),
                              max_depth=p.pop("max_depth", 7), tree_method="hist",
                              device="cuda" if use_gpu else "cpu", eval_metric="auc",
                              early_stopping_rounds=p.pop("early_stopping_rounds", 300),
                              enable_categorical=True, random_state=seed, n_jobs=4, **p)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1], int(m.best_iteration)
    if model == "lgb":
        import lightgbm as lgb
        es = p.pop("es", 300)
        m = lgb.LGBMClassifier(n_estimators=p.pop("n_estimators", 10000), learning_rate=p.pop("learning_rate", 0.03),
                               num_leaves=p.pop("num_leaves", 63), random_state=seed, n_jobs=4, verbose=-1, **p)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc", categorical_feature=cats or "auto",
              callbacks=[lgb.early_stopping(es, verbose=False)])
        return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1], int(m.best_iteration_)
    raise ValueError(model)


def gpu_available():
    try:
        import subprocess
        return subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
    except Exception:
        return False


USE_GPU = gpu_available() and not os.environ.get("S6E10_NOGPU")
SPACES = {
    "cat": lambda t: dict(depth=t.suggest_int("depth", 5, 10), learning_rate=t.suggest_float("learning_rate", 0.03, 0.15, log=True),
                          l2_leaf_reg=t.suggest_float("l2_leaf_reg", 0.5, 30, log=True),
                          random_strength=t.suggest_float("random_strength", 0.0, 5.0),
                          bagging_temperature=t.suggest_float("bagging_temperature", 0.0, 3.0),
                          border_count=t.suggest_categorical("border_count", [64, 128, 254])),
    "xgb": lambda t: dict(max_depth=t.suggest_int("max_depth", 4, 12), learning_rate=t.suggest_float("learning_rate", 0.02, 0.1, log=True),
                          min_child_weight=t.suggest_float("min_child_weight", 1, 64, log=True),
                          subsample=t.suggest_float("subsample", 0.6, 1.0), colsample_bytree=t.suggest_float("colsample_bytree", 0.3, 1.0),
                          reg_lambda=t.suggest_float("reg_lambda", 0.01, 30, log=True), reg_alpha=t.suggest_float("reg_alpha", 0.001, 10, log=True),
                          gamma=t.suggest_float("gamma", 0.0, 2.0), max_bin=t.suggest_categorical("max_bin", [128, 256, 512])),
    "lgb": lambda t: dict(num_leaves=t.suggest_int("num_leaves", 15, 511, log=True), learning_rate=t.suggest_float("learning_rate", 0.02, 0.08, log=True),
                          min_child_samples=t.suggest_int("min_child_samples", 5, 200, log=True), subsample=t.suggest_float("subsample", 0.6, 1.0),
                          subsample_freq=1, colsample_bytree=t.suggest_float("colsample_bytree", 0.3, 1.0),
                          reg_lambda=t.suggest_float("reg_lambda", 0.001, 30, log=True), reg_alpha=t.suggest_float("reg_alpha", 0.001, 10, log=True),
                          min_split_gain=t.suggest_float("min_split_gain", 0.0, 0.5), max_bin=t.suggest_categorical("max_bin", [127, 255, 511])),
}


def run_tune(tc):
    """Optuna TPE on one fold (train folds != k, validate fold k); same features, cats and data as the 5-fold runs."""
    import optuna
    model, fs, k = tc["model"], tc.get("features", "base"), tc.get("fold", 0)
    Xtr_all = add_features(train, fs, "train")
    Xo = add_features(orig, fs, "orig") if (tc.get("use_orig") and orig is not None) else None
    cats = cat_list(Xtr_all, tc.get("cats", "base" if model != "lgb" else "none"))
    tr, va = folds != k, folds == k
    Xtr, ytr, Xva = Xtr_all[tr].copy(), y[tr], Xtr_all[va].copy()
    for c in tc.get("te", []):
        Xtr[f"te_{c}"], (Xva[f"te_{c}"],) = te_fold(c, Xtr, ytr, [Xva])
    if Xo is not None:
        Xtr, ytr = pd.concat([Xtr, Xo], ignore_index=True), np.concatenate([ytr, orig[TARGET].to_numpy()])
    gpu = USE_GPU and model != "lgb"
    fixed = tc.get("fixed", {})

    def objective(t):
        params = {**SPACES[model](t), **fixed}
        pv, _, it = fit_predict(model, params, Xtr, ytr, Xva, y[va], Xva.head(1), cats, 0, gpu)
        t.set_user_attr("iter", it)
        return roc_auc_score(y[va], pv)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=tc.get("seed", 0)))
    if tc.get("enqueue"):
        study.enqueue_trial(tc["enqueue"])
    study.optimize(objective, n_trials=tc.get("n_trials", 200), timeout=tc["minutes"] * 60,
                   callbacks=[lambda s, t: log(tc["name"], "trial", t.number, "auc %.6f" % (t.value or 0), "best %.6f" % s.best_value)])
    study.trials_dataframe().to_csv(f"{OUT}/trials_{tc['name']}.csv", index=False)
    best = dict(name=tc["name"], model=model, fold=k, best_auc=study.best_value, best_params=study.best_params,
                best_iter=study.best_trial.user_attrs.get("iter"), n_trials=len(study.trials), fixed=fixed)
    json.dump(best, open(f"{OUT}/best_{tc['name']}.json", "w"), indent=1)
    log("TUNE DONE", json.dumps(best))


for tc in CONFIG.get("tune", []):
    run_tune(tc)
results = []
for exp in CONFIG.get("experiments", []):
    name, model, fs = exp["name"], exp["model"], exp.get("features", "base")
    Xtr_all, Xte = add_features(train, fs, "train"), add_features(test, fs, "test")
    Xo = add_features(orig, fs, "orig") if (exp.get("use_orig") and orig is not None) else None
    assert Xo is None or not ({"route", "og"} & set(fs.split("+"))), "route/og views are not defined for original rows"
    te_cols = exp.get("te", [])
    if Xo is not None and exp.get("orig_flag"):
        Xtr_all["is_orig"], Xte["is_orig"], Xo["is_orig"] = 0, 0, 1
    cats = cat_list(Xtr_all, exp.get("cats", "base" if model != "lgb" else "none"))
    gpu = USE_GPU and exp.get("gpu", True) and model != "lgb"
    for seed in exp.get("seeds", [0]):
        t0 = time.time()
        oof = np.zeros(len(train), dtype=np.float64)
        pte = np.zeros(len(test), dtype=np.float64)
        fold_auc, iters = [], []
        for k in range(NFOLDS):
            tr, va = folds != k, folds == k
            Xtr, ytr, Xva, Xtek = Xtr_all[tr].copy(), y[tr], Xtr_all[va].copy(), Xte.copy()
            for c in te_cols:
                Xtr[f"te_{c}"], (Xva[f"te_{c}"], Xtek[f"te_{c}"]) = te_fold(c, Xtr, ytr, [Xva, Xtek], seed=seed)
            if Xo is not None:
                Xtr = pd.concat([Xtr, Xo], ignore_index=True)
                ytr = np.concatenate([ytr, orig[TARGET].to_numpy()])
            pv, pt, it = fit_predict(model, exp.get("params", {}), Xtr, ytr, Xva, y[va], Xtek, cats, seed, gpu)
            oof[va] = pv
            pte += pt / NFOLDS
            fold_auc.append(roc_auc_score(y[va], pv))
            iters.append(it)
            log(name, "seed", seed, "fold", k, "auc %.6f" % fold_auc[-1], "iter", it)
        auc = roc_auc_score(y, oof)
        tag = f"{name}_s{seed}"
        np.save(f"{OUT}/oof_{tag}.npy", oof.astype(np.float32))
        np.save(f"{OUT}/test_{tag}.npy", pte.astype(np.float32))
        rec = dict(name=name, seed=seed, model=model, features=fs, cats=exp.get("cats"), te=te_cols, use_orig=bool(Xo is not None),
                   params=exp.get("params", {}), gpu=gpu, oof_auc=auc, fold_auc=fold_auc,
                   fold_mean=float(np.mean(fold_auc)), fold_std=float(np.std(fold_auc)), iters=iters,
                   sec=round(time.time() - t0, 1))
        results.append(rec)
        log("DONE", tag, "oof auc %.6f" % auc, "sec", rec["sec"])
        json.dump(results, open(f"{OUT}/results.json", "w"), indent=1)
        if exp.get("submit_file"):
            pd.DataFrame({"id": test["id"], TARGET: pte}).to_csv(f"{OUT}/submission_{tag}.csv", index=False)
log("ALL DONE")
