# Darwin-Cage core, verbatim from Agnuxo1/s6e9-honest-ceiling src/honest_ceiling/darwin.py (Apache-2.0),
# minus the __future__ import so it can be inlined in a Kaggle script.
"""Darwin-Cage search: evolve feature programs from noise, rewarded only by the structure they find in the residual.

Nothing is prescribed. A *program* is a composition of up to four atoms (a categorical column, or an integer
transform of a numeric column: remainder, integer division, decimal digit, bit shift, raw value). Its *key* is
the joint value of its atoms. Fitness is out-of-fold: the shrunk group mean of the residual r = y - p(x) is fit
on 8 search folds and correlated with r on the 9th, rotating over the 9 search folds. The last fold is never
seen by the search; a program counts as a discovery only if its correlation there exceeds
`threshold = z / sqrt(n_confirm)` (z = 3 by default: 0.0116 for the 66,866 rows of S6E9's reserved fold).
Every hall-of-fame entry is confirmed once, so the family-wise false-positive rate is about
n_hall * P(Z > z); the report carries `n_hall` for that correction.

`plant` injects a synthetic structure of known size into the labels, which turns the search into a power
analysis: the smallest planted effect it recovers bounds what could have been missed on the real residual.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

MODS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 19, 23, 25, 29, 31, 37, 41, 50, 64, 97, 100, 101, 128, 250,
        256, 500, 1000, 1024, 2500, 5000, 10000]
DIVS = [2, 3, 5, 7, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000, 25000]
DIGITS = [1, 10, 100, 1000, 10000, 100000]
NUM_SCALE = {"Age": 1, "Annual_Income_USD": 1, "Daily_Commute_km": 10, "Number_of_Cars_Owned": 1,
             "Charging_Stations_Near_Home": 1, "Charging_Stations_Near_Work": 1, "Environmental_Concern_Level": 1}
CATS = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level"]


def integer_columns(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Integer view of the 13 raw columns used as the search alphabet."""
    D = {c: pd.factorize(df[c].astype(str))[0].astype(np.int64) for c in CATS}
    for c, s in NUM_SCALE.items():
        D[c] = np.round(df[c].to_numpy(np.float64) * s).astype(np.int64)
    return D


def random_atom(rng: np.random.Generator) -> tuple:
    if rng.random() < 0.25:
        return ("cat", str(rng.choice(CATS)))
    col = str(rng.choice(list(NUM_SCALE)))
    op = str(rng.choice(["mod", "div", "digit", "raw", "pshift"]))
    if op == "mod":
        return ("mod", col, int(rng.choice(MODS)))
    if op == "div":
        return ("div", col, int(rng.choice(DIVS)))
    if op == "digit":
        return ("digit", col, int(rng.choice(DIGITS)))
    if op == "pshift":
        return ("pshift", col, int(rng.integers(1, 7)), int(rng.choice([2, 3, 4, 8])))
    return ("raw", col)


def atom_values(atom: tuple, D: dict[str, np.ndarray]) -> np.ndarray:
    kind, col = atom[0], atom[1]
    v = D[col]
    if kind in ("cat", "raw"):
        return v
    if kind == "mod":
        return v % atom[2]
    if kind == "div":
        return v // atom[2]
    if kind == "digit":
        return (v // atom[2]) % 10
    if kind == "pshift":
        return (v >> atom[2]) % atom[3]
    raise ValueError(kind)


def program_key(prog: list[tuple], D: dict[str, np.ndarray]) -> np.ndarray:
    """Exact joint key of the program's atoms (mixed radix over factorised values)."""
    key = np.zeros(len(next(iter(D.values()))), dtype=np.int64)
    for atom in prog:
        codes = pd.factorize(atom_values(atom, D))[0].astype(np.int64)
        key = key * (int(codes.max()) + 1) + codes
        if key.max() > 2 ** 62:
            key = pd.factorize(key)[0].astype(np.int64)
    return pd.factorize(key)[0].astype(np.int64)


def prog_id(prog) -> str:
    return hashlib.md5(json.dumps([list(a) for a in prog]).encode()).hexdigest()[:12]


def mutate(prog, rng):
    prog = [tuple(a) for a in prog]
    t = rng.random()
    if t < 0.3 and len(prog) < 4:
        prog.append(random_atom(rng))
    elif t < 0.55 and len(prog) > 1:
        prog.pop(int(rng.integers(len(prog))))
    else:
        i = int(rng.integers(len(prog)))
        a = list(prog[i])
        if a[0] in ("mod", "div", "digit") and rng.random() < 0.6:
            a[2] = int(rng.choice({"mod": MODS, "div": DIVS, "digit": DIGITS}[a[0]]))
            prog[i] = tuple(a)
        else:
            prog[i] = random_atom(rng)
    return prog


def crossover(p1, p2, rng):
    child = list(p1[: int(rng.integers(1, len(p1) + 1))]) + list(p2[int(rng.integers(0, len(p2) + 1)):])
    return child[:4] if child else list(p1)


@dataclass
class DarwinSearch:
    D: dict[str, np.ndarray]
    r: np.ndarray
    fold: np.ndarray
    confirm_fold: int = 9
    z: float = 3.0
    threshold: float | None = None
    alpha: float = 5.0
    max_keys: int = 200_000
    hall_min_fit: float = 0.006
    seen: dict = field(default_factory=dict)
    hall: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.threshold is None:
            self.threshold = self.z / float(np.sqrt((self.fold == self.confirm_fold).sum()))

    def fitness(self, key: np.ndarray) -> tuple[float, float]:
        n_keys = int(key.max()) + 1
        cors = []
        for k in range(int(self.fold.max()) + 1):
            if k == self.confirm_fold:
                continue
            fit = (self.fold != k) & (self.fold != self.confirm_fold)
            ho = self.fold == k
            s = np.bincount(key[fit], weights=self.r[fit], minlength=n_keys)
            c = np.bincount(key[fit], minlength=n_keys)
            est = (s / (c + self.alpha))[key[ho]]
            cors.append(0.0 if est.std() < 1e-12 else float(np.corrcoef(est, self.r[ho])[0, 1]))
        return float(np.mean(cors)), float(np.min(cors))

    def confirm(self, key: np.ndarray) -> float:
        fit = self.fold != self.confirm_fold
        ho = self.fold == self.confirm_fold
        n_keys = int(key.max()) + 1
        s = np.bincount(key[fit], weights=self.r[fit], minlength=n_keys)
        c = np.bincount(key[fit], minlength=n_keys)
        est = (s / (c + self.alpha))[key[ho]]
        return 0.0 if est.std() < 1e-12 else float(np.corrcoef(est, self.r[ho])[0, 1])

    def null(self, reps: int = 20, n_groups: int = 500, seed: int = 0) -> tuple[float, float]:
        rng = np.random.default_rng(seed)
        v = [self.fitness(pd.factorize(rng.integers(0, n_groups, len(self.r)))[0])[0] for _ in range(reps)]
        return float(np.mean(v)), float(np.std(v))

    def evaluate(self, prog) -> tuple[float, float, int]:
        pid = prog_id(prog)
        if pid not in self.seen:
            key = program_key(prog, self.D)
            nk = int(key.max()) + 1
            f = (-1.0, -1.0) if nk > self.max_keys else self.fitness(key)
            self.seen[pid] = (f[0], f[1], nk)
            if f[0] > self.hall_min_fit and f[1] > 0:
                self.hall[pid] = dict(program=[list(a) for a in prog], fit_mean=f[0], fit_min=f[1], n_keys=nk,
                                      confirm=self.confirm(key))
        return self.seen[pid]

    def run(self, seconds: float | None = None, max_evals: int | None = None, pop: int = 80, seed: int = 2026,
            log_every: int = 25, log_path: str | Path | None = None, stop_on_confirm: bool = False) -> dict:
        rng = np.random.default_rng(seed)
        population = [[random_atom(rng)] for _ in range(pop)]
        t0, gen = time.time(), 0
        history = []
        while True:
            scored = sorted(((*self.evaluate(p), p) for p in population), key=lambda t: t[0], reverse=True)
            gen += 1
            confirmed = [h for h in self.hall.values() if h["confirm"] > self.threshold]
            if gen % log_every == 0 or confirmed:
                rec = dict(gen=gen, evaluated=len(self.seen), sec=round(time.time() - t0, 1), best_fit=scored[0][0],
                           best_prog=[list(a) for a in scored[0][3]], hall=len(self.hall), confirmed=len(confirmed))
                history.append(rec)
                if log_path:
                    with open(log_path, "a") as f:
                        f.write(json.dumps(rec) + "\n")
            done = (seconds is not None and time.time() - t0 > seconds) or \
                   (max_evals is not None and len(self.seen) >= max_evals) or (stop_on_confirm and confirmed)
            if done:
                break
            elite = [s[3] for s in scored[: max(6, pop // 5)]]
            nxt = list(elite)
            while len(nxt) < pop:
                u = rng.random()
                if u < 0.45:
                    nxt.append(mutate(elite[int(rng.integers(len(elite)))], rng))
                elif u < 0.8:
                    nxt.append(crossover(elite[int(rng.integers(len(elite)))], elite[int(rng.integers(len(elite)))], rng))
                else:
                    nxt.append([random_atom(rng) for _ in range(int(rng.integers(1, 4)))])
            population = nxt
        hall = sorted(self.hall.values(), key=lambda h: -h["confirm"])
        from math import erfc, sqrt
        p_one = 0.5 * erfc(self.z / sqrt(2))
        return dict(evaluated=len(self.seen), generations=gen, seconds=round(time.time() - t0, 1), threshold=self.threshold,
                    z=self.z, n_hall=len(self.hall), familywise_fp_expected=len(self.hall) * p_one,
                    confirmed=[h for h in hall if h["confirm"] > self.threshold], hall=hall[:100], history=history)


def plant(p: np.ndarray, D: dict[str, np.ndarray], program: list[tuple], effect: float, seed: int = 0):
    """Synthetic labels y' ~ Bernoulli(p + effect * s(key)), with s = +-1 drawn per key value.

    Returns (y_planted, shift) so that r' = y' - p contains a structure of known size living on `program`.
    """
    rng = np.random.default_rng(seed)
    key = program_key(program, D)
    sign = rng.choice([-1.0, 1.0], size=int(key.max()) + 1)
    shift = effect * sign[key]
    return rng.binomial(1, np.clip(p + shift, 1e-4, 1 - 1e-4)).astype(np.int8), shift
