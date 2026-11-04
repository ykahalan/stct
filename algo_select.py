"""Choose algorithm changes for STCT+ using training splits only.

    python algo_select.py

Every candidate is scored by 5-fold cross-validation inside the training split
of every dataset.  This script never loads a test split, so its decisions cannot
depend on test scores.  Candidates are added greedily by a rule fixed in advance.
A candidate passes if it helps one task without hurting the other:

* classification: mean accuracy up by at least 0.002 with more datasets improved
  than worsened, and regression geometric-mean RMSE ratio at most 1.002; or
* regression: ratio at most 0.995 with more datasets improved than worsened, and
  classification accuracy down by at most 0.0005.

Candidates above MAX_FEATURES at a long series are skipped.  Results go to
results/algo_select.csv and the choice to results/algo_selected.json.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

import bench
import stct_plus as SP

warnings.filterwarnings("ignore")
MAX_FEATURES = 600
CANDIDATES = ("orthonormal", "shape", "energy", "cross", "second")
# candidates that are not a single boolean option of stct_plus
NAMED = {"fixed_swap": dict(fractions=(1 / 8, 1 / 4), sample_windows=(8, 32)),
         "fixed_add": dict(sample_windows=(16,)),
         "resid_sign": dict(resid_kind="sign"),
         "resid_peak": dict(resid_kind="peak")}
CONFLICTS = {"fixed_swap": "fixed_add", "fixed_add": "fixed_swap", "resid_sign": "resid_peak", "resid_peak": "resid_sign"}


RECIPE = {}  # keyword arguments of the base recipe, set from the command line


def expand(flags):
    """Turn candidate names into stct_plus keyword arguments."""
    kw = dict(RECIPE)
    for name, on in flags.items():
        if on:
            kw.update(NAMED.get(name, {name: True}))
    return kw


def load_train(task, name):
    """Training split only."""
    if task == "classification":
        from aeon.datasets import load_classification as loader
    else:
        from aeon.datasets import load_regression as loader
    X, y = loader(name, split="train")
    X = bench._prep(X)
    return X, (y if task == "classification" else np.asarray(y, float))


def cv_score(task, X, y, flags, seed=bench.RNG):
    F = np.nan_to_num(SP.stct_plus(X, **expand(flags)))
    if task == "classification":
        est = ExtraTreesClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
        counts = np.unique(y, return_counts=True)[1]
        cv = (StratifiedKFold if counts.max() >= 5 else KFold)(5, shuffle=True, random_state=seed)
        return float((cross_val_predict(est, F, y, cv=cv) == y).mean())
    est = ExtraTreesRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
    pred = cross_val_predict(est, F, y, cv=KFold(5, shuffle=True, random_state=seed))
    return float(np.sqrt(((pred - y) ** 2).mean()))


def with_retry(fn, *args, attempts=30, wait=120):
    """Wait and retry when the machine is short of memory."""
    for attempt in range(attempts):
        try:
            return fn(*args)
        except MemoryError:
            gc.collect()
            print(f"memory short, waiting {wait}s (attempt {attempt + 1})", flush=True)
            time.sleep(wait)
    return fn(*args)


def n_features(flags):
    return SP.stct_plus(np.random.default_rng(0).standard_normal((2, 1024)), **expand(flags)).shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classification", default=None, help="comma-separated override, smoke tests only")
    ap.add_argument("--regression", default=None, help="comma-separated override, smoke tests only")
    ap.add_argument("--prefix", default="results/algo")
    ap.add_argument("--base", default="", help="comma-separated changes already adopted")
    ap.add_argument("--candidates", default=",".join(CANDIDATES))
    ap.add_argument("--recipe", default="ladder", choices=["ladder", "paper"])
    ap.add_argument("--noise-seeds", default="", help="comma-separated seeds: rescore the base to measure noise")
    a = ap.parse_args()
    if a.recipe == "paper":
        RECIPE["paper"] = True
    candidates = a.candidates.split(",")
    names = {"classification": a.classification.split(",") if a.classification else bench.UCR,
             "regression": a.regression.split(",") if a.regression else bench.TSER}
    data = {}
    for task, ns in names.items():
        data[task] = {}
        for n in ns:
            try:
                data[task][n] = load_train(task, n)
            except Exception as e:  # ragged datasets
                print(f"[skip] {n}: {e}", flush=True)

    csv = f"{a.prefix}_select.csv"
    if os.path.exists(csv):
        os.remove(csv)
    # per-dataset scores persist on disk, so an interrupted run resumes where it stopped
    scores_csv = f"{a.prefix}_scores.csv"
    known = {}
    if os.path.exists(scores_csv):
        prev = pd.read_csv(scores_csv, keep_default_na=False)
        known = {(r.key, r.task, r.dataset): r.score for r in prev.itertuples()}

    def score(flags):
        key = "+".join(sorted(k for k, v in flags.items() if v)) or "base"
        out = {}
        for t, d in data.items():
            vals = {}
            for n, (X, y) in d.items():
                if (key, t, n) not in known:
                    known[key, t, n] = with_retry(cv_score, t, X, y, flags)
                    row = dict(key=key, task=t, dataset=n, score=known[key, t, n])
                    pd.DataFrame([row]).to_csv(scores_csv, mode="a", header=not os.path.exists(scores_csv), index=False)
                vals[n] = known[key, t, n]
            out[t] = pd.Series(vals)
        print(f"scored {key}", flush=True)
        return out

    def compare(cand, base):
        d = cand["classification"] - base["classification"]
        r = (cand["regression"] + 1e-12) / (base["regression"] + 1e-12)
        return dict(acc_gain=float(d.mean()), acc_better=int((d > 0).sum()), acc_worse=int((d < 0).sum()),
                    rmse_ratio=float(np.exp(np.log(r).mean())), rmse_better=int((r < 1).sum()),
                    rmse_worse=int((r > 1).sum()))

    def passes(c):
        cls_win = c["acc_gain"] >= 0.002 and c["acc_better"] > c["acc_worse"] and c["rmse_ratio"] <= 1.002
        reg_win = c["rmse_ratio"] <= 0.995 and c["rmse_better"] > c["rmse_worse"] and c["acc_gain"] >= -0.0005
        return cls_win or reg_win

    chosen = {b: True for b in a.base.split(",") if b}
    base = score(chosen)
    for seed in [int(x) for x in a.noise_seeds.split(",") if x]:
        again = {t: pd.Series({n: with_retry(cv_score, t, X, y, chosen, seed) for n, (X, y) in d.items()})
                 for t, d in data.items()}
        c = compare(again, base)
        c.update(seed=seed)
        noise = f"{a.prefix}_noise.csv"
        pd.DataFrame([c]).to_csv(noise, mode="a", header=not os.path.exists(noise), index=False)
        print("NOISE", json.dumps(c), flush=True)
    step = 0
    while True:
        step += 1
        trials = []
        for cand in candidates:
            if cand in chosen or CONFLICTS.get(cand) in chosen:
                continue
            flags = dict(chosen, **{cand: True})
            feats = n_features(flags)
            if feats > MAX_FEATURES:
                print(f"step {step} {cand}: {feats} features, over budget", flush=True)
                continue
            c = compare(score(flags), base)
            c.update(step=step, candidate=cand, features=feats, passes=passes(c))
            trials.append(c)
            pd.DataFrame([c]).to_csv(csv, mode="a", header=not os.path.exists(csv), index=False)
            print(json.dumps(c), flush=True)
        ok = [t for t in trials if t["passes"]]
        if not ok:
            break
        best = max(ok, key=lambda t: (t["acc_gain"], -t["rmse_ratio"]))
        chosen[best["candidate"]] = True
        base = score(chosen)
        print(f"ADOPT {best['candidate']}", flush=True)
    out = dict(chosen=sorted(chosen), features=n_features(chosen), train_cv_accuracy=float(base["classification"].mean()),
               classification_datasets=len(data["classification"]), regression_datasets=len(data["regression"]))
    with open(f"{a.prefix}_selected.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("FINAL", json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
