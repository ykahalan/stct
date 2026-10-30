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
import json
import os
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


def load_train(task, name):
    """Training split only."""
    if task == "classification":
        from aeon.datasets import load_classification as loader
    else:
        from aeon.datasets import load_regression as loader
    X, y = loader(name, split="train")
    X = bench._prep(X)
    return X, (y if task == "classification" else np.asarray(y, float))


def cv_score(task, X, y, flags):
    F = np.nan_to_num(SP.stct_plus(X, **flags))
    if task == "classification":
        est = ExtraTreesClassifier(n_estimators=200, random_state=bench.RNG, n_jobs=-1)
        counts = np.unique(y, return_counts=True)[1]
        cv = (StratifiedKFold if counts.max() >= 5 else KFold)(5, shuffle=True, random_state=bench.RNG)
        return float((cross_val_predict(est, F, y, cv=cv) == y).mean())
    est = ExtraTreesRegressor(n_estimators=200, random_state=bench.RNG, n_jobs=-1)
    pred = cross_val_predict(est, F, y, cv=KFold(5, shuffle=True, random_state=bench.RNG))
    return float(np.sqrt(((pred - y) ** 2).mean()))


def n_features(flags):
    return SP.stct_plus(np.random.default_rng(0).standard_normal((2, 1024)), **flags).shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classification", default=None, help="comma-separated override, smoke tests only")
    ap.add_argument("--regression", default=None, help="comma-separated override, smoke tests only")
    ap.add_argument("--prefix", default="results/algo")
    ap.add_argument("--base", default="", help="comma-separated changes already adopted")
    ap.add_argument("--candidates", default=",".join(CANDIDATES))
    a = ap.parse_args()
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
    cache = {}

    def score(flags):
        key = tuple(sorted(k for k, v in flags.items() if v))
        if key not in cache:
            cache[key] = {t: pd.Series({n: cv_score(t, X, y, flags) for n, (X, y) in d.items()})
                          for t, d in data.items()}
            print(f"scored {key}", flush=True)
        return cache[key]

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
    step = 0
    while True:
        step += 1
        trials = []
        for cand in candidates:
            if cand in chosen:
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
