"""Test winner-count kernels on Chebyshev coefficients, using training splits only.

    python fast_compete.py
    python fast_compete.py --seeds 1,2

Each variant is scored with ridge alone and averaged with the fast recipe's
forest, against the forest alone and the threshold kernels on the same folds.
Winner counts are computed at every series length.  Results persist in
results/fast_compete_scores.csv.  No test split is loaded.
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.model_selection import KFold, StratifiedKFold

import bench
import stct_fast as SF
from algo_select import load_train, with_retry
from fast_search import oof_ridge
from vote_test import MIN_LENGTH

warnings.filterwarnings("ignore")
OUT = "results/fast_compete_scores.csv"
FOREST = {"classification": dict(n_estimators=100, max_features=0.1),
          "regression": dict(n_estimators=50, max_features=0.5, min_samples_leaf=2)}
# name: (largest group count, inputs, keep mean winning scores)
VARIANTS = {"R16": (16, (False,), True), "S16": (16, (True,), True), "RS8": (8, (False, True), True),
            "RS16": (16, (False, True), True), "RS16c": (16, (False, True), False)}


def score_dataset(task, X, y, seed):
    cls = task == "classification"
    T = X.shape[1]
    F = np.nan_to_num(SF.features(X)[0])
    if cls:
        stratified = np.unique(y, return_counts=True)[1].max() >= 5
        folds = list((StratifiedKFold if stratified else KFold)(5, shuffle=True, random_state=seed).split(F, y))
    else:
        folds = list(KFold(5, shuffle=True, random_state=seed).split(F))
    labels = np.unique(y) if cls else np.zeros(1)

    forest = np.zeros((len(y), len(labels))) if cls else np.zeros(len(y))
    for tr, te in folds:
        et = (ExtraTreesClassifier if cls else ExtraTreesRegressor)(random_state=seed, n_jobs=-1, **FOREST[task]).fit(F[tr], y[tr])
        if cls:
            forest[te[:, None], np.searchsorted(labels, et.classes_)] = et.predict_proba(F[te])
        else:
            forest[te] = et.predict(F[te])

    def score(p):
        return float((labels[p.argmax(axis=1)] == y).mean()) if cls else float(np.sqrt(((p - y) ** 2).mean()))

    rows = [dict(variant="forest", model="forest", score=score(forest), extract_s=0.0, ridge_s=0.0, kernel_features=0)]

    def add(name, K, extract_s):
        t = time.perf_counter()
        ridge = oof_ridge(task, K, y, folds, labels)
        ridge_s = time.perf_counter() - t
        if ridge is None:
            return
        for model, p in (("ridge", ridge), ("average", (forest + ridge) / 2)):
            rows.append(dict(variant=name, model=model, score=score(p), extract_s=extract_s, ridge_s=ridge_s,
                             kernel_features=K.shape[1]))

    if T >= MIN_LENGTH:
        t = time.perf_counter()
        kp = SF.kernel_fit(X, SF.n_kernels(T), seed, max_series=32)
        K = np.nan_to_num(SF.features(X, kp)[1])
        add("threshold", K, time.perf_counter() - t)
    for name, (gmax, inputs, soft) in VARIANTS.items():
        t = time.perf_counter()
        params = SF.compete_fit(X, seed, int(np.clip(T // 32, 1, gmax)))
        K = np.nan_to_num(SF.compete_features(X, params, inputs, soft=soft))
        add(name, K, time.perf_counter() - t)
    return [dict(r, seed=seed, task=task, T=T, n=len(y)) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=str(bench.RNG))
    a = ap.parse_args()
    known = set()
    if os.path.exists(OUT):
        p = pd.read_csv(OUT)
        known = set(zip(p.seed, p.task, p.dataset))
    for seed in [int(s) for s in a.seeds.split(",")]:
        for task, names in (("classification", bench.UCR), ("regression", bench.TSER)):
            for name in names:
                if (seed, task, name) in known:
                    continue
                try:
                    X, y = load_train(task, name)
                    rows = with_retry(score_dataset, task, X, y, seed)
                except Exception as e:
                    print(f"[failed] {seed} {task} {name}: {str(e)[:120]}", flush=True)
                    continue
                pd.DataFrame(rows).assign(dataset=name).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
                print(f"{seed} {task[:3]} {name}", flush=True)


if __name__ == "__main__":
    main()
