"""Test kernels on sequences of neighbouring windows, using training splits only.

    python fast_context.py
    python fast_context.py --seeds 1,2

Each kernel sees a window and the next context - 1 windows, one window length
apart, so it can respond to an ordered pair or triple of local shapes wherever it
occurs.  Kernel counts stay the same, at length 128 and up.  Results persist in
results/fast_context_scores.csv.  No test split is loaded.
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
OUT = "results/fast_context_scores.csv"
CONTEXTS = (1, 2, 3, 5)
FOREST = {"classification": dict(n_estimators=100, max_features=0.1),
          "regression": dict(n_estimators=50, max_features=0.5, min_samples_leaf=2)}


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

    rows = [dict(variant="forest", model="forest", score=score(forest), extract_s=0.0)]
    parts = {}
    for c in CONTEXTS if T >= MIN_LENGTH else ():
        t = time.perf_counter()
        params = SF.context_fit(X, SF.n_kernels(T), seed, c)
        K = np.nan_to_num(SF.context_features(X, params, c))
        seconds = time.perf_counter() - t
        ridge = oof_ridge(task, K, y, folds, labels)
        if ridge is None:
            continue
        parts[c] = ridge
        rows.append(dict(variant=f"C{c}", model="ridge", score=score(ridge), extract_s=seconds))
        rows.append(dict(variant=f"C{c}", model="average", score=score((forest + ridge) / 2), extract_s=seconds))
    if 1 in parts and 3 in parts:
        rows.append(dict(variant="C1+C3", model="average", score=score(forest / 2 + (parts[1] + parts[3]) / 4), extract_s=0.0))
    return [dict(r, seed=seed, task=task, T=T) for r in rows]


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
