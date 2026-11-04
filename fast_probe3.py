"""Probe level-free dilated kernels on the first-difference series, using training splits only.

    python fast_probe3.py

Ridge alone, out of fold, on development datasets: dilated kernels on the series
(D), on the series and its first differences with half the directions each (DD2,
same size as D), and with full directions each (DD, twice the size).  Results
persist in results/fast_probe3_scores.csv.  No test split is loaded.
"""
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

import bench
import stct_fast as SF
from algo_select import load_train, with_retry
from fast_dlf import dilated_every
from fast_probe import kernels
from fast_search import oof_ridge

warnings.filterwarnings("ignore")
OUT = "results/fast_probe3_scores.csv"


def score_dataset(X, y, seed):
    stratified = np.unique(y, return_counts=True)[1].max() >= 5
    folds = list((StratifiedKFold if stratified else KFold)(5, shuffle=True, random_state=seed).split(X, y))
    labels = np.unique(y)
    n = SF.n_kernels(X.shape[1])
    dX = np.diff(X, axis=1)
    sets = {"D": kernels(X, seed, dilated_every, True),
            "DD2": np.hstack([kernels(X, seed, dilated_every, True, max(1, n // 2)),
                              kernels(dX, seed + 1, dilated_every, True, max(1, n // 2))]),
            "DD": np.hstack([kernels(X, seed, dilated_every, True), kernels(dX, seed + 1, dilated_every, True)])}
    rows = []
    for name, K in sets.items():
        p = oof_ridge("classification", np.nan_to_num(K), y, folds, labels)
        if p is not None:
            rows.append(dict(model=name, score=float((labels[p.argmax(axis=1)] == y).mean()), features=K.shape[1]))
    return [dict(r, seed=seed, T=X.shape[1]) for r in rows]


def main():
    known = set(pd.read_csv(OUT).dataset) if os.path.exists(OUT) else set()
    for name in bench.DEV_UCR:
        if name in known:
            continue
        try:
            X, y = load_train("classification", name)
            rows = with_retry(score_dataset, X, y, bench.RNG)
        except Exception as e:
            print(f"[failed] {name}: {str(e)[:120]}", flush=True)
            continue
        pd.DataFrame(rows).assign(dataset=name).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
        print(name, flush=True)


if __name__ == "__main__":
    main()
