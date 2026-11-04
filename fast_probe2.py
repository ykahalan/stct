"""Probe the fit degree of level-free dilated kernels, using training splits only.

    python fast_probe2.py

Ridge alone, out of fold, on development datasets: level-free dilated kernels on
9 samples with Chebyshev degree 4 (as now), 6 or 7, same number of directions.
Results persist in results/fast_probe2_scores.csv.  No test split is loaded.
"""
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

import bench
import stct_fast as SF
from algo_select import load_train, with_retry
from fast_dilate import _sample
from fast_search import oof_ridge

warnings.filterwarnings("ignore")
OUT = "results/fast_probe2_scores.csv"
DEGREES = (4, 6, 7)


def level_free_kernels(X, seed, degree):
    rng = np.random.default_rng(seed)
    T, n_dir = X.shape[1], SF.n_kernels(X.shape[1])
    params = []
    for d in SF.dilations(T):
        Cx = SF.dilated(_sample(X, rng), d, 1, degree=degree)
        flat = Cx.transpose(1, 0, 2).reshape(Cx.shape[1], -1)
        mu, sd = flat.mean(axis=1), flat.std(axis=1) + 1e-12
        D = rng.standard_normal((n_dir, Cx.shape[1]))
        D[:, 0] = 0.0
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        params.append((mu, sd, D, np.quantile(D @ ((flat - mu[:, None]) / sd[:, None]), SF.QS, axis=1)))
    out = []
    for s in range(0, len(X), SF.BLOCK):
        Xb = X[s : s + SF.BLOCK]
        out.append(np.hstack([SF._kernels(SF.dilated(Xb, d, 1, degree=degree), *params[i]) for i, d in enumerate(SF.dilations(T))]))
    return np.vstack(out)


def score_dataset(X, y, seed):
    stratified = np.unique(y, return_counts=True)[1].max() >= 5
    folds = list((StratifiedKFold if stratified else KFold)(5, shuffle=True, random_state=seed).split(X, y))
    labels = np.unique(y)
    rows = []
    for deg in DEGREES:
        K = np.nan_to_num(level_free_kernels(X, seed, deg))
        p = oof_ridge("classification", K, y, folds, labels)
        if p is not None:
            rows.append(dict(model=f"D{deg}", score=float((labels[p.argmax(axis=1)] == y).mean()), features=K.shape[1]))
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
