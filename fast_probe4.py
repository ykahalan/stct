"""Probe how dilated kernels should spend a fixed feature count, using training splits only.

    python fast_probe4.py

Ridge alone, out of fold, on development datasets, all at the same feature count:
K6 is the current block (n directions x 5 thresholds + maximum), K3 has 2n
directions x 2 thresholds + maximum, K2 has 3n directions x 1 threshold (median)
+ maximum.  Results persist in results/fast_probe4_scores.csv with extraction
seconds.  No test split is loaded.
"""
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

import bench
import stct_fast as SF
from algo_select import load_train, with_retry
from fast_search import oof_ridge

warnings.filterwarnings("ignore")
OUT = "results/fast_probe4_scores.csv"
SPLITS = {"K6": (1, SF.QS), "K3": (2, np.array([1 / 3, 2 / 3])), "K2": (3, np.array([0.5]))}


def dilated_block(X, seed, mult, qs):
    """Level-free dilated kernels with mult * n_kernels directions and the given threshold quantiles."""
    rng = np.random.default_rng(seed)
    T = X.shape[1]
    n_dir = mult * SF.n_kernels(T)
    S = X[rng.choice(len(X), 32, replace=False)] if len(X) > 32 else X
    params = []
    for d in SF.dilations(T):
        Cx = SF.dilated(S, d, 1)
        flat = Cx.transpose(1, 0, 2).reshape(Cx.shape[1], -1)
        mu, sd = flat.mean(axis=1), flat.std(axis=1) + 1e-12
        D = rng.standard_normal((n_dir, Cx.shape[1]))
        D[:, 0] = 0.0
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        params.append((mu, sd, D, np.quantile(D @ ((flat - mu[:, None]) / sd[:, None]), qs, axis=1).reshape(len(qs), -1)))
    out = []
    for s in range(0, len(X), SF.BLOCK):
        Xb = X[s : s + SF.BLOCK]
        out.append(np.hstack([SF._kernels(SF.dilated(Xb, d, 1), *params[i]) for i, d in enumerate(SF.dilations(T))]))
    return np.vstack(out)


def score_dataset(X, y, seed):
    stratified = np.unique(y, return_counts=True)[1].max() >= 5
    folds = list((StratifiedKFold if stratified else KFold)(5, shuffle=True, random_state=seed).split(X, y))
    labels = np.unique(y)
    rows = []
    for name, (mult, qs) in SPLITS.items():
        t = time.perf_counter()
        K = np.nan_to_num(dilated_block(X, seed, mult, qs))
        seconds = time.perf_counter() - t
        p = oof_ridge("classification", K, y, folds, labels)
        if p is not None:
            rows.append(dict(model=name, score=float((labels[p.argmax(axis=1)] == y).mean()), features=K.shape[1], seconds=seconds))
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
