"""Test level-free dilated kernels for regression, using training splits only.

    python fast_dlf_reg.py --seeds 42,1,2

The fast regressor averages a forest (50 trees, half the features, 2 samples per
leaf) with a ridge on threshold kernels (T) from length 128 up.  Compared: the
ridge on level-free dilated kernels (D) instead, both ridges, and D at every
length.  Results persist in results/fast_dlf_reg_scores.csv.  No test split is
loaded.
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold

import bench
import stct_fast as SF
from algo_select import load_train, with_retry
from fast_dlf import dilated_every
from fast_probe import kernels
from fast_search import oof_ridge
from vote_test import MIN_LENGTH

warnings.filterwarnings("ignore")
OUT = "results/fast_dlf_reg_scores.csv"


def score_dataset(X, y, seed):
    T = X.shape[1]
    long_ = T >= MIN_LENGTH
    kp = SF.kernel_fit(X, SF.n_kernels(T), seed, max_series=32) if long_ else None
    F, K = SF.features(X, kp)
    F = np.nan_to_num(F)
    sets = {"D": np.nan_to_num(kernels(X, seed, dilated_every, True))}
    if long_:
        sets["T"] = np.nan_to_num(K)
    folds = list(KFold(5, shuffle=True, random_state=seed).split(F))
    forest = np.zeros(len(y))
    for tr, te in folds:
        et = ExtraTreesRegressor(n_estimators=50, max_features=0.5, min_samples_leaf=2, random_state=seed, n_jobs=-1)
        forest[te] = et.fit(F[tr], y[tr]).predict(F[te])
    ridges = {k: oof_ridge("regression", K, y, folds, np.zeros(1)) for k, K in sets.items()}
    rmse = lambda p: float(np.sqrt(((p - y) ** 2).mean()))
    blends = {"F": (), "fast": ("T",) if long_ else (), "F+D": ("D",), "F+D_long": ("D",) if long_ else (),
              "F+T+D": ("T", "D") if long_ else ("D",)}
    rows = []
    for name, parts in blends.items():
        parts = [ridges[b] for b in parts if ridges.get(b) is not None]
        rows.append(dict(blend=name, score=rmse((forest + sum(parts)) / (1 + len(parts)))))
    return [dict(r, seed=seed, T=T) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=str(bench.RNG))
    a = ap.parse_args()
    known = set()
    if os.path.exists(OUT):
        p = pd.read_csv(OUT)
        known = set(zip(p.seed, p.dataset))
    for seed in [int(s) for s in a.seeds.split(",")]:
        for name in bench.TSER:
            if (seed, name) in known:
                continue
            try:
                X, y = load_train("regression", name)
                rows = with_retry(score_dataset, X, y, seed)
            except Exception as e:
                print(f"[failed] {seed} {name}: {str(e)[:120]}", flush=True)
                continue
            pd.DataFrame(rows).assign(dataset=name).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
            print(f"{seed} {name}", flush=True)


if __name__ == "__main__":
    main()
