"""Second batch of regressor candidates, on training splits only (see fast_reg_tune.py).

    python fast_reg_tune2.py --seeds 42
    python fast_reg_tune2.py --seeds 42 --summary

Members, all fitted out of fold on the same folds as fast_reg_tune.py:

* ET (current forest) with all features per split (ETmf1), a third (ETmf3),
  1 or 5 samples per leaf (ETl1, ETl5), and on the views plus threshold kernels (ETFK)
* K (current ridge) and K2, the ridge on twice as many directions per scale
* blends with the forest weighted 2:1 or 1:2 against the ridge

Results persist in results/fast_reg_tune2_scores.csv.  No test split is loaded.
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold

import bench
import fast_reg_tune as R
import stct_fast as SF
from algo_select import load_train, with_retry
from vote_test import MIN_LENGTH

warnings.filterwarnings("ignore")
R.OUT = OUT = "results/fast_reg_tune2_scores.csv"
R.BLENDS = BLENDS = {
    "fast": ("ET", "K"),
    "2ET+K": ("ET", "ET", "K"),
    "ET+2K": ("ET", "K", "K"),
    "ETmf1+K": ("ETmf1", "K"),
    "ETmf3+K": ("ETmf3", "K"),
    "ETl1+K": ("ETl1", "K"),
    "ETl5+K": ("ETl5", "K"),
    "ETFK": ("ETFK",),
    "ETFK+K": ("ETFK", "K"),
    "ET+K2": ("ET", "K2"),
    "ETmf1+K2": ("ETmf1", "K2"),
}


def score_dataset(X, y, seed):
    T = X.shape[1]
    long_ = T >= MIN_LENGTH
    kp = SF.kernel_fit(X, SF.n_kernels(T), seed, max_series=32) if long_ else None
    F, K = SF.features(X, kp)
    F = np.nan_to_num(F)
    folds = list(KFold(5, shuffle=True, random_state=seed).split(F))
    et = lambda mf=0.5, leaf=2: (lambda: ExtraTreesRegressor(n_estimators=50, max_features=mf, min_samples_leaf=leaf,
                                                             random_state=seed, n_jobs=-1))
    ridge = lambda: bench.make_learner("ridge", "regression")
    m = {"ET": R.oof(et(), F, y, folds), "ETmf1": R.oof(et(1.0), F, y, folds), "ETmf3": R.oof(et(1 / 3), F, y, folds),
         "ETl1": R.oof(et(leaf=1), F, y, folds), "ETl5": R.oof(et(leaf=5), F, y, folds)}
    if long_:
        K = np.nan_to_num(K)
        _, K2 = SF.features(X, SF.kernel_fit(X, 2 * SF.n_kernels(T), seed + 1000, max_series=32))
        m.update(K=R.oof(ridge, K, y, folds), K2=R.oof(ridge, np.nan_to_num(K2), y, folds),
                 ETFK=R.oof(et(), np.hstack([F, K]), y, folds))
    else:
        m.update(K=(None, 0.0), K2=(None, 0.0), ETFK=m["ET"])
    rmse = lambda p: float(np.sqrt(((p - y) ** 2).mean()))
    rows = []
    for name, parts in BLENDS.items():
        preds = [m[p][0] for p in parts if m[p][0] is not None]
        rows.append(dict(blend=name, score=rmse(np.mean(preds, axis=0)), member_seconds=sum(m[p][1] for p in set(parts))))
    return [dict(r, seed=seed, T=T, n=len(y)) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=str(bench.RNG))
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    if a.summary:
        return R.summary(seeds)
    known = set()
    if os.path.exists(OUT):
        p = pd.read_csv(OUT)
        known = set(zip(p.seed, p.dataset))
    for seed in seeds:
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
