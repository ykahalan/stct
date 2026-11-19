"""The classification STCT+ features with regression learners, on training splits only.

    python fast_reg_plus.py --seeds 42

Compared by 5-fold cross-validation on each TSER training split:
  fast    the current regressor (forest on the views, ridge on threshold kernels from length 128 up);
  plus    STCT+'s feature pipeline unchanged (views; context, winner and dilated kernels from length 128 up,
          dilated kernels alone below) with ExtraTreesRegressor (50 trees, half the features, 2 per leaf)
          and one RidgeCV per kernel block, all averaged with equal weight;
  plus_w  the same with the dilated ridge at half weight below length 128, as in the classifier.
Rule fixed in advance: adopt only if the geometric-mean RMSE ratio to fast is at most 0.99 and more datasets
improve than worsen.  Results persist in results/fast_reg_plus_scores.csv.  No test split is loaded.
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import bench
from algo_select import load_train
from stct_models import STCTRegressor, _2d

warnings.filterwarnings("ignore")
OUT = "results/fast_reg_plus_scores.csv"


def predict_weighted(model, X):
    """Forest and ridges averaged with the classifier's weights (ridges at half weight below length 128)."""
    V, blocks = model._blocks(_2d(X))
    pred, used, w = model.forest_.predict(V), 1.0, model.ridge_weight_
    for r, B in zip(model.ridges_, blocks):
        if r is not None:
            pred, used = pred + w * r.predict(B), used + w
    return pred / used


def score_dataset(X, y, seed, n_jobs):
    folds = list(KFold(5, shuffle=True, random_state=seed).split(X))
    oof = {k: np.zeros(len(y)) for k in ("fast", "plus", "plus_w")}
    secs = {"fast": 0.0, "plus": 0.0}
    feats = {}
    for tr, te in folds:
        for recipe in ("fast", "plus"):
            t = time.perf_counter()
            m = STCTRegressor(recipe=recipe, random_state=seed, n_jobs=n_jobs).fit(X[tr], y[tr])
            oof[recipe][te] = m.predict(X[te])
            secs[recipe] += time.perf_counter() - t
            feats[recipe] = m.n_features(X[:1])
            if recipe == "plus":
                oof["plus_w"][te] = predict_weighted(m, X[te])
    rmse = {k: float(np.sqrt(((p - y) ** 2).mean())) for k, p in oof.items()}
    return [dict(variant=k, rmse=rmse[k], seconds=secs.get(k, secs["plus"]), n_features=feats.get(k, feats["plus"]),
                 seed=seed, T=X.shape[1]) for k in oof]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=str(bench.RNG))
    ap.add_argument("--n_jobs", type=int, default=2)
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
                rows = score_dataset(_2d(X), np.asarray(y, float), seed, a.n_jobs)
            except Exception as e:
                print(f"[failed] {seed} {name}: {type(e).__name__} {str(e)[:120]}", flush=True)
                continue
            pd.DataFrame(rows).assign(dataset=name).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
            r = {row["variant"]: row["rmse"] for row in rows}
            print(f"{seed} {name} T={X.shape[-1]} plus/fast {r['plus'] / r['fast']:.3f} "
                  f"plus_w/fast {r['plus_w'] / r['fast']:.3f}", flush=True)


if __name__ == "__main__":
    main()
