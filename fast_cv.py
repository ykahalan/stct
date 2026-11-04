"""Search fewer kernel windows and smaller forests for the averaged recipe, using training splits only.

    python fast_cv.py
    python fast_cv.py --seeds 1,2

Same folds as fast_search.py, whose scores give the baseline.  Thresholds come
from at most 32 series; kernels see at most about max_windows windows per scale
(0 means all).  Results persist in results/fast_cap_scores.csv.  No test split
is loaded.
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
MAX_SERIES = 32
MAX_WINDOWS = (0, 64, 32, 16)
FOREST = {"classification": dict(max_features=0.1, trees=(100, 200), leaves=(1,)),
          "regression": dict(max_features=0.5, trees=(50, 100), leaves=(1, 2))}
OUT = "results/fast_cap_scores.csv"


def score_dataset(task, X, y, seed):
    cls = task == "classification"
    F = np.nan_to_num(SF.features(X)[0])
    if cls:
        stratified = np.unique(y, return_counts=True)[1].max() >= 5
        folds = list((StratifiedKFold if stratified else KFold)(5, shuffle=True, random_state=seed).split(F, y))
    else:
        folds = list(KFold(5, shuffle=True, random_state=seed).split(F))
    labels = np.unique(y) if cls else np.zeros(1)
    ridges, kernel_s = {}, {}
    for w in MAX_WINDOWS if X.shape[1] >= MIN_LENGTH else ():
        t = time.perf_counter()
        kp = SF.kernel_fit(X, SF.n_kernels(X.shape[1]), seed, max_series=MAX_SERIES, max_windows=w or None)
        K = np.nan_to_num(SF.features(X, kp, max_windows=w or None)[1])
        kernel_s[w] = time.perf_counter() - t
        ridges[w] = oof_ridge(task, K, y, folds, labels)
    cfg, rows = FOREST[task], []
    for leaf in cfg["leaves"]:
        total = {n: np.zeros((len(y), len(labels))) if cls else np.zeros(len(y)) for n in cfg["trees"]}
        seconds = 0.0
        for tr, te in folds:
            t = time.perf_counter()
            et = (ExtraTreesClassifier if cls else ExtraTreesRegressor)(
                n_estimators=max(cfg["trees"]), max_features=cfg["max_features"], min_samples_leaf=leaf,
                random_state=seed, n_jobs=-1).fit(F[tr], y[tr])
            seconds += time.perf_counter() - t
            run = np.zeros((len(te), len(labels))) if cls else np.zeros(len(te))
            for i, tree in enumerate(et.estimators_, 1):
                if cls:
                    run[:, np.searchsorted(labels, et.classes_)] += tree.predict_proba(F[te])
                else:
                    run += tree.predict(F[te])
                if i in total:
                    total[i][te] = run / i
        for n, a in total.items():
            for w in MAX_WINDOWS:
                ridge = ridges.get(w)
                p = a if ridge is None else (a + ridge) / 2
                score = float((labels[p.argmax(axis=1)] == y).mean()) if cls else float(np.sqrt(((p - y) ** 2).mean()))
                rows.append(dict(leaf=leaf, trees=n, max_windows=w, seed=seed, task=task, score=score,
                                 forest_s=seconds * n / max(cfg["trees"]), kernel_s=kernel_s.get(w, 0.0)))
    return rows


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
