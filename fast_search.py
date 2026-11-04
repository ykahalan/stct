"""Search smaller forests for the averaged recipe, using training splits only.

    python fast_search.py
    python fast_search.py --seeds 1,2

Each setting is scored by the same 5-fold CV as ablate.py.  A forest of 200 trees
is fitted per max_features value and its first n trees give the smaller forests,
so all tree counts share folds.  Scores and relative forest times persist in
results/fast_scores.csv.  No test split is loaded.
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.model_selection import KFold, StratifiedKFold

import ablate
import bench
import stct_plus as SP
from algo_select import load_train, with_retry
from vote_test import MIN_LENGTH

warnings.filterwarnings("ignore")
TREES = (10, 25, 50, 100, 200)
MAX_FEATURES = {"classification": (0.1, 0.05, "sqrt"), "regression": (1.0, 0.5, 0.25, 0.1)}
OUT = "results/fast_scores.csv"


def oof_ridge(task, K, y, folds, labels):
    """Out-of-fold ridge outputs: class probabilities or predictions, None if ridge fails."""
    cls = task == "classification"
    out = np.zeros((len(y), len(labels))) if cls else np.zeros(len(y))
    try:
        for tr, te in folds:
            r = bench.make_learner("ridge", task).fit(K[tr], y[tr])
            if cls:
                out[te[:, None], np.searchsorted(labels, r.classes_)] = ablate.soft(r.decision_function(K[te]))
            else:
                out[te] = r.predict(K[te])
    except np.linalg.LinAlgError:
        return None
    return out


def score_dataset(task, X, y, seed):
    cls = task == "classification"
    F = np.nan_to_num(SP.stct_views(X))
    K = None
    if X.shape[1] >= MIN_LENGTH:
        kp = ablate.kernel_fit(X, int(np.clip(X.shape[1] // 8, 1, 64)), seed)
        K = np.nan_to_num(ablate.transform(X, {"KT"}, kp))
    if cls:
        stratified = np.unique(y, return_counts=True)[1].max() >= 5
        folds = list((StratifiedKFold if stratified else KFold)(5, shuffle=True, random_state=seed).split(F, y))
    else:
        folds = list(KFold(5, shuffle=True, random_state=seed).split(F))
    labels = np.unique(y) if cls else np.zeros(1)
    ridge = oof_ridge(task, K, y, folds, labels) if K is not None else None
    rows = []
    for mf in MAX_FEATURES[task]:
        total = {n: np.zeros((len(y), len(labels))) if cls else np.zeros(len(y)) for n in TREES}
        seconds = 0.0
        for tr, te in folds:
            t = time.perf_counter()
            et = (ExtraTreesClassifier if cls else ExtraTreesRegressor)(
                n_estimators=max(TREES), max_features=mf, random_state=seed, n_jobs=-1).fit(F[tr], y[tr])
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
            p = a if ridge is None else (a + ridge) / 2
            score = float((labels[p.argmax(axis=1)] == y).mean()) if cls else float(np.sqrt(((p - y) ** 2).mean()))
            rows.append(dict(max_features=str(mf), trees=n, seed=seed, task=task, score=score,
                             forest_s=seconds * n / max(TREES)))
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
