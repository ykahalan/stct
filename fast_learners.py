"""Try closed-form learners in place of the forest, using training splits only.

    python fast_learners.py
    python fast_learners.py --seeds 1,2

Same folds as fast_search.py, whose scores give the baseline.  Results persist
in results/fast_learner_scores.csv.  No test split is loaded.
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import ablate
import bench
import stct_fast as SF
from algo_select import load_train, with_retry
from vote_test import MIN_LENGTH

warnings.filterwarnings("ignore")
OUT = "results/fast_learner_scores.csv"


def ridge(task):
    return bench.make_learner("ridge", task)


def lda(task):
    return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))


def oof(task, make, Z, y, folds, labels):
    """Out-of-fold class scores or predictions, and fit plus predict seconds."""
    cls = task == "classification"
    out = np.zeros((len(y), len(labels))) if cls else np.zeros(len(y))
    t = time.perf_counter()
    for tr, te in folds:
        r = make(task).fit(Z[tr], y[tr])
        if cls:
            out[te[:, None], np.searchsorted(labels, r.classes_)] = ablate.soft(r.decision_function(Z[te]))
        else:
            out[te] = r.predict(Z[te])
    return out, time.perf_counter() - t


def score_dataset(task, X, y, seed):
    cls = task == "classification"
    kp = SF.kernel_fit(X, SF.n_kernels(X.shape[1]), seed, max_series=32) if X.shape[1] >= MIN_LENGTH else None
    F, K = SF.features(X, kp)
    F = np.nan_to_num(F)
    if cls:
        stratified = np.unique(y, return_counts=True)[1].max() >= 5
        folds = list((StratifiedKFold if stratified else KFold)(5, shuffle=True, random_state=seed).split(F, y))
    else:
        folds = list(KFold(5, shuffle=True, random_state=seed).split(F))
    labels = np.unique(y) if cls else np.zeros(1)
    Z = F if K is None else np.hstack([F, np.nan_to_num(K)])
    runs = {"ridge_views": oof(task, ridge, F, y, folds, labels), "ridge_all": oof(task, ridge, Z, y, folds, labels)}
    if K is not None:
        k, s = oof(task, ridge, np.nan_to_num(K), y, folds, labels)
        runs["ridge_vote"] = ((runs["ridge_views"][0] + k) / 2, runs["ridge_views"][1] + s)
    else:
        runs["ridge_vote"] = runs["ridge_views"]
    if cls:
        runs["lda_all"] = oof(task, lda, Z, y, folds, labels)
    rows = []
    for name, (p, s) in runs.items():
        score = float((labels[p.argmax(axis=1)] == y).mean()) if cls else float(np.sqrt(((p - y) ** 2).mean()))
        rows.append(dict(learner=name, seed=seed, task=task, score=score, seconds=s))
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
