"""Test three-model averages for classification under the feature limit, using training splits only.

    python fast_trio.py --seeds 42,1,2

Forest of 100 or 50 trees, averaged with ridges on threshold kernels (T), on
kernels over two neighbouring windows (C) and on winner counts (W).  Below length
128 only the forest and W are available.  Results persist in
results/fast_trio_scores.csv.  No test split is loaded.
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import KFold, StratifiedKFold

import bench
import stct_fast as SF
from algo_select import load_train, with_retry
from fast_search import oof_ridge
from vote_test import MIN_LENGTH

warnings.filterwarnings("ignore")
OUT = "results/fast_trio_scores.csv"
TREES = (50, 100)
winner_groups = SF.winner_groups


def score_dataset(X, y, seed):
    T = X.shape[1]
    F = np.nan_to_num(SF.features(X)[0])
    stratified = np.unique(y, return_counts=True)[1].max() >= 5
    folds = list((StratifiedKFold if stratified else KFold)(5, shuffle=True, random_state=seed).split(F, y))
    labels = np.unique(y)
    total = {n: np.zeros((len(y), len(labels))) for n in TREES}
    for tr, te in folds:
        et = ExtraTreesClassifier(n_estimators=max(TREES), max_features=0.1, random_state=seed, n_jobs=-1).fit(F[tr], y[tr])
        run = np.zeros((len(te), len(labels)))
        for i, tree in enumerate(et.estimators_, 1):
            run[:, np.searchsorted(labels, et.classes_)] += tree.predict_proba(F[te])
            if i in total:
                total[i][te] = run / i
    sets = {"W": SF.compete_features(X, SF.compete_fit(X, seed, winner_groups(T)))}
    if T >= MIN_LENGTH:
        sets["T"] = SF.features(X, SF.kernel_fit(X, SF.n_kernels(T), seed, max_series=32))[1]
        sets["C"] = SF.context_features(X, SF.context_fit(X, SF.n_kernels(T), seed, 2), 2)
    ridges = {k: oof_ridge("classification", np.nan_to_num(K), y, folds, labels) for k, K in sets.items()}
    rows = []
    for n, a in total.items():
        for name in ("F", "FT", "FC", "FW", "FTW", "FCW", "FTC"):
            parts = [ridges.get(c) for c in name[1:]]
            if any(p is None for p in parts):
                continue
            p = (a + sum(parts)) / (1 + len(parts))
            rows.append(dict(blend=name, trees=n, score=float((labels[p.argmax(axis=1)] == y).mean()), seed=seed, T=T,
                             features=F.shape[1] + sum(sets[c].shape[1] for c in name[1:])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=str(bench.RNG))
    a = ap.parse_args()
    known = set()
    if os.path.exists(OUT):
        p = pd.read_csv(OUT)
        known = set(zip(p.seed, p.dataset))
    for seed in [int(s) for s in a.seeds.split(",")]:
        for name in bench.UCR:
            if (seed, name) in known:
                continue
            try:
                X, y = load_train("classification", name)
                rows = with_retry(score_dataset, X, y, seed)
            except Exception as e:
                print(f"[failed] {seed} {name}: {str(e)[:120]}", flush=True)
                continue
            pd.DataFrame(rows).assign(dataset=name).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
            print(f"{seed} {name}", flush=True)


if __name__ == "__main__":
    main()
