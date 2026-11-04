"""Test level-free dilated STCT kernels in the classification blend, using training splits only.

    python fast_dlf.py --seeds 42,1,2

D: degree-4 Chebyshev fits to 9 samples spaced d apart (d = 1, 2, 4, ...) at every
sample, then threshold kernels whose random directions give no weight to the level
coefficient c0, so they respond to local shape and not to local level.  Dh takes
a window every d // 2 samples instead.  Blends of the forest with ridges on
context kernels (C), winner counts (W), D and Dh; C and W exist from length 128
up, D and Dh at every length.  Results persist in results/fast_dlf_scores.csv.
No test split is loaded.
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import KFold, StratifiedKFold

import bench
import stct_fast as SF
from algo_select import load_train, with_retry
from fast_probe import kernels
from fast_search import oof_ridge
from vote_test import MIN_LENGTH

warnings.filterwarnings("ignore")
OUT = "results/fast_dlf_scores.csv"
BLENDS = ("F", "F+C+W", "F+D", "F+Dh", "F+W+D", "F+W+Dh", "F+C+D", "F+C+W+D")


def dilated_every(X):
    return [SF.dilated(X, d, 1) for d in SF.dilations(X.shape[1])]


def dilated_half(X):
    return [SF.dilated(X, d, max(1, d // 2)) for d in SF.dilations(X.shape[1])]


def score_dataset(X, y, seed):
    T = X.shape[1]
    F = np.nan_to_num(SF.features(X)[0])
    makers = {"D": lambda: kernels(X, seed, dilated_every, True), "Dh": lambda: kernels(X, seed, dilated_half, True)}
    if T >= MIN_LENGTH:
        makers["C"] = lambda: SF.context_features(X, SF.context_fit(X, SF.n_kernels(T), seed, 2), 2)
        makers["W"] = lambda: SF.compete_features(X, SF.compete_fit(X, seed, SF.winner_groups(T)))
    sets = {}
    for k, f in makers.items():
        t = time.perf_counter()
        sets[k] = (np.nan_to_num(f()), time.perf_counter() - t)
    stratified = np.unique(y, return_counts=True)[1].max() >= 5
    folds = list((StratifiedKFold if stratified else KFold)(5, shuffle=True, random_state=seed).split(F, y))
    labels = np.unique(y)
    forest = np.zeros((len(y), len(labels)))
    for tr, te in folds:
        et = ExtraTreesClassifier(n_estimators=100, max_features=0.1, random_state=seed, n_jobs=-1).fit(F[tr], y[tr])
        forest[te[:, None], np.searchsorted(labels, et.classes_)] = et.predict_proba(F[te])
    ridges = {k: oof_ridge("classification", K, y, folds, labels) for k, (K, _) in sets.items()}
    rows = []
    for name in BLENDS:
        parts = name.split("+")[1:]
        if any(p not in sets for p in parts):
            continue
        p = forest + sum(ridges[c] for c in parts if ridges[c] is not None)
        rows.append(dict(blend=name, score=float((labels[p.argmax(axis=1)] == y).mean()),
                         features=F.shape[1] + sum(sets[c][0].shape[1] for c in parts),
                         seconds=sum(sets[c][1] for c in parts)))
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
