"""Fit the four-model blend under QUANT's feature count, using training splits only.

    python fast_quant.py --seeds 42,1,2

The budget is QUANT's feature count at the series length minus one.  "fill"
keeps winner counts (W) and full-size level-free dilated kernels (D) and gives
context kernels (C) the directions left over, dropping them when none fit.
"scale" shrinks C and D directions by one common factor until everything fits.
Each blend is scored with forests of 100 and 200 trees (the first 100 trees of
the 200-tree forest are the 100-tree forest).  Below length 128 every blend is
the forest alone.  Results persist in results/fast_quant_scores.csv.  No test
split is loaded.
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
from quant_sized import quant_features
from algo_select import load_train, with_retry
from fast_search import oof_ridge
from vote_test import MIN_LENGTH

warnings.filterwarnings("ignore")
OUT = "results/fast_quant_scores.csv"
TREES = (100, 200)


def sizes(T, n_views):
    """Directions per scale for C and D under each sizing, given the forest's feature count."""
    n, ns, nd = SF.n_kernels(T), len(SF.window_lengths(T)), len(SF.dilations(T))
    room = quant_features(T) - 1 - n_views - 32 * SF.winner_groups(T) * ns
    d_fill = int(min(n, max(0, room) // (6 * nd)))
    c_fill = int(min(n, max(0, room - 6 * nd * d_fill) // (6 * ns)))
    s = min(1.0, max(0, room) / (6 * n * (ns + nd)))
    return {"fill": (c_fill, d_fill), "scale": (int(n * s), int(n * s)), "full": (n, n)}


def score_dataset(X, y, seed):
    T = X.shape[1]
    F = np.nan_to_num(SF.features(X)[0])
    stratified = np.unique(y, return_counts=True)[1].max() >= 5
    folds = list((StratifiedKFold if stratified else KFold)(5, shuffle=True, random_state=seed).split(F, y))
    labels = np.unique(y)
    forest = {n: np.zeros((len(y), len(labels))) for n in TREES}
    for tr, te in folds:
        et = ExtraTreesClassifier(n_estimators=max(TREES), max_features=0.1, random_state=seed, n_jobs=-1).fit(F[tr], y[tr])
        run, cols = np.zeros((len(te), len(labels))), np.searchsorted(labels, et.classes_)
        for i, tree in enumerate(et.estimators_, 1):
            run[:, cols] += tree.predict_proba(F[te])
            if i in forest:
                forest[i][te] = run / i
    acc = lambda p: float((labels[p.argmax(axis=1)] == y).mean())
    budget = quant_features(T) - 1
    if T < MIN_LENGTH:
        return [dict(blend=b, trees=n, score=acc(forest[n]), features=F.shape[1], budget=budget, seed=seed, T=T)
                for b in ("trio", "four", "fill", "scale") for n in TREES]
    sz = sizes(T, F.shape[1])
    blocks, ridges = {}, {}

    def ridge(key, make):
        if key not in ridges:
            K = np.nan_to_num(make()) if key[1] > 0 else None
            blocks[key] = 0 if K is None else K.shape[1]
            ridges[key] = None if K is None else oof_ridge("classification", K, y, folds, labels)
        return ridges[key], blocks[key]

    W = lambda: SF.compete_features(X, SF.compete_fit(X, seed, SF.winner_groups(T)))
    C = lambda k: (lambda: SF.context_features(X, SF.context_fit(X, k, seed, 2), 2))
    D = lambda k: (lambda: SF.dilated_features(X, SF.dilated_fit(X, k, seed)))
    plan = {"trio": [("W", 1, W), ("C", sz["full"][0], C(sz["full"][0]))],
            "four": [("W", 1, W), ("C", sz["full"][0], C(sz["full"][0])), ("D", sz["full"][1], D(sz["full"][1]))],
            "fill": [("W", 1, W), ("C", sz["fill"][0], C(sz["fill"][0])), ("D", sz["fill"][1], D(sz["fill"][1]))],
            "scale": [("W", 1, W), ("C", sz["scale"][0], C(sz["scale"][0])), ("D", sz["scale"][1], D(sz["scale"][1]))]}
    rows = []
    for name, parts in plan.items():
        got = [ridge((b, k), make) for b, k, make in parts]
        for n in TREES:
            p = forest[n] + sum(r for r, _ in got if r is not None)
            rows.append(dict(blend=name, trees=n, score=acc(p), features=F.shape[1] + sum(f for _, f in got), budget=budget,
                             seed=seed, T=T, c_dirs=parts[1][1], d_dirs=parts[2][1] if len(parts) > 2 else 0))
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
