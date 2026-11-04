"""Fit level-free dilated kernels into the three-model budget, using training splits only.

    python fast_dlf2.py --seeds 42,1,2

Context kernels (C) and level-free dilated kernels (D) each get half the usual
directions per scale or dilation (C2, D2), so the four-model blend costs about
what the three-model recipe does.  Ridges on C2 and D2 are fitted apart, or as one
ridge on both (J).  Also scored: the full-size four-model blend with one joint
ridge on C and D.  Below length 128 only D exists, at full size.  Results persist
in results/fast_dlf2_scores.csv.  No test split is loaded.
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
from fast_dlf import dilated_every
from fast_probe import kernels
from fast_search import oof_ridge
from vote_test import MIN_LENGTH

warnings.filterwarnings("ignore")
OUT = "results/fast_dlf2_scores.csv"
# blend -> ridge inputs, each a tuple of blocks joined into one ridge
BLENDS = {"F+C+W": (("C",), ("W",)), "F+C2+W+D2": (("C2",), ("W",), ("D2",)), "F+W+J2": (("W",), ("C2", "D2")),
          "F+W+J": (("W",), ("C", "D")), "F+D": (("D",),)}


def score_dataset(X, y, seed):
    T = X.shape[1]
    n = SF.n_kernels(T)
    F = np.nan_to_num(SF.features(X)[0])
    makers = {"D": lambda: kernels(X, seed, dilated_every, True)}
    if T >= MIN_LENGTH:
        makers.update(
            D2=lambda: kernels(X, seed, dilated_every, True, max(1, n // 2)),
            C=lambda: SF.context_features(X, SF.context_fit(X, n, seed, 2), 2),
            C2=lambda: SF.context_features(X, SF.context_fit(X, max(1, n // 2), seed, 2), 2),
            W=lambda: SF.compete_features(X, SF.compete_fit(X, seed, SF.winner_groups(T))))
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
    cache, rows = {}, [dict(blend="F", score=float((labels[forest.argmax(axis=1)] == y).mean()), features=F.shape[1], seconds=0.0)]
    for name, inputs in BLENDS.items():
        if any(b not in sets for group in inputs for b in group):
            continue
        p = forest.copy()
        for group in inputs:
            if group not in cache:
                cache[group] = oof_ridge("classification", np.hstack([sets[b][0] for b in group]), y, folds, labels)
            if cache[group] is not None:
                p += cache[group]
        blocks = {b for group in inputs for b in group}
        rows.append(dict(blend=name, score=float((labels[p.argmax(axis=1)] == y).mean()),
                         features=F.shape[1] + sum(sets[b][0].shape[1] for b in blocks),
                         seconds=sum(sets[b][1] for b in blocks)))
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
