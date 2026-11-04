"""Compare an STCT-native kernel size rule with the QUANT-sized one, using training splits only.

    python fast_native.py --seeds 42,1,2

"quant" sizes context and dilated kernels so the total stays below QUANT's
feature count (and computes collapsed paper windows once at length 15).
"native" gives one kernel direction per 8 samples of series length, shared
equally among the kernel types in use, at most 64 per type: from length 128 up
context and dilated kernels get one per 16 samples each; below 128 the dilated
kernels, at half weight, get one per 8.  Both use 200 trees and winner counts
from length 128 up.  Only datasets where the two rules differ are computed; the
others give the same model.  Results persist in results/fast_native_scores.csv.
No test split is loaded.
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
from quant_sized import quant_features, scaled_directions
from algo_select import load_train, with_retry
from fast_search import oof_ridge
from vote_test import MIN_LENGTH

warnings.filterwarnings("ignore")
OUT = "results/fast_native_scores.csv"


def plans(T, V, V_distinct):
    """(views distinct?, directions) for each rule."""
    n, nd = SF.n_kernels(T), len(SF.dilations(T))
    if T >= MIN_LENGTH:
        return {"quant": (False, scaled_directions(T, V)), "native": (False, min(64, T // 16))}
    Q = quant_features(T)
    distinct = V >= Q
    v = V_distinct if distinct else V
    return {"quant": (distinct, int(min(n, max(0, Q - 1 - v) // (6 * nd)))), "native": (False, n)}


def score_dataset(X, y, seed):
    T = X.shape[1]
    long_ = T >= MIN_LENGTH
    V = SF.trio_features(X[:1])[0].shape[1]
    Vd = SF.trio_features(X[:1], distinct=True)[0].shape[1]
    plan = plans(T, V, Vd)
    if plan["quant"] == plan["native"]:
        return []
    stratified = np.unique(y, return_counts=True)[1].max() >= 5
    folds = list((StratifiedKFold if stratified else KFold)(5, shuffle=True, random_state=seed).split(X, y))
    labels = np.unique(y)
    forests, ridges = {}, {}

    def forest(distinct):
        if distinct not in forests:
            F = np.nan_to_num(SF.trio_features(X, distinct=distinct)[0])
            p = np.zeros((len(y), len(labels)))
            for tr, te in folds:
                et = ExtraTreesClassifier(n_estimators=200, max_features=0.1, random_state=seed, n_jobs=-1).fit(F[tr], y[tr])
                p[te[:, None], np.searchsorted(labels, et.classes_)] = et.predict_proba(F[te])
            forests[distinct] = (p, F.shape[1])
        return forests[distinct]

    def ridge(key, make):
        if key not in ridges:
            K = np.nan_to_num(make())
            ridges[key] = (oof_ridge("classification", K, y, folds, labels), K.shape[1])
        return ridges[key]

    rows = []
    for name, (distinct, k) in plan.items():
        p, nf = forest(distinct)
        parts = []
        if long_:
            parts.append(ridge(("W",), lambda: SF.compete_features(X, SF.compete_fit(X, seed, SF.winner_groups(T)))))
            if k > 0:
                parts.append(ridge(("C", k), lambda: SF.context_features(X, SF.context_fit(X, k, seed, 2), 2)))
        if k > 0:
            parts.append(ridge(("D", k), lambda: SF.dilated_features(X, SF.dilated_fit(X, k, seed))))
        w = 1.0 if long_ else 0.5
        total = p + w * sum(r for r, _ in parts if r is not None)
        rows.append(dict(rule=name, score=float((labels[total.argmax(axis=1)] == y).mean()), directions=k,
                         features=nf + sum(f for _, f in parts), seed=seed, T=T))
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
            if rows:
                pd.DataFrame(rows).assign(dataset=name).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
            print(f"{seed} {name} {'compared' if rows else 'same model'}", flush=True)


if __name__ == "__main__":
    main()
