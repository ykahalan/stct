"""Test additions for series shorter than 128, using training splits only.

    python fast_short.py --seeds 42,1,2

Below length 128 the classifiers are the forest alone, and QUANT keeps up with
the convolution methods there.  Candidate forest inputs, each kept only while the
total stays below QUANT's feature count:
  L  local sorted STCT: values sorted inside each of 2, 4 and 8 windows, then the
     degree-4 Chebyshev fit and residual RMS (a compact local quantile curve)
  Ld the same on the first difference
  Dv paper and pooled features of the first difference
  P  paper features (8 windows) of the log periodogram
Also a ridge on level-free dilated kernels blended at weight 1/2 or 1, forests of
200 and 500 trees, and QUANT itself (aeon, 200 trees, entropy) as the reference.
Results persist in results/fast_short_scores.csv.  No test split is loaded.
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
import stct_plus as SP
from algo_select import load_train, with_retry
from fast_search import oof_ridge
from vote_test import MIN_LENGTH

warnings.filterwarnings("ignore")
OUT = "results/fast_short_scores.csv"
TREES = (200, 500)
SETS = ("F", "F+L", "F+L+Ld", "F+Dv", "F+P", "F+L+P", "F+L+Ld+P", "F+L+Dv")


def local_sorted(X, counts=(2, 4, 8)):
    """Chebyshev fit and residual RMS of the sorted values inside each tiling window."""
    X = np.asarray(X, dtype=float)
    n = X.shape[1]
    parts = []
    for c in counts:
        if n // c < SF.DEGREE + 2:
            continue
        edges = np.round(np.linspace(0, n, c + 1)).astype(int)
        S = X.copy()
        for a, b in zip(edges[:-1], edges[1:]):
            S[:, a:b] = np.sort(S[:, a:b], axis=1)
        parts.append(SP.stct_ladder(S, (c,)))
    return np.hstack(parts) if parts else np.zeros((len(X), 0))


def paper_pooled(X):
    """Paper features and pooled statistics of a series, as in the views."""
    T = X.shape[1]
    ppv_residual = 7 * (SF.DEGREE + 2) + SF.DEGREE + 1
    pooled = [np.delete(SF._pooled(SF._sliding(X, *SF._setup(T, L))), ppv_residual, axis=1) for L in SF.window_lengths(T)]
    return np.hstack([SP.stct_fixed(X)] + pooled)


def blocks(X):
    dX = np.diff(X, axis=1)
    return {"F": SF.features(X)[0], "L": local_sorted(X), "Ld": local_sorted(dX), "Dv": paper_pooled(dX),
            "P": SP.stct_fixed(SP.log_periodogram(X), (8,))}


def oof_quant(X, y, folds, labels):
    from aeon.transformations.collection.interval_based import QUANTTransformer

    out = np.zeros((len(y), len(labels)))
    for tr, te in folds:
        q = QUANTTransformer()
        Ftr = np.asarray(q.fit_transform(X[tr][:, None, :], y[tr]))
        Fte = np.asarray(q.transform(X[te][:, None, :]))
        et = ExtraTreesClassifier(n_estimators=200, max_features=0.1, criterion="entropy", random_state=bench.RNG, n_jobs=-1)
        et.fit(Ftr, y[tr])
        out[te[:, None], np.searchsorted(labels, et.classes_)] = et.predict_proba(Fte)
    return out


def score_dataset(X, y, seed):
    T = X.shape[1]
    budget = quant_features(T) - 1
    B = {k: np.nan_to_num(v) for k, v in blocks(X).items()}
    stratified = np.unique(y, return_counts=True)[1].max() >= 5
    folds = list((StratifiedKFold if stratified else KFold)(5, shuffle=True, random_state=seed).split(X, y))
    labels = np.unique(y)
    acc = lambda p: float((labels[p.argmax(axis=1)] == y).mean())
    rows = [dict(model="QUANT", trees=200, score=acc(oof_quant(X, y, folds, labels)), features=quant_features(T))]
    probs = {}
    for name in SETS:
        parts = name.split("+")
        F = np.hstack([B[p] for p in parts])
        if F.shape[1] > budget or any(B[p].shape[1] == 0 for p in parts):
            continue
        probs[name] = {n: np.zeros((len(y), len(labels))) for n in TREES}
        for tr, te in folds:
            et = ExtraTreesClassifier(n_estimators=max(TREES), max_features=0.1, random_state=seed, n_jobs=-1).fit(F[tr], y[tr])
            run, cols = np.zeros((len(te), len(labels))), np.searchsorted(labels, et.classes_)
            for i, tree in enumerate(et.estimators_, 1):
                run[:, cols] += tree.predict_proba(F[te])
                if i in probs[name]:
                    probs[name][i][te] = run / i
        for n in TREES:
            rows.append(dict(model=name, trees=n, score=acc(probs[name][n]), features=F.shape[1]))
    # level-free dilated kernels in whatever room the plain views leave
    nd = len(SF.dilations(T))
    k = int(min(SF.n_kernels(T), max(0, budget - B["F"].shape[1]) // (6 * nd)))
    if k > 0:
        K = np.nan_to_num(SF.dilated_features(X, SF.dilated_fit(X, k, seed)))
        r = oof_ridge("classification", K, y, folds, labels)
        if r is not None:
            for w in (0.5, 1.0):
                for n in TREES:
                    rows.append(dict(model=f"F+D{w:g}", trees=n, score=acc(probs["F"][n] + w * r), features=B["F"].shape[1] + K.shape[1]))
    return [dict(r, seed=seed, T=T, budget=budget) for r in rows]


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
                if X.shape[1] >= MIN_LENGTH:
                    continue
                rows = with_retry(score_dataset, X, y, seed)
            except Exception as e:
                print(f"[failed] {seed} {name}: {str(e)[:120]}", flush=True)
                continue
            pd.DataFrame(rows).assign(dataset=name).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
            print(f"{seed} {name}", flush=True)


if __name__ == "__main__":
    main()
