"""Probe what makes MiniRocket's ridge stronger than STCT kernels, using training splits only.

    python fast_probe.py

Ridge alone, out of fold, on development datasets: MiniRocket at its default
size and at about 2,500 features, STCT context kernels (C) and dilated kernels at
every sample (DK1), each with random directions as now or level-free (no weight
on the level coefficient c0, or for two windows only on their level
difference).  Results persist in results/fast_probe_scores.csv.  No test split is
loaded.
"""
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

import ablate
import bench
import stct_fast as SF
from algo_select import load_train, with_retry
from fast_dilate import _sample
from fast_search import oof_ridge
from vote_test import MIN_LENGTH

warnings.filterwarnings("ignore")
OUT = "results/fast_probe_scores.csv"
CH = SF.DEGREE + 2


def directions(rng, n_dir, ch, level_free):
    D = rng.standard_normal((n_dir, ch))
    if level_free:
        if ch == CH:
            D[:, 0] = 0.0
        else:  # windows in context: weight only level differences
            lv = D[:, ::CH]
            D[:, ::CH] = lv - lv.mean(axis=1, keepdims=True)
    return D / np.linalg.norm(D, axis=1, keepdims=True)


def kernel_params(sources, n_dir, rng, level_free):
    params = []
    for Cx in sources:
        flat = Cx.transpose(1, 0, 2).reshape(Cx.shape[1], -1)
        mu, sd = flat.mean(axis=1), flat.std(axis=1) + 1e-12
        D = directions(rng, n_dir, Cx.shape[1], level_free)
        params.append((mu, sd, D, np.quantile(D @ ((flat - mu[:, None]) / sd[:, None]), SF.QS, axis=1)))
    return params


def context_sources(X):
    T = X.shape[1]
    return [SF.in_context(SF._sliding(X, *SF._setup(T, L)), L, 2) for L in SF.window_lengths(T)]


def dilated_sources(X):
    return [SF.dilated(X, d, 1) for d in SF.dilations(X.shape[1])]


def kernels(X, seed, make_sources, level_free, n_dir=None):
    rng = np.random.default_rng(seed)
    n_dir = SF.n_kernels(X.shape[1]) if n_dir is None else n_dir
    params = kernel_params(make_sources(_sample(X, rng)), n_dir, rng, level_free)
    out = []
    for s in range(0, len(X), SF.BLOCK):
        out.append(np.hstack([SF._kernels(Cx, *params[i]) for i, Cx in enumerate(make_sources(X[s : s + SF.BLOCK]))]))
    return np.vstack(out)


def oof_minirocket(X, y, folds, labels, num_kernels):
    from aeon.transformations.collection.convolution_based import MiniRocket

    out = np.zeros((len(y), len(labels)))
    for tr, te in folds:
        m = MiniRocket(n_kernels=num_kernels, n_jobs=-1, random_state=bench.RNG)
        Ftr, Fte = m.fit_transform(X[tr][:, None, :]), m.transform(X[te][:, None, :])
        r = bench.make_learner("ridge", "classification").fit(Ftr, y[tr])
        out[te[:, None], np.searchsorted(labels, r.classes_)] = ablate.soft(r.decision_function(Fte))
    return out, Ftr.shape[1]


def score_dataset(X, y, seed):
    stratified = np.unique(y, return_counts=True)[1].max() >= 5
    folds = list((StratifiedKFold if stratified else KFold)(5, shuffle=True, random_state=seed).split(X, y))
    labels = np.unique(y)
    acc = lambda p: float((labels[p.argmax(axis=1)] == y).mean())
    rows = []
    for name, n in (("MR", 10000), ("MR2500", 2500)):
        p, f = oof_minirocket(X, y, folds, labels, n)
        rows.append(dict(model=name, score=acc(p), features=f))
    variants = {"DK1": (dilated_sources, False), "DK1_lf": (dilated_sources, True)}
    if X.shape[1] >= MIN_LENGTH:
        variants.update({"C": (context_sources, False), "C_lf": (context_sources, True)})
    for name, (src, lf) in variants.items():
        K = np.nan_to_num(kernels(X, seed, src, lf))
        p = oof_ridge("classification", K, y, folds, labels)
        if p is not None:
            rows.append(dict(model=name, score=acc(p), features=K.shape[1]))
    return [dict(r, seed=seed, T=X.shape[1]) for r in rows]


def main():
    known = set(pd.read_csv(OUT).dataset) if os.path.exists(OUT) else set()
    for name in bench.DEV_UCR:
        if name in known:
            continue
        try:
            X, y = load_train("classification", name)
            rows = with_retry(score_dataset, X, y, bench.RNG)
        except Exception as e:
            print(f"[failed] {name}: {str(e)[:120]}", flush=True)
            continue
        pd.DataFrame(rows).assign(dataset=name).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
        print(name, flush=True)


if __name__ == "__main__":
    main()
