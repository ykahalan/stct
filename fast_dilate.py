"""Screen dilated STCT blocks, using training splits only.

    python fast_dilate.py --datasets dev --seeds 42

Dilated STCT fits the degree-4 Chebyshev polynomial to 9 samples spaced d apart,
d = 1, 2, 4, ..., the receptive fields of MiniRocket's kernels.  The coefficients
feed the existing threshold kernels (DK: one window every d samples; DK1: every
sample) and winner counts (DW).  Out-of-fold accuracy of each ridge alone, of
MiniRocket + ridge for reference, and of the forest blended with every set of up
to three ridges.  Results persist in results/fast_dilate_scores.csv.  No test
split is loaded.
"""
import argparse
import itertools
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import KFold, StratifiedKFold

import ablate
import bench
import stct_fast as SF
from algo_select import load_train, with_retry
from fast_search import oof_ridge
from vote_test import MIN_LENGTH

warnings.filterwarnings("ignore")
OUT = "results/fast_dilate_scores.csv"


def _sample(X, rng, max_series=32):
    return X[rng.choice(len(X), max_series, replace=False)] if len(X) > max_series else X


def dk_fit(X, n_dir, seed, div):
    """Weightings and thresholds per dilation, from at most 32 series, no labels."""
    rng = np.random.default_rng(seed)
    S, params = _sample(X, rng), []
    for d in SF.dilations(X.shape[1]):
        Cx = SF.dilated(S, d, max(1, d // div))
        flat = Cx.transpose(1, 0, 2).reshape(Cx.shape[1], -1)
        mu, sd = flat.mean(axis=1), flat.std(axis=1) + 1e-12
        D = rng.standard_normal((n_dir, Cx.shape[1]))
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        params.append((mu, sd, D, np.quantile(D @ ((flat - mu[:, None]) / sd[:, None]), SF.QS, axis=1)))
    return params


def dk_features(X, params, div):
    T, out = X.shape[1], []
    for s in range(0, len(X), SF.BLOCK):
        Xb = X[s : s + SF.BLOCK]
        out.append(np.hstack([SF._kernels(SF.dilated(Xb, d, max(1, d // div)), *params[i])
                              for i, d in enumerate(SF.dilations(T))]))
    return np.vstack(out)


def dw_fit(X, seed, groups, div, k=8):
    rng = np.random.default_rng(seed)
    S, params = _sample(X, rng), []
    for d in SF.dilations(X.shape[1]):
        flat = SF.dilated(S, d, max(1, d // div)).transpose(1, 0, 2).reshape(SF.DEGREE + 2, -1)
        D = {}
        for shape, dim in ((False, SF.DEGREE + 2), (True, SF.DEGREE)):
            D[shape] = rng.standard_normal((groups * k, dim))
            D[shape] /= np.linalg.norm(D[shape], axis=1, keepdims=True)
        params.append((flat.mean(axis=1), flat.std(axis=1) + 1e-12, D))
    return params


def dw_features(X, params, div, k=8):
    T, out = X.shape[1], []
    for s in range(0, len(X), SF.BLOCK):
        Xb, row = X[s : s + SF.BLOCK], []
        for i, d in enumerate(SF.dilations(T)):
            Cx = SF.dilated(Xb, d, max(1, d // div))
            mu, sd, D = params[i]
            for shape in (False, True):
                F = SF._compete(SF.compete_inputs(Cx, mu, sd, shape), D[shape], k)
                m = F.shape[1] // 2
                row.append(np.hstack([np.sqrt(F[:, :m]), F[:, m:]]))
        out.append(np.hstack(row))
    return np.vstack(out)


def blocks(X, seed):
    """Feature blocks with their extraction seconds."""
    T, out = X.shape[1], {}
    makers = {"DK": lambda: dk_features(X, dk_fit(X, SF.n_kernels(T), seed, 1), 1),
              "DK1": lambda: dk_features(X, dk_fit(X, SF.n_kernels(T), seed, 10**6), 10**6),
              "DW": lambda: dw_features(X, dw_fit(X, seed, SF.winner_groups(T), 1), 1)}
    if T >= MIN_LENGTH:
        makers["C"] = lambda: SF.context_features(X, SF.context_fit(X, SF.n_kernels(T), seed, 2), 2)
        makers["W"] = lambda: SF.compete_features(X, SF.compete_fit(X, seed, SF.winner_groups(T)))
    for k, f in makers.items():
        t = time.perf_counter()
        out[k] = (np.nan_to_num(f()), time.perf_counter() - t)
    return out


def oof_minirocket(X, y, folds, labels):
    from aeon.transformations.collection.convolution_based import MiniRocket

    out = np.zeros((len(y), len(labels)))
    for tr, te in folds:
        m = MiniRocket(n_jobs=-1, random_state=bench.RNG)
        Ftr, Fte = m.fit_transform(X[tr][:, None, :]), m.transform(X[te][:, None, :])
        r = bench.make_learner("ridge", "classification").fit(Ftr, y[tr])
        out[te[:, None], np.searchsorted(labels, r.classes_)] = ablate.soft(r.decision_function(Fte))
    return out


def score_dataset(X, y, seed):
    T = X.shape[1]
    F = np.nan_to_num(SF.features(X)[0])
    stratified = np.unique(y, return_counts=True)[1].max() >= 5
    folds = list((StratifiedKFold if stratified else KFold)(5, shuffle=True, random_state=seed).split(F, y))
    labels = np.unique(y)
    acc = lambda p: float((labels[p.argmax(axis=1)] == y).mean())
    forest = np.zeros((len(y), len(labels)))
    for tr, te in folds:
        et = ExtraTreesClassifier(n_estimators=100, max_features=0.1, random_state=seed, n_jobs=-1).fit(F[tr], y[tr])
        forest[te[:, None], np.searchsorted(labels, et.classes_)] = et.predict_proba(F[te])
    sets = blocks(X, seed)
    ridges = {k: oof_ridge("classification", K, y, folds, labels) for k, (K, _) in sets.items()}
    ridges = {k: v for k, v in ridges.items() if v is not None}
    rows = [dict(blend="F", model="blend", score=acc(forest), features=F.shape[1], seconds=0.0),
            dict(blend="MR", model="ridge", score=acc(oof_minirocket(X, y, folds, labels)), features=9996, seconds=0.0)]
    for k, p in ridges.items():
        rows.append(dict(blend=k, model="ridge", score=acc(p), features=sets[k][0].shape[1], seconds=sets[k][1]))
    for n in (1, 2, 3):
        for combo in itertools.combinations(sorted(ridges), n):
            p = forest + sum(ridges[c] for c in combo)
            rows.append(dict(blend="F+" + "+".join(combo), model="blend", score=acc(p),
                             features=F.shape[1] + sum(sets[c][0].shape[1] for c in combo),
                             seconds=sum(sets[c][1] for c in combo)))
    return [dict(r, seed=seed, T=T) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=str(bench.RNG))
    ap.add_argument("--datasets", default="dev")
    a = ap.parse_args()
    names = bench.DEV_UCR if a.datasets == "dev" else bench.UCR
    known = set()
    if os.path.exists(OUT):
        p = pd.read_csv(OUT)
        known = set(zip(p.seed, p.dataset))
    for seed in [int(s) for s in a.seeds.split(",")]:
        for name in names:
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
