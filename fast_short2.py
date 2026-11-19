"""Second round for series shorter than 128, using training splits only.

    python fast_short2.py --seeds 42
    python fast_short2.py --seeds 42,1,2 --summary

STCT+ below length 128 averages the forest (200 trees, 10% of features per
split) on the views with a half-weight ridge on level-free dilated kernels
(9 samples, degree 4, n0 = min(64, max(1, T // 8)) directions per dilation).
The earlier round (fast_short.py) kept everything below QUANT's feature count;
here the kernel budget may grow while the total stays far below MiniROCKET's.
Candidates, each blended with the same out-of-fold forest:

* D9xk: the current dilated kernels with k times the directions (k = 2, 4) or 64
* D7, D5: stencils of 7 samples (degree 4) and 5 samples (degree 3), which allow
  more dilations on short series; D9+D7 shares one ridge
* K: threshold kernels on the plain sliding fits, as its own ridge or with D9
* weight 1 instead of 1/2 for the larger blocks

Results persist in results/fast_short2_scores.csv.  No test split is loaded.

D9x64 at weight 1 passed on three seeds (+0.41 on short series) and was scored
once on the test splits (results/short64_test_classification.csv): 86.74 vs
86.72 for STCT+, 13/7 on short series (p = 0.50), with 231 more features on
average.  It was not adopted.
"""
import argparse
import os
import time
import warnings

import numpy as np
import numpy.polynomial.chebyshev as C
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import KFold, StratifiedKFold

import bench
import stct_fast as SF
from algo_select import load_train, with_retry
from fast_search import oof_ridge
from vote_test import MIN_LENGTH

warnings.filterwarnings("ignore")
OUT = "results/fast_short2_scores.csv"


def stencil_fit(X, n_dir, seed, taps, degree, max_series=32):
    """dilated_fit for a stencil of taps samples and the given degree."""
    rng = np.random.default_rng(seed)
    S = X[rng.choice(len(X), max_series, replace=False)] if len(X) > max_series else X
    params = []
    for d in SF.dilations(X.shape[1], taps):
        Cx = SF.dilated(S, d, 1, taps, degree)
        flat = Cx.transpose(1, 0, 2).reshape(Cx.shape[1], -1)
        mu, sd = flat.mean(axis=1), flat.std(axis=1) + 1e-12
        D = rng.standard_normal((n_dir, Cx.shape[1]))
        D[:, 0] = 0.0
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        params.append((mu, sd, D, np.quantile(D @ ((flat - mu[:, None]) / sd[:, None]), SF.QS, axis=1)))
    return params


def stencil_features(X, params, taps, degree):
    return np.hstack([SF._kernels(SF.dilated(X, d, 1, taps, degree), *params[i])
                      for i, d in enumerate(SF.dilations(X.shape[1], taps))])


def score_dataset(X, y, seed):
    T = X.shape[1]
    n0 = int(min(64, max(1, T // 8)))
    t0 = time.perf_counter()
    F = np.nan_to_num(SF.features(X)[0])
    blocks = {}
    for name, (k, taps, deg) in {"D9": (n0, 9, 4), "D9x2": (2 * n0, 9, 4), "D9x4": (4 * n0, 9, 4), "D9x64": (64, 9, 4),
                                 "D7": (n0, 7, 4), "D7x2": (2 * n0, 7, 4), "D5": (n0, 5, 3)}.items():
        blocks[name] = np.nan_to_num(stencil_features(X, stencil_fit(X, k, seed, taps, deg), taps, deg))
    blocks["K"] = np.nan_to_num(SF.features(X, SF.kernel_fit(X, n0, seed, max_series=32))[1])
    blocks["D9+D7"] = np.hstack([blocks["D9"], blocks["D7"]])
    blocks["D9x2+D7x2"] = np.hstack([blocks["D9x2"], blocks["D7x2"]])
    blocks["D9+K"] = np.hstack([blocks["D9"], blocks["K"]])
    t_feat = time.perf_counter() - t0
    stratified = np.unique(y, return_counts=True)[1].max() >= 5
    folds = list((StratifiedKFold if stratified else KFold)(5, shuffle=True, random_state=seed).split(X, y))
    labels = np.unique(y)
    forest = np.zeros((len(y), len(labels)))
    for tr, te in folds:
        et = ExtraTreesClassifier(n_estimators=200, max_features=0.1, random_state=seed, n_jobs=-1).fit(F[tr], y[tr])
        forest[te[:, None], np.searchsorted(labels, et.classes_)] = et.predict_proba(F[te])
    ridges = {k: oof_ridge("classification", B, y, folds, labels) for k, B in blocks.items()}
    acc = lambda p: float((labels[p.argmax(axis=1)] == y).mean())
    blends = {"F": [], "cur": [("D9", 0.5)], "D9x2": [("D9x2", 0.5)], "D9x4": [("D9x4", 0.5)],
              "D9x64": [("D9x64", 0.5)], "D9x4w1": [("D9x4", 1.0)], "D9x64w1": [("D9x64", 1.0)],
              "D7": [("D7", 0.5)], "D5": [("D5", 0.5)], "D9+D7": [("D9+D7", 0.5)],
              "D9x2+D7x2": [("D9x2+D7x2", 0.5)], "D9x2+D7x2w1": [("D9x2+D7x2", 1.0)],
              "cur+K": [("D9", 0.5), ("K", 0.5)], "D9+K": [("D9+K", 0.5)]}
    rows = []
    for name, parts in blends.items():
        if any(ridges[b] is None for b, _ in parts):
            continue
        p = forest + sum(w * ridges[b] for b, w in parts)
        feats = F.shape[1] + sum(blocks[b].shape[1] for b, _ in parts)
        rows.append(dict(blend=name, score=acc(p), features=feats))
    return [dict(r, seed=seed, T=T, n=len(y), feat_seconds=t_feat) for r in rows]


def summary(seeds):
    d = pd.read_csv(OUT)
    d = d[d.seed.isin(seeds)].drop_duplicates(["seed", "dataset", "blend"], keep="last")
    piv = d.pivot_table(index=["seed", "dataset"], columns="blend", values="score")
    feats = d.pivot_table(index="dataset", columns="blend", values="features").mean()
    base = piv["cur"]
    rows = []
    for b in piv.columns:
        diff = (piv[b] - base) * 100
        rows.append(dict(blend=b, gain=diff.mean(), by_seed=" ".join(f"{v:+.2f}" for v in diff.groupby(level="seed").mean()),
                         better=int((diff > 0).sum()), worse=int((diff < 0).sum()), n=int(diff.notna().sum()),
                         features=feats[b]))
    print(pd.DataFrame(rows).sort_values("gain", ascending=False).to_string(index=False, float_format=lambda v: f"{v:.2f}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=str(bench.RNG))
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    if a.summary:
        return summary(seeds)
    known = set()
    if os.path.exists(OUT):
        p = pd.read_csv(OUT)
        known = set(zip(p.seed, p.dataset))
    for seed in seeds:
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
