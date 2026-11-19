"""A linear learner on position-aligned Chebyshev fits, on training splits only.

    python fast_aligned.py --seeds 42
    python fast_aligned.py --seeds 42 --summary

Every ridge in STCT+ sees position-free counts (kernels pooled over the
series); position-aligned features (fixed windows) reach only the forest.  On
ChlorineConcentration a ridge on the raw series scores 79.7 out of fold against
66.6 for STCT+.  Variants add to the STCT+ blend (STCTClassifier()) one more ridge,
at the weight of the others, on the fit coefficients and residual RMS of fixed
windows of about L samples (T // L windows, at most 128):

* A5: windows of 5 samples, where the degree-4 fit is exact (an invertible
  transform of the series; longer windows plus residual RMS above 640 samples)
* A8, A16: windows of about 8 or 16 samples
* A8+16: both window lengths in one ridge
* F+A16: the A16 features added to the forest input instead

Results persist in results/fast_aligned_scores.csv.  No test split is loaded.
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from numpy.polynomial import chebyshev as C
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import bench
import stct_fast as SF
import stct_plus as SP
from algo_select import load_train, with_retry
from fast_params import A3, kernel_blocks
from stct_models import _soft
from vote_test import MIN_LENGTH

warnings.filterwarnings("ignore")
OUT = "results/fast_aligned_scores.csv"
MAX_WINDOWS = 128


def aligned(X, L):
    return SP.stct_fixed(X, (int(np.clip(X.shape[1] // L, 1, MAX_WINDOWS)),))


def exact(X, L=5):
    """Degree-4 fits in tiling windows of 5 samples (an exact, invertible transform),
    or longer windows with their residual RMS when 5 would give more than MAX_WINDOWS."""
    n, T = X.shape
    L = max(L, -(-T // MAX_WINDOWS))
    W = T // L
    V = C.chebvander(np.linspace(-1, 1, L), min(SF.DEGREE, L - 1))
    P = np.linalg.pinv(V)
    S = X[:, : W * L].reshape(n, W, L)
    Z = S @ P.T
    out = [Z.reshape(n, -1)]
    if L > SF.DEGREE + 1:
        out.append(np.sqrt(((S - Z @ V.T) ** 2).mean(axis=-1)))
    return np.hstack(out)


def score_dataset(X, y, seed):
    T = X.shape[1]
    long_ = T >= MIN_LENGTH
    stratified = np.unique(y, return_counts=True)[1].max() >= 5
    folds = list((StratifiedKFold if stratified else KFold)(5, shuffle=True, random_state=seed).split(X, y))
    labels = np.unique(y)
    acc = lambda p: float((labels[p.argmax(axis=1)] == y).mean())
    V = np.nan_to_num(SF.features(X)[0])
    t0 = time.perf_counter()
    A = {L: np.nan_to_num(aligned(X, L)) for L in (8, 16)}
    A["8+16"] = np.hstack([A[8], A[16]])
    a_secs = time.perf_counter() - t0
    A[5] = exact(X)

    def forest(F):
        out = np.zeros((len(y), len(labels)))
        for tr, te in folds:
            et = ExtraTreesClassifier(n_estimators=200, max_features=0.1, random_state=seed, n_jobs=-1).fit(F[tr], y[tr])
            out[te[:, None], np.searchsorted(labels, et.classes_)] = et.predict_proba(F[te])
        return out

    def ridge(B):
        out = np.zeros((len(y), len(labels)))
        try:
            for tr, te in folds:
                r = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=A3)).fit(B[tr], y[tr])
                out[te[:, None], np.searchsorted(labels, r.classes_)] = _soft(r.decision_function(B[te]))
        except np.linalg.LinAlgError:
            return None
        return out

    blocks = {k: np.nan_to_num(B) for k, B in kernel_blocks(X, seed, SF.QS).items()}
    rps = [r for r in (ridge(B) for B in blocks.values()) if r is not None]
    kfeats = sum(B.shape[1] for B in blocks.values())
    w = 1.0 if long_ else 0.5
    blend = lambda fp, extra: (fp + w * sum(rps + extra)) / (1 + w * len(rps + extra))
    f0 = forest(V)
    rows = [dict(variant="base", score=acc(blend(f0, [])), features=V.shape[1] + kfeats)]
    for k in (5, 8, 16, "8+16"):
        ra = ridge(A[k])
        rows.append(dict(variant=f"A{k}", score=acc(blend(f0, [] if ra is None else [ra])), features=V.shape[1] + kfeats + A[k].shape[1]))
        if ra is not None:
            rows.append(dict(variant=f"ridge:A{k}", score=acc(ra), features=A[k].shape[1]))
    rows.append(dict(variant="F+A16", score=acc(blend(forest(np.hstack([V, A[16]])), [])), features=V.shape[1] + kfeats + A[16].shape[1]))
    return [dict(r, seed=seed, T=T, n=len(y), aligned_seconds=a_secs) for r in rows]


def summary(seeds):
    d = pd.read_csv(OUT)
    d = d[d.seed.isin(seeds)].drop_duplicates(["seed", "dataset", "variant"], keep="last")
    piv = d.pivot_table(index=["seed", "dataset"], columns="variant", values="score")
    T = d.drop_duplicates(["seed", "dataset"]).set_index(["seed", "dataset"])["T"].reindex(piv.index)
    n = d.drop_duplicates(["seed", "dataset"]).set_index(["seed", "dataset"])["n"].reindex(piv.index)
    feats = d.groupby("variant").features.mean()
    rows = []
    for v in piv.columns:
        if v.startswith("ridge:"):
            continue
        g = (piv[v] - piv["base"]) * 100
        rows.append(dict(variant=v, gain=g.mean(), by_seed=" ".join(f"{x:+.2f}" for x in g.groupby(level="seed").mean()),
                         short=g[T < MIN_LENGTH].mean(), long=g[T >= MIN_LENGTH].mean(), large_n=g[n > 250].mean(),
                         better=int((g > 0).sum()), worse=int((g < 0).sum()), features=feats[v]))
    print(pd.DataFrame(rows).sort_values("gain", ascending=False).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    best = max((r for r in rows if r["variant"] != "base"), key=lambda r: r["gain"])["variant"]
    g = (piv[best] - piv["base"]) * 100
    print(f"\nlargest changes for {best}:\n", g.sort_values().iloc[[*range(5), *range(-8, 0)]].round(2).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=str(bench.RNG))
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--datasets", default=None)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    if a.summary:
        return summary(seeds)
    known = set()
    if os.path.exists(OUT):
        p = pd.read_csv(OUT)
        known = set(zip(p.seed, p.dataset))
    names = a.datasets.split(",") if a.datasets else bench.UCR
    for seed in seeds:
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
