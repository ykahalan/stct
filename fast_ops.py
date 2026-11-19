"""Pooling operators for Chebyshev kernels and pooled dilated views, on training splits only.

    python fast_ops.py --task classification --seeds 42
    python fast_ops.py --task regression --seeds 42
    python fast_ops.py --task classification --seeds 42 --summary

Two ideas not tried before:

* Pooling operators (MultiROCKET): every threshold kernel gives the share of
  positions above five quantile thresholds and the maximum.  At the middle
  threshold, also the mean excess of the positive values (MPV), the mean
  position of the positive values (MIPV) and the longest run of positive values
  (LSPV), each divided by the number of positions.  Variants add LSPV, MPV or
  all three to the context and dilated kernels (classification) or to the
  threshold kernels (regression).
* Pooled dilated views (PD): the ten statistics per channel of the dilated fits
  at each dilation, 59 per dilation, added to the forest input.

Everything else is the STCT+ of the paper tables (STCTClassifier()), scored out of
fold.  Results persist in results/fast_ops_<task>_scores.csv.  No test split is loaded.
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from numba import njit
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.model_selection import KFold, StratifiedKFold

import bench
import stct_fast as SF
from algo_select import load_train, with_retry
from fast_search import oof_ridge
from vote_test import MIN_LENGTH

warnings.filterwarnings("ignore")
MID = 2  # index of the middle threshold among the five quantiles 1/6..5/6
PPV_RESIDUAL = 7 * (SF.DEGREE + 2) + SF.DEGREE + 1


@njit(cache=True)
def _kernels_ops(Cx, mu, sd, D, B, mid):
    """_kernels plus MPV, MIPV and LSPV at threshold mid: (n, m * (q + 4)) laid out as
    [PPV q*m | max m | MPV m | MIPV m | LSPV m]."""
    n, ch, t = Cx.shape
    m, q = D.shape[0], B.shape[0]
    F = np.zeros((n, m * (q + 4)))
    Dc = D.T.copy()
    p = np.empty(m)
    counts = np.empty((q, m))
    top, excess, index, run, best = np.empty(m), np.empty(m), np.empty(m), np.empty(m), np.empty(m)
    for i in range(n):
        counts[:] = 0.0
        top[:] = -np.inf
        excess[:] = 0.0
        index[:] = 0.0
        run[:] = 0.0
        best[:] = 0.0
        for w in range(t):
            p[:] = 0.0
            for c in range(ch):
                z = (Cx[i, c, w] - mu[c]) / sd[c]
                dc = Dc[c]
                for k in range(m):
                    p[k] += dc[k] * z
            for j in range(q):
                b = B[j]
                for k in range(m):
                    counts[j, k] += p[k] > b[k]
            bm = B[mid]
            for k in range(m):
                top[k] = max(top[k], p[k])
                if p[k] > bm[k]:
                    excess[k] += p[k] - bm[k]
                    index[k] += w
                    run[k] += 1.0
                    best[k] = max(best[k], run[k])
                else:
                    run[k] = 0.0
        for k in range(m):
            for j in range(q):
                F[i, k * q + j] = counts[j, k] / t
            F[i, m * q + k] = top[k]
            npos = counts[mid, k]
            F[i, m * (q + 1) + k] = excess[k] / npos if npos > 0 else 0.0
            F[i, m * (q + 2) + k] = index[k] / (npos * max(t - 1, 1)) if npos > 0 else -1.0
            F[i, m * (q + 3) + k] = best[k] / t
    return F


def select(F, m, q, ops):
    """Columns of one _kernels_ops chunk: the base six per direction plus the named operators."""
    keep = [F[:, : m * (q + 1)]]
    for name, slot in (("mpv", 1), ("mipv", 2), ("lspv", 3)):
        if name in ops:
            keep.append(F[:, m * (q + slot) : m * (q + slot + 1)])
    return np.hstack(keep)


VARIANTS = {"base": (), "lspv": ("lspv",), "mpv": ("mpv",), "ops3": ("mpv", "mipv", "lspv")}


def kernel_blocks(chunks):
    """chunks: list of (Cx, params) per scale or dilation -> {variant: features}."""
    raw = [(_kernels_ops(Cx, *prm, MID), prm[2].shape[0], prm[3].shape[0]) for Cx, prm in chunks]
    return {v: np.nan_to_num(np.hstack([select(F, m, q, ops) for F, m, q in raw])) for v, ops in VARIANTS.items()}


def pooled_dilated(X):
    return np.hstack([np.delete(SF._pooled(SF.dilated(X, d, 1)), PPV_RESIDUAL, axis=1) for d in SF.dilations(X.shape[1])])


def score_classification(X, y, seed):
    T = X.shape[1]
    long_ = T >= MIN_LENGTH
    stratified = np.unique(y, return_counts=True)[1].max() >= 5
    folds = list((StratifiedKFold if stratified else KFold)(5, shuffle=True, random_state=seed).split(X, y))
    labels = np.unique(y)
    acc = lambda p: float((labels[p.argmax(axis=1)] == y).mean())
    kd = min(64, T // 16) if long_ else int(min(64, max(1, T // 8)))
    dp = SF.dilated_fit(X, kd, seed)
    Dv = kernel_blocks([(SF.dilated(X, d, 1), dp[i]) for i, d in enumerate(SF.dilations(T))])
    if long_:
        cp = SF.context_fit(X, kd, seed, 2)
        V, _, W = SF.trio_features(X, None, SF.compete_fit(X, seed, SF.winner_groups(T)))
        Cv = kernel_blocks([(SF.in_context(SF._sliding(X, *SF._setup(T, L)), L, 2), cp[i])
                            for i, L in enumerate(SF.window_lengths(T))])
        rw = oof_ridge("classification", np.nan_to_num(W), y, folds, labels)
    else:
        V = SF.features(X)[0]
    V = np.nan_to_num(V)
    PD = np.nan_to_num(pooled_dilated(X))

    def forest(F):
        out = np.zeros((len(y), len(labels)))
        for tr, te in folds:
            et = ExtraTreesClassifier(n_estimators=200, max_features=0.1, random_state=seed, n_jobs=-1).fit(F[tr], y[tr])
            out[te[:, None], np.searchsorted(labels, et.classes_)] = et.predict_proba(F[te])
        return out

    forests = {"F": forest(V), "F+PD": forest(np.hstack([V, PD]))}
    w = 1.0 if long_ else 0.5
    rows = []
    for v in VARIANTS:
        parts = [oof_ridge("classification", Dv[v], y, folds, labels)]
        feats = Dv[v].shape[1]
        if long_:
            parts += [oof_ridge("classification", Cv[v], y, folds, labels), rw]
            feats += Cv[v].shape[1] + W.shape[1]
        parts = [p for p in parts if p is not None]
        for fname, fp in forests.items():
            if fname == "F+PD" and v not in ("base", "ops3"):
                continue
            nf = V.shape[1] + (PD.shape[1] if fname == "F+PD" else 0)
            rows.append(dict(variant=f"{fname}|{v}", score=acc((fp + w * sum(parts)) / (1 + w * len(parts))), features=nf + feats))
    return [dict(r, seed=seed, T=T, n=len(y)) for r in rows]


def score_regression(X, y, seed):
    T = X.shape[1]
    long_ = T >= MIN_LENGTH
    folds = list(KFold(5, shuffle=True, random_state=seed).split(X))
    V = np.nan_to_num(SF.features(X)[0])
    PD = np.nan_to_num(pooled_dilated(X))

    def forest(F):
        out = np.zeros(len(y))
        for tr, te in folds:
            et = ExtraTreesRegressor(n_estimators=50, max_features=0.5, min_samples_leaf=2, random_state=seed, n_jobs=-1)
            out[te] = et.fit(F[tr], y[tr]).predict(F[te])
        return out

    forests = {"F": forest(V), "F+PD": forest(np.hstack([V, PD]))}
    rmse = lambda p: float(np.sqrt(((p - y) ** 2).mean()))
    rows = []
    if long_:
        kp = SF.kernel_fit(X, SF.n_kernels(T), seed, max_series=32)
        Kv = kernel_blocks([(SF._sliding(X, *SF._setup(T, L)), kp[i]) for i, L in enumerate(SF.window_lengths(T))])
    for fname, fp in forests.items():
        nf = V.shape[1] + (PD.shape[1] if fname == "F+PD" else 0)
        if not long_:
            rows.append(dict(variant=f"{fname}|base", score=rmse(fp), features=nf))
            continue
        for v in VARIANTS:
            r = oof_ridge("regression", Kv[v], y, folds, np.zeros(1))
            rows.append(dict(variant=f"{fname}|{v}", score=rmse(fp if r is None else (fp + r) / 2), features=nf + Kv[v].shape[1]))
    return [dict(r, seed=seed, T=T, n=len(y)) for r in rows]


def summary(task, seeds):
    d = pd.read_csv(f"results/fast_ops_{task}_scores.csv")
    d = d[d.seed.isin(seeds)].drop_duplicates(["seed", "dataset", "variant"], keep="last")
    piv = d.pivot_table(index=["seed", "dataset"], columns="variant", values="score")
    feats = d.groupby("variant").features.mean()
    base = piv["F|base"]
    T = d.drop_duplicates(["seed", "dataset"]).set_index(["seed", "dataset"])["T"].reindex(piv.index)
    rows = []
    for v in piv.columns:
        ok = piv[v].notna()
        if task == "classification":
            g = (piv.loc[ok, v] - base[ok]) * 100
            stat = dict(gain=g.mean(), by_seed=" ".join(f"{x:+.2f}" for x in g.groupby(level="seed").mean()),
                        short=g[T[ok] < MIN_LENGTH].mean(), long=g[T[ok] >= MIN_LENGTH].mean(),
                        better=int((g > 0).sum()), worse=int((g < 0).sum()))
        else:
            r = np.log(piv.loc[ok, v] / base[ok])
            stat = dict(ratio=np.exp(r.mean()), by_seed=" ".join(f"{np.exp(x):.3f}" for x in r.groupby(level="seed").mean()),
                        better=int((r < 0).sum()), worse=int((r > 0).sum()))
        rows.append(dict(variant=v, **stat, n=int(ok.sum()), features=feats[v]))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["classification", "regression"], required=True)
    ap.add_argument("--seeds", default=str(bench.RNG))
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    if a.summary:
        return summary(a.task, seeds)
    out = f"results/fast_ops_{a.task}_scores.csv"
    known = set()
    if os.path.exists(out):
        p = pd.read_csv(out)
        known = set(zip(p.seed, p.dataset))
    score = score_classification if a.task == "classification" else score_regression
    for seed in seeds:
        for name in (bench.UCR if a.task == "classification" else bench.TSER):
            if (seed, name) in known:
                continue
            try:
                X, y = load_train(a.task, name)
                rows = with_retry(score, X, y, seed)
            except Exception as e:
                print(f"[failed] {seed} {name}: {str(e)[:120]}", flush=True)
                continue
            pd.DataFrame(rows).assign(dataset=name).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
            print(f"{seed} {name}", flush=True)


if __name__ == "__main__":
    main()
