"""Tune the STCT+ regressor on training splits only.

    python fast_reg_tune.py --seeds 42
    python fast_reg_tune.py --seeds 42,1,2 --summary

The current regressor averages ExtraTrees (50 trees, half the features, 2 per
leaf) on the views (F) with a ridge on threshold kernels (K) from length 128 up.
Every member below is fitted out of fold (5-fold CV inside the training split)
on the same features and seeds; blends are plain averages of members.  Members:

* ET: the current forest; ET100: 100 trees; HGB: histogram gradient boosting
* a "log" suffix: the member is fitted to log(y) and exponentiated, when all y > 0
* K: ridge on threshold kernels from length 128 up (current); Kall: at every length
* C: ridge on context kernels from length 128 up; RF: ridge on the views

Fit plus predict seconds of each member are stored so runtime changes can be
checked.  Results persist in results/fast_reg_tune_scores.csv.  No test split
is loaded.
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import KFold

import bench
import stct_fast as SF
from algo_select import load_train, with_retry
from vote_test import MIN_LENGTH

warnings.filterwarnings("ignore")
OUT = "results/fast_reg_tune_scores.csv"

BLENDS = {
    "fast": ("ET", "K"),
    "ET": ("ET",),
    "HGB": ("HGB",),
    "ET+HGB": ("ET", "HGB"),
    "HGB+K": ("HGB", "K"),
    "ET+HGB+K": ("ET", "HGB", "K"),
    "ET100+K": ("ET100", "K"),
    "ETlog+K": ("ETlog", "K"),
    "HGBlog+K": ("HGBlog", "K"),
    "ETlog+HGBlog+K": ("ETlog", "HGBlog", "K"),
    "ET+Kall": ("ET", "Kall"),
    "ET+C": ("ET", "C"),
    "ET+K+C": ("ET", "K", "C"),
    "ET+K+RF": ("ET", "K", "RF"),
    "ET+HGB+Kall": ("ET", "HGB", "Kall"),
}


def oof(make, Z, y, folds, log=False):
    """Out-of-fold predictions and total fit + predict seconds; None if the member cannot be fitted."""
    if log and (y <= 0).any():
        return None, 0.0
    target = np.log(y) if log else y
    out, t0 = np.zeros(len(y)), time.perf_counter()
    try:
        for tr, te in folds:
            out[te] = make().fit(Z[tr], target[tr]).predict(Z[te])
    except np.linalg.LinAlgError:
        return None, 0.0
    return (np.exp(out) if log else out), time.perf_counter() - t0


def score_dataset(X, y, seed):
    T = X.shape[1]
    long_ = T >= MIN_LENGTH
    t0 = time.perf_counter()
    kp = SF.kernel_fit(X, SF.n_kernels(T), seed, max_series=32)
    F, Kall = SF.features(X, kp)
    F, Kall = np.nan_to_num(F), np.nan_to_num(Kall)
    t_feat = time.perf_counter() - t0
    C = None
    if long_:
        cp = SF.context_fit(X, min(64, T // 16), seed, 2)
        _, C, _ = SF.trio_features(X, cp, None)
        C = np.nan_to_num(C)
    folds = list(KFold(5, shuffle=True, random_state=seed).split(F))
    et = lambda n: (lambda: ExtraTreesRegressor(n_estimators=n, max_features=0.5, min_samples_leaf=2,
                                                random_state=seed, n_jobs=-1))
    hgb = lambda: HistGradientBoostingRegressor(random_state=seed)
    ridge = lambda: bench.make_learner("ridge", "regression")
    members = {"ET": oof(et(50), F, y, folds), "ET100": oof(et(100), F, y, folds),
               "HGB": oof(hgb, F, y, folds), "ETlog": oof(et(50), F, y, folds, log=True),
               "HGBlog": oof(hgb, F, y, folds, log=True), "Kall": oof(ridge, Kall, y, folds),
               "RF": oof(ridge, F, y, folds)}
    members["K"] = members["Kall"] if long_ else (None, 0.0)
    members["C"] = oof(ridge, C, y, folds) if long_ else (None, 0.0)
    rmse = lambda p: float(np.sqrt(((p - y) ** 2).mean()))
    rows = []
    for name, parts in BLENDS.items():
        preds = [members[p][0] for p in parts if members[p][0] is not None]
        if len(preds) < len([p for p in parts if not (p in ("K", "C") and not long_)]):
            continue  # a member that should exist failed (e.g. log target with y <= 0)
        secs = sum(members[p][1] for p in parts)
        rows.append(dict(blend=name, score=rmse(np.mean(preds, axis=0)), member_seconds=secs))
    return [dict(r, seed=seed, T=T, n=len(y), feat_seconds=t_feat) for r in rows]


def summary(seeds):
    d = pd.read_csv(OUT)
    d = d[d.seed.isin(seeds)].drop_duplicates(["seed", "dataset", "blend"], keep="last")
    piv = d.pivot_table(index=["seed", "dataset"], columns="blend", values="score")
    base = piv["fast"]
    rows = []
    for b in piv.columns:
        ok = piv[b].notna() & base.notna()
        r = np.log(piv.loc[ok, b] / base[ok])
        per_seed = r.groupby(level="seed").mean().apply(np.exp)
        sec = d[d.blend == b].member_seconds.sum() / d[d.blend == "fast"].member_seconds.sum()
        rows.append(dict(blend=b, ratio=np.exp(r.mean()), by_seed=" ".join(f"{v:.3f}" for v in per_seed),
                         better=int((r < 0).sum()), worse=int((r > 0).sum()), n=int(ok.sum()), time_vs_fast=sec))
    print(pd.DataFrame(rows).sort_values("ratio").to_string(index=False, float_format=lambda v: f"{v:.3f}"))


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
        for name in bench.TSER:
            if (seed, name) in known:
                continue
            try:
                X, y = load_train("regression", name)
                rows = with_retry(score_dataset, X, y, seed)
            except Exception as e:
                print(f"[failed] {seed} {name}: {str(e)[:120]}", flush=True)
                continue
            pd.DataFrame(rows).assign(dataset=name).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
            print(f"{seed} {name}", flush=True)


if __name__ == "__main__":
    main()
