"""Does the kernel budget limit STCT+ where training sets are large?  Training splits only.

    python fast_budget.py --seeds 42
    python fast_budget.py --seeds 42 --summary

On the test splits STCT+ leads the convolution methods on small training sets
and trails them on large ones and on lengths 128-300, where its context and
dilated kernels get only min(64, T // 16) directions per scale or dilation
(10 at length 166).  On the hardest datasets the dilated ridge with 64
directions alone beats the whole blend.  For series of length 128 and up this
scores, out of fold, the current blend (forest + context, winner and dilated
ridges at equal weight) against blends with more dilated directions, fewer
weak blocks, or a heavier dilated weight.  Results persist in results/fast_budget_scores.csv.
No test split is loaded.
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
from fast_search import oof_ridge
from vote_test import MIN_LENGTH

warnings.filterwarnings("ignore")
OUT = "results/fast_budget_scores.csv"


def score_dataset(X, y, seed):
    T = X.shape[1]
    stratified = np.unique(y, return_counts=True)[1].max() >= 5
    folds = list((StratifiedKFold if stratified else KFold)(5, shuffle=True, random_state=seed).split(X, y))
    labels = np.unique(y)
    V, _, W = SF.trio_features(X, None, SF.compete_fit(X, seed, SF.winner_groups(T)))
    V, W = np.nan_to_num(V), np.nan_to_num(W)
    forest = np.zeros((len(y), len(labels)))
    for tr, te in folds:
        et = ExtraTreesClassifier(n_estimators=200, max_features=0.1, random_state=seed, n_jobs=-1).fit(V[tr], y[tr])
        forest[te[:, None], np.searchsorted(labels, et.classes_)] = et.predict_proba(V[te])
    acc = lambda p: float((labels[p.argmax(axis=1)] == y).mean())
    k0 = min(64, T // 16)
    blocks, secs = {"W": W}, {"W": 0.0}
    for k in sorted({k0, 32, 64, 128}):
        t0 = time.perf_counter()
        blocks[f"C{k}"] = np.nan_to_num(SF.trio_features(X, SF.context_fit(X, k, seed, 2), None)[1])
        secs[f"C{k}"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        blocks[f"D{k}"] = np.nan_to_num(SF.dilated_features(X, SF.dilated_fit(X, k, seed)))
        secs[f"D{k}"] = time.perf_counter() - t0
    ridge = {b: oof_ridge("classification", B, y, folds, labels) for b, B in blocks.items()}
    # members joined by "+", each at weight 1 unless prefixed by a weight like "2*"; "F" is the forest
    blends = {"cur": f"F+C{k0}+W+D{k0}", "F+C+W+D32": f"F+C{k0}+W+D32", "F+C+W+D64": f"F+C{k0}+W+D64",
              "F+D64": "F+D64", "F+D128": "F+D128", "F+C+D64": f"F+C{k0}+D64", "F+C32+D64": "F+C32+D64",
              "F+2D64": "F+2*D64", "F+C+W+2D64": f"F+C{k0}+W+2*D64", "D64": "D64", "F+C64+W+D64": "F+C64+W+D64",
              "F+C32+W+D64": "F+C32+W+D64"}
    rows = []
    for name, spec in blends.items():
        p, used, feats, t = 0.0, 0.0, V.shape[1] if "F" in spec.split("+") else 0, 0.0
        for part in spec.split("+"):
            w, b = (float(part.split("*")[0]), part.split("*")[1]) if "*" in part else (1.0, part)
            if b == "F":
                p, used = p + w * forest, used + w
            elif ridge[b] is not None:
                p, used, feats, t = p + w * ridge[b], used + w, feats + blocks[b].shape[1], t + secs[b]
        rows.append(dict(budget=name, k=k0, score=acc(p / used), features=feats, kernel_seconds=t))
    return [dict(r, seed=seed, T=T, n=len(y)) for r in rows]


def summary(seeds):
    d = pd.read_csv(OUT)
    d = d[d.seed.isin(seeds)].drop_duplicates(["seed", "dataset", "budget"], keep="last")
    piv = d.pivot_table(index=["seed", "dataset"], columns="budget", values="score")
    n = d.drop_duplicates(["seed", "dataset"]).set_index(["seed", "dataset"])["n"].reindex(piv.index)
    T = d.drop_duplicates(["seed", "dataset"]).set_index(["seed", "dataset"])["T"].reindex(piv.index)
    feats = d.groupby("budget").features.mean()
    diff = (piv.sub(piv["cur"], axis=0) * 100).drop(columns="cur")
    print("mean gain over the current budget (points), better/worse, mean features")
    for b in diff.columns:
        print(f"  {b:5s} {diff[b].mean():+.2f}  {int((diff[b] > 0).sum())}/{int((diff[b] < 0).sum())}  {feats[b]:.0f} (cur {feats['cur']:.0f})")
    print("\nby training size")
    print(diff.groupby(pd.cut(n, [0, 100, 250, 500, 1e6])).mean().round(2).assign(count=diff.groupby(pd.cut(n, [0, 100, 250, 500, 1e6])).size()).to_string())
    print("\nby length")
    print(diff.groupby(pd.cut(T, [127, 300, 700, 1e6])).mean().round(2).to_string())


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
                if X.shape[1] < MIN_LENGTH:
                    continue
                rows = with_retry(score_dataset, X, y, seed)
            except Exception as e:
                print(f"[failed] {seed} {name}: {str(e)[:120]}", flush=True)
                continue
            pd.DataFrame(rows).assign(dataset=name).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
            print(f"{seed} {name}", flush=True)


if __name__ == "__main__":
    main()
