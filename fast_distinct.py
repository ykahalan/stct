"""Test computing collapsed paper windows once on short series, using training splits only.

    python fast_distinct.py --seeds 42,1,2

Below length 128 every recipe is the forest alone.  On short series the 8- and
16-window paper fits can give the same windows, which duplicates their columns.
Out-of-fold accuracy of the forest (100 and 200 trees) with and without the
duplicates.  Results persist in results/fast_distinct_scores.csv.  No test split
is loaded.
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
from algo_select import load_train, with_retry
from vote_test import MIN_LENGTH

warnings.filterwarnings("ignore")
OUT = "results/fast_distinct_scores.csv"
TREES = (100, 200)


def score_dataset(X, y, seed):
    stratified = np.unique(y, return_counts=True)[1].max() >= 5
    folds = list((StratifiedKFold if stratified else KFold)(5, shuffle=True, random_state=seed).split(X, y))
    labels = np.unique(y)
    rows = []
    for distinct in (False, True):
        F = np.nan_to_num(SF.trio_features(X, distinct=distinct)[0])
        hits = {n: np.zeros(len(y), bool) for n in TREES}
        for tr, te in folds:
            et = ExtraTreesClassifier(n_estimators=max(TREES), max_features=0.1, random_state=seed, n_jobs=-1).fit(F[tr], y[tr])
            run, cols = np.zeros((len(te), len(labels))), np.searchsorted(labels, et.classes_)
            for i, tree in enumerate(et.estimators_, 1):
                run[:, cols] += tree.predict_proba(F[te])
                if i in hits:
                    hits[i][te] = labels[run.argmax(axis=1)] == y[te]
        for n in TREES:
            rows.append(dict(distinct=distinct, trees=n, score=float(hits[n].mean()), features=F.shape[1],
                             budget=quant_features(X.shape[1]) - 1, seed=seed, T=X.shape[1]))
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
