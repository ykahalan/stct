"""Test Chebyshev words and sharper ridge probabilities, using training splits only.

    python fast_words.py --seeds 42,1,2

Words (D): each sliding fit's coefficients 1 to 3 are cut into 4 equal-frequency
bins, and the share of windows in each of the 64 words is counted per scale.
Sharpening multiplies ridge decision values by 2 or 4 before the softmax, the
same factor for every dataset.  Out-of-fold scores are compared with the current
recipes ("trio" for classification, "fast" for regression).  Results persist in
results/fast_words_scores.csv.  No test split is loaded.
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.model_selection import KFold, StratifiedKFold

import ablate
import bench
import stct_fast as SF
from algo_select import load_train, with_retry
from fast_search import oof_ridge
from vote_test import MIN_LENGTH

warnings.filterwarnings("ignore")
OUT = "results/fast_words_scores.csv"
FOREST = {"classification": dict(n_estimators=100, max_features=0.1),
          "regression": dict(n_estimators=50, max_features=0.5, min_samples_leaf=2)}
SHARPEN = (1, 2, 4)


def oof_decision(K, y, folds, labels):
    """Out-of-fold ridge decision values per class, None if ridge fails."""
    out = np.zeros((len(y), len(labels)))
    try:
        for tr, te in folds:
            r = bench.make_learner("ridge", "classification").fit(K[tr], y[tr])
            d = r.decision_function(K[te])
            d = d if d.ndim > 1 else np.column_stack([-d, d])
            out[te[:, None], np.searchsorted(labels, r.classes_)] = d
    except np.linalg.LinAlgError:
        return None
    return out


def score_dataset(task, X, y, seed):
    cls = task == "classification"
    T = X.shape[1]
    long_ = T >= MIN_LENGTH
    cp = SF.context_fit(X, SF.n_kernels(T), seed, 2) if cls and long_ else None
    wp = SF.compete_fit(X, seed, SF.winner_groups(T)) if cls and long_ else None
    F, Cf, Wf = SF.trio_features(X, cp, wp)
    sets = {"D": SF.word_features(X, SF.word_fit(X, seed))}
    if cls and long_:
        sets.update(C=Cf, W=Wf)
    if not cls and long_:
        sets["T"] = SF.features(X, SF.kernel_fit(X, SF.n_kernels(T), seed, max_series=32))[1]
    F = np.nan_to_num(F)
    if cls:
        stratified = np.unique(y, return_counts=True)[1].max() >= 5
        folds = list((StratifiedKFold if stratified else KFold)(5, shuffle=True, random_state=seed).split(F, y))
    else:
        folds = list(KFold(5, shuffle=True, random_state=seed).split(F))
    labels = np.unique(y) if cls else np.zeros(1)
    forest = np.zeros((len(y), len(labels))) if cls else np.zeros(len(y))
    for tr, te in folds:
        et = (ExtraTreesClassifier if cls else ExtraTreesRegressor)(random_state=seed, n_jobs=-1, **FOREST[task]).fit(F[tr], y[tr])
        if cls:
            forest[te[:, None], np.searchsorted(labels, et.classes_)] = et.predict_proba(F[te])
        else:
            forest[te] = et.predict(F[te])
    rows = []
    if cls:
        dec = {k: oof_decision(np.nan_to_num(K), y, folds, labels) for k, K in sets.items()}
        base = "CW" if long_ else ""
        for name, blocks in (("trio", base), ("trio_D", base + "D")):
            for s in SHARPEN:
                parts = [ablate.soft(s * dec[b]) for b in blocks if dec.get(b) is not None]
                p = forest + sum(parts)
                rows.append(dict(blend=f"{name}_s{s}", score=float((labels[p.argmax(axis=1)] == y).mean()),
                                 features=F.shape[1] + sum(sets[b].shape[1] for b in blocks)))
    else:
        pred = {k: oof_ridge(task, np.nan_to_num(K), y, folds, labels) for k, K in sets.items()}
        base = "T" if long_ else ""
        for name, blocks in (("fast", base), ("fast_D", base + "D")):
            parts = [pred[b] for b in blocks if pred.get(b) is not None]
            p = (forest + sum(parts)) / (1 + len(parts))
            rows.append(dict(blend=name, score=float(np.sqrt(((p - y) ** 2).mean())),
                             features=F.shape[1] + sum(sets[b].shape[1] for b in blocks)))
    return [dict(r, seed=seed, task=task, T=T) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=str(bench.RNG))
    a = ap.parse_args()
    known = set()
    if os.path.exists(OUT):
        p = pd.read_csv(OUT)
        known = set(zip(p.seed, p.task, p.dataset))
    for seed in [int(s) for s in a.seeds.split(",")]:
        for task, names in (("classification", bench.UCR), ("regression", bench.TSER)):
            for name in names:
                if (seed, task, name) in known:
                    continue
                try:
                    X, y = load_train(task, name)
                    rows = with_retry(score_dataset, task, X, y, seed)
                except Exception as e:
                    print(f"[failed] {seed} {task} {name}: {str(e)[:120]}", flush=True)
                    continue
                pd.DataFrame(rows).assign(dataset=name).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
                print(f"{seed} {task[:3]} {name}", flush=True)


if __name__ == "__main__":
    main()
