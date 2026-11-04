"""Test refinements of the three-model and fast recipes, using training splits only.

    python fast_refine.py --seeds 42,1,2

Out-of-fold outputs of the forest and of ridges on context kernels (C), winner
counts (W), threshold kernels (T) and on the forest's own STCT views (V, no extra
features).  Blends are scored per dataset; the current recipes are "trio" and
"fast".  Context kernels are also computed below length 128 to test them there.
Results persist in results/fast_refine_scores.csv.  No test split is loaded.
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.model_selection import KFold, StratifiedKFold

import bench
import stct_fast as SF
from algo_select import load_train, with_retry
from fast_search import oof_ridge
from fast_trio import winner_groups
from vote_test import MIN_LENGTH

warnings.filterwarnings("ignore")
OUT = "results/fast_refine_scores.csv"
FOREST = {"classification": dict(n_estimators=100, max_features=0.1),
          "regression": dict(n_estimators=50, max_features=0.5, min_samples_leaf=2)}


def blends(task, long_):
    """Blend name -> (forest weight, ridge blocks)."""
    if task == "classification":
        out = {"trio": (1, "CW" if long_ else ""), "trio_F2": (2, "CW" if long_ else ""),
               "trio_V": (1, "CWV" if long_ else "V"), "trio_Vlong": (1, "CWV" if long_ else ""),
               "trio_Vshort": (1, "CW" if long_ else "V"), "trio_Cshort": (1, "CW" if long_ else "C"),
               "trio_CVshort": (1, "CW" if long_ else "CV"), "forest": (1, "")}
    else:
        out = {"fast": (1, "T" if long_ else ""), "fast_V": (1, "TV" if long_ else "V"),
               "fast_Vshort": (1, "T" if long_ else "V"), "fast_F2": (2, "T" if long_ else ""),
               "fast_C": (1, "C" if long_ else ""), "forest": (1, "")}
    return out


def score_dataset(task, X, y, seed):
    cls = task == "classification"
    T = X.shape[1]
    long_ = T >= MIN_LENGTH
    cp = SF.context_fit(X, SF.n_kernels(T), seed, 2, max_series=32)
    wp = SF.compete_fit(X, seed, winner_groups(T)) if cls and long_ else None
    F, Cf, Wf = SF.trio_features(X, cp, wp)
    sets = {"V": F, "C": Cf}
    if wp is not None:
        sets["W"] = Wf
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
    ridges = {k: oof_ridge(task, np.nan_to_num(K), y, folds, labels) for k, K in sets.items()}

    def score(p):
        return float((labels[p.argmax(axis=1)] == y).mean()) if cls else float(np.sqrt(((p - y) ** 2).mean()))

    rows = []
    for name, (wf, blocks) in blends(task, long_).items():
        parts = [ridges.get(b) for b in blocks if ridges.get(b) is not None]
        p = (wf * forest + sum(parts)) / (wf + len(parts))
        extra = sum(sets[b].shape[1] for b in blocks if b != "V" and ridges.get(b) is not None)
        rows.append(dict(blend=name, score=score(p), features=F.shape[1] + extra))
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
