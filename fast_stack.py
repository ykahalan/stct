"""Test a forest weight chosen inside the training data, using training splits only.

    python fast_stack.py --datasets dev --seeds 42

Outer 5-fold cross-validation scores each recipe.  Inside every outer training
part, a 2-fold split with a 20-tree forest and the same ridges picks the forest's
weight against the summed ridge probabilities (ties go to weight 1); the full
models then use that weight.  Ridge blocks: context kernels and winner counts (CW,
the three-model recipe), and the same plus level-free dilated kernels (CWD).
Below length 128 only D exists.  Results persist in results/fast_stack_scores.csv.
No test split is loaded.
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import KFold, StratifiedKFold

import ablate
import bench
import stct_fast as SF
from algo_select import load_train, with_retry
from fast_probe import dilated_sources, kernels
from vote_test import MIN_LENGTH

warnings.filterwarnings("ignore")
OUT = "results/fast_stack_scores.csv"
WEIGHTS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, np.inf)


def split(y, k, seed):
    stratified = np.unique(y, return_counts=True)[1].min() >= k
    return list((StratifiedKFold if stratified else KFold)(k, shuffle=True, random_state=seed).split(y, y))


def models(F, R, y, tr, te, labels, trees, seed):
    """Forest probabilities and summed ridge probabilities on te, aligned to labels."""
    et = ExtraTreesClassifier(n_estimators=trees, max_features=0.1, random_state=seed, n_jobs=-1).fit(F[tr], y[tr])
    pf = np.zeros((len(te), len(labels)))
    pf[:, np.searchsorted(labels, et.classes_)] = et.predict_proba(F[te])
    pr = np.zeros((len(te), len(labels)))
    for K in R:
        try:
            r = bench.make_learner("ridge", "classification").fit(K[tr], y[tr])
        except np.linalg.LinAlgError:
            continue
        pr[:, np.searchsorted(labels, r.classes_)] += ablate.soft(r.decision_function(K[te]))
    return pf, pr


def blend(pf, pr, w):
    return pf if np.isinf(w) else w * pf + pr


def choose_weight(F, R, y, idx, labels, seed):
    """Forest weight with the best 2-fold accuracy inside idx, preferring 1 on ties."""
    hits = np.zeros(len(WEIGHTS))
    for a, b in split(y[idx], 2, seed):
        pf, pr = models(F, R, y, idx[a], idx[b], labels, 20, seed)
        for j, w in enumerate(WEIGHTS):
            hits[j] += (labels[blend(pf, pr, w).argmax(axis=1)] == y[idx[b]]).sum()
    best = np.flatnonzero(hits == hits.max())
    one = WEIGHTS.index(1.0)
    return WEIGHTS[one] if one in best else WEIGHTS[best[0]]


def score_dataset(X, y, seed):
    T = X.shape[1]
    long_ = T >= MIN_LENGTH
    cp = SF.context_fit(X, SF.n_kernels(T), seed, 2) if long_ else None
    wp = SF.compete_fit(X, seed, SF.winner_groups(T)) if long_ else None
    F, Cf, Wf = SF.trio_features(X, cp, wp)
    F = np.nan_to_num(F)
    D = np.nan_to_num(kernels(X, seed, dilated_sources, True))
    sets = {"CW": [np.nan_to_num(Cf), np.nan_to_num(Wf)] if long_ else [], "CWD": ([np.nan_to_num(Cf), np.nan_to_num(Wf)] if long_ else []) + [D]}
    labels = np.unique(y)
    rows = []
    for name, R in sets.items():
        fixed, chosen, ws = np.zeros(len(y), bool), np.zeros(len(y), bool), []
        for tr, te in split(y, 5, seed):
            w = choose_weight(F, R, y, tr, labels, seed) if R else np.inf
            pf, pr = models(F, R, y, tr, te, labels, 100, seed)
            fixed[te] = labels[blend(pf, pr, 1.0 if R else np.inf).argmax(axis=1)] == y[te]
            chosen[te] = labels[blend(pf, pr, w).argmax(axis=1)] == y[te]
            ws.append(w)
        extra = sum(K.shape[1] for K in R)
        rows.append(dict(blocks=name, weighting="fixed", score=fixed.mean(), features=F.shape[1] + extra, weights=""))
        rows.append(dict(blocks=name, weighting="chosen", score=chosen.mean(), features=F.shape[1] + extra,
                         weights=" ".join(str(w) for w in ws)))
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
