"""Test how to combine the forest with ridges on threshold and winner kernels, using training splits only.

    python fast_blend.py
    python fast_blend.py --seeds 1,2

Blends use equal weights, or weights from accuracy or RMSE measured inside each
training fold (ridge leave-one-out and forest out-of-bag) raised to the fourth
power, as HC2 does.  The forest is bootstrapped for the weighted blends.  Results
persist in results/fast_blend_scores.csv.  No test split is loaded.
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import RidgeClassifierCV, RidgeCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

import ablate
import bench
import stct_fast as SF
from algo_select import load_train, with_retry
from vote_test import MIN_LENGTH

warnings.filterwarnings("ignore")
OUT = "results/fast_blend_scores.csv"
ALPHAS = np.logspace(-3, 3, 10)
FOREST = {"classification": dict(n_estimators=100, max_features=0.1),
          "regression": dict(n_estimators=50, max_features=0.5, min_samples_leaf=2)}


def ridge_fold(cls, K, y, tr, te, labels):
    """Test outputs and leave-one-out accuracy or RMSE of a ridge fitted on one training fold."""
    sc = StandardScaler().fit(K[tr])
    Ktr, Kte = sc.transform(K[tr]), sc.transform(K[te])
    if cls:
        r = RidgeClassifierCV(alphas=ALPHAS, store_cv_results=True).fit(Ktr, y[tr])
        cv = r.cv_results_  # samples, outputs, alphas
        best = np.argmax([(pick(cv[:, :, a], r.classes_) == y[tr]).mean() for a in range(len(ALPHAS))])
        quality = (pick(cv[:, :, best], r.classes_) == y[tr]).mean()
        out = np.zeros((len(te), len(labels)))
        out[:, np.searchsorted(labels, r.classes_)] = ablate.soft(r.decision_function(Kte))
        return out, quality
    r = RidgeCV(alphas=ALPHAS, store_cv_results=True).fit(Ktr, y[tr])
    errors = np.sqrt(((r.cv_results_ - y[tr][:, None]) ** 2).mean(axis=0))
    return r.predict(Kte), errors.min()


def pick(scores, classes):
    return classes[(scores > 0).astype(int).ravel()] if scores.shape[1] == 1 else classes[scores.argmax(axis=1)]


def forest_fold(cls, F, y, tr, te, labels, kw, seed, bootstrap):
    et = (ExtraTreesClassifier if cls else ExtraTreesRegressor)(random_state=seed, n_jobs=-1, bootstrap=bootstrap,
                                                                oob_score=bootstrap, **kw).fit(F[tr], y[tr])
    if cls:
        out = np.zeros((len(te), len(labels)))
        out[:, np.searchsorted(labels, et.classes_)] = et.predict_proba(F[te])
    else:
        out = et.predict(F[te])
    if not bootstrap:
        return out, None
    if cls:
        quality = et.oob_score_
    else:
        oob = np.nan_to_num(et.oob_prediction_.ravel(), nan=y[tr].mean())
        quality = float(np.sqrt(((oob - y[tr]) ** 2).mean()))
    return out, quality


def score_dataset(task, X, y, seed):
    cls = task == "classification"
    T = X.shape[1]
    F = np.nan_to_num(SF.features(X)[0])
    if cls:
        stratified = np.unique(y, return_counts=True)[1].max() >= 5
        folds = list((StratifiedKFold if stratified else KFold)(5, shuffle=True, random_state=seed).split(F, y))
    else:
        folds = list(KFold(5, shuffle=True, random_state=seed).split(F))
    labels = np.unique(y) if cls else np.zeros(1)
    kernels = {}
    if T >= MIN_LENGTH:
        kernels["T"] = np.nan_to_num(SF.features(X, SF.kernel_fit(X, SF.n_kernels(T), seed, max_series=32))[1])
    kernels["W"] = np.nan_to_num(SF.compete_features(X, SF.compete_fit(X, seed, int(np.clip(T // 32, 1, 16)))))
    empty = lambda: np.zeros((len(y), len(labels))) if cls else np.zeros(len(y))
    fplain, fboot, rids = empty(), empty(), {k: empty() for k in kernels}
    blends = {}

    def weigh(names, parts, qualities, te):
        q = np.array(qualities, dtype=float)
        w = q**4 if cls else (1 / np.maximum(q, 1e-12)) ** 4
        w = w / w.sum()
        return sum(a * b for a, b in zip(w, parts))

    for tr, te in folds:
        fplain[te] = forest_fold(cls, F, y, tr, te, labels, FOREST[task], seed, False)[0]
        fb, fq = forest_fold(cls, F, y, tr, te, labels, FOREST[task], seed, True)
        fboot[te] = fb
        parts, quality = {}, {}
        for k, K in kernels.items():
            parts[k], quality[k] = ridge_fold(cls, K, y, tr, te, labels)
            rids[k][te] = parts[k]
        for combo in (("T",), ("W",), ("T", "W")):
            if all(k in parts for k in combo):
                name = "cawpe_F" + "".join(combo)
                blends.setdefault(name, empty())[te] = weigh(combo, [fb] + [parts[k] for k in combo], [fq] + [quality[k] for k in combo], te)

    def score(p):
        return float((labels[p.argmax(axis=1)] == y).mean()) if cls else float(np.sqrt(((p - y) ** 2).mean()))

    rows = {"F": fplain, "Fboot": fboot, **blends}
    for k in rids:
        rows[f"equal_F{k}"] = (fplain + rids[k]) / 2
        rows[f"equalboot_F{k}"] = (fboot + rids[k]) / 2
    if len(rids) == 2:
        rows["equal_FTW"] = (fplain + rids["T"] + rids["W"]) / 3
        rows["half_FTW"] = fplain / 2 + (rids["T"] + rids["W"]) / 4
    return [dict(blend=k, score=score(v), seed=seed, task=task, T=T) for k, v in rows.items()]


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
