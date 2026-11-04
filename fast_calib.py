"""Test ridges calibrated on their own leave-one-out outputs, using training splits only.

    python fast_calib.py --seeds 42,1,2

Classification only, series length 128 and up (shorter series use the forest
alone in both recipes).  Within each fold, every ridge sets its softmax sharpness
from leave-one-out outputs on the fold's training part.  Compared: the three-model
recipe (forest + context kernels + winner counts) and the fast recipe (forest +
threshold kernels), each with plain and calibrated ridges.  Results persist in
results/fast_calib_scores.csv.  No test split is loaded.
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

import bench
import stct_fast as SF
from algo_select import load_train, with_retry
from stct_models import ALPHAS, _soft
from vote_test import MIN_LENGTH

warnings.filterwarnings("ignore")
OUT = "results/fast_calib_scores.csv"
SHARPNESS = np.logspace(-1, 2, 25)


class CalibratedRidge:
    """Ridge classifier whose softmax sharpness is fitted to its own leave-one-out outputs.

    Same model as a standardized RidgeClassifierCV: one-vs-rest targets of +1 and
    -1, alpha chosen by leave-one-out squared error.  The leave-one-out outputs at
    that alpha come free with the alpha search and set the sharpness that minimises
    their log loss.
    """

    def fit(self, K, y):
        self.scaler_ = StandardScaler().fit(K)
        self.classes_, codes = np.unique(y, return_inverse=True)
        Y = np.where(codes[:, None] == np.arange(len(self.classes_))[None], 1.0, -1.0)
        self.ridge_ = RidgeCV(alphas=ALPHAS, scoring="neg_mean_squared_error", store_cv_results=True)
        self.ridge_.fit(self.scaler_.transform(K), Y)
        loo = self.ridge_.cv_results_[:, :, list(ALPHAS).index(self.ridge_.alpha_)]
        loss = [-np.log(_soft(s * loo)[np.arange(len(y)), codes] + 1e-12).mean() for s in SHARPNESS]
        self.sharpness_ = float(SHARPNESS[int(np.argmin(loss))])
        del self.ridge_.cv_results_
        return self

    def decision_function(self, K):
        return self.ridge_.predict(self.scaler_.transform(K))


def oof(K, y, folds, labels):
    """Out-of-fold probabilities with sharpness 1 and calibrated, and the mean sharpness."""
    plain, cal, sharp = np.zeros((len(y), len(labels))), np.zeros((len(y), len(labels))), []
    for tr, te in folds:
        r = CalibratedRidge().fit(K[tr], y[tr])
        cols = np.searchsorted(labels, r.classes_)
        d = r.decision_function(K[te])
        plain[te[:, None], cols] = _soft(d)
        cal[te[:, None], cols] = _soft(r.sharpness_ * d)
        sharp.append(r.sharpness_)
    return plain, cal, float(np.exp(np.mean(np.log(sharp))))


def score_dataset(X, y, seed):
    T = X.shape[1]
    cp = SF.context_fit(X, SF.n_kernels(T), seed, 2)
    wp = SF.compete_fit(X, seed, SF.winner_groups(T))
    F, Cf, Wf = SF.trio_features(X, cp, wp)
    Tf = SF.features(X, SF.kernel_fit(X, SF.n_kernels(T), seed, max_series=32))[1]
    F = np.nan_to_num(F)
    stratified = np.unique(y, return_counts=True)[1].max() >= 5
    folds = list((StratifiedKFold if stratified else KFold)(5, shuffle=True, random_state=seed).split(F, y))
    labels = np.unique(y)
    forest = np.zeros((len(y), len(labels)))
    for tr, te in folds:
        et = ExtraTreesClassifier(n_estimators=100, max_features=0.1, random_state=seed, n_jobs=-1).fit(F[tr], y[tr])
        forest[te[:, None], np.searchsorted(labels, et.classes_)] = et.predict_proba(F[te])
    out = {k: oof(np.nan_to_num(K), y, folds, labels) for k, K in (("C", Cf), ("W", Wf), ("T", Tf))}
    acc = lambda p: float((labels[p.argmax(axis=1)] == y).mean())
    rows = [dict(blend="forest", score=acc(forest))]
    for name, blocks in (("trio", "CW"), ("fast", "T")):
        rows.append(dict(blend=name, score=acc(forest + sum(out[b][0] for b in blocks))))
        rows.append(dict(blend=name + "_cal", score=acc(forest + sum(out[b][1] for b in blocks))))
    sharp = {f"sharp_{b}": out[b][2] for b in out}
    return [dict(r, seed=seed, T=T, **sharp) for r in rows]


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
