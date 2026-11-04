"""Score the fast recipe on the default test splits, one seed, resumable.

Compiled extraction with thresholds from at most 32 series.  ExtraTrees of 100
trees (classification) or 50 trees, half the features and 2 samples per leaf
(regression), averaged with ridge on kernels from series length 128 up.  Rows
are appended to results/full_<task>.csv as paper_fast:et_ridge, on the datasets
scored for paper_vote:et_ridge.
"""
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

import ablate
import bench
import stct_fast as SF
from vote_test import MIN_LENGTH

METHOD = "paper_fast:et_ridge"
MAX_SERIES = 32
FOREST = {"classification": dict(n_estimators=100, max_features=0.1),
          "regression": dict(n_estimators=50, max_features=0.5, min_samples_leaf=2)}


def fit_predict(task, Xtr, ytr, Xte):
    """Predictions, feature count and extraction, fit and predict times."""
    cls = task == "classification"
    t0 = time.time()
    average = Xtr.shape[1] >= MIN_LENGTH
    kp = SF.kernel_fit(Xtr, SF.n_kernels(Xtr.shape[1]), bench.RNG, max_series=MAX_SERIES) if average else None
    Ftr, Ktr = SF.features(Xtr, kp)
    Fte, Kte = SF.features(Xte, kp)
    Ftr, Fte = np.nan_to_num(Ftr), np.nan_to_num(Fte)
    t_ext = time.time() - t0
    n_features = Ftr.shape[1] + (Ktr.shape[1] if average else 0)

    t0 = time.time()
    et = (ExtraTreesClassifier if cls else ExtraTreesRegressor)(random_state=bench.RNG, n_jobs=-1, **FOREST[task]).fit(Ftr, ytr)
    ridge = None
    if average:
        try:
            ridge = bench.make_learner("ridge", task).fit(np.nan_to_num(Ktr), ytr)
        except np.linalg.LinAlgError:
            print("  [warn] ridge failed, ExtraTrees alone", flush=True)
    t_fit = time.time() - t0

    t0 = time.time()
    if cls:
        probs = et.predict_proba(Fte)
        if ridge is not None:
            probs = (probs + ablate.soft(ridge.decision_function(np.nan_to_num(Kte)))) / 2
        pred = et.classes_[probs.argmax(axis=1)]
    else:
        pred = et.predict(Fte)
        if ridge is not None:
            pred = (pred + ridge.predict(np.nan_to_num(Kte))) / 2
    return pred, n_features, t_ext, t_fit, time.time() - t0


def main():
    for task in ("classification", "regression"):
        out = f"results/full_{task}.csv"
        prev = bench.read_results(out, task)
        names = prev.loc[prev.method == "paper_vote:et_ridge", "dataset"].drop_duplicates().tolist()
        done = set(prev.loc[prev.method == METHOD, "dataset"])
        cols = pd.read_csv(out, nrows=0).columns.tolist()
        for i, name in enumerate(names):
            if name in done:
                continue
            Xtr, ytr, Xte, yte = bench.load(task, name)
            pred, n_features, t_ext, t_fit, t_pred = fit_predict(task, Xtr, ytr, Xte)
            res = bench.score(task, yte, pred)
            row = {"dataset": name, "n_train": len(Xtr), "n_test": len(Xte), "length": Xtr.shape[1],
                   "method": METHOD, "features": "paper_fast", "learner": "et_ridge", **res,
                   "n_features": n_features, "extract_time_s": t_ext, "fit_time_s": t_fit, "predict_time_s": t_pred}
            pd.DataFrame([{c: row.get(c, np.nan) for c in cols}], columns=cols).to_csv(
                out, mode="a", header=False, index=False)
            print(f"[{i + 1}/{len(names)}] {name} {res.get('accuracy', res.get('rmse')):.4f} feats={n_features}", flush=True)


if __name__ == "__main__":
    assert os.path.exists("results/full_classification.csv")
    main()
