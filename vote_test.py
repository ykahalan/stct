"""Score the averaged recipe on the default test splits, one seed, resumable.

ExtraTrees on stct_views, averaged with ridge on random coefficient kernels from
series length 128 up (one kernel per 8 samples, at most 64 per scale).  Shorter
series use ExtraTrees alone.  Rows are appended to results/full_<task>.csv as
paper_vote:et_ridge, on the datasets scored for paper_views:et_tuned.
"""
import os
import time

import numpy as np
import pandas as pd

import ablate
import bench
import stct_plus as SP

METHOD = "paper_vote:et_ridge"
MIN_LENGTH = 128


def kernel_features(Xtr, Xte, seed=bench.RNG):
    kp = ablate.kernel_fit(Xtr, int(np.clip(Xtr.shape[1] // 8, 1, 64)), seed)
    return ablate.transform(Xtr, {"KT"}, kp), ablate.transform(Xte, {"KT"}, kp)


def fit_predict(task, Xtr, ytr, Xte):
    """Predictions, feature count and extraction, fit and predict times."""
    t0 = time.time()
    Ftr, Fte = np.nan_to_num(SP.stct_views(Xtr)), np.nan_to_num(SP.stct_views(Xte))
    average = Xtr.shape[1] >= MIN_LENGTH
    if average:
        Ktr, Kte = (np.nan_to_num(F) for F in kernel_features(Xtr, Xte))
    t_ext = time.time() - t0
    n_features = Ftr.shape[1] + (Ktr.shape[1] if average else 0)

    t0 = time.time()
    et = bench.make_learner("et_tuned", task).fit(Ftr, ytr)
    ridge = None
    if average:
        try:
            ridge = bench.make_learner("ridge", task).fit(Ktr, ytr)
        except np.linalg.LinAlgError:
            print("  [warn] ridge failed, ExtraTrees alone", flush=True)
    t_fit = time.time() - t0

    t0 = time.time()
    if task == "classification":
        labels = et.classes_
        probs = et.predict_proba(Fte)
        if ridge is not None:
            probs = (probs + ablate.soft(ridge.decision_function(Kte))) / 2
        pred = labels[probs.argmax(axis=1)]
    else:
        pred = et.predict(Fte)
        if ridge is not None:
            pred = (pred + ridge.predict(Kte)) / 2
    return pred, n_features, t_ext, t_fit, time.time() - t0


def main():
    for task in ("classification", "regression"):
        out = f"results/full_{task}.csv"
        prev = bench.read_results(out, task)
        names = prev.loc[prev.method == "paper_views:et_tuned", "dataset"].drop_duplicates().tolist()
        done = set(prev.loc[prev.method == METHOD, "dataset"])
        cols = pd.read_csv(out, nrows=0).columns.tolist()
        for i, name in enumerate(names):
            if name in done:
                continue
            Xtr, ytr, Xte, yte = bench.load(task, name)
            pred, n_features, t_ext, t_fit, t_pred = fit_predict(task, Xtr, ytr, Xte)
            res = bench.score(task, yte, pred)
            row = {"dataset": name, "n_train": len(Xtr), "n_test": len(Xte), "length": Xtr.shape[1],
                   "method": METHOD, "features": "paper_vote", "learner": "et_ridge", **res,
                   "n_features": n_features, "extract_time_s": t_ext, "fit_time_s": t_fit, "predict_time_s": t_pred}
            pd.DataFrame([{c: row.get(c, np.nan) for c in cols}], columns=cols).to_csv(
                out, mode="a", header=False, index=False)
            print(f"[{i + 1}/{len(names)}] {name} {res.get('accuracy', res.get('rmse')):.4f} feats={n_features}", flush=True)


if __name__ == "__main__":
    assert os.path.exists("results/full_classification.csv")
    main()
