"""Score the three-model classification recipe on the default test splits, one seed, resumable.

ExtraTrees of 100 trees on stct_views, averaged from series length 128 up with
ridges on kernels over two neighbouring windows and on winner counts.  Shorter
series use ExtraTrees alone.  Rows are appended to results/full_classification.csv
as paper_trio:et_ridge, on the datasets scored for paper_fast:et_ridge.
"""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

import ablate
import bench
import stct_fast as SF
from fast_trio import winner_groups
from vote_test import MIN_LENGTH

METHOD = "paper_trio:et_ridge"


def fit_predict(Xtr, ytr, Xte):
    T = Xtr.shape[1]
    t0 = time.time()
    Ftr, Fte = np.nan_to_num(SF.features(Xtr)[0]), np.nan_to_num(SF.features(Xte)[0])
    sets = []
    if T >= MIN_LENGTH:
        cp = SF.context_fit(Xtr, SF.n_kernels(T), bench.RNG, 2)
        sets.append((SF.context_features(Xtr, cp, 2), SF.context_features(Xte, cp, 2)))
        wp = SF.compete_fit(Xtr, bench.RNG, winner_groups(T))
        sets.append((SF.compete_features(Xtr, wp), SF.compete_features(Xte, wp)))
    t_ext = time.time() - t0
    n_features = Ftr.shape[1] + sum(a.shape[1] for a, _ in sets)

    t0 = time.time()
    et = ExtraTreesClassifier(n_estimators=100, max_features=0.1, random_state=bench.RNG, n_jobs=-1).fit(Ftr, ytr)
    ridges = []
    for a, b in sets:
        try:
            ridges.append((bench.make_learner("ridge", "classification").fit(np.nan_to_num(a), ytr), np.nan_to_num(b)))
        except np.linalg.LinAlgError:
            print("  [warn] ridge failed, left out", flush=True)
    t_fit = time.time() - t0

    t0 = time.time()
    probs = et.predict_proba(Fte)
    for r, b in ridges:
        probs = probs + ablate.soft(r.decision_function(b))
    return et.classes_[probs.argmax(axis=1)], n_features, t_ext, t_fit, time.time() - t0


def main():
    out = "results/full_classification.csv"
    prev = bench.read_results(out, "classification")
    names = prev.loc[prev.method == "paper_fast:et_ridge", "dataset"].drop_duplicates().tolist()
    done = set(prev.loc[prev.method == METHOD, "dataset"])
    cols = pd.read_csv(out, nrows=0).columns.tolist()
    for i, name in enumerate(names):
        if name in done:
            continue
        Xtr, ytr, Xte, yte = bench.load("classification", name)
        pred, n_features, t_ext, t_fit, t_pred = fit_predict(Xtr, ytr, Xte)
        res = bench.score("classification", yte, pred)
        row = {"dataset": name, "n_train": len(Xtr), "n_test": len(Xte), "length": Xtr.shape[1],
               "method": METHOD, "features": "paper_trio", "learner": "et_ridge", **res,
               "n_features": n_features, "extract_time_s": t_ext, "fit_time_s": t_fit, "predict_time_s": t_pred}
        pd.DataFrame([{c: row.get(c, np.nan) for c in cols}], columns=cols).to_csv(out, mode="a", header=False, index=False)
        print(f"[{i + 1}/{len(names)}] {name} {res['accuracy']:.4f} feats={n_features}", flush=True)


if __name__ == "__main__":
    main()
