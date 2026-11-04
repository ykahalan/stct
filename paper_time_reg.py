"""Time regressors and count their features on one thread, same machine, same pass.

    python paper_time_reg.py --part 0 --parts 6
    python paper_time_reg.py --methods RotF,FreshPRINCE --part 0 --parts 6

For each TSER dataset, the STCT+ regressor and the published methods that can run
here are fitted on the training split and predict the test split one after
another in the same process, one thread each.  The references use the
configurations of the published benchmark code (tsml-eval): aeon's RocketRegressor
and 1-NN DTW with a 10% window, and scikit-learn's RidgeCV, 100-tree random forest
and grid-searched SVR on the raw series.  aeon's RotationForestRegressor and
FreshPRINCERegressor (which needs tsfresh) run only when named in --methods: their
published fit + predict times total 15.6 and 14.5 hours, and the paper table
lists them as not run.  Test targets are read only to check RMSE against the
published results.  Appends to results/paper_time_reg.csv and skips finished
(dataset, method) pairs on resume.

Not included: DrCIF and the deep networks (published fit + predict times total
9.3 to 175 hours), XGBoost and FPCR (their packages are not installed here).
"""
import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[var] = "1"

import argparse
import time
import warnings

import numpy as np
import pandas as pd
from aeon.regression.convolution_based import RocketRegressor
from aeon.regression.distance_based import KNeighborsTimeSeriesRegressor
from aeon.regression.feature_based import FreshPRINCERegressor
from aeon.regression.sklearn import RotationForestRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

import bench
from stct_models import STCTRegressor

warnings.filterwarnings("ignore")
OUT = "results/paper_time_reg.csv"
SVR_GRID = [{"kernel": ["rbf", "sigmoid"], "C": [0.1, 1, 10, 100], "gamma": [0.001, 0.01, 0.1, 1]}]
# name -> (factory, input): "stct" takes (n, T), "raw" the flattened series, "aeon" (n, 1, T)
METHODS = {
    "STCT+": (lambda: STCTRegressor(random_state=bench.RNG, n_jobs=1), "stct"),
    "Ridge": (lambda: RidgeCV(alphas=np.logspace(-3, 3, 10)), "raw"),
    "RandF": (lambda: RandomForestRegressor(random_state=bench.RNG, n_jobs=1), "raw"),
    "GridSVR": (lambda: GridSearchCV(SVR(), SVR_GRID, scoring="neg_mean_squared_error", cv=3, n_jobs=1), "raw"),
    "ROCKET": (lambda: RocketRegressor(random_state=bench.RNG, n_jobs=1), "aeon"),
    "1NN-DTW": (lambda: KNeighborsTimeSeriesRegressor(distance="dtw", distance_params={"window": 0.1}, n_jobs=1), "aeon"),
    "RotF": (lambda: RotationForestRegressor(random_state=bench.RNG, n_jobs=1), "raw"),
    "FreshPRINCE": (lambda: FreshPRINCERegressor(random_state=bench.RNG, n_jobs=1), "aeon"),
}
SLOW = ("RotF", "FreshPRINCE")  # run only when named in --methods


def n_features(name, model, X):
    """Width of the representation the model's final learner sees."""
    if name == "STCT+":
        return model.n_features(X)
    if name in ("Ridge", "RandF", "GridSVR", "RotF"):
        return X.shape[1]
    if name == "1NN-DTW":
        return 0
    if name == "FreshPRINCE":
        return 783
    est = getattr(model, "estimator_", None) or getattr(model, "_estimator", None)
    if hasattr(est, "steps"):
        est = est.steps[-1][1]
    return int(est.n_features_in_) if hasattr(est, "n_features_in_") else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--methods", default=",".join(m for m in METHODS if m not in SLOW))
    a = ap.parse_args()
    methods = a.methods.split(",")
    known = set()
    if os.path.exists(OUT):
        p = pd.read_csv(OUT)
        known = set(zip(p.dataset, p.method))
    for i, name in enumerate(bench.TSER):
        if i % a.parts != a.part or all((name, m) in known for m in methods):
            continue
        Xtr, ytr, Xte, yte = bench.load("regression", name)
        inputs = {"stct": (Xtr, Xte), "raw": (Xtr, Xte), "aeon": (Xtr[:, None, :], Xte[:, None, :])}
        for method in methods:
            if (name, method) in known:
                continue
            make, kind = METHODS[method]
            A, B = inputs[kind]
            try:
                t = time.perf_counter()
                model = make()
                pred = model.fit(A, ytr).predict(B)
                seconds = time.perf_counter() - t
                row = dict(dataset=name, method=method, seconds=seconds, n_features=n_features(method, model, Xtr),
                           rmse=float(np.sqrt(((pred - yte) ** 2).mean())), n_train=len(Xtr), n_test=len(Xte), T=Xtr.shape[1])
            except Exception as e:
                print(f"[failed] {name} {method}: {type(e).__name__} {str(e)[:120]}", flush=True)
                continue
            pd.DataFrame([row]).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
            print(f"{name} {method} {seconds:.1f}s feats={row['n_features']} rmse={row['rmse']:.4f}", flush=True)


if __name__ == "__main__":
    main()
