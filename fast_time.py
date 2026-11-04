"""Time the fast recipe and MiniRocket + ridge on one thread, without scoring anything.

    python fast_time.py

Both run in the same pass, so they see the same machine load.  Test series are
transformed and predicted for timing, but no test label is read.  Writes
results/fast_time.csv, one row per dataset, and skips finished datasets on resume.
"""
import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[var] = "1"

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import RidgeClassifierCV, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import bench
import single_thread as ST
import stct_fast as SF
from vote_test import MIN_LENGTH

warnings.filterwarnings("ignore")
OUT = "results/fast_time.csv"
MAX_SERIES = 32
FORESTS = {"classification": dict(big=dict(n_estimators=200, max_features=0.1), small=dict(n_estimators=100, max_features=0.1)),
           "regression": dict(big=dict(n_estimators=100, max_features=0.5, min_samples_leaf=2), small=dict(n_estimators=50, max_features=0.5, min_samples_leaf=2))}


def stages(task, Xtr, ytr, Xte):
    cls, clock, row = task == "classification", time.perf_counter, {}
    use_k = Xtr.shape[1] >= MIN_LENGTH
    t = clock()
    kp = SF.kernel_fit(Xtr, SF.n_kernels(Xtr.shape[1]), bench.RNG, max_series=MAX_SERIES) if use_k else None
    row["kernel_fit"] = clock() - t
    t = clock()
    Ftr, Ktr = SF.features(Xtr, kp)
    Fte, Kte = SF.features(Xte, kp)
    row["extract"] = clock() - t
    Ftr, Fte = np.nan_to_num(Ftr), np.nan_to_num(Fte)
    row["features"] = Ftr.shape[1] + (Ktr.shape[1] if use_k else 0)
    t = clock()
    if use_k:
        alphas = np.logspace(-3, 3, 10)
        ridge = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=alphas) if cls else RidgeCV(alphas=alphas))
        ridge.fit(np.nan_to_num(Ktr), ytr)
        (ridge.decision_function if cls else ridge.predict)(np.nan_to_num(Kte))
    row["ridge"] = clock() - t
    for name, kw in FORESTS[task].items():
        t = clock()
        et = (ExtraTreesClassifier if cls else ExtraTreesRegressor)(random_state=bench.RNG, n_jobs=1, **kw).fit(Ftr, ytr)
        (et.predict_proba if cls else et.predict)(Fte)
        row[name] = clock() - t
    return row


def main():
    known = set(pd.read_csv(OUT).dataset) if os.path.exists(OUT) else set()
    for task, names in (("classification", bench.UCR), ("regression", bench.TSER)):
        for name in names:
            if name in known:
                continue
            try:
                Xtr, ytr, Xte, _ = bench.load(task, name)
            except Exception as e:
                print(f"[skip] {name}: {e}", flush=True)
                continue
            try:
                row = dict(task=task, dataset=name, n=len(Xtr) + len(Xte), T=Xtr.shape[1], **stages(task, Xtr, ytr, Xte))
                row["minirocket"] = ST.run_minirocket(Xtr, ytr, Xte, task == "classification")[1]
            except Exception as e:
                print(f"[failed] {name}: {str(e)[:120]}", flush=True)
                continue
            pd.DataFrame([row]).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
            print(row, flush=True)


if __name__ == "__main__":
    main()
