"""Time the three-model classification recipe and MiniRocket + ridge on one thread, without scoring.

    python trio_time.py

Both run in the same pass.  Test series are transformed and predicted for timing,
but no test label is read.  Writes results/trio_time.csv and skips finished
datasets on resume.
"""
import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[var] = "1"

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import bench
import single_thread as ST
import stct_fast as SF
from fast_trio import winner_groups
from vote_test import MIN_LENGTH

warnings.filterwarnings("ignore")
OUT = "results/trio_time.csv"


def ridge_time(Ktr, ytr, Kte):
    t = time.perf_counter()
    r = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))).fit(np.nan_to_num(Ktr), ytr)
    r.decision_function(np.nan_to_num(Kte))
    return time.perf_counter() - t


def stages(Xtr, ytr, Xte):
    clock, row, T = time.perf_counter, {}, Xtr.shape[1]
    t = clock()
    Ftr, Fte = np.nan_to_num(SF.features(Xtr)[0]), np.nan_to_num(SF.features(Xte)[0])
    row["views"] = clock() - t
    row["features"] = Ftr.shape[1]
    row["context_s"] = row["winners_s"] = row["ridge_s"] = 0.0
    if T >= MIN_LENGTH:
        t = clock()
        cp = SF.context_fit(Xtr, SF.n_kernels(T), bench.RNG, 2)
        Ctr, Cte = SF.context_features(Xtr, cp, 2), SF.context_features(Xte, cp, 2)
        row["context_s"] = clock() - t
        t = clock()
        wp = SF.compete_fit(Xtr, bench.RNG, winner_groups(T))
        Wtr, Wte = SF.compete_features(Xtr, wp), SF.compete_features(Xte, wp)
        row["winners_s"] = clock() - t
        row["ridge_s"] = ridge_time(Ctr, ytr, Cte) + ridge_time(Wtr, ytr, Wte)
        row["features"] += Ctr.shape[1] + Wtr.shape[1]
    for n in (100, 50):
        t = clock()
        ExtraTreesClassifier(n_estimators=n, max_features=0.1, random_state=bench.RNG, n_jobs=1).fit(Ftr, ytr).predict_proba(Fte)
        row[f"trees{n}"] = clock() - t
    return row


def main():
    known = set(pd.read_csv(OUT).dataset) if os.path.exists(OUT) else set()
    for name in bench.UCR:
        if name in known:
            continue
        try:
            Xtr, ytr, Xte, _ = bench.load("classification", name)
            row = dict(dataset=name, n=len(Xtr) + len(Xte), T=Xtr.shape[1], **stages(Xtr, ytr, Xte))
            row["minirocket"] = ST.run_minirocket(Xtr, ytr, Xte, True)[1]
        except Exception as e:
            print(f"[failed] {name}: {str(e)[:120]}", flush=True)
            continue
        pd.DataFrame([row]).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
        print(row, flush=True)


if __name__ == "__main__":
    main()
