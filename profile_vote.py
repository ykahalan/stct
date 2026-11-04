"""Time each stage of the averaged recipe on one thread, without scoring anything.

    python profile_vote.py

Uses every fourth dataset.  Test series are transformed for timing, but no test
label is read and no prediction is scored.  Writes results/profile_vote.csv.
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

import ablate
import bench
import stct_plus as SP
from vote_test import MIN_LENGTH

warnings.filterwarnings("ignore")
OUT = "results/profile_vote.csv"


def stages(Xtr, ytr, Xte, cls):
    row, clock = {}, time.perf_counter

    def timed(name, fn):
        t = clock()
        out = fn()
        row[name] = clock() - t
        return out

    Ftr = timed("views_train", lambda: np.nan_to_num(SP.stct_views(Xtr)))
    Fte = timed("views_test", lambda: np.nan_to_num(SP.stct_views(Xte)))
    kw = dict(n_estimators=500, random_state=bench.RNG, n_jobs=1, max_features=0.1 if cls else 1.0)
    et = (ExtraTreesClassifier if cls else ExtraTreesRegressor)(**kw)
    timed("trees_fit", lambda: et.fit(Ftr, ytr))
    timed("trees_predict", lambda: et.predict(Fte))
    if Xtr.shape[1] >= MIN_LENGTH:
        n_dir = int(np.clip(Xtr.shape[1] // 8, 1, 64))
        kp = timed("kernel_fit", lambda: ablate.kernel_fit(Xtr, n_dir, bench.RNG))
        Ktr = timed("kernel_train", lambda: np.nan_to_num(ablate.transform(Xtr, {"KT"}, kp)))
        Kte = timed("kernel_test", lambda: np.nan_to_num(ablate.transform(Xte, {"KT"}, kp)))
        alphas = np.logspace(-3, 3, 10)
        ridge = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=alphas) if cls else RidgeCV(alphas=alphas))
        timed("ridge_fit", lambda: ridge.fit(Ktr, ytr))
        timed("ridge_predict", lambda: ridge.predict(Kte))
    return row


def main():
    for task, names in (("classification", bench.UCR[::4]), ("regression", bench.TSER[::4])):
        for name in names:
            try:
                Xtr, ytr, Xte, _ = bench.load(task, name)
            except Exception as e:
                print(f"[skip] {name}: {e}", flush=True)
                continue
            row = dict(task=task, dataset=name, n=len(Xtr) + len(Xte), T=Xtr.shape[1],
                       **stages(Xtr, ytr, Xte, task == "classification"))
            print(row, flush=True)
            ROWS.append(row)
            pd.DataFrame(ROWS).to_csv(OUT, index=False)


ROWS = []
if __name__ == "__main__":
    if os.path.exists(OUT):
        os.remove(OUT)
    main()
