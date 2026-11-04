"""Time STCT+ and references on one thread, on this machine, for all datasets.

    python single_thread.py

Published baseline times come from other hardware, so MiniRocket and QUANT are
run here too: they give a same-machine comparison, and MiniRocket's ratio to its
published time estimates the hardware difference.  Times cover feature
extraction, fit and predict.  Rows are appended to results/single_thread_<task>.csv
and finished pairs are skipped on resume.  QUANT is classification only here.
"""
import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[var] = "1"

import argparse
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import RidgeClassifierCV, RidgeCV
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import bench
import stct_plus as SP

warnings.filterwarnings("ignore")


def trees(cls, **kw):
    return (ExtraTreesClassifier if cls else ExtraTreesRegressor)(n_estimators=500, random_state=bench.RNG, n_jobs=1, **kw)


def run_stct(fn, Xtr, ytr, Xte, cls, **tree_kw):
    t = time.perf_counter()
    Ftr, Fte = np.nan_to_num(fn(Xtr)), np.nan_to_num(fn(Xte))
    pred = trees(cls, **tree_kw).fit(Ftr, ytr).predict(Fte)
    return pred, time.perf_counter() - t, Ftr.shape[1]


def run_vote(Xtr, ytr, Xte, cls):
    """ExtraTrees on stct_views, averaged with ridge on kernels from length 128 up."""
    import ablate
    from vote_test import MIN_LENGTH, kernel_features

    t = time.perf_counter()
    Ftr, Fte = np.nan_to_num(SP.stct_views(Xtr)), np.nan_to_num(SP.stct_views(Xte))
    et = trees(cls, max_features=0.1 if cls else 1.0).fit(Ftr, ytr)
    n = Ftr.shape[1]
    if Xtr.shape[1] < MIN_LENGTH:
        return et.predict(Fte), time.perf_counter() - t, n
    Ktr, Kte = (np.nan_to_num(F) for F in kernel_features(Xtr, Xte))
    alphas = np.logspace(-3, 3, 10)
    ridge = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=alphas) if cls else RidgeCV(alphas=alphas)).fit(Ktr, ytr)
    if cls:
        pred = et.classes_[(et.predict_proba(Fte) + ablate.soft(ridge.decision_function(Kte))).argmax(axis=1)]
    else:
        pred = (et.predict(Fte) + ridge.predict(Kte)) / 2
    return pred, time.perf_counter() - t, n + Ktr.shape[1]


def run_minirocket(Xtr, ytr, Xte, cls):
    from aeon.transformations.collection.convolution_based import MiniRocket

    t = time.perf_counter()
    tr = MiniRocket(n_jobs=1, random_state=bench.RNG)
    Ftr = tr.fit_transform(Xtr[:, None, :])
    Fte = tr.transform(Xte[:, None, :])
    alphas = np.logspace(-3, 3, 10)
    est = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=alphas) if cls else RidgeCV(alphas=alphas))
    pred = est.fit(Ftr, ytr).predict(Fte)
    return pred, time.perf_counter() - t, Ftr.shape[1]


def run_quant(Xtr, ytr, Xte, cls):
    from aeon.transformations.collection.interval_based import QUANTTransformer

    t = time.perf_counter()
    tr = QUANTTransformer()
    Ftr = tr.fit_transform(Xtr[:, None, :], ytr)
    Fte = tr.transform(Xte[:, None, :])
    Ftr, Fte = [np.asarray(F.numpy() if hasattr(F, "numpy") else F) for F in (Ftr, Fte)]
    est = ExtraTreesClassifier(n_estimators=200, max_features=0.1, criterion="entropy", random_state=bench.RNG, n_jobs=1)
    pred = est.fit(Ftr, ytr).predict(Fte)
    return pred, time.perf_counter() - t, Ftr.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classification", default=None, help="comma-separated override, smoke tests only")
    ap.add_argument("--regression", default=None, help="comma-separated override, smoke tests only")
    ap.add_argument("--prefix", default="results/single_thread")
    a = ap.parse_args()
    lists = {"classification": a.classification.split(",") if a.classification else bench.UCR,
             "regression": a.regression.split(",") if a.regression else bench.TSER}
    for task, names in lists.items():
        cls = task == "classification"
        out = f"{a.prefix}_{task}.csv"
        done = set()
        if os.path.exists(out):
            prev = pd.read_csv(out)
            done = set(zip(prev.dataset, prev.method))
        methods = {"stct_plus": lambda *x: run_stct(lambda X: SP.stct_plus(X, orthonormal=True), *x),
                   "paper_pooled": lambda *x: run_stct(lambda X: SP.stct_plus(X, paper=True), *x),
                   "paper_plus": lambda *x: run_stct(
                       lambda X: SP.stct_plus(X, paper=True, resid_kind="peak", global_q=True), *x),
                   "paper_tuned": lambda *x: run_stct(
                       lambda X: SP.stct_plus(X, paper=True, fractions=(1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4)),
                       *x, max_features=0.1 if x[-1] else 1.0),
                   "paper_views": lambda *x: run_stct(SP.stct_views, *x, max_features=0.1 if x[-1] else 1.0),
                   "paper_vote": run_vote,
                   "fixed": lambda *x: run_stct(SP.stct_fixed, *x),
                   "minirocket": run_minirocket}
        if cls:
            methods["quant"] = run_quant
        for name in names:
            try:
                Xtr, ytr, Xte, yte = bench.load(task, name)
            except Exception as e:
                print(f"[skip] {name}: {e}", flush=True)
                continue
            for m, fn in methods.items():
                if (name, m) in done:
                    continue
                try:
                    pred, secs, feats = fn(Xtr, ytr, Xte, cls)
                    score = accuracy_score(yte, pred) if cls else float(np.sqrt(mean_squared_error(yte, pred)))
                    row = dict(dataset=name, method=m, seconds=round(secs, 3), score=score, features=feats)
                except Exception as e:
                    print(f"[warn] {name}/{m}: {str(e)[:120]}", flush=True)
                    continue
                pd.DataFrame([row]).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
                print(f"{task[:3]} {name:30s} {m:11s} {secs:8.2f}s score={score:.4f} feats={feats}", flush=True)


if __name__ == "__main__":
    main()
