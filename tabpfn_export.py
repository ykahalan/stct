"""Export the STCT+ features of every TSER (or UCR) dataset for a TabPFN run in another environment.

    python tabpfn_export.py --out E:/stct-tmp/stct_feats
    python tabpfn_export.py --task classification --out E:/stct-tmp/stct_feats_cls
    python tabpfn_export.py --recipe plus --out E:/stct-tmp/stct_feats_plus

For each dataset the STCT+ model (regressor recipe "fast", classifier recipe
"plus", seed 42) is fitted on the training split as in the paper; its forest
views V and kernel blocks K are computed for both splits and saved with the
targets.  For regression the regressor's own test predictions are saved too (so
a blend can reuse its kernel ridge).  With --recipe plus a regression dataset
gets the classifier's features instead: the same views plus its context, winner
and dilated kernel blocks (saved as Ktr, Kte), whose parameters need no labels.
Kernel parameters come from training series only.  Feature extraction time is
recorded for the runtime comparison.
"""
import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[var] = "1"

import argparse
import time

import numpy as np

import bench
from stct_models import STCTClassifier, STCTRegressor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", default="regression", choices=("regression", "classification"))
    ap.add_argument("--recipe", default=None, choices=("plus",), help="regression only: the classifier's features")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    reg = a.task == "regression"
    if a.recipe and not reg:
        ap.error("--recipe is for regression")
    for name in bench.TSER if reg else bench.UCR:
        path = os.path.join(a.out, f"{name}.npz")
        if os.path.exists(path):
            continue
        try:
            Xtr, ytr, Xte, yte = bench.load(a.task, name)
        except Exception as e:  # noqa: BLE001 - a dataset that fails to load is skipped, as in the paper
            print(f"{name} [failed to load] {e}", flush=True)
            continue
        t = time.perf_counter()
        if a.recipe:
            m = STCTClassifier(recipe=a.recipe, random_state=bench.RNG, n_jobs=1)._fit_kernels(Xtr)
            fit_s = time.perf_counter() - t
            t = time.perf_counter()
            (Vtr, Btr), (Vte, Bte) = m._blocks(Xtr), m._blocks(Xte)
            feat_s = time.perf_counter() - t
            Ktr, Kte = np.hstack(Btr), np.hstack(Bte)  # recipe plus always has dilated kernels
            np.savez_compressed(path, Vtr=Vtr, Vte=Vte, Ktr=Ktr, Kte=Kte, ytr=ytr, yte=yte, n_kernel=Ktr.shape[1],
                                stct_fit_s=fit_s, feat_s=feat_s)
            print(f"{name} V={Vtr.shape[1]} K={Ktr.shape[1]} n={len(ytr)} feat {feat_s:.1f}s", flush=True)
            continue
        if reg:
            m = STCTRegressor(random_state=bench.RNG, n_jobs=1).fit(Xtr, ytr)
        else:
            m = STCTClassifier(recipe="plus", random_state=bench.RNG, n_jobs=1).fit(Xtr, ytr)
        fit_s = time.perf_counter() - t
        t = time.perf_counter()
        Vtr, Btr = m._blocks(Xtr)
        Vte, Bte = m._blocks(Xte)
        feat_s = time.perf_counter() - t
        extra = {}
        if reg:
            ridge = [r.predict(B) for r, B in zip(m.ridges_, Bte) if r is not None]
            extra = dict(forest_te=m.forest_.predict(Vte), ridge_te=ridge[0] if ridge else np.full(len(yte), np.nan))
        np.savez_compressed(path, Vtr=Vtr, Vte=Vte, ytr=ytr, yte=yte, n_kernel=sum(B.shape[1] for B in Btr),
                            stct_fit_s=fit_s, feat_s=feat_s, **extra)
        print(f"{name} V={Vtr.shape[1]} K={sum(B.shape[1] for B in Btr)} n={len(ytr)} feat {feat_s:.1f}s", flush=True)


if __name__ == "__main__":
    main()
