"""Save the STCT+ views of every TSER (or UCR) dataset for STCT-TabPFN, which runs in an environment with tabpfn.

    python tabpfn_features.py --task regression --out feats_regression
    python tabpfn_features.py --task classification --out feats_classification

For each dataset the STCT+ model (seed 42) is fitted on the training split; its
forest views are computed for both splits and saved with the targets, one .npz
per dataset.  The views need no labels.  The extraction time and the size of
the kernel blocks STCT+ adds are saved for the runtime and feature comparison.
"""
import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[var] = "1"

import argparse
import time

import numpy as np

import data
from stct_models import STCTClassifier, STCTRegressor

SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", default="regression", choices=("regression", "classification"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    reg = a.task == "regression"
    for name in data.DATASETS[a.task]:
        path = os.path.join(a.out, f"{name}.npz")
        if os.path.exists(path):
            continue
        Xtr, ytr, Xte, yte = data.load(a.task, name)
        t = time.perf_counter()
        m = (STCTRegressor if reg else STCTClassifier)(random_state=SEED, n_jobs=1).fit(Xtr, ytr)
        fit_s = time.perf_counter() - t
        t = time.perf_counter()
        Vtr, Btr = m._blocks(Xtr)
        Vte, _ = m._blocks(Xte)
        feat_s = time.perf_counter() - t
        n_kernel = sum(B.shape[1] for B in Btr)
        np.savez_compressed(path, Vtr=Vtr, Vte=Vte, ytr=ytr, yte=yte, n_kernel=n_kernel, stct_fit_s=fit_s, feat_s=feat_s)
        print(f"{name} views={Vtr.shape[1]} kernels={n_kernel} n={len(ytr)} features {feat_s:.1f}s", flush=True)


if __name__ == "__main__":
    main()
