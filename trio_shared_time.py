"""Time the three-model extraction in separate passes and in one shared pass, on one thread.

    python trio_shared_time.py

Both run on the same datasets in the same run; no label is read.  Writes
results/trio_shared_time.csv and skips finished datasets on resume.
"""
import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[var] = "1"

import time

import numpy as np
import pandas as pd

import bench
import stct_fast as SF
from fast_trio import winner_groups
from vote_test import MIN_LENGTH

OUT = "results/trio_shared_time.csv"


def main():
    known = set(pd.read_csv(OUT).dataset) if os.path.exists(OUT) else set()
    for name in bench.UCR:
        if name in known:
            continue
        try:
            Xtr, _, Xte, _ = bench.load("classification", name)
        except Exception as e:
            print(f"[failed] {name}: {str(e)[:120]}", flush=True)
            continue
        T, clock = Xtr.shape[1], time.perf_counter
        cp = SF.context_fit(Xtr, SF.n_kernels(T), bench.RNG, 2) if T >= MIN_LENGTH else None
        wp = SF.compete_fit(Xtr, bench.RNG, winner_groups(T)) if T >= MIN_LENGTH else None
        t = clock()
        for X in (Xtr, Xte):
            SF.features(X)
            if cp is not None:
                SF.context_features(X, cp, 2), SF.compete_features(X, wp)
        separate = clock() - t
        t = clock()
        for X in (Xtr, Xte):
            SF.trio_features(X, cp, wp)
        shared = clock() - t
        row = dict(dataset=name, T=T, n=len(Xtr) + len(Xte), separate=separate, shared=shared)
        pd.DataFrame([row]).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
        print(row, flush=True)


if __name__ == "__main__":
    main()
