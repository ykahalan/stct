"""Time STCT classifiers and MiniRocket + ridge on one thread, in the same pass.

    python dilated_time.py
    python dilated_time.py --recipes trio,quant --out results/quant_time.csv
    python dilated_time.py --recipes trio,plus --out results/plus_time.csv

Each recipe and MiniRocket run on each dataset in turn: fit on the training split,
then predict the test split, without reading test labels.  "quant" is the
QUANT-sized variant without its short-series ridge (quant_sized.py).  Skips
finished datasets on resume.
"""
import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[var] = "1"

import argparse
import time
import warnings

import pandas as pd

import bench
import single_thread as ST
from quant_sized import QuantSizedClassifier
from stct_models import STCTClassifier

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipes", default="trio,dilated,four")
    ap.add_argument("--out", default="results/dilated_time.csv")
    a = ap.parse_args()
    known = set(pd.read_csv(a.out).dataset) if os.path.exists(a.out) else set()
    for name in bench.UCR:
        if name in known:
            continue
        try:
            Xtr, ytr, Xte, _ = bench.load("classification", name)
            row = dict(dataset=name, n=len(Xtr) + len(Xte), T=Xtr.shape[1])
            for recipe in a.recipes.split(","):
                t = time.perf_counter()
                model = (QuantSizedClassifier(bench.RNG, 1, short_ridge=False) if recipe == "quant"
                         else STCTClassifier(recipe, random_state=bench.RNG, n_jobs=1))
                model.fit(Xtr, ytr).predict(Xte)
                row[recipe] = time.perf_counter() - t
            row["minirocket"] = ST.run_minirocket(Xtr, ytr, Xte, True)[1]
        except Exception as e:
            print(f"[failed] {name}: {str(e)[:120]}", flush=True)
            continue
        pd.DataFrame([row]).to_csv(a.out, mode="a", header=not os.path.exists(a.out), index=False)
        print(row, flush=True)


if __name__ == "__main__":
    main()
