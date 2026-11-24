"""Score STCT+ on the default train/test splits of the 116 UCR and 26 TSER datasets.

    python benchmark.py --task classification --seeds 42,1,2
    python benchmark.py --task regression --seeds 42,1,2

Writes results/stct_plus_<task>.csv, one row per (dataset, seed), and skips
finished rows on resume.  The reported results use seed 42; seeds 1 and 2 show
the spread over random seeds.  Uses all cores; single-thread runtimes are in
timing_classification.py and timing_regression.py.
"""
import argparse
import os
import time
import warnings

import pandas as pd

import data
from stct_models import STCTClassifier, STCTRegressor

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["classification", "regression"], required=True)
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--datasets", default="all", help="comma-separated names, or all")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    names = data.DATASETS[a.task] if a.datasets == "all" else a.datasets.split(",")
    out = os.path.join("results", f"stct_plus_{a.task}.csv")
    done = data.done_keys(out, ["dataset", "seed"])
    score = "accuracy" if a.task == "classification" else "rmse"
    for name in names:
        todo = [s for s in seeds if (name, str(s)) not in done]
        if not todo:
            continue
        Xtr, ytr, Xte, yte = data.load(a.task, name)
        for seed in todo:
            model = (STCTClassifier if a.task == "classification" else STCTRegressor)(random_state=seed)
            t0 = time.perf_counter()
            model.fit(Xtr, ytr)
            t1 = time.perf_counter()
            pred = model.predict(Xte)
            t2 = time.perf_counter()
            value = float((pred == yte).mean()) if a.task == "classification" else data.rmse(yte, pred)
            row = dict(dataset=name, seed=seed, n_train=len(Xtr), n_test=len(Xte), length=Xtr.shape[1],
                       **{score: value}, n_features=model.n_features(Xte), fit_seconds=t1 - t0,
                       predict_seconds=t2 - t1)
            pd.DataFrame([row]).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
            print(f"{name:32s} seed {seed:3d}  {score} {value:.4f}  features {row['n_features']}", flush=True)


if __name__ == "__main__":
    main()
