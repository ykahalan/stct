"""STCT-TabPFN regression: TabPFN 2.5 on the STCT+ views (run in an environment with tabpfn installed).

    python tabpfn_regression.py --feats feats_regression --out results/stct_tabpfn_regression.csv --device cpu

Reads the files written by tabpfn_features.py --task regression.  STCT-TabPFN is
the TabPFN 2.5 regressor at its version defaults (seed 42) on the STCT views, the
version TS2TabPFN used for tsfresh-, MultiROCKET- and catch22-TabPFN (its weights
need a Prior Labs account that accepted the 2.5 licence).  Each dataset is
scored once on its test split.  Appends to --out and skips finished datasets; a
dataset that fails (for example out of memory) is reported and left for the next run.
"""
import argparse
import csv
import glob
import os
import time

import numpy as np
import torch
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    a = ap.parse_args()
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            done = {r["dataset"] for r in csv.DictReader(f)}
    for path in sorted(glob.glob(os.path.join(a.feats, "*.npz"))):
        name = os.path.splitext(os.path.basename(path))[0]
        if name in done:
            continue
        d = np.load(path)
        try:
            sync(a.device)
            t = time.perf_counter()
            model = TabPFNRegressor.create_default_for_version(ModelVersion.V2_5, random_state=42, device=a.device,
                                                               ignore_pretraining_limits=True)
            pred = model.fit(d["Vtr"], d["ytr"]).predict(d["Vte"])
            sync(a.device)
            model_s = time.perf_counter() - t
        except Exception as e:
            print(f"[failed] {name}: {type(e).__name__} {str(e)[:160]}", flush=True)
            continue
        row = dict(dataset=name, method="STCT-TabPFN", rmse=float(np.sqrt(((pred - d["yte"]) ** 2).mean())),
                   device=a.device, threads=torch.get_num_threads(), model_seconds=model_s,
                   cpu_feature_seconds=float(d["feat_s"]), n_views=d["Vtr"].shape[1], n_kernel=int(d["n_kernel"]),
                   n_train=len(d["ytr"]))
        new = not os.path.exists(a.out)
        with open(a.out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"{name} views={d['Vtr'].shape[1]} {a.device} {model_s:.1f}s rmse {row['rmse']:.4f}", flush=True)


if __name__ == "__main__":
    main()
