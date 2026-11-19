"""TabPFN on the STCT+ regressor's features (run in an environment with tabpfn installed).

    python tabpfn_run.py --feats E:/stct-tmp/stct_feats --out results/stct_tabpfn_regression.csv --device cpu

Reads the files written by tabpfn_export.py.  Two variants, fixed before any test score was seen:
  STCT-TabPFN        TabPFN 2.5 regressor at its version defaults (seed 42) on the STCT views,
                     the version TS2TabPFN used for tsfresh-, MultiROCKET- and catch22-TabPFN
                     (its weights need a Prior Labs API key whose account accepted the 2.5 licence);
  STCT-TabPFN+ridge  its prediction averaged with the STCT+ kernel ridge from length 128 up, as STCT+
                     averages its forest with that ridge (the TabPFN prediction alone below 128).
With --kernels (files from tabpfn_export.py --recipe plus) one variant instead:
  STCT-TabPFN-plus   the same TabPFN on the STCT+ classifier's features, the views and its context,
                     winner and dilated kernel blocks side by side (TabPFN gives each of its
                     estimators at most 500 of them, balanced across estimators).
    python tabpfn_run.py --kernels --feats E:/stct-tmp/stct_feats_plus --out results/stct_tabpfn_plus_regression.csv
Each dataset is scored once on its test split.  Appends to --out and skips finished datasets; a dataset
that fails (for example out of memory) is reported and left for the next run.
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
    ap.add_argument("--kernels", action="store_true", help="add the saved kernel blocks Ktr, Kte to the views")
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
        Xtr, Xte = (np.hstack([d["Vtr"], d["Ktr"]]), np.hstack([d["Vte"], d["Kte"]])) if a.kernels else (d["Vtr"], d["Vte"])
        try:
            sync(a.device)
            t = time.perf_counter()
            model = TabPFNRegressor.create_default_for_version(ModelVersion.V2_5, random_state=42, device=a.device,
                                                               ignore_pretraining_limits=True)
            pred = model.fit(Xtr, d["ytr"]).predict(Xte)
            sync(a.device)
            model_s = time.perf_counter() - t
        except Exception as e:
            print(f"[failed] {name}: {type(e).__name__} {str(e)[:160]}", flush=True)
            continue
        if a.kernels:
            variants = [("STCT-TabPFN-plus", pred)]
        else:
            ridge = d["ridge_te"]
            variants = [("STCT-TabPFN", pred), ("STCT-TabPFN+ridge", pred if np.isnan(ridge).all() else (pred + ridge) / 2)]
        rmse = lambda p: float(np.sqrt(((p - d["yte"]) ** 2).mean()))
        rows = [dict(dataset=name, method=m, rmse=rmse(p), device=a.device, threads=torch.get_num_threads(),
                     model_seconds=model_s, cpu_feature_seconds=float(d["feat_s"]),
                     n_views=d["Vtr"].shape[1], n_kernel=int(d["n_kernel"]), n_train=len(d["ytr"]))
                for m, p in variants]
        new = not os.path.exists(a.out)
        with open(a.out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            if new:
                w.writeheader()
            w.writerows(rows)
        print(f"{name} features={Xtr.shape[1]} {a.device} {model_s:.1f}s "
              + " ".join(f"{r['method']} {r['rmse']:.4f}" for r in rows), flush=True)


if __name__ == "__main__":
    main()
