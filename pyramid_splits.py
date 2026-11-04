"""Compare split pyramids with the paper's windows inside the tuned recipe.

    python pyramid_splits.py

Each pyramid replaces the paper's 8 and 16 windows and concatenates the fit at
every level.  The pooled block and learner stay as tuned.  Training splits only,
with three seeds per configuration, using the thresholds of param_search.py on the
mean over seeds.  Results go to results/pyramid_trials.csv.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd

import bench
import stct_plus as SP
from algo_select import load_train, with_retry
from param_search import cv_score

warnings.filterwarnings("ignore")
SEEDS = (bench.RNG, 1, 2)
MAX_FEATURES = 600
PYRAMIDS = {
    "dyadic_1_2_4_8_16": (1, 2, 4, 8, 16),
    "ternary_1_3_9_27": (1, 3, 9, 27),
    "flat_primes_1_2_3_5_7_11": (1, 2, 3, 5, 7, 11),
    "nested_primes_1_2_6_30": (1, 2, 6, 30),
    "nested_primes_1_3_6_30": (1, 3, 6, 30),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classification", default=None, help="comma-separated override, smoke tests only")
    ap.add_argument("--regression", default=None, help="comma-separated override, smoke tests only")
    ap.add_argument("--prefix", default="results/pyramid")
    ap.add_argument("--base-scores", default="results/length_scores.csv", help="saved scores of the tuned recipe to reuse")
    a = ap.parse_args()
    names = {"classification": a.classification.split(",") if a.classification else bench.UCR,
             "regression": a.regression.split(",") if a.regression else bench.TSER}
    data = {}
    for task, ns in names.items():
        data[task] = {}
        for n in ns:
            try:
                data[task][n] = load_train(task, n)
            except Exception as e:
                print(f"[skip] {n}: {e}", flush=True)
    chosen = json.load(open("results/param_selected.json", encoding="utf-8"))
    base_cfg, learners = chosen["config"], chosen["learners"]

    scores_csv = f"{a.prefix}_scores.csv"
    known = {}
    for path in (a.base_scores, scores_csv):
        if os.path.exists(path):
            p = pd.read_csv(path, keep_default_na=False)
            p = p[p.key == "tuned"] if path == a.base_scores else p
            known.update({(r.key, r.seed, r.task, r.dataset): r.score for r in p.itertuples()})

    def scores(name, cfg, task):
        rows = []
        for seed in SEEDS:
            vals = {}
            for n, (X, y) in data[task].items():
                k = (name, seed, task, n)
                if k not in known:
                    known[k] = with_retry(cv_score, task, X, y, cfg, learners[task], seed)
                    row = dict(key=name, seed=seed, task=task, dataset=n, score=known[k])
                    pd.DataFrame([row]).to_csv(scores_csv, mode="a", header=not os.path.exists(scores_csv), index=False)
                vals[n] = known[k]
            rows.append(pd.Series(vals))
        print(f"scored {name} {task}", flush=True)
        return rows

    base = {t: scores("tuned", base_cfg, t) for t in data}
    long_series = np.random.default_rng(0).standard_normal((2, 1024))
    out = []
    for name, counts in PYRAMIDS.items():
        cfg = dict(base_cfg, pyramid_counts=list(counts))
        feats = SP.stct_plus(long_series, paper=True, **cfg).shape[1]
        if feats > MAX_FEATURES:
            print(f"{name}: {feats} features, over the limit", flush=True)
            continue
        cand = {t: scores(name, cfg, t) for t in data}
        d = sum(cand["classification"]) / 3 - sum(base["classification"]) / 3
        logs = [np.log(c + 1e-12) - np.log(b + 1e-12) for c, b in zip(cand["regression"], base["regression"])]
        r = sum(logs) / 3
        row = dict(pyramid=name, features=feats, acc_gain_points=round(float(d.mean()) * 100, 3), acc_better=int((d > 0).sum()), acc_worse=int((d < 0).sum()),
                   rmse_ratio=round(float(np.exp(r.mean())), 4), rmse_better=int((r < 0).sum()), rmse_worse=int((r > 0).sum()),
                   seed_gains=[round(float((c - b).mean()) * 100, 2) for c, b in zip(cand["classification"], base["classification"])])
        row["passes"] = bool((d.mean() >= 0.003 and row["acc_better"] > row["acc_worse"] and row["rmse_ratio"] <= 1.006)
                             or (row["rmse_ratio"] <= 0.99 and row["rmse_better"] > row["rmse_worse"] and d.mean() >= -0.0025))
        out.append(row)
        print(json.dumps(row), flush=True)
    pd.DataFrame(out).to_csv(f"{a.prefix}_trials.csv", index=False)
    print("FINAL", json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
