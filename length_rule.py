"""Compare series-length rules for pooled window lengths with the tuned recipe.

    python length_rule.py

Training splits only, with three seeds per configuration.  Thresholds match
param_search.py and are applied to the mean over seeds, with more datasets better
than worse on that mean.  Results go to results/length_trials.csv.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd

import bench
from algo_select import load_train, with_retry
from param_search import cv_score

warnings.filterwarnings("ignore")
SEEDS = (bench.RNG, 1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classification", default=None, help="comma-separated override, smoke tests only")
    ap.add_argument("--regression", default=None, help="comma-separated override, smoke tests only")
    ap.add_argument("--prefix", default="results/length")
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
    if os.path.exists(scores_csv):
        p = pd.read_csv(scores_csv, keep_default_na=False)
        known = {(r.key, r.seed, r.task, r.dataset): r.score for r in p.itertuples()}

    def mean_scores(name, cfg, task):
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

    base = {t: mean_scores("tuned", base_cfg, t) for t in data}
    out = []
    for rule in ("pow2", "halve", "pow3", "pow5", "pow2_3", "prime"):
        cfg = dict(base_cfg, length_rule=rule)
        cand = {t: mean_scores(rule, cfg, t) for t in data}
        d = sum(cand["classification"]) / 3 - sum(base["classification"]) / 3
        logs = [np.log(c + 1e-12) - np.log(b + 1e-12) for c, b in zip(cand["regression"], base["regression"])]
        r = np.exp(sum(logs) / 3)
        row = dict(rule=rule, acc_gain_points=round(float(d.mean()) * 100, 3), acc_better=int((d > 0).sum()), acc_worse=int((d < 0).sum()),
                   rmse_ratio=round(float(r.mean() if False else np.exp(np.log(r).mean())), 4), rmse_better=int((r < 1).sum()), rmse_worse=int((r > 1).sum()),
                   seed_gains=[round(float((c - b).mean()) * 100, 2) for c, b in zip(cand["classification"], base["classification"])])
        row["passes"] = bool((d.mean() >= 0.003 and row["acc_better"] > row["acc_worse"] and row["rmse_ratio"] <= 1.006)
                             or (row["rmse_ratio"] <= 0.99 and row["rmse_better"] > row["rmse_worse"] and d.mean() >= -0.0025))
        length = {n: X.shape[1] for n, (X, _) in data["classification"].items()}
        bins = pd.cut(pd.Series(length), [0, 100, 300, 1000, 1e6], labels=["<=100", "101-300", "301-1000", ">1000"])
        row["gain_by_length"] = (d * 100).groupby(bins, observed=True).mean().round(2).to_dict()
        out.append(row)
        print(json.dumps(row), flush=True)
    pd.DataFrame(out).to_csv(f"{a.prefix}_trials.csv", index=False)
    print("FINAL", json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
