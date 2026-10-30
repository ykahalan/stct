"""Tabulate the dev-set ablations (results/dev_*.csv) into results/dev_ablations.md."""
from __future__ import annotations

import pandas as pd

import bench

SLOW = ("rotf", "hgb")


def block(task, path, methods=None, title=""):
    df = bench.read_results(path, task)
    if methods is not None:
        df = df[df.method.isin(methods)]
    key = "accuracy" if task == "classification" else "rmse"
    piv = df.pivot_table(index="dataset", columns="method", values=key).dropna()
    ranks = piv.rank(axis=1, ascending=(task != "classification"))
    tt = df.assign(total=df[["extract_time_s", "fit_time_s", "predict_time_s"]].sum(axis=1))
    s = pd.DataFrame({
        "mean_rank": ranks.mean(),
        "n_features": df.pivot_table(index="dataset", columns="method", values="n_features").mean(),
        "extract_s": df.pivot_table(index="dataset", columns="method", values="extract_time_s").mean(),
        "fit+predict_s": tt.pivot_table(index="dataset", columns="method", values="total").mean()
        - df.pivot_table(index="dataset", columns="method", values="extract_time_s").mean(),
    })
    if task == "classification":
        s.insert(0, "mean_accuracy", piv.mean())
    else:
        s.insert(0, "mean_rel_rmse", piv.div(piv.min(axis=1), axis=0).mean())
    s = s.sort_values("mean_rank")
    return [f"### {title} ({len(piv)} datasets)", "", s.round(3).to_markdown(), ""]


lines = ["# Dev-set ablations", "",
         "Configuration was chosen on these subsets only (38 small UCR datasets, 16 TSER datasets); "
         "the full-archive numbers in summary_*.md are reported on the complete archives.",
         "Times are seconds per dataset on a shared, loaded machine: compare ratios, not absolutes.", ""]
for task, path in [("classification", "results/dev_classification.csv"),
                   ("regression", "results/dev_regression.csv")]:
    df = bench.read_results(path, task)
    fast = sorted(m for m in df.method.unique() if not any(s in m for s in SLOW))
    lines += [f"## {task}", ""]
    lines += block(task, path, fast, "Fast learners (RF / ExtraTrees / ridge), every feature set")
    lines += block(task, path, None, "Including the slow learners (RotationForest, HistGB) on the "
                                     "datasets they finished before being dropped")
with open("results/dev_ablations.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n".join(lines))
