"""Tabulate the dev-set ablations (results/dev_*.csv) into results/dev_ablations.md."""
from __future__ import annotations

import pandas as pd

import bench

SLOW = ("rotf", "hgb")


def load_all(task):
    """All dev rounds for a task (results/dev*_<task>.csv)."""
    import glob

    paths = sorted(glob.glob(f"results/dev*_{task}.csv"))
    return pd.concat([bench.read_results(p, task) for p in paths], ignore_index=True)


def block(task, path, methods=None, title=""):
    df = load_all(task) if path is None else bench.read_results(path, task)
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
for task in ["classification", "regression"]:
    df = load_all(task)
    fast = sorted(m for m in df.method.unique() if not any(s in m for s in SLOW))
    ridge = sorted(m for m in fast if m.startswith("rppv") or m == "minirocket:ridge")
    ladders = sorted(m for m in fast if m.startswith(("ms", "lap", "pyr", "base", "pooledpyr"))
                     or m in ("fixed:rf", "fixed4:rf", "fixed+pooled:et", "minirocket:ridge"))
    lines += [f"## {task}", ""]
    lines += block(task, None, ladders, "Multi-scale ladders, overlap, position pyramids (rounds 8-11)")
    lines += block(task, None, ridge, "Random polynomial kernel PPV recipes (rounds 4-7)")
    lines += block(task, None, fast, "Fast learners (RF / ExtraTrees / ridge), every feature set")
    lines += block(task, None, None, "Including the slow learners (RotationForest, HistGB) on the "
                                     "datasets they finished before being dropped")
with open("results/dev_ablations.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n".join(lines))
