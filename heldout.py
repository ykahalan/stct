"""One STCT recipe at a time against the published methods, on all and on held-out datasets.

Each recipe is ranked alone with the references, so several similar STCT
variants cannot push the references down the table.  Held-out datasets are
those whose test scores were not used when the recipe structure was chosen
(not in bench.DEV_UCR / DEV_TSER).
Writes results/heldout_<task>.md.
"""
import numpy as np
import pandas as pd
from scipy import stats

import bench

CONFIG = {
    "classification": dict(
        key="accuracy", dev=bench.DEV_UCR,
        recipes=["fixed:rf", "fixed:et", "paper_pooled:et", "paper_tuned:et_tuned", "minirocket:ridge"],
        refs=["HC2", "MR-Hydra", "MultiROCKET", "RDST", "WEASEL-2.0", "MiniROCKET", "InceptionTime", "ROCKET",
              "QUANT", "FreshPRINCE", "Catch22", "1NN-DTW"]),
    "regression": dict(
        key="rmse", dev=bench.DEV_TSER,
        recipes=["fixed:rf", "fixed:et", "paper_pooled:et", "paper_tuned:et_tuned", "minirocket:ridge"],
        refs=["DrCIF", "FreshPRINCE", "InceptionTime", "RotF", "ROCKET", "Ridge", "RandF", "XGBoost", "GridSVR",
              "ResNet", "FPCR", "1NN-DTW", "FCN"]),
}

for task, c in CONFIG.items():
    higher = task == "classification"
    mine = bench.read_results(f"results/full_{task}.csv", task).drop_duplicates(["dataset", "method"])
    piv = mine.pivot_table(index="dataset", columns="method", values=c["key"])
    pub = pd.read_csv(f"results/published_{task}_{'accuracy' if higher else 'rmse'}.csv", index_col=0)[c["refs"]]
    lines = [f"# {task}: one recipe at a time vs published methods", ""]
    for label, drop in [("all datasets", []), ("held-out datasets", c["dev"])]:
        rows = []
        for r in c["recipes"]:
            d = piv[[r]].join(pub, how="inner").dropna()
            d = d.loc[d.index.difference(drop)]
            ranks = d.rank(axis=1, ascending=not higher)
            row = {"recipe": r, "datasets": len(d), "score": d[r].mean() if higher else d[r].div(d.min(axis=1)).median(),
                   "mean_rank": ranks[r].mean(), "place": int((ranks.mean() < ranks[r].mean()).sum() + 1)}
            for ref in (c["refs"][:2] if higher else c["refs"][:2]):
                x = d[r] - d[ref]
                w, l = ((x > 0).sum(), (x < 0).sum()) if higher else ((x < 0).sum(), (x > 0).sum())
                row[f"vs {ref}"] = f"{w}/{l} p={stats.wilcoxon(d[r], d[ref]).pvalue:.3f}"
            rows.append(row)
        t = pd.DataFrame(rows).round(4)
        head = "mean accuracy" if higher else "median RMSE relative to best"
        lines += [f"## {label} (score = {head}; place out of {len(c['refs']) + 1})", "", t.to_markdown(index=False), ""]
        base = piv.join(pub, how="inner")[c["refs"]].dropna()
        base = base.loc[base.index.difference(drop)]
        rk = base.rank(axis=1, ascending=not higher).mean().sort_values()
        lines += ["Published methods alone, mean rank: " + ", ".join(f"{k} {v:.1f}" for k, v in rk.items()), ""]
    # timing and feature counts for the recipes
    tt = mine.assign(t=mine[["extract_time_s", "fit_time_s", "predict_time_s"]].sum(axis=1))
    g = tt[tt.method.isin(c["recipes"])].groupby("method").agg(features=("n_features", "mean"), mean_s=("t", "mean"), median_s=("t", "median"))
    lines += ["## cost", "", g.round(2).to_markdown(), ""]
    text = "\n".join(lines)
    open(f"results/heldout_{task}.md", "w", encoding="utf-8").write(text)
    print(text)
