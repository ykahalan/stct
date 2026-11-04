"""Paper table for regression: error, mean rank, feature count and runtime of every regressor.

    python paper_table_reg.py

Error is RMSE relative to the best of the 14 methods on each of the 26 TSER
datasets (median over datasets; 1.0 would be best everywhere), with published
results for the references and the STCT+ regressor run here; the table is sorted
by it, lowest first.  Feature counts and runtimes of the methods that run here
come from paper_time_reg.py: fit on the training split and predict the test
split, one thread, all methods on a dataset in the same process, same machine.
The others show their published fit + predict time from timeseriesclassification.com,
measured on other hardware.  Writes results/paper_table_regression.md.
"""
import numpy as np
import pandas as pd
from scipy import stats

import bench

OURS = "paper_fast:et_ridge"  # STCTRegressor()
REFS = ["DrCIF", "FreshPRINCE", "InceptionTime", "RotF", "ROCKET", "Ridge", "RandF", "XGBoost", "GridSVR", "ResNet",
        "FPCR", "1NN-DTW", "FCN"]
DRCIF_PER_TREE = {20: 150, 24: 150, 57: 170, 84: 200, 91: 200, 100: 200, 144: 220, 266: 270, 365: 290, 1716: 480,
                  2307: 590, 2542: 590, 3578: 650, 7500: 890}  # measured with aeon's DrCIFRegressor defaults
NOT_RUN = {"DrCIF": "drcif", "FreshPRINCE": "783", "InceptionTime": "learned: 5 networks x 128", "RotF": "raw",
           "XGBoost": "raw", "ResNet": "learned: 128", "FPCR": "functional principal components of the raw series",
           "FCN": "learned: 128"}
WHY = {"XGBoost": "not installed", "FPCR": "not installed"}


def published_hours(method, datasets):
    path = f"results/published_times_regression/{method}"
    t = ((pd.read_csv(f"{path}_fittime.csv", index_col=0)["0"] + pd.read_csv(f"{path}_predicttime.csv", index_col=0)["0"])
         / 1000).reindex(datasets)
    if (t < 0).any():
        return "corrupted (negative times)"
    return f"{t.sum() / 3600:.1f} on {int(t.notna().sum())} datasets"


def main():
    mine = bench.read_results("results/full_regression.csv", "regression").drop_duplicates(["dataset", "method"])
    ours = mine[mine.method == OURS].set_index("dataset").rmse.rename("STCT+")
    lengths = mine.drop_duplicates("dataset").set_index("dataset").length.astype(int)
    pub = pd.read_csv("results/published_regression_rmse.csv", index_col=0)[REFS]
    d = pub.join(ours, how="inner").dropna()
    held = d.loc[d.index.difference(bench.DEV_TSER)]
    rel = d.div(d.min(axis=1), axis=0).median()
    rel_held = held.div(held.min(axis=1), axis=0).median()
    rank = d.rank(axis=1, ascending=True).mean()
    t = pd.read_csv("results/paper_time_reg.csv")
    t = t[t.dataset.isin(d.index)]
    raw = f"raw series ({lengths.reindex(d.index).mean():,.0f} values on average)"
    rows = []
    for m in d.columns:
        g = t[t.method == m]
        x = d[m] - d["STCT+"]
        p = stats.wilcoxon(x[x != 0]).pvalue if m != "STCT+" else np.nan
        row = {"Method": m, "RMSE vs best": rel[m], "Held out": rel_held[m], "Mean rank": rank[m],
               "STCT+ better / worse": "" if m == "STCT+" else f"{int((x > 0).sum())} / {int((x < 0).sum())} (p = {p:.3f})"}
        if len(g) == len(d):
            n = g.n_features
            row["Features"] = ("none (raw series)" if m == "1NN-DTW" else raw if m in ("Ridge", "RandF", "GridSVR")
                               else f"{int(n.iloc[0]):,}" if n.nunique() == 1 else f"{n.mean():,.0f} ({n.min():,.0f}-{n.max():,.0f})")
            row["Runtime here, total (h)"] = f"{g.seconds.sum() / 3600:.2f}"
            row["Median per dataset (s)"] = f"{g.seconds.median():.1f}"
        else:
            feats = NOT_RUN.get(m, "")
            if feats == "raw":
                feats = raw
            elif feats == "drcif":
                per_tree = lengths.reindex(d.index).map(DRCIF_PER_TREE)
                feats = f"200 trees x {per_tree.min()}-{per_tree.max()} interval features ({200 * per_tree.mean():,.0f} on average)"
            row["Features"] = feats
            row["Runtime here, total (h)"] = f"not run ({WHY.get(m, 'published time over 9 hours')})" if len(g) == 0 else f"incomplete ({len(g)})"
            row["Median per dataset (s)"] = ""
        row["Published runtime (h)"] = "" if m == "STCT+" else published_hours(m, d.index)
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("RMSE vs best")
    table.insert(0, "#", range(1, len(table) + 1))
    for c in ("RMSE vs best", "Held out"):
        table[c] = table[c].map("{:.3f}".format)
    table["Mean rank"] = table["Mean rank"].map("{:.2f}".format)
    check = t.merge(pub.rename_axis("dataset").stack().rename("published").reset_index().rename(columns={"level_1": "method"}),
                    on=["dataset", "method"], how="inner")
    ratio = np.exp(np.log(check.rmse / check.published).groupby(check.method).mean())
    lines = ["# Regression: error, features and runtime", "",
             f"{len(d)} TSER datasets, {len(held)} of them outside the development set; sorted by median RMSE relative to the best method on each dataset.", "",
             table.to_markdown(index=False), "",
             "Runtime here: fit on the training split plus predict on the test split, one thread, summed over the datasets; "
             "all methods on a dataset ran in the same process on the same machine (paper_time_reg.py). Published runtime: "
             "fit + predict times from timeseriesclassification.com (other hardware; not comparable with the runtime here).", "",
             "RMSE measured here divided by published RMSE, geometric mean per method (configuration check): "
             + ", ".join(f"{m} {v:.3f}" for m, v in ratio.items()), ""]
    open("results/paper_table_regression.md", "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
