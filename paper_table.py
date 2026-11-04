"""Paper table: accuracy, mean rank, feature count and runtime of every classifier, sorted by accuracy.

    python paper_table.py

Accuracy and mean rank cover the 112 UCR datasets common to all methods (published
results for the references, STCT+ run here).  Feature counts and runtimes for
STCT+ and the eight references that run here come from paper_time.py: fit on the
training split and predict the test split, one thread, all methods on a dataset in
the same process, same machine.  HC2, InceptionTime, 1NN-DTW and FreshPRINCE were
not run here; their runtime is the published fit + predict time from
timeseriesclassification.com, measured on other hardware.  Writes
results/paper_table_classification.md.
"""
import numpy as np
import pandas as pd

import bench

OURS = "stct_plus:et_ridge"
REFS = ["HC2", "MR-Hydra", "MultiROCKET", "RDST", "WEASEL-2.0", "MiniROCKET", "InceptionTime", "ROCKET", "QUANT",
        "FreshPRINCE", "Catch22", "1NN-DTW"]
NOT_RUN = {"HC2": "over 100,000 (its Arsenal alone: 25 x 4,000)", "InceptionTime": "learned: 5 networks x 128",
           "1NN-DTW": "none (raw series)", "FreshPRINCE": "783"}


def published_hours(method, datasets):
    fit = pd.read_csv(f"results/published_times/{method}_fittime.csv", index_col=0)["0"]
    pred = pd.read_csv(f"results/published_times/{method}_predicttime.csv", index_col=0)["0"]
    return float(((fit + pred) / 1000).reindex(datasets).sum() / 3600)


def main():
    mine = bench.read_results("results/full_classification.csv", "classification").drop_duplicates(["dataset", "method"])
    acc = mine[mine.method == OURS].set_index("dataset").accuracy.rename("STCT+")
    pub = pd.read_csv("results/published_classification_accuracy.csv", index_col=0)[REFS]
    d = pub.join(acc, how="inner").dropna()
    held = d.loc[d.index.difference(bench.DEV_UCR)]
    rank = d.rank(axis=1, ascending=False).mean()
    t = pd.read_csv("results/paper_time.csv")
    t = t[t.dataset.isin(d.index)]
    rows = []
    for m in d.columns:
        g = t[t.method == m]
        row = {"Method": m, "Accuracy (%)": 100 * d[m].mean(), "Held out (%)": 100 * held[m].mean(), "Mean rank": rank[m]}
        if m in NOT_RUN:
            row["Features"] = NOT_RUN[m]
            row["Runtime here (h)"] = "not run"
            row["Median per dataset (s)"] = ""
            row["Published runtime (h)"] = f"{published_hours(m, d.index):.1f}"
        else:
            n = g.n_features
            row["Features"] = f"{int(n.iloc[0]):,}" if n.nunique() == 1 else f"{n.mean():,.0f} (mean; {n.min():,.0f}-{n.max():,.0f})"
            row["Runtime here (h)"] = f"{g.seconds.sum() / 3600:.2f}" if len(g) == len(d) else f"incomplete ({len(g)})"
            row["Median per dataset (s)"] = f"{g.seconds.median():.1f}"
            row["Published runtime (h)"] = "" if m == "STCT+" else f"{published_hours(m, d.index):.1f}"
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("Accuracy (%)", ascending=False)
    table.insert(0, "#", range(1, len(table) + 1))
    for c in ("Accuracy (%)", "Held out (%)"):
        table[c] = table[c].map("{:.2f}".format)
    table["Mean rank"] = table["Mean rank"].map("{:.2f}".format)
    published = pub.rename_axis("dataset").stack().rename("published").reset_index().rename(columns={"level_1": "method"})
    check = t.merge(published, on=["dataset", "method"], how="inner")
    gap = (100 * (check.accuracy - check.published)).groupby(check.method).mean()
    lines = ["# Classification: accuracy, features and runtime", "",
             f"{len(d)} UCR datasets, {len(held)} of them outside the development set; sorted by mean accuracy.", "",
             table.to_markdown(index=False), "",
             "Runtime here: fit on the training split plus predict on the test split, one thread, summed over the "
             "datasets; all methods on a dataset ran in the same process on the same machine (paper_time.py). "
             "Published runtime: fit + predict times from timeseriesclassification.com (other hardware; not comparable "
             "with the runtime here).", "",
             "Accuracy measured here minus published accuracy, per method (configuration check): "
             + ", ".join(f"{m} {v:+.2f}" for m, v in gap.items()), ""]
    open("results/paper_table_classification.md", "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
