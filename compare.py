"""Compare bench.py results with published reference results (timeseriesclassification.com).

    python compare.py --task classification --results results/full_classification.csv \
        --methods fixed:rf,plus:rotf --refs HC2,MR-Hydra,InceptionTime,MiniROCKET,FreshPRINCE

Writes results/summary_<task>.md (markdown tables) and results/perdataset_<task>.csv.
Only datasets for which every compared method has a score are used, so all
numbers in a table are on the same dataset set and the same default
train/test split.
"""
from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

PUBLISHED = {
    "classification": "results/published_classification_accuracy.csv",
    "regression": "results/published_regression_rmse.csv",
}


def load_mine(path, task, methods):
    from bench import read_results

    df = read_results(path, task)
    key = "accuracy" if task == "classification" else "rmse"
    piv = df.pivot_table(index="dataset", columns="method", values=key)
    cols = [m for m in methods if m in piv.columns] if methods else list(piv.columns)
    return piv[cols]


def pairwise(piv, a, b, higher_better):
    x, y = piv[a].values, piv[b].values
    d = x - y
    wins = int((d > 1e-12).sum() if higher_better else (d < -1e-12).sum())
    losses = int((d < -1e-12).sum() if higher_better else (d > 1e-12).sum())
    ties = len(d) - wins - losses
    try:
        p = stats.wilcoxon(x, y).pvalue if np.any(d != 0) else 1.0
    except ValueError:
        p = float("nan")
    return wins, ties, losses, p


def table(piv, task):
    higher = task == "classification"
    ranks = piv.rank(axis=1, ascending=not higher)
    out = pd.DataFrame({"mean": piv.mean(), "mean_rank": ranks.mean(),
                        "best_on": (ranks == 1).sum()})
    if not higher:
        out["mean_rel_rmse"] = piv.div(piv.min(axis=1), axis=0).mean()
        out["median_rel_rmse"] = piv.div(piv.min(axis=1), axis=0).median()
    return out.sort_values("mean_rank")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["classification", "regression"])
    ap.add_argument("--results", required=True)
    ap.add_argument("--methods", default=None)
    ap.add_argument("--refs", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    methods = a.methods.split(",") if a.methods else None
    mine = load_mine(a.results, a.task, methods)
    pub = pd.read_csv(PUBLISHED[a.task], index_col=0)
    refs = a.refs.split(",") if a.refs else [c for c in pub.columns if pub[c].notna().sum() > 0]
    pub = pub[refs]
    both = mine.join(pub, how="inner").dropna()
    higher = a.task == "classification"
    key = "accuracy" if higher else "RMSE"

    lines = [f"# {a.task}: STCT variants vs published results", "",
             f"{len(both)} datasets common to all methods (default train/test split).", ""]
    t = table(both, a.task)
    lines += ["## Overall", "", t.round(4).to_markdown(), ""]
    lines += [f"## Pairwise: each STCT variant vs each reference (wins/ties/losses on {key}, Wilcoxon p)", ""]
    rows = []
    for m in mine.columns:
        for r in refs:
            w, ti, l, p = pairwise(both, m, r, higher)
            rows.append({"variant": m, "reference": r, "wins": w, "ties": ti, "losses": l,
                         "wilcoxon_p": p, "mean_diff": (both[m] - both[r]).mean()})
    pw = pd.DataFrame(rows)
    lines += [pw.round(4).to_markdown(index=False), ""]
    if len(mine.columns) > 1:
        lines += ["## Pairwise among STCT variants", ""]
        rows = []
        for m1, m2 in combinations(mine.columns, 2):
            w, ti, l, p = pairwise(both, m1, m2, higher)
            rows.append({"a": m1, "b": m2, "a_wins": w, "ties": ti, "b_wins": l, "wilcoxon_p": p})
        lines += [pd.DataFrame(rows).round(4).to_markdown(index=False), ""]
    if not higher:
        lines += ["## Mean rank is the headline for regression: raw RMSE is not comparable across datasets.", ""]
    md = "\n".join(lines)
    out = a.out or f"results/summary_{a.task}.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    both.round(4).to_csv(f"results/perdataset_{a.task}.csv")
    print(md)


if __name__ == "__main__":
    main()
