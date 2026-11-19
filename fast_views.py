"""One forest per view instead of one forest over all views, on training splits only.

    python fast_views.py --seeds 42
    python fast_views.py --seeds 42 --summary

On the test splits the paper features alone (fixed windows of the series, one
ExtraTrees) beat STCT+ on 19 datasets, by 6-7 points on ChlorineConcentration,
FaceAll and EthanolLevel, although STCT+ contains those features: in one forest
over about 700 view features they are rarely drawn.  Same features and the same
200 trees, split differently, out of fold:

* split4: 50 trees on each view (fixed windows of the series, pooled sliding
  fits, fixed windows of the sorted series, of the autocorrelation), averaged
* split2: 100 trees on the fixed windows of the series, 100 on the rest
* add_fixed: the current forest averaged with 50 more trees on the fixed windows

Each forest replaces the current one in the STCT+ blend (STCTClassifier()); the
"forest" variants score the forests alone.  Results persist in
results/fast_views_scores.csv.  No test split is loaded.
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import bench
import stct_fast as SF
import stct_plus as SP
from algo_select import load_train, with_retry
from fast_params import A3, kernel_blocks
from stct_models import _soft
from vote_test import MIN_LENGTH

warnings.filterwarnings("ignore")
OUT = "results/fast_views_scores.csv"


def view_slices(X):
    """Column ranges of the four views inside SF.features(X)[0]."""
    nP = SF._paper(X, False).shape[1]
    nR = SF._paper(SP.autocorrelation(X), False).shape[1]
    total = SF.features(X[:2])[0].shape[1]
    nS = total - 2 * nP - nR
    edges = np.cumsum([0, nP, nS, nP, nR])
    return {v: slice(a, b) for v, a, b in zip(("P", "S", "Ps", "Pr"), edges[:-1], edges[1:])}


def score_dataset(X, y, seed):
    T = X.shape[1]
    long_ = T >= MIN_LENGTH
    stratified = np.unique(y, return_counts=True)[1].max() >= 5
    folds = list((StratifiedKFold if stratified else KFold)(5, shuffle=True, random_state=seed).split(X, y))
    labels = np.unique(y)
    V = np.nan_to_num(SF.features(X)[0])
    cols = view_slices(X)
    acc = lambda p: float((labels[p.argmax(axis=1)] == y).mean())
    secs = {}

    def forest(F, trees, key):
        out = np.zeros((len(y), len(labels)))
        t0 = time.perf_counter()
        for tr, te in folds:
            et = ExtraTreesClassifier(n_estimators=trees, max_features=0.1, random_state=seed, n_jobs=-1).fit(F[tr], y[tr])
            out[te[:, None], np.searchsorted(labels, et.classes_)] = et.predict_proba(F[te])
        secs[key] = secs.get(key, 0.0) + time.perf_counter() - t0
        return out

    def ridge(B):
        out = np.zeros((len(y), len(labels)))
        try:
            for tr, te in folds:
                r = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=A3)).fit(B[tr], y[tr])
                out[te[:, None], np.searchsorted(labels, r.classes_)] = _soft(r.decision_function(B[te]))
        except np.linalg.LinAlgError:
            return None
        return out

    rps = [r for r in (ridge(np.nan_to_num(B)) for B in kernel_blocks(X, seed, SF.QS).values()) if r is not None]
    w = 1.0 if long_ else 0.5
    blend = lambda fp: (fp + w * sum(rps)) / (1 + w * len(rps))
    per_view = {v: forest(V[:, c], 50, f"f50_{v}") for v, c in cols.items()}
    rest = np.hstack([V[:, c] for v, c in cols.items() if v != "P"])
    forests = {"base": forest(V, 200, "base"),
               "split4": sum(per_view.values()) / 4,
               "split2": (forest(V[:, cols["P"]], 100, "split2") + forest(rest, 100, "split2")) / 2}
    forests["add_fixed"] = (4 * forests["base"] + per_view["P"]) / 5
    seconds = {"base": secs["base"], "split4": sum(s for k, s in secs.items() if k.startswith("f50_")),
               "split2": secs["split2"], "add_fixed": secs["base"] + secs["f50_P"]}
    rows = []
    for name, fp in forests.items():
        rows.append(dict(variant=name, score=acc(blend(fp)), forest_seconds=seconds[name]))
        rows.append(dict(variant=f"forest:{name}", score=acc(fp), forest_seconds=seconds[name]))
    for v, fp in per_view.items():
        rows.append(dict(variant=f"forest:{v}50", score=acc(fp), forest_seconds=secs[f"f50_{v}"]))
    return [dict(r, seed=seed, T=T, n=len(y)) for r in rows]


def summary(seeds):
    d = pd.read_csv(OUT)
    d = d[d.seed.isin(seeds)].drop_duplicates(["seed", "dataset", "variant"], keep="last")
    piv = d.pivot_table(index=["seed", "dataset"], columns="variant", values="score")
    T = d.drop_duplicates(["seed", "dataset"]).set_index(["seed", "dataset"])["T"].reindex(piv.index)
    n = d.drop_duplicates(["seed", "dataset"]).set_index(["seed", "dataset"])["n"].reindex(piv.index)
    secs = d.groupby("variant").forest_seconds.sum()
    rows = []
    for v in piv.columns:
        ref = "forest:base" if v.startswith("forest:") else "base"
        g = (piv[v] - piv[ref]) * 100
        rows.append(dict(variant=v, vs=ref, gain=g.mean(), by_seed=" ".join(f"{x:+.2f}" for x in g.groupby(level="seed").mean()),
                         short=g[T < MIN_LENGTH].mean(), long=g[T >= MIN_LENGTH].mean(), large_n=g[n > 250].mean(),
                         better=int((g > 0).sum()), worse=int((g < 0).sum()), forest_time=secs[v] / secs[ref]))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    g = (piv["split4"] - piv["base"]) * 100
    print("\nlargest split4 changes:\n", g.sort_values().iloc[[*range(5), *range(-5, 0)]].round(2).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=str(bench.RNG))
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--datasets", default=None)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    if a.summary:
        return summary(seeds)
    known = set()
    if os.path.exists(OUT):
        p = pd.read_csv(OUT)
        known = set(zip(p.seed, p.dataset))
    names = a.datasets.split(",") if a.datasets else bench.UCR
    for seed in seeds:
        for name in names:
            if (seed, name) in known:
                continue
            try:
                X, y = load_train("classification", name)
                rows = with_retry(score_dataset, X, y, seed)
            except Exception as e:
                print(f"[failed] {seed} {name}: {str(e)[:120]}", flush=True)
                continue
            pd.DataFrame(rows).assign(dataset=name).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
            print(f"{seed} {name}", flush=True)


if __name__ == "__main__":
    main()
