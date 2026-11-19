"""Learner and combination parameters of STCT+, on training splits only.

    python fast_params.py --seeds 42
    python fast_params.py --seeds 42 --summary

Features are those of STCT+ in the paper tables (STCTClassifier()).  One change
at a time from that recipe, out of fold:

* forest: 5%, 20% or sqrt features per split, entropy criterion, 400 trees
* ridges: penalties logspace(-1, 5, 13) instead of logspace(-3, 3, 10)
* combination: geometric instead of arithmetic mean of probabilities; forest
  weight 1/2 or 2 relative to each ridge
* kernel thresholds: 3 or 9 quantiles instead of 5 (context and dilated kernels)

Results persist in results/fast_params_scores.csv.  No test split is loaded.
"""
import argparse
import os
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
from algo_select import load_train, with_retry
from stct_models import _soft
from vote_test import MIN_LENGTH

warnings.filterwarnings("ignore")
OUT = "results/fast_params_scores.csv"
A3, A5 = np.logspace(-3, 3, 10), np.logspace(-1, 5, 13)
QSETS = {"q5": SF.QS.copy(), "q3": np.linspace(0, 1, 5)[1:-1], "q9": np.linspace(0, 1, 11)[1:-1]}


def kernel_blocks(X, seed, qs):
    """Context, winner and dilated blocks of STCT+ with the given threshold quantiles."""
    T, saved = X.shape[1], SF.QS
    SF.QS = qs
    try:
        long_ = T >= MIN_LENGTH
        k = min(64, T // 16) if long_ else int(min(64, max(1, T // 8)))
        D = SF.dilated_features(X, SF.dilated_fit(X, k, seed))
        if not long_:
            return {"D": D}
        _, C, W = SF.trio_features(X, SF.context_fit(X, k, seed, 2), SF.compete_fit(X, seed, SF.winner_groups(T)))
        return {"C": C, "W": W, "D": D}
    finally:
        SF.QS = saved


def score_dataset(X, y, seed):
    T = X.shape[1]
    long_ = T >= MIN_LENGTH
    stratified = np.unique(y, return_counts=True)[1].max() >= 5
    folds = list((StratifiedKFold if stratified else KFold)(5, shuffle=True, random_state=seed).split(X, y))
    labels = np.unique(y)
    V = np.nan_to_num(SF.features(X)[0])
    acc = lambda p: float((labels[p.argmax(axis=1)] == y).mean())

    def forest(**kw):
        out = np.zeros((len(y), len(labels)))
        opts = dict(n_estimators=200, max_features=0.1, random_state=seed, n_jobs=-1) | kw
        for tr, te in folds:
            et = ExtraTreesClassifier(**opts).fit(V[tr], y[tr])
            out[te[:, None], np.searchsorted(labels, et.classes_)] = et.predict_proba(V[te])
        return out

    def ridge(B, alphas):
        out = np.zeros((len(y), len(labels)))
        try:
            for tr, te in folds:
                r = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=alphas)).fit(B[tr], y[tr])
                out[te[:, None], np.searchsorted(labels, r.classes_)] = _soft(r.decision_function(B[te]))
        except np.linalg.LinAlgError:
            return None
        return out

    def blend(fp, rps, wf=1.0, geometric=False):
        w = 1.0 if long_ else 0.5
        rps = [r for r in rps if r is not None]
        if geometric:
            logs = wf * np.log(fp + 1e-6) + w * sum(np.log(r + 1e-6) for r in rps)
            return logs / (wf + w * len(rps))
        return (wf * fp + w * sum(rps)) / (wf + w * len(rps))

    blocks = {q: {k: np.nan_to_num(B) for k, B in kernel_blocks(X, seed, qs).items()} for q, qs in QSETS.items()}
    base_r = [ridge(B, A3) for B in blocks["q5"].values()]
    f0 = forest()
    rows = {"base": acc(blend(f0, base_r))}
    for name, kw in {"mf05": dict(max_features=0.05), "mf20": dict(max_features=0.2), "mfsqrt": dict(max_features="sqrt"),
                     "entropy": dict(criterion="entropy"), "trees400": dict(n_estimators=400)}.items():
        rows[name] = acc(blend(forest(**kw), base_r))
    rows["alpha5"] = acc(blend(f0, [ridge(B, A5) for B in blocks["q5"].values()]))
    rows["geometric"] = acc(blend(f0, base_r, geometric=True))
    rows["forest_w0.5"] = acc(blend(f0, base_r, wf=0.5))
    rows["forest_w2"] = acc(blend(f0, base_r, wf=2.0))
    for q in ("q3", "q9"):
        rows[q] = acc(blend(f0, [ridge(B, A3) for B in blocks[q].values()]))
    feats = {"base": sum(B.shape[1] for B in blocks["q5"].values()), "q3": sum(B.shape[1] for B in blocks["q3"].values()),
             "q9": sum(B.shape[1] for B in blocks["q9"].values())}
    return [dict(variant=k, score=v, features=V.shape[1] + feats.get(k, feats["base"]), seed=seed, T=T, n=len(y))
            for k, v in rows.items()]


def summary(seeds):
    d = pd.read_csv(OUT)
    d = d[d.seed.isin(seeds)].drop_duplicates(["seed", "dataset", "variant"], keep="last")
    piv = d.pivot_table(index=["seed", "dataset"], columns="variant", values="score")
    T = d.drop_duplicates(["seed", "dataset"]).set_index(["seed", "dataset"])["T"].reindex(piv.index)
    n = d.drop_duplicates(["seed", "dataset"]).set_index(["seed", "dataset"])["n"].reindex(piv.index)
    feats = d.groupby("variant").features.mean()
    rows = []
    for v in piv.columns:
        g = (piv[v] - piv["base"]) * 100
        rows.append(dict(variant=v, gain=g.mean(), by_seed=" ".join(f"{x:+.2f}" for x in g.groupby(level="seed").mean()),
                         short=g[T < MIN_LENGTH].mean(), long=g[T >= MIN_LENGTH].mean(), large_n=g[n > 250].mean(),
                         better=int((g > 0).sum()), worse=int((g < 0).sum()), features=feats[v]))
    print(pd.DataFrame(rows).sort_values("gain", ascending=False).to_string(index=False, float_format=lambda x: f"{x:.2f}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=str(bench.RNG))
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    if a.summary:
        return summary(seeds)
    known = set()
    if os.path.exists(OUT):
        p = pd.read_csv(OUT)
        known = set(zip(p.seed, p.dataset))
    for seed in seeds:
        for name in bench.UCR:
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
