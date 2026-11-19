"""A gated aligned ridge, on training splits only.

    python fast_gate.py --seeds 42
    python fast_gate.py --seeds 42 --summary

fast_aligned.py showed a ridge on exact 5-sample fits (A5) is strong on a few
datasets (ChlorineConcentration 80.5 alone vs 66.6 for STCT+) but lowers the
blend on most (-0.25, 25/40).  Here A5 joins the STCT+ blend (STCTClassifier())
only when, inside the training part of each outer fold, its 3-fold accuracy is
at least that of the best kernel ridge (G5max) or of their mean (G5mean).
A8 (windows of about 8 samples) is gated the same way (G8max).  Results persist
in results/fast_gate_scores.csv.  No test split is loaded.
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
from fast_aligned import aligned, exact
from fast_params import A3, kernel_blocks
from stct_models import _soft
from vote_test import MIN_LENGTH

warnings.filterwarnings("ignore")
OUT = "results/fast_gate_scores.csv"


def splitter(y, k, seed):
    stratified = np.unique(y, return_counts=True)[1].min() >= k
    return (StratifiedKFold if stratified else KFold)(k, shuffle=True, random_state=seed)


def ridge_model():
    return make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=A3))


def inner_acc(B, y, seed):
    """3-fold accuracy of a ridge on the given training rows."""
    hits = 0
    try:
        for tr, te in splitter(y, 3, seed).split(B, y):
            hits += (ridge_model().fit(B[tr], y[tr]).predict(B[te]) == y[te]).sum()
    except (np.linalg.LinAlgError, ValueError):
        return -1.0
    return hits / len(y)


def score_dataset(X, y, seed):
    T = X.shape[1]
    long_ = T >= MIN_LENGTH
    stratified = np.unique(y, return_counts=True)[1].max() >= 5
    folds = list((StratifiedKFold if stratified else KFold)(5, shuffle=True, random_state=seed).split(X, y))
    labels = np.unique(y)
    acc = lambda p: float((labels[p.argmax(axis=1)] == y).mean())
    V = np.nan_to_num(SF.features(X)[0])
    K = [np.nan_to_num(B) for B in kernel_blocks(X, seed, SF.QS).values()]
    A = {"5": np.nan_to_num(exact(X)), "8": np.nan_to_num(aligned(X, 8))}
    w = 1.0 if long_ else 0.5
    variants = ("base", "G5max", "G5mean", "G8max")
    P = {v: np.zeros((len(y), len(labels))) for v in variants}
    used = {v: 0 for v in variants}
    for tr, te in folds:
        et = ExtraTreesClassifier(n_estimators=200, max_features=0.1, random_state=seed, n_jobs=-1).fit(V[tr], y[tr])
        cols = np.searchsorted(labels, et.classes_)
        fp = np.zeros((len(te), len(labels)))
        fp[:, cols] = et.predict_proba(V[te])

        def member(B):
            try:
                r = ridge_model().fit(B[tr], y[tr])
            except np.linalg.LinAlgError:
                return None
            out = np.zeros((len(te), len(labels)))
            out[:, np.searchsorted(labels, r.classes_)] = _soft(r.decision_function(B[te]))
            return out

        kr = [p for p in (member(B) for B in K) if p is not None]
        k_inner = [inner_acc(B[tr], y[tr], seed) for B in K]
        a_inner = {k: inner_acc(B[tr], y[tr], seed) for k, B in A.items()}
        gates = {"G5max": ("5", max(k_inner)), "G5mean": ("5", np.mean(k_inner)), "G8max": ("8", max(k_inner))}
        members = {"base": kr}
        for v, (k, bar) in gates.items():
            members[v] = kr + [member(A[k])] if a_inner[k] >= bar else kr
            used[v] += a_inner[k] >= bar
        for v, ms in members.items():
            ms = [m for m in ms if m is not None]
            P[v][te] = (fp + w * sum(ms)) / (1 + w * len(ms))
    return [dict(variant=v, score=acc(P[v]), gate_folds=used[v], seed=seed, T=T, n=len(y)) for v in variants]


def summary(seeds):
    d = pd.read_csv(OUT)
    d = d[d.seed.isin(seeds)].drop_duplicates(["seed", "dataset", "variant"], keep="last")
    piv = d.pivot_table(index=["seed", "dataset"], columns="variant", values="score")
    T = d.drop_duplicates(["seed", "dataset"]).set_index(["seed", "dataset"])["T"].reindex(piv.index)
    gates = d.pivot_table(index=["seed", "dataset"], columns="variant", values="gate_folds")
    rows = []
    for v in piv.columns:
        g = (piv[v] - piv["base"]) * 100
        rows.append(dict(variant=v, gain=g.mean(), by_seed=" ".join(f"{x:+.2f}" for x in g.groupby(level="seed").mean()),
                         short=g[T < MIN_LENGTH].mean(), long=g[T >= MIN_LENGTH].mean(),
                         better=int((g > 0).sum()), worse=int((g < 0).sum()),
                         gated_in=f"{(gates[v] > 0).sum()} datasets, {gates[v].sum() / (5 * len(gates)):.0%} of folds"))
    print(pd.DataFrame(rows).sort_values("gain", ascending=False).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    g = (piv["G5max"] - piv["base"]) * 100
    print("\nlargest changes for G5max:\n", g[g != 0].sort_values().round(2).to_string())


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
