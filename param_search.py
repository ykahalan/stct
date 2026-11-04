"""Search the settings of paper features + pooled using training splits only.

    python param_search.py

Each setting is tried in turn from the current best configuration, scored by
5-fold cross-validation inside the training split of every dataset.  No test
split is loaded.  The rule is fixed in advance and sits above the seed noise
measured on this recipe (up to 0.25 accuracy points and a 0.6 % RMSE change):

* classification win: accuracy up by at least 0.3 points with more datasets
  better than worse, and regression RMSE ratio at most 1.006; or
* regression win: RMSE ratio at most 0.99 with more datasets better than worse,
  and accuracy down by at most 0.25 points; or
* simpler: at least 20 % fewer features, accuracy down by at most 0.05 points and
  RMSE ratio at most 1.002.

Feature settings are shared by both tasks.  ExtraTrees settings are then chosen
per task, using only that task's condition.  Configurations above MAX_FEATURES
at length 1024 are skipped.  The final choice is rescored against the start with
two other seeds.  Scores persist in results/param_scores.csv, so runs resume.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

import bench
import stct_plus as SP
from algo_select import load_train, with_retry

warnings.filterwarnings("ignore")
MAX_FEATURES = 600
S = SP.POOL_STATS
START = dict(paper_counts=(8, 16), degree=4, fractions=SP.POOL_FRACTIONS, overlap=8, pool_degree=4, stats=S, resid_norm="rms")
FEATURE_AXES = {
    "paper_counts": [(8,), (16,), (4, 8), (16, 32), (4, 8, 16), (8, 16, 32), (2, 4, 8, 16), (4, 8, 16, 32)],
    "degree": [2, 3, 5, 6],
    "fractions": [(1 / 16, 1 / 8, 1 / 4), (1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4), (1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2),
                  (1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2), (1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4)],
    "overlap": [2, 4, 16, 32],
    "pool_degree": [1, 2, 3, 5, 6],
    "stats": [S[:8], S[:4], ("mean", "std", "min", "max", "argmax", "argmin"),
              ("mean", "std", "min", "max", "q10", "q25", "q50", "q75", "q90", "ppv", "argmax", "argmin")],
    "resid_norm": ["mae", "max"],
}
LEARNER_AXES = {
    "classification": {"max_features": [0.1, 0.25, 0.5], "criterion": ["entropy"], "min_samples_leaf": [2, 3]},
    "regression": {"max_features": ["sqrt", 0.1, 0.25, 0.5], "min_samples_leaf": [2, 3]},
}
START_LEARNER = {"classification": dict(max_features="sqrt", criterion="gini", min_samples_leaf=1),
                 "regression": dict(max_features=1.0, min_samples_leaf=1)}


def features(X, cfg):
    return np.nan_to_num(SP.stct_plus(X, paper=True, **cfg))


def n_features(cfg):
    return features(np.random.default_rng(0).standard_normal((2, 1024)), cfg).shape[1]


def cv_score(task, X, y, cfg, learner, seed):
    F = features(X, cfg)
    if task == "classification":
        est = ExtraTreesClassifier(n_estimators=200, random_state=seed, n_jobs=-1, **learner)
        cv = (StratifiedKFold if np.unique(y, return_counts=True)[1].max() >= 5 else KFold)(5, shuffle=True, random_state=seed)
        return float((cross_val_predict(est, F, y, cv=cv) == y).mean())
    est = ExtraTreesRegressor(n_estimators=200, random_state=seed, n_jobs=-1, **learner)
    pred = cross_val_predict(est, F, y, cv=KFold(5, shuffle=True, random_state=seed))
    return float(np.sqrt(((pred - y) ** 2).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classification", default=None, help="comma-separated override, smoke tests only")
    ap.add_argument("--regression", default=None, help="comma-separated override, smoke tests only")
    ap.add_argument("--prefix", default="results/param")
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

    scores_csv, trials_csv = f"{a.prefix}_scores.csv", f"{a.prefix}_trials.csv"
    known = {}
    if os.path.exists(scores_csv):
        p = pd.read_csv(scores_csv, keep_default_na=False)
        known = {(r.key, r.task, r.dataset): r.score for r in p.itertuples()}
    if os.path.exists(trials_csv):
        os.remove(trials_csv)

    def score_task(task, cfg, learner, seed=bench.RNG):
        key = json.dumps([cfg, learner, seed], sort_keys=True)
        vals, secs = {}, 0.0
        for n, (X, y) in data[task].items():
            if (key, task, n) not in known:
                t0 = time.time()
                known[key, task, n] = with_retry(cv_score, task, X, y, cfg, learner, seed)
                secs += time.time() - t0
                row = dict(key=key, task=task, dataset=n, score=known[key, task, n])
                pd.DataFrame([row]).to_csv(scores_csv, mode="a", header=not os.path.exists(scores_csv), index=False)
            vals[n] = known[key, task, n]
        return pd.Series(vals), secs

    def compare(task, cand, base):
        if task == "classification":
            d = cand - base
            return float(d.mean()), int((d > 0).sum()), int((d < 0).sum())
        r = (cand + 1e-12) / (base + 1e-12)
        return float(np.exp(np.log(r).mean())), int((r < 1).sum()), int((r > 1).sum())

    def record(row):
        pd.DataFrame([row]).to_csv(trials_csv, mode="a", header=not os.path.exists(trials_csv), index=False)
        print(json.dumps(row), flush=True)

    # shared feature settings
    cfg = dict(START)
    learners = {t: dict(START_LEARNER[t]) for t in data}
    base = {t: score_task(t, cfg, learners[t])[0] for t in data}
    base_feats = n_features(cfg)
    start_scores = dict(base)
    print(f"start: {base_feats} features, accuracy {base['classification'].mean():.4f}", flush=True)
    for axis, values in FEATURE_AXES.items():
        trials = []
        for v in values:
            cand = dict(cfg, **{axis: v})
            feats = n_features(cand)
            if feats > MAX_FEATURES:
                print(f"{axis} {v}: {feats} features, over the limit", flush=True)
                continue
            sc = {t: score_task(t, cand, learners[t])[0] for t in data}
            gain, ab, aw = compare("classification", sc["classification"], base["classification"])
            ratio, rb, rw = compare("regression", sc["regression"], base["regression"])
            cls_win = gain >= 0.003 and ab > aw and ratio <= 1.006
            reg_win = ratio <= 0.99 and rb > rw and gain >= -0.0025
            simpler = feats <= 0.8 * base_feats and gain >= -0.0005 and ratio <= 1.002
            t = dict(stage="features", axis=axis, value=str(v), features=feats, acc_gain=round(gain, 5), acc_better=ab, acc_worse=aw,
                     rmse_ratio=round(ratio, 4), rmse_better=rb, rmse_worse=rw, win=bool(cls_win or reg_win), simpler=bool(simpler))
            trials.append(dict(t, cfg=cand, sc=sc))
            record(t)
        wins = [t for t in trials if t["win"]]
        simple = [t for t in trials if t["simpler"]]
        choice = max(wins, key=lambda t: t["acc_gain"] - 0.1 * (t["rmse_ratio"] - 1)) if wins else (min(simple, key=lambda t: t["features"]) if simple else None)
        if choice:
            cfg, base, base_feats = choice["cfg"], choice["sc"], choice["features"]
            print(f"ADOPT {axis} = {choice['value']}", flush=True)

    # per-task ExtraTrees settings
    for task in data:
        for axis, values in LEARNER_AXES[task].items():
            trials = []
            for v in values:
                cand = dict(learners[task], **{axis: v})
                sc = score_task(task, cfg, cand)[0]
                m, b, w = compare(task, sc, base[task])
                win = (m >= 0.003 if task == "classification" else m <= 0.99) and b > w
                t = dict(stage=task, axis=axis, value=str(v), metric=round(m, 5), better=b, worse=w, win=bool(win))
                trials.append(dict(t, learner=cand, sc=sc))
                record(t)
            wins = [t for t in trials if t["win"]]
            if wins:
                choice = max(wins, key=lambda t: t["metric"]) if task == "classification" else min(wins, key=lambda t: t["metric"])
                learners[task], base[task] = choice["learner"], choice["sc"]
                print(f"ADOPT {task} {axis} = {choice['value']}", flush=True)

    # recheck the final choice against the start with other seeds
    checks = []
    for seed in (1, 2):
        row = dict(seed=seed)
        for t in data:
            final = score_task(t, cfg, learners[t], seed)[0]
            start = score_task(t, START, START_LEARNER[t], seed)[0]
            m, b, w = compare(t, final, start)
            row.update({f"{t}_metric": round(m, 5), f"{t}_better": b, f"{t}_worse": w})
        checks.append(row)
        print("RECHECK", json.dumps(row), flush=True)
    out = dict(config={k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}, learners=learners, features=base_feats,
               start_accuracy=float(start_scores["classification"].mean()), final_accuracy=float(base["classification"].mean()), rechecks=checks)
    with open(f"{a.prefix}_selected.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("FINAL", json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
