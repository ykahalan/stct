"""Ablate STCT blocks and their combinations using training splits only.

    python ablate.py
    python ablate.py --configs P+S,P+S+C --seeds 1,2

Blocks, combined with "+":

    P    paper features: fit coefficients and residual RMS in 8 and 16 windows
    S    pooled statistics of the sliding fit at six window lengths
    SY   S, plus the same statistics over each half of the series (a position pyramid)
    K    8 random coefficient weightings per scale: proportion above five training
         quantiles plus the maximum (the useful part of the earlier kernel recipe)
    KY   K, plus the same over each half of the series
    K32  K with 32 weightings per scale
    C    correlation between coefficient channels over positions, per scale
    D    pooled statistics of the first difference series
    H    histogram of 8 local shapes (signs of coefficients 1 to 3) per scale
    G    depth-3 path signature of (time, value) over the series, halves and quarters
    W    10 statistics of each DWT level
    F    mean, standard deviation and maximum of each STFT frequency band over time
    Pd   paper features of the first difference series
    Pc   paper features of the cumulative sum of the mean-removed series
    Dc   pooled statistics of that cumulative sum
    Pa, Da  paper features or pooled statistics of the absolute first difference
    Ps      paper features of the sorted series (its value distribution)
    Pr      paper features of the autocorrelation function
    P2, D2  paper features or pooled statistics of the second difference
    K64  K with 64 weightings per scale, or 32 on series shorter than 300
    KT   K with one weighting per 8 samples of series length, 1 to 64 per scale
    SQ   pooled statistics with q5, q10, q90 and q95 added, in place of S
    Sd   pooled statistics of fits of degree 2 and 6, added to S
    Sr   pooled statistics of the autocorrelation function
    A|B  averages ExtraTrees on blocks A with ridge on blocks B
    WG   paper windows + pooled statistics of a Chebyshev fit with Gaussian weights (window centre)
    WH   the same with a Hann taper (window centre, reaching zero at the edges)
    WC   the same with the Chebyshev weight 1 / sqrt(1 - x^2) (window edges)
    WS   the same with exponential weights (window start)
    WE   the same with exponential weights (window end)
    N1, N4, Nm, Nh  residual L1, L4, maximum or L1/2 norm in paper windows and pooled statistics
    Ns   the same with the scale-free residual ratios L1 / L2 and maximum / L2
    Wr   raw DWT coefficients, as in the paper notebooks
    Fr   raw STFT magnitudes, as in the paper notebooks

A ":ridge" suffix uses a ridge learner instead of ExtraTrees.  Kernel thresholds use
training data without labels.  No test split is loaded.  Scores persist in
results/ablate_scores.csv and feature counts in results/ablate_features.csv.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import warnings

import numpy as np
import pandas as pd
from numpy.polynomial import chebyshev as C
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import Ridge, RidgeClassifier, RidgeClassifierCV, RidgeCV
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import bench
import stct_plus as SP
from algo_select import load_train, with_retry

warnings.filterwarnings("ignore")
FRACTIONS = (1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4)
DEGREE, OVERLAP, BLOCK = 4, 8, 256
QS = np.linspace(0, 1, 7)[1:-1]


def window_lengths(T):
    out = []
    for f in FRACTIONS:
        L = max(DEGREE + 2, min(int(round(T * f)), T))
        if L not in out:
            out.append(L)
    return out


def pooled(Cx):
    F = SP._pool(Cx, SP.POOL_STATS)
    ch = Cx.shape[1]
    return np.delete(F, SP.POOL_STATS.index("ppv") * ch + ch - 1, axis=1)


def halves(A):
    t = A.shape[2]
    if t < 2:
        return [A, A]
    return [A[:, :, : t // 2], A[:, :, t // 2 :]]


def cumulative(X):
    """Cumulative sum of each mean-removed series."""
    return np.cumsum(X - X.mean(axis=1, keepdims=True), axis=1)


def autocorrelation(X):
    """Normalised autocorrelation of each series, lags 0 to T // 2."""
    T = X.shape[1]
    Z = X - X.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(Z, n=2 * T, axis=1)) ** 2
    acf = np.fft.irfft(power, axis=1)[:, : max(T // 2 + 1, 8)]
    return acf / (acf[:, :1] + 1e-12)


def shape_histogram(Cx):
    """Share of windows in each of 8 shapes given by the signs of coefficients 1, 2 and 3."""
    label = (Cx[:, 1] > 0).astype(int) + 2 * (Cx[:, 2] > 0) + 4 * (Cx[:, 3] > 0)
    return np.stack([(label == b).mean(axis=1) for b in range(8)], axis=1)


def signature(path):
    """Left-point iterated sums to depth 3 of a path (n, t, d): d + d^2 + d^3 values per series."""
    n, _, d = path.shape
    dX = np.diff(path, axis=1)
    A1 = np.cumsum(dX, axis=1)
    prev1 = np.concatenate([np.zeros((n, 1, d)), A1[:, :-1]], axis=1)
    inc2 = prev1[:, :, :, None] * dX[:, :, None, :]  # (n, t, i, j)
    A2 = np.cumsum(inc2, axis=1)
    prev2 = np.concatenate([np.zeros((n, 1, d, d)), A2[:, :-1]], axis=1)
    S3 = (prev2[:, :, :, :, None] * dX[:, :, None, None, :]).sum(axis=1)
    return np.hstack([A1[:, -1], A2[:, -1].reshape(n, -1), S3.reshape(n, -1)])


def signature_pyramid(X):
    """Signatures of the (time, value) path over the series, its halves and its quarters."""
    n, T = X.shape
    t = np.broadcast_to(np.linspace(0, 1, T), (n, T))
    path = np.stack([t, X], axis=2)
    parts = []
    for k in (1, 2, 4):
        edges = np.round(np.linspace(0, T, k + 1)).astype(int)
        for a, b in zip(edges[:-1], edges[1:]):
            parts.append(signature(path[:, a : min(b + 1, T)]) if b - a >= 1 else np.zeros((n, 14)))
    return np.hstack(parts)


def dwt_summary(X):
    import pywt

    lvl = min(4, pywt.dwt_max_level(X.shape[1], pywt.Wavelet("db4").dec_len))
    levels = pywt.wavedec(X, "db4", level=lvl, axis=1)
    return np.hstack([SP._pool(c[:, None, :], SP.POOL_STATS) for c in levels])


def stft_summary(X):
    from scipy.signal import stft

    seg = min(32, X.shape[1])
    _, _, Z = stft(X, nperseg=seg, noverlap=min(16, seg - 1), axis=1)
    return SP._pool(np.abs(Z), ("mean", "std", "max"))


def kernel_summary(P, B):
    ppv = np.count_nonzero(P[:, :, None, :] > B.T[None, :, :, None], axis=3) / P.shape[2]
    return np.hstack([ppv.reshape(len(P), -1), P.max(axis=2)])


def kernel_fit(X, n_dir, seed):
    """Unit weightings and quantile thresholds per scale, from at most 512 series, no labels."""
    rng = np.random.default_rng(seed)
    S = X[rng.choice(len(X), 512, replace=False)] if len(X) > 512 else X
    params = []
    for L in window_lengths(X.shape[1]):
        Cx = SP._windowed_coefficients(S, L, DEGREE, max(1, L // OVERLAP))
        ch = Cx.shape[1]
        flat = Cx.transpose(1, 0, 2).reshape(ch, -1)
        mu, sd = flat.mean(axis=1), flat.std(axis=1) + 1e-12
        D = rng.standard_normal((n_dir, ch))
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        P = D @ ((Cx - mu[None, :, None]) / sd[None, :, None])
        params.append((mu, sd, D, np.quantile(P.transpose(1, 0, 2).reshape(n_dir, -1), QS, axis=1)))
    return params


def basis_design(L, name, degree=DEGREE):
    """Chebyshev design matrix and square-root weights of a window of L samples."""
    x = np.linspace(-1, 1, L)
    weights = {"WG": np.exp(-((2.5 * x) ** 2) / 2),
               "WH": 0.5 * (1 + np.cos(np.pi * x)) + 1e-3,
               "WC": 1 / np.sqrt(1 - np.minimum(x**2, 0.99)),
               "WS": np.exp(-3 * (x + 1)),
               "WE": np.exp(3 * (x - 1))}[name]
    return C.chebvander(x, degree), np.sqrt(weights)


def basis_windows(X, L, starts, name):
    """Coefficients and residual RMS of a weighted fit at the given starts: (n, 6, windows)."""
    Phi, w = basis_design(L, name)
    pinv = np.linalg.pinv(Phi * w[:, None]) * w[None, :]  # (d + 1, L)
    idx = np.asarray(starts)[:, None] + np.arange(L)[None, :]
    out = np.empty((len(X), DEGREE + 2, len(starts)))
    step = max(1, 1_000_000 // (len(starts) * L))
    for s in range(0, len(X), step):
        W = X[s : s + step][:, idx]
        c = W @ pinv.T
        out[s : s + step, : DEGREE + 1] = c.transpose(0, 2, 1)
        out[s : s + step, DEGREE + 1] = np.sqrt(((W - c @ Phi.T) ** 2).mean(axis=-1))
    return out


def basis_block(X, name):
    """Paper windows and pooled statistics, with the fit replaced by the named basis."""
    T = X.shape[1]
    parts = []
    for n0 in (8, 16):
        nw, _ = SP._safe_cheby_params(T, n0, DEGREE)
        L = min(max(DEGREE + 2, T // nw), T)
        step = max(1, (T - L) // max(1, nw - 1))
        starts = list(range(0, T - L + 1, step))[:nw]
        parts.append(basis_windows(X, L, starts, name).reshape(len(X), -1))
    for L in window_lengths(T):
        stride = max(1, L // OVERLAP)
        starts = np.arange(0, T - L + 1, stride)
        if starts[-1] != T - L:
            starts = np.append(starts, T - L)
        parts.append(pooled(basis_windows(X, L, starts, name)))
    return np.hstack(parts)


def residual_measures(R, name):
    """Norms of the fit residuals along the last axis: (..., channels)."""
    A = np.abs(R)
    if name == "N1":
        return A.mean(axis=-1)[..., None]
    if name == "N4":
        return ((A**4).mean(axis=-1) ** 0.25)[..., None]
    if name == "Nm":
        return A.max(axis=-1)[..., None]
    if name == "Nh":
        return (np.sqrt(A).mean(axis=-1) ** 2)[..., None]
    if name == "Ns":
        l2 = np.sqrt((A**2).mean(axis=-1)) + 1e-12
        return np.stack([A.mean(axis=-1) / l2, A.max(axis=-1) / l2], axis=-1)
    raise ValueError(name)


def residual_windows(X, L, starts, names):
    """Residual measures of the Chebyshev fit at the given starts: (n, channels, windows)."""
    Phi = C.chebvander(np.linspace(-1, 1, L), DEGREE)
    pinv = np.linalg.pinv(Phi)
    idx = np.asarray(starts)[:, None] + np.arange(L)[None, :]
    rows = []
    step = max(1, 1_000_000 // (len(starts) * L))
    for s in range(0, len(X), step):
        W = X[s : s + step][:, idx]
        R = W - (W @ pinv.T) @ Phi.T
        rows.append(np.concatenate([residual_measures(R, n) for n in names], axis=-1).transpose(0, 2, 1))
    return np.vstack(rows)


def residual_block(X, names):
    """The named residual measures in paper windows and as pooled statistics at each scale."""
    T = X.shape[1]
    parts = []
    for n0 in (8, 16):
        nw, _ = SP._safe_cheby_params(T, n0, DEGREE)
        L = min(max(DEGREE + 2, T // nw), T)
        step = max(1, (T - L) // max(1, nw - 1))
        starts = list(range(0, T - L + 1, step))[:nw]
        parts.append(residual_windows(X, L, starts, names).reshape(len(X), -1))
    stats = tuple(st for st in SP.POOL_STATS if st != "ppv")
    for L in window_lengths(T):
        stride = max(1, L // OVERLAP)
        starts = np.arange(0, T - L + 1, stride)
        if starts[-1] != T - L:
            starts = np.append(starts, T - L)
        parts.append(SP._pool(residual_windows(X, L, starts, names), stats))
    return np.hstack(parts)


RICH_STATS = SP.POOL_STATS + ("q5", "q10", "q90", "q95")


def transform(X, blocks, kparams):
    X = np.asarray(X, dtype=float)
    parts = [SP.stct_fixed(X)] if "P" in blocks else []
    if "D" in blocks:
        parts.append(SP.stct_pooled(np.diff(X, axis=1), FRACTIONS))
    if "Pd" in blocks:
        parts.append(SP.stct_fixed(np.diff(X, axis=1)))
    if "Pc" in blocks:
        parts.append(SP.stct_fixed(cumulative(X)))
    if "Dc" in blocks:
        parts.append(SP.stct_pooled(cumulative(X), FRACTIONS))
    for name in ("WG", "WH", "WC", "WS", "WE"):
        if name in blocks:
            parts.append(basis_block(X, name))
    names = [n for n in ("N1", "N4", "Nm", "Nh", "Ns") if n in blocks]
    if names:
        parts.append(residual_block(X, names))
    if "Sr" in blocks:
        parts.append(SP.stct_pooled(SP.autocorrelation(X), FRACTIONS))
    if "Pa" in blocks:
        parts.append(SP.stct_fixed(np.abs(np.diff(X, axis=1))))
    if "Da" in blocks:
        parts.append(SP.stct_pooled(np.abs(np.diff(X, axis=1)), FRACTIONS))
    if "Ps" in blocks:
        parts.append(SP.stct_fixed(np.sort(X, axis=1)))
    if "Pr" in blocks:
        parts.append(SP.stct_fixed(autocorrelation(X)))
    if "P2" in blocks:
        parts.append(SP.stct_fixed(np.diff(X, n=2, axis=1)))
    if "D2" in blocks:
        parts.append(SP.stct_pooled(np.diff(X, n=2, axis=1), FRACTIONS))
    if "G" in blocks:
        parts.append(signature_pyramid(X))
    if "W" in blocks:
        parts.append(dwt_summary(X))
    if "F" in blocks:
        parts.append(stft_summary(X))
    if "Wr" in blocks:
        parts.append(bench.wavelet_features(X))
    if "Fr" in blocks:
        parts.append(bench.stft_features(X))
    sliding = blocks & {"S", "SY", "C", "K", "KY", "K32", "K64", "KT", "H", "SQ", "Sd"}
    if sliding:
        rows = []
        for s in range(0, len(X), BLOCK):
            Xb, feats = X[s : s + BLOCK], []
            for i, L in enumerate(window_lengths(X.shape[1])):
                Cx = SP._windowed_coefficients(Xb, L, DEGREE, max(1, L // OVERLAP))
                if blocks & {"S", "SY"}:
                    feats.append(pooled(Cx))
                if "SY" in blocks:
                    feats += [pooled(h) for h in halves(Cx)]
                if "SQ" in blocks:
                    F = SP._pool(Cx, RICH_STATS)
                    feats.append(np.delete(F, RICH_STATS.index("ppv") * Cx.shape[1] + Cx.shape[1] - 1, axis=1))
                if "Sd" in blocks:
                    for deg in (2, 6):
                        if L >= deg + 2:
                            feats.append(pooled(SP._windowed_coefficients(Xb, L, deg, max(1, L // OVERLAP))))
                if "C" in blocks:
                    feats.append(SP._cross_correlation(Cx))
                if "H" in blocks:
                    feats.append(shape_histogram(Cx))
                if kparams:
                    mu, sd, D, B = kparams[i]
                    P = D @ ((Cx - mu[None, :, None]) / sd[None, :, None])
                    feats.append(kernel_summary(P, B))
                    if "KY" in blocks:
                        feats += [kernel_summary(h, B) for h in halves(P)]
            rows.append(np.hstack(feats))
        parts.append(np.vstack(rows))
    return np.nan_to_num(np.hstack(parts))


def parse(config):
    name, _, learner = config.partition(":")
    return set(name.split("+")), learner or "et"


def features(X, config, seed):
    if "|" in config:
        return np.hstack([features(X, part, seed) for part in config.split("|")])
    blocks, _ = parse(config)
    n_dir = (64 if X.shape[1] >= 300 else 32) if "K64" in blocks else (32 if "K32" in blocks else 8)
    if "KT" in blocks:
        n_dir = int(np.clip(X.shape[1] // 8, 1, 64))
    kp = kernel_fit(X, n_dir, seed) if blocks & {"K", "KY", "K32", "K64", "KT"} else None
    return transform(X, blocks, kp)


def soft(decision):
    """Class probabilities from ridge decision values."""
    d = decision if decision.ndim > 1 else np.column_stack([-decision, decision])
    e = np.exp(d - d.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def vote_score(task, X, y, config, seed):
    """Average ExtraTrees on the left blocks with ridge on the right blocks, same folds."""
    left, right = config.split("|")
    FL, FR = features(X, left, seed), features(X, right, seed)
    cls = task == "classification"
    alphas = np.logspace(-3, 3, 10)
    if cls:
        cv = (StratifiedKFold if np.unique(y, return_counts=True)[1].max() >= 5 else KFold)(5, shuffle=True, random_state=seed)
        et = ExtraTreesClassifier(n_estimators=200, max_features=0.1, random_state=seed, n_jobs=-1)
        ridge = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=alphas))
        labels = np.unique(y)
        probs = np.zeros((len(y), len(labels)))
        for tr, te in cv.split(FL, y):
            et.fit(FL[tr], y[tr])
            pe = np.zeros((len(te), len(labels)))
            pe[:, np.searchsorted(labels, et.classes_)] = et.predict_proba(FL[te])
            try:
                ridge.fit(FR[tr], y[tr])
                pr = np.zeros_like(pe)
                pr[:, np.searchsorted(labels, ridge.classes_)] = soft(ridge.decision_function(FR[te]))
            except np.linalg.LinAlgError:
                pr = pe
            probs[te] = (pe + pr) / 2
        score = float((labels[probs.argmax(axis=1)] == y).mean())
    else:
        cv = KFold(5, shuffle=True, random_state=seed)
        et = ExtraTreesRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
        ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
        a = cross_val_predict(et, FL, y, cv=cv)
        try:
            b = cross_val_predict(ridge, FR, y, cv=cv)
        except np.linalg.LinAlgError:
            b = a
        score = float(np.sqrt((((a + b) / 2 - y) ** 2).mean()))
    return score, FL.shape[1] + FR.shape[1]


def cv_score(task, X, y, config, seed):
    if "|" in config:
        return vote_score(task, X, y, config, seed)
    F = features(X, config, seed)
    _, learner = parse(config)
    cls = task == "classification"
    cv = ((StratifiedKFold if np.unique(y, return_counts=True)[1].max() >= 5 else KFold) if cls else KFold)(5, shuffle=True, random_state=seed)
    if learner == "ridge":
        alphas = np.logspace(-3, 3, 10)
        est = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=alphas) if cls else RidgeCV(alphas=alphas))
    elif cls:
        est = ExtraTreesClassifier(n_estimators=200, max_features=0.1, random_state=seed, n_jobs=-1)
    else:
        est = ExtraTreesRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
    try:
        pred = cross_val_predict(est, F, y, cv=cv)
    except np.linalg.LinAlgError:  # the SVD in the cross-validated ridge can fail to converge
        print("ridge SVD failed, using a fixed-penalty iterative ridge for this dataset", flush=True)
        fallback = RidgeClassifier(solver="lsqr") if cls else Ridge(solver="lsqr")
        pred = cross_val_predict(make_pipeline(StandardScaler(), fallback), F, y, cv=cv)
    return (float((pred == y).mean()) if cls else float(np.sqrt(((pred - y) ** 2).mean()))), F.shape[1]


def all_configs():
    out = []
    for p, s, k, c in itertools.product(["", "P"], ["", "S", "SY"], ["", "K", "KY"], ["", "C"]):
        parts = [x for x in (p, s, k, c) if x]
        if parts:
            out.append("+".join(parts))
    return out + ["K:ridge", "KY:ridge", "K32:ridge", "P+K:ridge", "P+KY:ridge"]


V = "P+S+Ps+Pr"
BIG_CONFIGS = [V + "+Sr", "P+SQ+Ps+Pr", V + "+Sd", V + "+K32", V + "+KY", V + "+K32+C", V + "+SY",
               V + "|K32", V + "|K64", V + "+C|K64", "P+SQ+Ps+Pr|K64", "P+S|K64"]
BASIS_CONFIGS = [V + "+WG", V + "+WH", V + "+WC", V + "+WS", V + "+WE", V + "+WS+WE", V + "+WG+WC"]
NORM_CONFIGS = [V + "+N1", V + "+N4", V + "+Nm", V + "+Nh", V + "+Ns", V + "+N1+Nm", V + "+Nh+N1+N4+Nm"]
NEW_BLOCK_CONFIGS = ["D", "H", "G", "W", "F", "Wr", "Fr", "P+S+D", "P+S+H", "P+S+G", "P+S+W", "P+S+F", "P+S+Wr", "P+S+Fr",
                     "P+S+H+D", "P+S+H+G", "P+S+W+F", "P+S+H+W+F", "P+S+H+G+W+F",
                     "Pd", "Pc", "Dc", "P+S+Pd", "P+S+Pc", "P+S+Dc", "P+S+Pd+Pc", "P+S+D+Dc", "P+S+D+Pd", "P+S+Dc+Pc",
                     "P+S+D+Pd+Dc+Pc",
                     # double views, without the raw series or without one block
                     "Pd+D", "Pc+Dc", "Pd+D+Pc+Dc", "S+D", "S+Dc", "P+D", "P+Dc", "P+Pd", "P+Pc",
                     # other views on top of paper + pooled
                     "Ps", "Pr", "Pa", "P+S+Ps", "P+S+Pr", "P+S+Pa", "P+S+P2", "P+S+Da", "P+S+D2", "P+S+Ps+Pr", "P+S+Ps+Pr+Pd"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classification", default=None, help="comma-separated override, smoke tests only")
    ap.add_argument("--regression", default=None, help="comma-separated override, smoke tests only")
    ap.add_argument("--configs", default=None)
    ap.add_argument("--seeds", default=str(bench.RNG))
    ap.add_argument("--prefix", default="results/ablate")
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
    path = f"{a.prefix}_scores.csv"
    known = set()
    if os.path.exists(path):
        p = pd.read_csv(path, keep_default_na=False)
        known = set(zip(p.config, p.seed, p.task, p.dataset))
    named = {"new": NEW_BLOCK_CONFIGS, "big": BIG_CONFIGS, "basis": BASIS_CONFIGS, "norm": NORM_CONFIGS}
    configs = named[a.configs] if a.configs in named else (a.configs.split(",") if a.configs else all_configs())
    # cheapest first, so partial results are already useful
    probe = np.random.default_rng(0).standard_normal((20, 1024)).cumsum(axis=1)
    sizes = {c: features(probe, c, 0).shape[1] for c in configs}
    print(json.dumps(sizes), flush=True)
    for config in sorted(configs, key=sizes.get):
        for seed in [int(s) for s in a.seeds.split(",")]:
            for task, d in data.items():
                for n, (X, y) in d.items():
                    if (config, seed, task, n) in known:
                        continue
                    try:
                        score, feats = with_retry(cv_score, task, X, y, config, seed)
                    except Exception as e:  # one failure must not stop the whole queue
                        print(f"[failed] {config} seed {seed} {task} {n}: {str(e)[:120]}", flush=True)
                        continue
                    row = dict(config=config, seed=seed, task=task, dataset=n, score=score, features=feats)
                    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
        print(f"scored {config} ({sizes[config]} features at length 1024)", flush=True)


if __name__ == "__main__":
    main()
