"""Compiled feature extraction for the averaged STCT+ recipe.

Produces the same values as stct_plus.stct_views and the kernel block of
ablate.py, up to rounding.  Sliding fits are computed once per scale and shared
by the pooled statistics and the kernels.
"""
import numpy as np
from numba import njit
from numpy.polynomial import chebyshev as C

import stct_plus as SP

FRACTIONS = SP.TUNED_FRACTIONS
DEGREE, OVERLAP, BLOCK = 4, 8, 256
QS = np.linspace(0, 1, 7)[1:-1]


def window_lengths(T):
    out = []
    for f in FRACTIONS:
        L = max(DEGREE + 2, min(int(round(T * f)), T))
        if L not in out:
            out.append(L)
    return out


def n_kernels(T):
    """One kernel per 8 samples of series length, 1 to 64 per scale."""
    return int(np.clip(T // 8, 1, 64))


def winner_groups(T):
    """Winner groups of 8 directions per scale and input: one per 72 samples of series length, 1 to 8."""
    return int(np.clip(T // 72, 1, 8))


def _setup(T, L):
    Q, R = np.linalg.qr(C.chebvander(np.linspace(-1, 1, L), DEGREE))
    stride = max(1, L // OVERLAP)
    starts = np.arange(0, T - L + 1, stride)
    if starts[-1] != T - L:  # always cover the end of the series
        starts = np.append(starts, T - L)
    return starts, Q, np.linalg.inv(R).T


@njit(cache=True)
def _sliding(X, starts, Q, RinvT, dilation=1):
    """Coefficients and residual RMS at the given starts: (n, degree + 2, windows).

    A window is Q.shape[0] samples spaced ``dilation`` apart.
    """
    n, nw, L, d = X.shape[0], len(starts), Q.shape[0], Q.shape[1]
    out = np.empty((n, d + 1, nw))
    proj = np.empty(d)
    for i in range(n):
        x = X[i]
        for w in range(nw):
            s = starts[w]
            proj[:] = 0.0
            energy = 0.0
            for t in range(L):
                v = x[s + t * dilation]
                energy += v * v
                for j in range(d):
                    proj[j] += v * Q[t, j]
            for j in range(d):
                energy -= proj[j] * proj[j]
            for k in range(d):
                c = 0.0
                for j in range(d):
                    c += proj[j] * RinvT[j, k]
                out[i, k, w] = c
            out[i, d, w] = np.sqrt(max(energy, 0.0) / L)
    return out


@njit(cache=True)
def _pooled(Cx):
    """Ten statistics per channel, without the proportion of positive residuals."""
    n, ch, t = Cx.shape
    F = np.empty((n, 10 * ch))
    span = max(t - 1, 1)
    pos = np.array([0.25, 0.5, 0.75]) * (t - 1)
    for i in range(n):
        for c in range(ch):
            a = Cx[i, c]
            total, positive, lo, hi, argmin, argmax = 0.0, 0, a[0], a[0], 0, 0
            for w in range(t):
                v = a[w]
                total += v
                positive += v > 0
                if v < lo:
                    lo, argmin = v, w
                if v > hi:
                    hi, argmax = v, w
            mean = total / t
            var = 0.0
            for w in range(t):
                var += (a[w] - mean) ** 2
            b = np.sort(a)
            F[i, c] = mean
            F[i, ch + c] = np.sqrt(var / t)
            F[i, 2 * ch + c] = lo
            F[i, 3 * ch + c] = hi
            for q in range(3):
                k = int(pos[q])
                g = pos[q] - k
                F[i, (4 + q) * ch + c] = b[k] + g * (b[k + 1] - b[k]) if g > 0 else b[k]
            F[i, 7 * ch + c] = positive / t
            F[i, 8 * ch + c] = argmax / span
            F[i, 9 * ch + c] = argmin / span
    return F


@njit(cache=True)
def _kernels(Cx, mu, sd, D, B):
    """Proportion above each threshold and the maximum, per random weighting."""
    n, ch, t = Cx.shape
    m, q = D.shape[0], B.shape[0]
    F = np.zeros((n, m * (q + 1)))
    Dc = D.T.copy()
    Bj = B.copy()
    p = np.empty(m)
    counts = np.empty((q, m))
    top = np.empty(m)
    for i in range(n):
        counts[:] = 0.0
        top[:] = -np.inf
        for w in range(t):
            p[:] = 0.0
            for c in range(ch):
                z = (Cx[i, c, w] - mu[c]) / sd[c]
                dc = Dc[c]
                for k in range(m):
                    p[k] += dc[k] * z
            for j in range(q):
                row, b = counts[j], Bj[j]
                for k in range(m):
                    row[k] += p[k] > b[k]
            for k in range(m):
                top[k] = max(top[k], p[k])
        for k in range(m):
            for j in range(q):
                F[i, k * q + j] = counts[j, k] / t
            F[i, m * q + k] = top[k]
    return F


@njit(cache=True)
def _compete(Z, D, k):
    """Share of windows each direction wins within its group of k, and its mean winning score."""
    n, dim, t = Z.shape
    m = D.shape[0]
    g = m // k
    F = np.zeros((n, 2 * m))
    Dc = D.T.copy()
    p = np.empty(m)
    for i in range(n):
        for w in range(t):
            p[:] = 0.0
            for c in range(dim):
                z, dc = Z[i, c, w], Dc[c]
                for j in range(m):
                    p[j] += dc[j] * z
            for a in range(g):
                best, top = a * k, p[a * k]
                for j in range(a * k + 1, (a + 1) * k):
                    if p[j] > top:
                        best, top = j, p[j]
                F[i, best] += 1.0
                F[i, m + best] += top
    return F / t


def compete_inputs(Cx, mu, sd, shape):
    """Standardized coefficients, or coefficients 1 to degree scaled to unit length per window."""
    if not shape:
        return (Cx - mu[None, :, None]) / sd[None, :, None]
    Z = Cx[:, 1 : DEGREE + 1] / sd[None, 1 : DEGREE + 1, None]
    return Z / (np.sqrt((Z**2).sum(axis=1, keepdims=True)) + 1e-12)


def compete_fit(X, seed, groups, k=8, max_series=32):
    """Scaling and unit directions per scale, for raw and shape inputs, no labels."""
    rng = np.random.default_rng(seed)
    S = X[rng.choice(len(X), max_series, replace=False)] if len(X) > max_series else X
    params = []
    for L in window_lengths(X.shape[1]):
        flat = _sliding(S, *_setup(X.shape[1], L)).transpose(1, 0, 2).reshape(DEGREE + 2, -1)
        D = {}
        for shape, dim in ((False, DEGREE + 2), (True, DEGREE)):
            D[shape] = rng.standard_normal((groups * k, dim))
            D[shape] /= np.linalg.norm(D[shape], axis=1, keepdims=True)
        params.append((flat.mean(axis=1), flat.std(axis=1) + 1e-12, D))
    return params


def compete_features(X, params, inputs=(False, True), k=8, soft=True):
    """Square-root win shares, and mean winning scores when soft, per scale and input."""
    X = np.asarray(X, dtype=float)
    T, out = X.shape[1], []
    for s in range(0, len(X), BLOCK):
        Xb, row = X[s : s + BLOCK], []
        for i, L in enumerate(window_lengths(T)):
            Cx = _sliding(Xb, *_setup(T, L))
            mu, sd, D = params[i]
            for shape in inputs:
                F = _compete(compete_inputs(Cx, mu, sd, shape), D[shape], k)
                m = F.shape[1] // 2
                row.append(np.hstack([np.sqrt(F[:, :m]), F[:, m:]]) if soft else np.sqrt(F[:, :m]))
        out.append(np.hstack(row))
    return np.vstack(out)


TAPS = 9


def dilations(T, taps=TAPS):
    """Dilations 1, 2, 4, ... while taps samples that far apart fit in the series."""
    out, d = [1], 2
    while (taps - 1) * d <= T - 1:
        out.append(d)
        d *= 2
    return out


def dilated(X, dilation, stride, taps=TAPS, degree=DEGREE):
    """Coefficients and residual RMS of fits to taps samples spaced dilation apart, every stride samples."""
    Q, R = np.linalg.qr(C.chebvander(np.linspace(-1, 1, taps), degree))
    starts = np.arange(0, max(1, X.shape[1] - (taps - 1) * dilation), stride)
    return _sliding(X, starts, Q, np.linalg.inv(R).T, dilation)


def dilated_fit(X, n_dir, seed, max_series=32):
    """Level-free weightings and thresholds per dilation, from at most max_series series, no labels.

    Directions give no weight to the level coefficient c0, so a kernel responds to
    the shape of the 9 samples and not to their level.
    """
    rng = np.random.default_rng(seed)
    S = X[rng.choice(len(X), max_series, replace=False)] if len(X) > max_series else X
    params = []
    for d in dilations(X.shape[1]):
        Cx = dilated(S, d, 1)
        flat = Cx.transpose(1, 0, 2).reshape(Cx.shape[1], -1)
        mu, sd = flat.mean(axis=1), flat.std(axis=1) + 1e-12
        D = rng.standard_normal((n_dir, Cx.shape[1]))
        D[:, 0] = 0.0
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        params.append((mu, sd, D, np.quantile(D @ ((flat - mu[:, None]) / sd[:, None]), QS, axis=1)))
    return params


def dilated_features(X, params):
    """Threshold kernels on dilated fits at every sample."""
    X = np.asarray(X, dtype=float)
    T, out = X.shape[1], []
    for s in range(0, len(X), BLOCK):
        Xb = X[s : s + BLOCK]
        out.append(np.hstack([_kernels(dilated(Xb, d, 1), *params[i]) for i, d in enumerate(dilations(T))]))
    return np.vstack(out)


def word_fit(X, seed, max_series=32, coefs=(1, 2, 3), bins=4):
    """Equal-frequency bin edges per scale for the chosen coefficients, no labels."""
    rng = np.random.default_rng(seed)
    S = X[rng.choice(len(X), max_series, replace=False)] if len(X) > max_series else X
    qs = np.arange(1, bins) / bins
    params = []
    for L in window_lengths(X.shape[1]):
        Cx = _sliding(S, *_setup(X.shape[1], L))[:, list(coefs)]
        params.append(np.quantile(Cx.transpose(1, 0, 2).reshape(len(coefs), -1), qs, axis=1).T)
    return params, coefs, bins


def words(Cx, edges, coefs, bins):
    """Square-root share of windows in each word of quantized coefficients."""
    n, _, t = Cx.shape
    idx = np.zeros((n, t), dtype=np.int64)
    for j, c in enumerate(coefs):
        idx = idx * bins + np.searchsorted(edges[j], Cx[:, c], side="right")
    m = bins ** len(coefs)
    counts = np.bincount((idx + m * np.arange(n)[:, None]).ravel(), minlength=n * m).reshape(n, m)
    return np.sqrt(counts / t)


def word_features(X, wparams):
    X = np.asarray(X, dtype=float)
    T, (edges, coefs, bins), out = X.shape[1], wparams, []
    for s in range(0, len(X), BLOCK):
        Xb = X[s : s + BLOCK]
        out.append(np.hstack([words(_sliding(Xb, *_setup(T, L)), edges[i], coefs, bins)
                              for i, L in enumerate(window_lengths(T))]))
    return np.vstack(out)


def in_context(Cx, L, context):
    """Stack each window's channels with those of the next context - 1 windows, one window length apart."""
    if context == 1:
        return Cx
    n, ch, t = Cx.shape
    step = min(L // max(1, L // OVERLAP), (t - 1) // (context - 1))
    t2 = t - step * (context - 1)
    return np.concatenate([Cx[:, :, i * step : i * step + t2] for i in range(context)], axis=1)


def context_fit(X, n_dir, seed, context, max_series=32):
    """Weightings and thresholds per scale on windows stacked in context, no labels."""
    rng = np.random.default_rng(seed)
    S = X[rng.choice(len(X), max_series, replace=False)] if len(X) > max_series else X
    params = []
    for L in window_lengths(X.shape[1]):
        Cx = in_context(_sliding(S, *_setup(X.shape[1], L)), L, context)
        flat = Cx.transpose(1, 0, 2).reshape(Cx.shape[1], -1)
        mu, sd = flat.mean(axis=1), flat.std(axis=1) + 1e-12
        D = rng.standard_normal((n_dir, Cx.shape[1]))
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        params.append((mu, sd, D, np.quantile(D @ ((flat - mu[:, None]) / sd[:, None]), QS, axis=1)))
    return params


def context_features(X, params, context):
    X = np.asarray(X, dtype=float)
    T, out = X.shape[1], []
    for s in range(0, len(X), BLOCK):
        Xb, row = X[s : s + BLOCK], []
        for i, L in enumerate(window_lengths(T)):
            row.append(_kernels(in_context(_sliding(Xb, *_setup(T, L)), L, context), *params[i]))
        out.append(np.hstack(row))
    return np.vstack(out)


def thin(Cx, max_windows):
    """Keep evenly spaced windows, at most about max_windows of them."""
    return Cx if max_windows is None else Cx[:, :, :: int(np.ceil(Cx.shape[2] / max_windows))]


def kernel_fit(X, n_dir, seed, max_series=512, max_windows=None):
    """Unit weightings and quantile thresholds per scale, from at most max_series series, no labels."""
    rng = np.random.default_rng(seed)
    S = X[rng.choice(len(X), max_series, replace=False)] if len(X) > max_series else X
    params = []
    for L in window_lengths(X.shape[1]):
        Cx = thin(_sliding(S, *_setup(X.shape[1], L)), max_windows)
        ch = Cx.shape[1]
        flat = Cx.transpose(1, 0, 2).reshape(ch, -1)
        mu, sd = flat.mean(axis=1), flat.std(axis=1) + 1e-12
        D = rng.standard_normal((n_dir, ch))
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        P = D @ ((flat - mu[:, None]) / sd[:, None])
        params.append((mu, sd, D, np.quantile(P, QS, axis=1)))
    return params


def features(X, kparams=None, max_windows=None):
    """stct_views features, and kernel features when parameters are given (else None)."""
    X = np.asarray(X, dtype=float)
    T = X.shape[1]
    ppv_residual = 7 * (DEGREE + 2) + DEGREE + 1
    pooled, kernels = [], []
    for s in range(0, len(X), BLOCK):
        Xb, row_p, row_k = X[s : s + BLOCK], [], []
        for i, L in enumerate(window_lengths(T)):
            Cx = _sliding(Xb, *_setup(T, L))
            row_p.append(np.delete(_pooled(Cx), ppv_residual, axis=1))
            if kparams is not None:
                row_k.append(_kernels(thin(Cx, max_windows), *kparams[i]))
        pooled.append(np.hstack(row_p))
        if kparams is not None:
            kernels.append(np.hstack(row_k))
    return _views(X, np.vstack(pooled)), np.vstack(kernels) if kparams is not None else None


def trio_features(X, cparams=None, wparams=None, context=2, distinct=False):
    """stct_views, context kernels and winner counts from one sliding fit per scale.

    Same values as features, context_features and compete_features; a block is
    None when its parameters are not given.  ``distinct`` computes paper windows
    that collapse to the same count on short series only once.
    """
    X = np.asarray(X, dtype=float)
    T = X.shape[1]
    ppv_residual = 7 * (DEGREE + 2) + DEGREE + 1
    blocks = {"P": [], "C": [], "W": []}
    for s in range(0, len(X), BLOCK):
        Xb, row = X[s : s + BLOCK], {"P": [], "C": [], "W": []}
        for i, L in enumerate(window_lengths(T)):
            Cx = _sliding(Xb, *_setup(T, L))
            row["P"].append(np.delete(_pooled(Cx), ppv_residual, axis=1))
            if cparams is not None:
                row["C"].append(_kernels(in_context(Cx, L, context), *cparams[i]))
            if wparams is not None:
                mu, sd, D = wparams[i]
                for shape in (False, True):
                    F = _compete(compete_inputs(Cx, mu, sd, shape), D[shape], 8)
                    m = F.shape[1] // 2
                    row["W"].append(np.hstack([np.sqrt(F[:, :m]), F[:, m:]]))
        for k, v in row.items():
            if v:
                blocks[k].append(np.hstack(v))
    views = _views(X, np.vstack(blocks["P"]), distinct)
    return (views, np.vstack(blocks["C"]) if cparams is not None else None,
            np.vstack(blocks["W"]) if wparams is not None else None)


def _paper(X, distinct):
    """Paper features at 8 and 16 windows; with distinct, once when both counts give the same windows."""
    same = SP._safe_cheby_params(X.shape[1], 8, DEGREE) == SP._safe_cheby_params(X.shape[1], 16, DEGREE)
    return SP.stct_fixed(X, (8,) if distinct and same else (8, 16))


def _views(X, pooled, distinct=False):
    return np.hstack([_paper(X, distinct), pooled, _paper(np.sort(X, axis=1), distinct),
                      _paper(SP.autocorrelation(X), distinct)])
