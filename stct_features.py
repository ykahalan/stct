"""STCT+ features: every feature is a statistic of degree-4 Chebyshev least-squares fits.

* ``stct_fixed`` is the original STCT: each series is cut into 8 and 16 windows,
  and every window is described by the coefficients of a degree-4 Chebyshev fit
  plus the RMS fit residual.
* The forest views (``views``) are STCT of the series, of its sorted values and
  of its autocorrelation, plus ten statistics of sliding fits pooled over every
  position at up to six window lengths (1/128 to 1/4 of the series).
* The kernel blocks project each window's fit descriptor (coefficients and
  residual) on random directions: threshold kernels (``kernel_fit``), context
  kernels on pairs of neighbouring windows (``context_fit``), winner counts in
  groups of eight competing directions (``compete_fit``) and dilated kernels on
  fits to 9 samples spaced 1, 2, 4, ... apart (``dilated_fit``).  Their
  directions and thresholds come from at most 32 training series, without labels.

Sliding fits are computed once per window length and shared by the pooled
statistics and the kernels; the inner loops are compiled with Numba.
"""
import numpy as np
from numba import njit
from numpy.polynomial import chebyshev as C

DEGREE = 4  # Chebyshev fit degree
OVERLAP = 8  # sliding windows move by 1/8 of their length
BLOCK = 256  # series per extraction block
FRACTIONS = (1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4)  # sliding window lengths, as fractions of the series
QS = np.linspace(0, 1, 7)[1:-1]  # kernel thresholds: quantiles 1/6 to 5/6
TAPS = 9  # samples per dilated fit
MAX_SERIES = 32  # training series used to set kernel scalings and thresholds


# --------------------------------------------------------------------------
# STCT on fixed windows
# --------------------------------------------------------------------------


def _safe_cheby_params(n, n_windows=8, degree=DEGREE):
    degree = max(1, min(degree, n - 2))
    if n >= degree + 2:
        n_windows = max(1, min(n_windows, n // (degree + 2)))
    else:
        n_windows = 1
        degree = max(1, n - 2)
    return n_windows, degree


def _stct_single_scale(X, n_windows, degree):
    n_samples, n = X.shape
    nw, deg = _safe_cheby_params(n, n_windows, degree)
    win_len = min(max(deg + 2, n // nw), n)
    step = max(1, (n - win_len) // max(1, nw - 1))
    starts = np.array(list(range(0, n - win_len + 1, step))[:nw])
    Phi = C.chebvander(np.linspace(-1, 1, win_len), deg)
    pinv = np.linalg.pinv(Phi)
    windows = X[:, starts[:, None] + np.arange(win_len)[None, :]]
    coefs = windows @ pinv.T
    resid = np.sqrt(((windows - coefs @ Phi.T) ** 2).mean(axis=-1))[..., None]
    block = np.concatenate([coefs, resid], axis=-1)
    if len(starts) < nw:
        block = np.concatenate([block, np.zeros((n_samples, nw - len(starts), block.shape[-1]))], axis=1)
    return block.reshape(n_samples, nw * block.shape[-1])


def stct_fixed(X, window_counts=(8, 16), degree=DEGREE):
    """STCT: fit coefficients and residual RMS in 8 and 16 windows."""
    X = np.asarray(X, dtype=float)
    return np.concatenate([_stct_single_scale(X, nw, degree) for nw in window_counts], axis=1)


def autocorrelation(X):
    """Normalised autocorrelation of each series, lags 0 to T // 2 (at least 8 lags)."""
    T = X.shape[1]
    Z = X - X.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(Z, n=2 * T, axis=1)) ** 2
    acf = np.fft.irfft(power, axis=1)[:, : max(T // 2 + 1, 8)]
    return acf / (acf[:, :1] + 1e-12)


# --------------------------------------------------------------------------
# sliding fits
# --------------------------------------------------------------------------


def window_lengths(T):
    """Distinct sliding window lengths for a series of length T, at least DEGREE + 2 samples."""
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
    """Ten statistics of each channel over all windows: mean, std, min, max, quartiles,
    proportion positive, and the relative positions of the maximum and minimum."""
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


# the proportion of positive residuals is always 1 and is dropped
_PPV_RESIDUAL = 7 * (DEGREE + 2) + DEGREE + 1


def views(X, pooled):
    """Forest input: STCT of the series, the pooled statistics, and STCT of the sorted series and its autocorrelation."""
    return np.hstack([stct_fixed(X), pooled, stct_fixed(np.sort(X, axis=1)), stct_fixed(autocorrelation(X))])


# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------


@njit(cache=True)
def _kernels(Cx, mu, sd, D, B):
    """Proportion of windows above each threshold and the maximum, per random direction."""
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


def _subset(X, rng, max_series):
    return X[rng.choice(len(X), max_series, replace=False)] if len(X) > max_series else X


def _directions(rng, n_dir, dim):
    D = rng.standard_normal((n_dir, dim))
    return D / np.linalg.norm(D, axis=1, keepdims=True)


def _thresholds(D, flat, mu, sd):
    return np.quantile(D @ ((flat - mu[:, None]) / sd[:, None]), QS, axis=1)


def kernel_fit(X, n_dir, seed, max_series=MAX_SERIES):
    """Threshold kernels: unit directions and quantile thresholds per window length, no labels."""
    rng = np.random.default_rng(seed)
    S = _subset(X, rng, max_series)
    params = []
    for L in window_lengths(X.shape[1]):
        Cx = _sliding(S, *_setup(X.shape[1], L))
        flat = Cx.transpose(1, 0, 2).reshape(Cx.shape[1], -1)
        mu, sd = flat.mean(axis=1), flat.std(axis=1) + 1e-12
        D = _directions(rng, n_dir, Cx.shape[1])
        params.append((mu, sd, D, _thresholds(D, flat, mu, sd)))
    return params


def in_context(Cx, L, context):
    """Stack each window's channels with those of the next context - 1 windows, one window length apart."""
    if context == 1:
        return Cx
    n, ch, t = Cx.shape
    step = min(L // max(1, L // OVERLAP), (t - 1) // (context - 1))
    t2 = t - step * (context - 1)
    return np.concatenate([Cx[:, :, i * step : i * step + t2] for i in range(context)], axis=1)


def context_fit(X, n_dir, seed, context=2, max_series=MAX_SERIES):
    """Context kernels: directions and thresholds on pairs of neighbouring windows, no labels."""
    rng = np.random.default_rng(seed)
    S = _subset(X, rng, max_series)
    params = []
    for L in window_lengths(X.shape[1]):
        Cx = in_context(_sliding(S, *_setup(X.shape[1], L)), L, context)
        flat = Cx.transpose(1, 0, 2).reshape(Cx.shape[1], -1)
        mu, sd = flat.mean(axis=1), flat.std(axis=1) + 1e-12
        D = _directions(rng, n_dir, Cx.shape[1])
        params.append((mu, sd, D, _thresholds(D, flat, mu, sd)))
    return params


def compete_inputs(Cx, mu, sd, shape):
    """Standardized descriptors, or coefficients 1 to degree scaled to unit length per window (shape)."""
    if not shape:
        return (Cx - mu[None, :, None]) / sd[None, :, None]
    Z = Cx[:, 1 : DEGREE + 1] / sd[None, 1 : DEGREE + 1, None]
    return Z / (np.sqrt((Z**2).sum(axis=1, keepdims=True)) + 1e-12)


def compete_fit(X, seed, groups, k=8, max_series=MAX_SERIES):
    """Winner counts: scaling and unit directions per window length, for raw and shape inputs, no labels."""
    rng = np.random.default_rng(seed)
    S = _subset(X, rng, max_series)
    params = []
    for L in window_lengths(X.shape[1]):
        flat = _sliding(S, *_setup(X.shape[1], L)).transpose(1, 0, 2).reshape(DEGREE + 2, -1)
        D = {shape: _directions(rng, groups * k, dim) for shape, dim in ((False, DEGREE + 2), (True, DEGREE))}
        params.append((flat.mean(axis=1), flat.std(axis=1) + 1e-12, D))
    return params


def dilations(T, taps=TAPS):
    """Dilations 1, 2, 4, ... while taps samples that far apart fit in the series."""
    out, d = [1], 2
    while (taps - 1) * d <= T - 1:
        out.append(d)
        d *= 2
    return out


def dilated(X, dilation, stride=1, taps=TAPS):
    """Coefficients and residual RMS of fits to taps samples spaced dilation apart, every stride samples."""
    Q, R = np.linalg.qr(C.chebvander(np.linspace(-1, 1, taps), DEGREE))
    starts = np.arange(0, max(1, X.shape[1] - (taps - 1) * dilation), stride)
    return _sliding(X, starts, Q, np.linalg.inv(R).T, dilation)


def dilated_fit(X, n_dir, seed, max_series=MAX_SERIES):
    """Dilated kernels: level-free directions and thresholds per dilation, no labels.

    Directions give no weight to the level coefficient c0, so a kernel responds to
    the shape of the 9 samples and not to their level.
    """
    rng = np.random.default_rng(seed)
    S = _subset(X, rng, max_series)
    params = []
    for d in dilations(X.shape[1]):
        Cx = dilated(S, d)
        flat = Cx.transpose(1, 0, 2).reshape(Cx.shape[1], -1)
        mu, sd = flat.mean(axis=1), flat.std(axis=1) + 1e-12
        D = rng.standard_normal((n_dir, Cx.shape[1]))
        D[:, 0] = 0.0
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        params.append((mu, sd, D, _thresholds(D, flat, mu, sd)))
    return params


def dilated_features(X, params):
    """Threshold kernels on dilated fits at every sample."""
    X = np.asarray(X, dtype=float)
    T, out = X.shape[1], []
    for s in range(0, len(X), BLOCK):
        Xb = X[s : s + BLOCK]
        out.append(np.hstack([_kernels(dilated(Xb, d), *params[i]) for i, d in enumerate(dilations(T))]))
    return np.vstack(out)


# --------------------------------------------------------------------------
# feature sets of the two models
# --------------------------------------------------------------------------


def regressor_features(X, kparams=None):
    """Forest views, and threshold kernels when their parameters are given (else None)."""
    X = np.asarray(X, dtype=float)
    T = X.shape[1]
    pooled, kernels = [], []
    for s in range(0, len(X), BLOCK):
        Xb, row_p, row_k = X[s : s + BLOCK], [], []
        for i, L in enumerate(window_lengths(T)):
            Cx = _sliding(Xb, *_setup(T, L))
            row_p.append(np.delete(_pooled(Cx), _PPV_RESIDUAL, axis=1))
            if kparams is not None:
                row_k.append(_kernels(Cx, *kparams[i]))
        pooled.append(np.hstack(row_p))
        if kparams is not None:
            kernels.append(np.hstack(row_k))
    return views(X, np.vstack(pooled)), np.vstack(kernels) if kparams is not None else None


def classifier_features(X, cparams=None, wparams=None):
    """Forest views, context kernels and winner counts from one sliding fit per window length.

    A kernel block is None when its parameters are not given.
    """
    X = np.asarray(X, dtype=float)
    T = X.shape[1]
    blocks = {"P": [], "C": [], "W": []}
    for s in range(0, len(X), BLOCK):
        Xb, row = X[s : s + BLOCK], {"P": [], "C": [], "W": []}
        for i, L in enumerate(window_lengths(T)):
            Cx = _sliding(Xb, *_setup(T, L))
            row["P"].append(np.delete(_pooled(Cx), _PPV_RESIDUAL, axis=1))
            if cparams is not None:
                row["C"].append(_kernels(in_context(Cx, L, 2), *cparams[i]))
            if wparams is not None:
                mu, sd, D = wparams[i]
                for shape in (False, True):
                    F = _compete(compete_inputs(Cx, mu, sd, shape), D[shape], 8)
                    m = F.shape[1] // 2
                    row["W"].append(np.hstack([np.sqrt(F[:, :m]), F[:, m:]]))
        for k, v in row.items():
            if v:
                blocks[k].append(np.hstack(v))
    return (views(X, np.vstack(blocks["P"])), np.vstack(blocks["C"]) if cparams is not None else None,
            np.vstack(blocks["W"]) if wparams is not None else None)
