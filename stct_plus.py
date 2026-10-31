"""STCT+ : the Short-Time Chebyshev Transform with a prime ladder and pooled statistics.

Both blocks are closed-form and training-free, and their size does not depend
on series length.

* ``stct_ladder`` -- each series is cut into 1, 2, 3, 5, 7, 11 and 13 tiling
  windows, and every window is described by the coefficients of a low-degree
  Chebyshev least-squares fit plus the RMS fit residual.

* ``stct_pooled`` -- the same fit is slid along the series at several window
  lengths, and each coefficient channel's time course is collapsed into
  shift-invariant statistics.

``stct_plus`` concatenates the two.  ``stct_fixed`` is the paper's original
representation, kept as the baseline.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial import chebyshev as C

PRIMES = (1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31)
LADDER_COUNTS = PRIMES[:7]
POOL_FRACTIONS = (1 / 32, 1 / 16, 1 / 8, 1 / 4)
POOL_STATS = ("mean", "std", "min", "max", "q25", "q50", "q75", "ppv", "argmax", "argmin")


def _cheby_design(win_len, deg):
    x = np.linspace(-1, 1, win_len)
    Phi = C.chebvander(x, deg)
    return Phi, np.linalg.pinv(Phi)


def _residual_measure(res, kind):
    """Summarise fit residuals along the last axis: sign-change rate or peak relative to RMS."""
    if kind == "sign":
        return (np.diff(np.sign(res), axis=-1) != 0).mean(axis=-1)
    if kind == "peak":
        return np.abs(res).max(axis=-1) / (np.sqrt((res**2).mean(axis=-1)) + 1e-12)
    raise ValueError(kind)


def _scale_degree(win_len):
    """Fit degree that grows with window length: floor(log2(length)) - 2, limited to 2..8."""
    return int(np.clip(int(np.log2(max(win_len, 1))) - 2, 2, 8))


# --------------------------------------------------------------------------
# Original STCT (verbatim logic from the paper notebooks)
# --------------------------------------------------------------------------


def _safe_cheby_params(n, n_windows=8, degree=4):
    degree = max(1, min(degree, n - 2))
    if n >= degree + 2:
        n_windows = max(1, min(n_windows, n // (degree + 2)))
    else:
        n_windows = 1
        degree = max(1, n - 2)
    return n_windows, degree


def _stct_single_scale(X, n_windows, degree, include_residual=True, orthonormal=False, adaptive_degree=False,
                       resid_kind=None):
    X = np.asarray(X, dtype=float)
    n_samples, n = X.shape
    if adaptive_degree:
        degree = _scale_degree(n // n_windows)
    nw, deg = _safe_cheby_params(n, n_windows, degree)
    win_len = min(max(deg + 2, n // nw), n)
    step = max(1, (n - win_len) // max(1, nw - 1))
    starts = np.array(list(range(0, n - win_len + 1, step))[:nw])
    actual_nw = len(starts)

    Phi, pinv = _cheby_design(win_len, deg)
    if orthonormal:
        Q, _ = np.linalg.qr(Phi)
        Phi, pinv = Q, Q.T
    idx = starts[:, None] + np.arange(win_len)[None, :]
    windows = X[:, idx]
    coefs = windows @ pinv.T
    if include_residual:
        recon = coefs @ Phi.T
        resid_rms = np.sqrt(((windows - recon) ** 2).mean(axis=-1, keepdims=True))
        block = np.concatenate([coefs, resid_rms], axis=-1)
        if resid_kind:
            extra = _residual_measure(windows - recon, resid_kind)[..., None]
            block = np.concatenate([block, extra], axis=-1)
    else:
        block = coefs
    if actual_nw < nw:
        pad = np.zeros((n_samples, nw - actual_nw, block.shape[-1]))
        block = np.concatenate([block, pad], axis=1)
    return block.reshape(n_samples, nw * block.shape[-1])


def stct_fixed(X, window_counts=(8, 16), degree=4, include_residual=True, orthonormal=False, adaptive_degree=False,
               resid_kind=None):
    """Original STCT features (paper default: degree 4, 8 and 16 windows)."""
    parts = [_stct_single_scale(X, nw, degree, include_residual, orthonormal, adaptive_degree, resid_kind)
             for nw in window_counts]
    return np.concatenate(parts, axis=1)


# --------------------------------------------------------------------------
# Prime ladder: tiling windows at several window counts
# --------------------------------------------------------------------------


def stct_ladder(X, counts=LADDER_COUNTS, degree=4, orthonormal=False, adaptive_degree=False):
    """Fit coefficients and residual RMS in every tiling window of every count.

    Counts whose windows would be shorter than degree + 2 samples are skipped.
    ``orthonormal`` reports coefficients in an orthonormal basis of the same
    polynomial space, which decorrelates them; the residual is unchanged.
    ``adaptive_degree`` replaces ``degree`` with :func:`_scale_degree` at each count.
    """
    X = np.asarray(X, dtype=float)
    n_samples, n = X.shape
    designs = {}
    feats = []
    for nw in counts:
        deg = _scale_degree(n // nw) if adaptive_degree else degree
        if n // nw < deg + 2:
            continue
        edges = np.round(np.linspace(0, n, nw + 1)).astype(int)
        for a, b in zip(edges[:-1], edges[1:]):
            L = b - a
            if (L, deg) not in designs:
                Phi, pinv = _cheby_design(L, deg)
                if orthonormal:
                    Q, _ = np.linalg.qr(Phi)
                    Phi, pinv = Q, Q.T
                designs[(L, deg)] = (Phi, pinv)
            Phi, pinv = designs[(L, deg)]
            W = X[:, a:b]
            c = W @ pinv.T
            res = W - c @ Phi.T
            feats += [c, np.sqrt((res**2).mean(axis=1, keepdims=True))]
    if not feats:
        return np.zeros((n_samples, 0))
    return np.concatenate(feats, axis=1)


# --------------------------------------------------------------------------
# Pooled block: sliding fit + shift-invariant statistics
# --------------------------------------------------------------------------


def _windowed_coefficients(X, win_len, degree, stride=1, max_block_elems=1_000_000, orthonormal=False):
    """Coefficients and residual RMS for windows starting every ``stride`` samples.

    Returns (n_samples, degree + 2, n_windows).  Each block is projected onto an
    orthonormal basis Q of the Chebyshev space in one matmul; the Chebyshev
    coefficients are a fixed linear map of that projection and the residual
    energy is ||w||^2 - ||Q^T w||^2, so no window is ever reconstructed.
    """
    X = np.asarray(X, dtype=float)
    n_samples, T = X.shape
    L = win_len
    Phi = C.chebvander(np.linspace(-1, 1, L), degree)
    Q, R = np.linalg.qr(Phi)
    RinvT = np.linalg.inv(R).T
    starts = np.arange(0, T - L + 1, stride)
    if starts[-1] != T - L:  # always cover the end of the series
        starts = np.append(starts, T - L)
    nw = len(starts)
    idx = starts[:, None] + np.arange(L)[None, :]
    out = np.empty((n_samples, degree + 2, nw))
    block = max(1, max_block_elems // (nw * L))
    for s in range(0, n_samples, block):
        W = X[s : s + block][:, idx]
        proj = W @ Q
        coefs = proj if orthonormal else proj @ RinvT
        energy = (W * W).sum(axis=-1) - (proj * proj).sum(axis=-1)
        out[s : s + block, : degree + 1] = coefs.transpose(0, 2, 1)
        out[s : s + block, degree + 1] = np.sqrt(np.maximum(energy, 0.0) / L)
    return out


def _pool(Cx, stats):
    """Collapse (n, ch, t) to (n, ch * len(stats)) with shift-invariant statistics."""
    t = Cx.shape[2]
    quantiles = [s for s in stats if s.startswith("q")]
    if quantiles:
        Qv = np.quantile(Cx, [int(s[1:]) / 100 for s in quantiles], axis=2)
    feats = []
    for s in stats:
        if s == "mean":
            feats.append(Cx.mean(axis=2))
        elif s == "std":
            feats.append(Cx.std(axis=2))
        elif s == "min":
            feats.append(Cx.min(axis=2))
        elif s == "max":
            feats.append(Cx.max(axis=2))
        elif s == "ppv":
            feats.append((Cx > 0).mean(axis=2))
        elif s == "argmax":
            feats.append(Cx.argmax(axis=2) / max(t - 1, 1))
        elif s == "argmin":
            feats.append(Cx.argmin(axis=2) / max(t - 1, 1))
        elif s.startswith("q"):
            feats.append(Qv[quantiles.index(s)])
        else:
            raise ValueError(s)
    return np.concatenate(feats, axis=1)


def _sliding_residual_measure(X, win_len, degree, stride, kind, max_block_elems=1_000_000):
    """Residual measure at every sliding window: (n_samples, 1, n_windows)."""
    n_samples, T = X.shape
    L = win_len
    Q, _ = np.linalg.qr(C.chebvander(np.linspace(-1, 1, L), degree))
    starts = np.arange(0, T - L + 1, stride)
    if starts[-1] != T - L:
        starts = np.append(starts, T - L)
    idx = starts[:, None] + np.arange(L)[None, :]
    out = np.empty((n_samples, 1, len(starts)))
    block = max(1, max_block_elems // (len(starts) * L))
    for s in range(0, n_samples, block):
        W = X[s : s + block][:, idx]
        out[s : s + block, 0] = _residual_measure(W - (W @ Q) @ Q.T, kind)
    return out


def _shape_normalise(Cx):
    """Divide non-constant coefficients by the window variation, separating shape from amplitude."""
    Cx = Cx.copy()
    d = Cx.shape[1] - 2
    scale = np.sqrt((Cx[:, 1 : d + 1] ** 2).sum(axis=1, keepdims=True) + Cx[:, d + 1 :] ** 2) + 1e-12
    Cx[:, 1 : d + 1] /= scale
    return Cx


def _energy(Cx):
    """RMS of each non-constant coefficient over positions: sign-free energy per polynomial degree."""
    d = Cx.shape[1] - 2
    return np.sqrt((Cx[:, 1 : d + 1] ** 2).mean(axis=2))


def _cross_correlation(Cx):
    """Correlation over positions between each pair of non-constant channels."""
    Z = Cx[:, 1:]
    Z = (Z - Z.mean(axis=2, keepdims=True)) / (Z.std(axis=2, keepdims=True) + 1e-12)
    corr = Z @ Z.transpose(0, 2, 1) / Z.shape[2]
    i, j = np.triu_indices(Z.shape[1], k=1)
    return corr[:, i, j]


def _second_order(Cx, degree=2):
    """Low-degree fit over each channel time course: how the local shape changes along the series."""
    _, pinv = _cheby_design(Cx.shape[2], degree)
    return (Cx @ pinv.T).reshape(len(Cx), -1)


def stct_pooled(X, fractions=POOL_FRACTIONS, degree=4, stats=POOL_STATS, overlap=8, orthonormal=False,
                shape=False, energy=False, cross=False, second=False, adaptive_degree=False, sample_windows=(),
                resid_kind=None):
    """Sliding STCT at window lengths given as fractions of the series length.

    Windows slide with stride window // overlap, and each channel time course
    is summarised by ``stats``.  The proportion of positive residuals is dropped
    because a residual RMS is never negative.  The remaining flags are the
    optional algorithmic additions described on their helper functions.
    ``sample_windows`` adds window lengths given in samples, independent of series
    length.  ``adaptive_degree`` sets each window's degree with :func:`_scale_degree`.
    """
    X = np.asarray(X, dtype=float)
    T = X.shape[1]
    shortest = 4 if adaptive_degree else degree + 2
    lengths = []
    for L in [int(round(T * f)) for f in fractions] + list(sample_windows):
        L = max(shortest, min(L, T))
        if L not in lengths:
            lengths.append(L)
    parts = []
    for L in lengths:
        deg = _scale_degree(L) if adaptive_degree else degree
        Cx = _windowed_coefficients(X, L, deg, max(1, L // overlap), orthonormal=orthonormal)
        pooled = _pool(_shape_normalise(Cx) if shape else Cx, stats)
        if "ppv" in stats:
            ch = Cx.shape[1]
            pooled = np.delete(pooled, stats.index("ppv") * ch + ch - 1, axis=1)
        parts.append(pooled)
        if resid_kind:
            R = _sliding_residual_measure(X, L, deg, max(1, L // overlap), resid_kind)
            parts.append(_pool(R, tuple(st for st in stats if st != "ppv")))
        if energy:
            parts.append(_energy(Cx))
        if cross:
            parts.append(_cross_correlation(Cx))
        if second:
            parts.append(_second_order(Cx))
    return np.concatenate(parts, axis=1)


# --------------------------------------------------------------------------
# The STCT+ recipe
# --------------------------------------------------------------------------


def log_periodogram(X):
    """Log power at each non-zero frequency of the mean-removed series."""
    X = np.asarray(X, dtype=float)
    X = X - X.mean(axis=1, keepdims=True)
    return np.log(np.abs(np.fft.rfft(X, axis=1))[:, 1:] ** 2 + 1e-8)


def global_quantiles(X, degree=4):
    """Deciles of the whole series and of its residual after a whole-series fit."""
    X = np.asarray(X, dtype=float)
    Phi, pinv = _cheby_design(X.shape[1], degree)
    res = X - (X @ pinv.T) @ Phi.T
    qs = np.arange(1, 10) / 10
    return np.hstack([np.quantile(X, qs, axis=1).T, np.quantile(res, qs, axis=1).T])


def window_quantiles(X, counts=PRIMES[:5]):
    """Quartiles of the raw values inside every tiling window of every count."""
    X = np.asarray(X, dtype=float)
    n = X.shape[1]
    feats = []
    for nw in counts:
        if n // nw < 4:
            continue
        edges = np.round(np.linspace(0, n, nw + 1)).astype(int)
        for a, b in zip(edges[:-1], edges[1:]):
            feats.append(np.quantile(X[:, a:b], (0.25, 0.5, 0.75), axis=1).T)
    return np.hstack(feats)


def level_quantiles(X, counts=PRIMES[1:7], degree=4, orthonormal=False):
    """Quartiles of each coefficient channel across the tiling windows of each ladder level."""
    X = np.asarray(X, dtype=float)
    ch = degree + 2
    parts = []
    for nw in counts:
        F = stct_ladder(X, (nw,), degree, orthonormal)
        if F.shape[1]:
            parts.append(_pool(F.reshape(len(X), -1, ch).transpose(0, 2, 1), ("q25", "q50", "q75")))
    return np.hstack(parts) if parts else np.zeros((len(X), 0))


def stct_plus(X, counts=LADDER_COUNTS, degree=4, fractions=POOL_FRACTIONS, stats=POOL_STATS, overlap=8,
              orthonormal=False, shape=False, energy=False, cross=False, second=False, spectrum=False,
              spectrum_counts=PRIMES[:4], global_q=False, window_q=False, level_q=False,
              adaptive_degree=False, sample_windows=(), paper=False, resid_kind=None, znorm=False):
    """Prime ladder or paper features (``paper=True``) + pooled statistics, plus optional candidate blocks.

    ``spectrum`` applies the same fit to frequency bands instead of time windows.
    ``global_q``, ``window_q`` and ``level_q`` add the quantile blocks above.
    """
    X = np.asarray(X, dtype=float)
    if znorm:
        X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
    first = (stct_fixed(X, (8, 16), degree, True, orthonormal, adaptive_degree, resid_kind) if paper
             else stct_ladder(X, counts, degree, orthonormal, adaptive_degree))
    parts = [first,
             stct_pooled(X, fractions, degree, stats, overlap, orthonormal, shape, energy, cross, second,
                         adaptive_degree, sample_windows, resid_kind)]
    if spectrum:
        parts.append(stct_ladder(log_periodogram(X), spectrum_counts, degree, orthonormal))
    if global_q:
        parts.append(global_quantiles(X, degree))
    if window_q:
        parts.append(window_quantiles(X))
    if level_q:
        parts.append(level_quantiles(X, degree=degree, orthonormal=orthonormal))
    return np.hstack(parts)


class STCTPlus:
    """Minimal sklearn-style transformer around :func:`stct_plus`."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X)
        if X.ndim == 3:
            X = X[:, 0, :]
        return stct_plus(X, **self.kwargs)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_params(self, deep=True):
        return dict(self.kwargs)

    def set_params(self, **p):
        self.kwargs.update(p)
        return self


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.standard_normal((3, 200)).cumsum(axis=1)
    L, d = 25, 4
    Cx = _windowed_coefficients(X, L, d, stride=1)
    Cs = _windowed_coefficients(X, L, d, stride=3)
    Phi, pinv = _cheby_design(L, d)
    for s in (0, 17, 175):
        w = X[:, s : s + L]
        c = w @ pinv.T
        r = np.sqrt(((w - c @ Phi.T) ** 2).mean(axis=1))
        assert np.allclose(Cx[:, : d + 1, s], c, atol=1e-8), s
        assert np.allclose(Cx[:, d + 1, s], r, atol=1e-8), s
    assert np.allclose(Cs[:, :, :-1], Cx[:, :, ::3]) and np.allclose(Cs[:, :, -1], Cx[:, :, -1])
    # orthonormal coefficients span the same fit, so the residual must not change
    Co = _windowed_coefficients(X, L, d, stride=1, orthonormal=True)
    assert np.allclose(Co[:, d + 1], Cx[:, d + 1], atol=1e-8)
    assert np.allclose(stct_ladder(X, (3,), orthonormal=True)[:, 5::6], stct_ladder(X, (3,))[:, 5::6])
    assert np.abs(_cross_correlation(Cx)).max() <= 1 + 1e-9
    assert np.abs(_shape_normalise(Cx)[:, 1 : d + 1]).max() <= 1 + 1e-9
    for T in (15, 24, 84, 96, 601, 1460):
        F = stct_plus(rng.standard_normal((4, T)).cumsum(axis=1), orthonormal=True, shape=True, energy=True,
                      cross=True, second=True, spectrum=True)
        assert np.isfinite(F).all(), T
        print(T, F.shape)
    # quantile blocks: a level's median must match the median of its raw ladder coefficients
    lvl = stct_ladder(X, (5,)).reshape(3, 5, 6)
    assert np.allclose(level_quantiles(X, (5,))[:, 6:12], np.median(lvl, axis=1))
    assert np.allclose(global_quantiles(X)[:, 4], np.median(X, axis=1))
    assert window_quantiles(X).shape[1] == 54 and global_quantiles(X).shape[1] == 18
    for T in (15, 24, 84):
        S = rng.standard_normal((4, T)).cumsum(axis=1)
        assert np.isfinite(stct_plus(S, orthonormal=True, global_q=True, window_q=True, level_q=True)).all(), T
    # paper base: first block is exactly the paper features, and the pooled block follows it
    P = stct_plus(X, paper=True)
    assert np.allclose(P[:, :144], stct_fixed(X)) and np.allclose(P[:, 144:], stct_pooled(X))
    # orthonormal paper coefficients keep the same residual column
    assert np.allclose(stct_fixed(X, (8,), orthonormal=True)[:, 5::6], stct_fixed(X, (8,))[:, 5::6])
    for T in (15, 24, 84, 601, 2000):
        S = rng.standard_normal((4, T)).cumsum(axis=1)
        for kw in (dict(orthonormal=True), dict(adaptive_degree=True), dict(shape=True, energy=True, cross=True, second=True),
                   dict(spectrum=True, global_q=True, window_q=True, level_q=True),
                   dict(fractions=(1 / 8, 1 / 4), sample_windows=(8, 32)), dict(sample_windows=(16,))):
            assert np.isfinite(stct_plus(S, paper=True, **kw)).all(), (T, kw)
    # residual measures: an alternating signal changes sign at every step and has a peak ratio of 1
    alt = np.array([[1.0, -1.0] * 8])
    assert _residual_measure(alt, "sign")[0] == 1.0 and np.isclose(_residual_measure(alt, "peak")[0], 1.0)
    # the sliding measure must match a direct fit at one window
    Rs = _sliding_residual_measure(X, 25, 4, 1, "sign")
    Ph, Pi = _cheby_design(25, 4)
    w = X[:, 40:65]
    assert np.allclose(Rs[:, 0, 40], _residual_measure(w - (w @ Pi.T) @ Ph.T, "sign"))
    # new options add one channel to each block, and z-normalisation removes scale and offset
    assert stct_fixed(X, resid_kind="sign").shape[1] == 168
    assert np.allclose(stct_plus(5 * X + 3, paper=True, znorm=True), stct_plus(X, paper=True, znorm=True), atol=1e-6)
    for T in (15, 24, 84, 601, 2000):
        S = rng.standard_normal((4, T)).cumsum(axis=1)
        for kw in (dict(resid_kind="sign"), dict(resid_kind="peak"), dict(znorm=True)):
            assert np.isfinite(stct_plus(S, paper=True, orthonormal=True, **kw)).all(), (T, kw)
    # degree follows window length; defaults must be unaffected by the new options
    assert [_scale_degree(L) for L in (4, 6, 8, 64, 256, 1024, 100000)] == [2, 2, 2, 4, 6, 8, 8]
    assert np.allclose(stct_pooled(X, sample_windows=()), stct_pooled(X))
    assert np.allclose(stct_pooled(X, fractions=(), sample_windows=(25,)),
                       stct_pooled(X, fractions=(25 / 200,)))
    for T in (15, 24, 84, 601, 2000):
        S = rng.standard_normal((4, T)).cumsum(axis=1)
        for kw in (dict(adaptive_degree=True), dict(fractions=(1 / 8, 1 / 4), sample_windows=(8, 32)),
                   dict(sample_windows=(16,)), dict(adaptive_degree=True, sample_windows=(16,))):
            F = stct_plus(S, orthonormal=True, **kw)
            assert np.isfinite(F).all(), (T, kw)
            if T == 2000:
                print(kw, F.shape[1])
    # a pure tone must put its largest periodogram value at its own frequency
    tone = np.sin(2 * np.pi * 10 * np.arange(200) / 200)[None]
    assert log_periodogram(tone).argmax() == 9
    assert stct_plus(np.ones((2, 200)), spectrum=True).shape[1] == stct_plus(X, spectrum=True).shape[1]
    print("stct_plus OK")
