"""STCT+ : the Short-Time Chebyshev Transform and a minimal expansion of it.

Two feature families, both closed-form and training-free:

* ``fixed``  -- the original STCT exactly as in the paper notebooks: each of
  ``n_windows`` windows at fixed positions is summarised by the coefficients of
  a degree-``degree`` Chebyshev least-squares fit plus the RMS fit residual.
  Feature count = sum(window_counts) * (degree + 2).

* ``pooled`` -- the *dense* STCT.  The same Chebyshev fit is evaluated at every
  position (stride 1) for several window lengths, which turns the transform
  into a bank of ``degree + 1`` orthogonal-polynomial convolution kernels per
  scale plus a residual channel.  Each channel's time course is then collapsed
  with translation-invariant statistics (quantiles, extrema and their
  positions, proportion of positive values).  This addresses STCT's main
  weakness (features tied to absolute window positions, so a time shift of a
  pattern changes every feature) at a feature count that still does not
  depend on series length.

Optionally both families are also computed on the first difference of the
series (``diff=True``), which exposes local slope structure that the raw
series hides; MultiRocket does the same.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial import chebyshev as C


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


def _cheby_design(win_len, deg):
    x = np.linspace(-1, 1, win_len)
    Phi = C.chebvander(x, deg)
    return Phi, np.linalg.pinv(Phi)


def _stct_single_scale(X, n_windows, degree, include_residual=True):
    X = np.asarray(X, dtype=float)
    n_samples, n = X.shape
    nw, deg = _safe_cheby_params(n, n_windows, degree)
    win_len = min(max(deg + 2, n // nw), n)
    step = max(1, (n - win_len) // max(1, nw - 1))
    starts = np.array(list(range(0, n - win_len + 1, step))[:nw])
    actual_nw = len(starts)

    Phi, pinv = _cheby_design(win_len, deg)
    idx = starts[:, None] + np.arange(win_len)[None, :]
    windows = X[:, idx]
    coefs = windows @ pinv.T
    if include_residual:
        recon = coefs @ Phi.T
        resid_rms = np.sqrt(((windows - recon) ** 2).mean(axis=-1, keepdims=True))
        block = np.concatenate([coefs, resid_rms], axis=-1)
    else:
        block = coefs
    if actual_nw < nw:
        pad = np.zeros((n_samples, nw - actual_nw, block.shape[-1]))
        block = np.concatenate([block, pad], axis=1)
    return block.reshape(n_samples, nw * block.shape[-1])


def stct_fixed(X, window_counts=(8, 16), degree=4, include_residual=True):
    """Original STCT features (paper default: degree 4, 8 and 16 windows)."""
    parts = [_stct_single_scale(X, nw, degree, include_residual) for nw in window_counts]
    return np.concatenate(parts, axis=1)


# --------------------------------------------------------------------------
# Dense STCT + translation-invariant pooling
# --------------------------------------------------------------------------

POOL_STATS = ("mean", "std", "min", "max", "q25", "q50", "q75", "ppv", "argmax", "argmin")


def _windowed_coefficients(X, win_len, degree, stride=1, max_block_elems=1_000_000):
    """Chebyshev LS coefficients and residual RMS for windows starting every ``stride``.

    Returns an array of shape (n_samples, degree + 2, n_windows).  ``stride=1``
    is the fully dense transform.  Each block of series is gathered into
    (b, n_windows, L) and projected onto an orthonormal basis Q of the
    Chebyshev space in one matmul; the Chebyshev coefficients are a fixed
    linear map of that projection and the residual energy is
    ||w||^2 - ||Q^T w||^2, so no window is ever reconstructed.
    """
    X = np.asarray(X, dtype=float)
    n_samples, T = X.shape
    L = win_len
    Phi = C.chebvander(np.linspace(-1, 1, L), degree)  # (L, d+1)
    Q, R = np.linalg.qr(Phi)  # Phi = Q R, Q orthonormal
    RinvT = np.linalg.inv(R).T  # coefs = (Q^T w) R^{-1}
    starts = np.arange(0, T - L + 1, stride)
    if starts[-1] != T - L:  # always cover the end of the series
        starts = np.append(starts, T - L)
    nw = len(starts)
    idx = starts[:, None] + np.arange(L)[None, :]
    out = np.empty((n_samples, degree + 2, nw))
    block = max(1, max_block_elems // (nw * L))
    for s in range(0, n_samples, block):
        W = X[s : s + block][:, idx]  # (b, nw, L)
        proj = W @ Q  # (b, nw, d+1)
        coefs = proj @ RinvT
        energy = (W * W).sum(axis=-1) - (proj * proj).sum(axis=-1)
        out[s : s + block, : degree + 1] = coefs.transpose(0, 2, 1)
        out[s : s + block, degree + 1] = np.sqrt(np.maximum(energy, 0.0) / L)
    return out


def _pool(Cx, stats):
    """Collapse (n, ch, t) -> (n, ch * len(stats)) with translation-invariant stats."""
    n, ch, t = Cx.shape
    feats = []
    need_q = [s for s in stats if s.startswith("q")]
    if need_q:
        qs = [int(s[1:]) / 100 for s in need_q]
        Qv = np.quantile(Cx, qs, axis=2)  # (nq, n, ch)
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
        elif s == "absmean":
            feats.append(np.abs(Cx).mean(axis=2))
        elif s == "rough":  # mean absolute change between consecutive windows
            feats.append(np.abs(np.diff(Cx, axis=2)).mean(axis=2) if t > 1 else np.zeros((n, ch)))
        elif s.startswith("q"):
            feats.append(Qv[need_q.index(s)])
        else:
            raise ValueError(s)
    return np.concatenate(feats, axis=1)  # stat-major, channel-minor


def _window_lengths(T, fractions, degree, min_len=None):
    min_len = min_len or (degree + 2)
    out = []
    for f in fractions:
        L = int(round(T * f))
        L = max(min_len, min(L, T))
        if L not in out:
            out.append(L)
    return out


def stct_pooled(
    X,
    fractions=(1 / 32, 1 / 16, 1 / 8, 1 / 4),
    degree=4,
    stats=POOL_STATS,
    min_len=None,
    overlap=8,
):
    """Sliding multi-scale STCT with translation-invariant pooling.

    Feature count = len(fractions) * (degree + 2) * len(stats), independent of
    series length.  Window lengths are fractions of the series length so the
    scales are comparable across datasets (as in the original n_windows
    parameterisation: 1/8 and 1/16 correspond to 8 and 16 windows).
    Windows are slid with stride ``L // overlap`` (``overlap=None`` gives the
    fully dense stride-1 transform); 8x overlap gives the same pooled
    statistics up to noise at a small fraction of the cost.
    """
    X = np.asarray(X, dtype=float)
    n_samples, T = X.shape
    parts = []
    for L in _window_lengths(T, fractions, degree, min_len):
        deg = max(1, min(degree, L - 2))
        stride = 1 if overlap is None else max(1, L // overlap)
        Cx = _windowed_coefficients(X, L, deg, stride)
        if deg < degree:  # pad channels so the layout is stable
            pad = np.zeros((n_samples, degree - deg, Cx.shape[2]))
            Cx = np.concatenate([Cx[:, : deg + 1], pad, Cx[:, deg + 1 :]], axis=1)
        parts.append(_pool(Cx, stats))
    return np.concatenate(parts, axis=1)


# --------------------------------------------------------------------------
# Combined extractor
# --------------------------------------------------------------------------


def stct_plus(
    X,
    fixed=True,
    pooled=True,
    diff=True,
    window_counts=(8, 16),
    fractions=(1 / 32, 1 / 16, 1 / 8, 1 / 4),
    degree=4,
    stats=POOL_STATS,
    overlap=8,
):
    """STCT+ feature vector: [fixed | pooled] on x, optionally also on diff(x)."""
    X = np.asarray(X, dtype=float)
    series = [X]
    if diff:
        series.append(np.diff(X, axis=1))
    parts = []
    for S in series:
        if fixed:
            parts.append(stct_fixed(S, window_counts, degree))
        if pooled:
            parts.append(stct_pooled(S, fractions, degree, stats, overlap=overlap))
    return np.concatenate(parts, axis=1)


# --------------------------------------------------------------------------
# sklearn transformer wrapper
# --------------------------------------------------------------------------


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
    # sanity: dense coefficients at a window start must equal the fixed-window fit
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
    for T in (24, 84, 96, 601, 1460):
        F = stct_plus(rng.standard_normal((4, T)).cumsum(axis=1))
        assert np.isfinite(F).all(), T
        print(T, F.shape)
    print("stct_plus OK")
