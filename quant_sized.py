"""QUANT's feature count, and the earlier classifier sized to stay below it.  Not part of STCT+.

``quant_features`` reproduces aeon QUANT's feature count; the benchmark scripts
use it only to report how STCT+ feature counts compare with QUANT's.

``QuantSizedClassifier`` is the earlier four-model variant scored as
paper_quant:et_ridge (``short_ridge=False``) and paper_quant2:et_ridge: context and
dilated kernels shrink by one common factor until the total is at most QUANT's
count minus one, paper windows that collapse to the same count are computed once
where the views alone would reach QUANT's count, the forest has 200 trees, and
below length 128 dilated kernels fill the room left under QUANT's count at half
weight.  STCT+ (``stct_models.STCTClassifier()``) replaced it with a size rule of
its own and matched it on training splits (``fast_native.py``).
"""
import numpy as np

import stct_fast as SF
from stct_models import MIN_LENGTH, STCTClassifier


def quant_features(T, depth=6, divisor=4):
    """QUANT's feature count for a series of length T (aeon defaults: interval depth 6, quantile divisor 4).

    Four representations (series, smoothed first difference, second difference,
    periodogram magnitude), dyadic intervals plus half-shifted ones, and
    1 + (length - 1) // 4 quantiles per interval.  Torch takes the lower middle
    value as the median, so the shift test does too.
    """
    total = 0
    for L in (T, T - 1, T - 2, T // 2 + 1):
        for n in 2 ** np.arange(min(depth, int(np.log2(L)) + 1)):
            idx = np.linspace(0, L, n + 1, dtype=np.float32).astype(np.int64)
            iv = np.stack([idx[:-1], idx[1:]], 1)
            m = np.sort(iv[:, 1] - iv[:, 0])
            ivs = [iv]
            if n > 1 and m[(len(m) - 1) // 2] > 1:
                ivs.append(iv[:-1] + int(np.ceil(L / n / 2)))
            total += sum(1 if b - a == 1 else 1 + (b - a - 1) // divisor for a, b in np.vstack(ivs))
    return int(total)


def scaled_directions(T, n_views):
    """Directions per scale for context and dilated kernels so the four-model blend stays below QUANT.

    The forest's views and the winner counts keep their size; context and
    dilated kernels shrink by one common factor until the total is at most
    QUANT's feature count minus one.
    """
    n = SF.n_kernels(T)
    ns, nd = len(SF.window_lengths(T)), len(SF.dilations(T))
    room = quant_features(T) - 1 - n_views - 32 * SF.winner_groups(T) * ns
    return int(n * min(1.0, max(0, room) / (6 * n * (ns + nd))))


class QuantSizedClassifier(STCTClassifier):
    def __init__(self, random_state=42, n_jobs=-1, short_ridge=True):
        super().__init__("four", random_state, n_jobs)
        self.short_ridge = short_ridge

    def _n_trees(self):
        return 200

    def _distinct(self, X):
        return SF.trio_features(X[:1])[0].shape[1] >= quant_features(X.shape[1])

    def _kernel_sizes(self, X):
        T = X.shape[1]
        n_views = SF.trio_features(X[:1], distinct=self.distinct_)[0].shape[1]
        if T >= MIN_LENGTH:
            k = scaled_directions(T, n_views)
            return k, k
        if not self.short_ridge:
            return 0, 0
        room = quant_features(T) - 1 - n_views
        return 0, int(min(SF.n_kernels(T), max(0, room) // (6 * len(SF.dilations(T)))))
