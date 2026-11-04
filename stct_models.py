"""STCT+ classifiers and regressor with an sklearn-style interface.

``STCTClassifier()`` is STCT+: ExtraTrees of 200 trees on the STCT views
(Chebyshev fits on fixed windows, pooled statistics of sliding fits, and the
paper features of the sorted series and of its autocorrelation), averaged with
ridges on Chebyshev kernels.  Kernels get one random direction per 8 samples of
series length, shared equally among the kernel types in use, at most 64 per type:

* from length 128 up, context kernels (pairs of neighbouring windows) and dilated
  kernels (degree-4 fits to 9 samples spaced 1, 2, 4, ... apart, with no weight
  on the level coefficient) get one direction per 16 samples each, and winner
  counts (groups of 8 directions competing over each window's coefficients) are
  added, each block with its own ridge;
* below length 128 the dilated kernels alone get one direction per 8 samples and
  enter at half the forest's weight.

Earlier recipes stay available: ``recipe="trio"`` (100 trees, context kernels and
winner counts), ``"four"`` (the three-model recipe plus full-size dilated
kernels), ``"dilated"`` (the three-model budget, half-size context and dilated
kernels sharing one ridge) and ``"fast"`` (one ridge on threshold kernels).
``STCTRegressor`` is the fast recipe for regression: ExtraTrees of 50 trees,
half the features per split and 2 samples per leaf, averaged with a ridge on
threshold kernels from length 128 up.  Shorter series use the forest alone.

Kernel weightings and thresholds come from at most 32 training series, without labels.
"""
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import RidgeClassifierCV, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import stct_fast as SF

MIN_LENGTH = 128
ALPHAS = np.logspace(-3, 3, 10)
RECIPES = ("plus", "trio", "four", "dilated", "fast")


def _2d(X):
    X = np.asarray(X, dtype=float)
    return X[:, 0, :] if X.ndim == 3 else X


def _soft(decision):
    """Class probabilities from ridge decision values."""
    d = decision if decision.ndim > 1 else np.column_stack([-decision, decision])
    e = np.exp(d - d.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class _STCTModel:
    def __init__(self, recipe="fast", random_state=42, n_jobs=-1):
        self.recipe = recipe
        self.random_state = random_state
        self.n_jobs = n_jobs

    def get_params(self, deep=True):
        return dict(recipe=self.recipe, random_state=self.random_state, n_jobs=self.n_jobs)

    def set_params(self, **p):
        for k, v in p.items():
            setattr(self, k, v)
        return self

    def _n_trees(self):
        return 200 if self.recipe == "plus" else 100

    def _distinct(self, X):
        """Whether paper windows that collapse to the same count are computed once (never in these recipes)."""
        return False

    def _kernel_sizes(self, X):
        """Directions per scale or dilation for context and dilated kernels; 0 leaves a block out."""
        T = X.shape[1]
        n = SF.n_kernels(T)
        if self.recipe == "plus":
            return (min(64, T // 16),) * 2 if T >= MIN_LENGTH else (0, n)
        if T < MIN_LENGTH:
            return 0, 0
        return {"trio": (n, 0), "four": (n, n), "dilated": (max(1, n // 2),) * 2}[self.recipe]

    def _blocks(self, X):
        """Forest features and the ridge feature blocks of this recipe."""
        if self.recipe == "fast":
            V, K = SF.features(X, self.kparams_)
            return np.nan_to_num(V), [] if K is None else [np.nan_to_num(K)]
        V, C, W = SF.trio_features(X, self.cparams_, self.wparams_, distinct=self.distinct_)
        blocks = [B for B in (C, W) if B is not None]
        if self.dparams_ is not None:
            D = SF.dilated_features(X, self.dparams_)
            blocks = [W, np.hstack([C, D])] if self.recipe == "dilated" else blocks + [D]
        return np.nan_to_num(V), [np.nan_to_num(B) for B in blocks]

    def _fit(self, X, y, forest, ridge):
        if self.recipe not in RECIPES:
            raise ValueError(f"recipe must be one of {RECIPES}")
        X = _2d(X)
        T, seed = X.shape[1], self.random_state
        long_ = T >= MIN_LENGTH
        self.kparams_ = self.cparams_ = self.wparams_ = self.dparams_ = None
        self.distinct_ = self._distinct(X)
        if self.recipe == "fast":
            if long_:
                self.kparams_ = SF.kernel_fit(X, SF.n_kernels(T), seed, max_series=32)
        else:
            c, d = self._kernel_sizes(X)
            self.cparams_ = SF.context_fit(X, c, seed, 2) if c > 0 else None
            self.wparams_ = SF.compete_fit(X, seed, SF.winner_groups(T)) if long_ else None
            self.dparams_ = SF.dilated_fit(X, d, seed) if d > 0 else None
        # below length 128 the dilated kernels are the only ridge and enter at half the forest's weight
        self.ridge_weight_ = 1.0 if long_ else 0.5
        V, blocks = self._blocks(X)
        self.forest_ = forest.fit(V, y)
        self.ridges_ = []
        for B in blocks:
            try:
                self.ridges_.append(make_pipeline(StandardScaler(), ridge(alphas=ALPHAS)).fit(B, y))
            except np.linalg.LinAlgError:  # leave a failed ridge out
                self.ridges_.append(None)
        return self

    def n_features(self, X):
        """Total features the fitted model computes per series."""
        V, blocks = self._blocks(_2d(X)[:1])
        return V.shape[1] + sum(B.shape[1] for B in blocks)


class STCTClassifier(_STCTModel):
    def __init__(self, recipe="plus", random_state=42, n_jobs=-1):
        super().__init__(recipe, random_state, n_jobs)

    def fit(self, X, y):
        forest = ExtraTreesClassifier(n_estimators=self._n_trees(), max_features=0.1,
                                      random_state=self.random_state, n_jobs=self.n_jobs)
        self._fit(X, y, forest, RidgeClassifierCV)
        self.classes_ = self.forest_.classes_
        return self

    def predict_proba(self, X):
        V, blocks = self._blocks(_2d(X))
        probs, used, w = self.forest_.predict_proba(V), 1.0, self.ridge_weight_
        for r, B in zip(self.ridges_, blocks):
            if r is not None:
                probs, used = probs + w * _soft(r.decision_function(B)), used + w
        return probs / used

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def score(self, X, y):
        return float((self.predict(X) == np.asarray(y)).mean())


class STCTRegressor(_STCTModel):
    def __init__(self, recipe="fast", random_state=42, n_jobs=-1):
        super().__init__(recipe, random_state, n_jobs)

    def fit(self, X, y):
        forest = ExtraTreesRegressor(n_estimators=50, max_features=0.5, min_samples_leaf=2,
                                     random_state=self.random_state, n_jobs=self.n_jobs)
        return self._fit(X, y, forest, RidgeCV)

    def predict(self, X):
        V, blocks = self._blocks(_2d(X))
        preds = [self.forest_.predict(V)] + [r.predict(B) for r, B in zip(self.ridges_, blocks) if r is not None]
        return np.mean(preds, axis=0)
