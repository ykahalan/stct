"""STCT+ classifier and regressor with an sklearn-style interface.

``STCTClassifier`` averages an ExtraTrees forest of 200 trees (10% of features
per split) on the STCT views with ridge classifiers on Chebyshev kernel blocks:

* from length 128 up, context kernels and dilated kernels with
  min(64, T // 16) directions per window length or dilation, and winner counts,
  each block with its own ridge at the forest's weight;
* below length 128, dilated kernels alone with min(64, T // 8) directions, at
  half the forest's weight.

``STCTRegressor`` averages an ExtraTrees forest of 50 trees (half the features
per split, 2 samples per leaf) on the same views with a ridge on threshold
kernels from length 128 up; shorter series use the forest alone.

Kernel directions and thresholds come from at most 32 training series, without labels.
"""
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import RidgeClassifierCV, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import stct_features as SF

MIN_LENGTH = 128  # series at least this long get the full kernel set
ALPHAS = np.logspace(-3, 3, 10)


def _2d(X):
    X = np.asarray(X, dtype=float)
    return X[:, 0, :] if X.ndim == 3 else X


def _soft(decision):
    """Class probabilities from ridge decision values."""
    d = decision if decision.ndim > 1 else np.column_stack([-decision, decision])
    e = np.exp(d - d.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class _STCTModel:
    def __init__(self, random_state=42, n_jobs=-1):
        self.random_state = random_state
        self.n_jobs = n_jobs

    def get_params(self, deep=True):
        return dict(random_state=self.random_state, n_jobs=self.n_jobs)

    def set_params(self, **p):
        for k, v in p.items():
            setattr(self, k, v)
        return self

    def _fit(self, X, y, forest, ridge):
        X = _2d(X)
        V, blocks = self._fit_kernels(X)._blocks(X)
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
    def _fit_kernels(self, X):
        T, seed = X.shape[1], self.random_state
        if T >= MIN_LENGTH:
            n = min(64, T // 16)
            self.cparams_ = SF.context_fit(X, n, seed)
            self.wparams_ = SF.compete_fit(X, seed, SF.winner_groups(T))
            self.dparams_ = SF.dilated_fit(X, n, seed)
            self.ridge_weight_ = 1.0
        else:
            self.cparams_ = self.wparams_ = None
            self.dparams_ = SF.dilated_fit(X, SF.n_kernels(T), seed)
            self.ridge_weight_ = 0.5
        return self

    def _blocks(self, X):
        V, Cb, W = SF.classifier_features(X, self.cparams_, self.wparams_)
        blocks = [B for B in (Cb, W) if B is not None] + [SF.dilated_features(X, self.dparams_)]
        return np.nan_to_num(V), [np.nan_to_num(B) for B in blocks]

    def fit(self, X, y):
        forest = ExtraTreesClassifier(n_estimators=200, max_features=0.1,
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
    def _fit_kernels(self, X):
        T = X.shape[1]
        self.kparams_ = SF.kernel_fit(X, SF.n_kernels(T), self.random_state) if T >= MIN_LENGTH else None
        return self

    def _blocks(self, X):
        V, K = SF.regressor_features(X, self.kparams_)
        return np.nan_to_num(V), [] if K is None else [np.nan_to_num(K)]

    def fit(self, X, y):
        forest = ExtraTreesRegressor(n_estimators=50, max_features=0.5, min_samples_leaf=2,
                                     random_state=self.random_state, n_jobs=self.n_jobs)
        return self._fit(X, y, forest, RidgeCV)

    def predict(self, X):
        V, blocks = self._blocks(_2d(X))
        preds = [self.forest_.predict(V)] + [r.predict(B) for r, B in zip(self.ridges_, blocks) if r is not None]
        return np.mean(preds, axis=0)
