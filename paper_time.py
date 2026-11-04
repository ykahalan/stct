"""Time classifiers and count their features on one thread, same machine, same pass.

    python paper_time.py --part 0 --parts 4

For each dataset, STCT+ and the published methods that can run here (aeon
implementations, configured as published) are fitted on the training split and
predict the test split one after another in the same process, one thread each.
Test labels are read only to check accuracy against the published results.
Datasets are spread over --parts processes.  Appends to results/paper_time.csv
and skips finished (dataset, method) pairs on resume.

HC2, InceptionTime, 1NN-DTW and FreshPRINCE are not run: their published fit
times total 39 to 290 hours on the UCR archive.
"""
import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[var] = "1"

import argparse
import time
import warnings

import numpy as np
import pandas as pd
import torch
from aeon.classification.convolution_based import (MiniRocketClassifier, MultiRocketClassifier,
                                                   MultiRocketHydraClassifier, RocketClassifier)
from aeon.classification.convolution_based._hydra import _SparseScaler
from aeon.classification.dictionary_based import WEASEL_V2
from aeon.classification.feature_based import Catch22Classifier
from aeon.classification.interval_based import QUANTClassifier
from aeon.classification.shapelet_based import RDSTClassifier
from aeon.transformations.collection.convolution_based import MultiRocket
from aeon.transformations.collection.convolution_based._hydra import HydraTransformer
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler

import bench
from stct_models import STCTClassifier

warnings.filterwarnings("ignore")
torch.set_num_threads(1)
OUT = "results/paper_time.csv"
MR_KERNELS = 6250  # MultiROCKET as published: 6,250 kernels, 49,728 features


class MRHydra(MultiRocketHydraClassifier):
    """aeon's MultiRocket-Hydra with MultiROCKET at its published size."""

    def _fit(self, X, y):
        self._n_jobs = 1
        self._transform_hydra = HydraTransformer(n_kernels=self.n_kernels, n_groups=self.n_groups, n_jobs=1,
                                                 random_state=self.random_state)
        self._scale_hydra = _SparseScaler()
        Xh = self._scale_hydra.fit_transform(self._transform_hydra.fit_transform(X))
        self._transform_multirocket = MultiRocket(n_kernels=MR_KERNELS, n_jobs=1, random_state=self.random_state)
        self._scale_multirocket = StandardScaler()
        Xm = self._scale_multirocket.fit_transform(self._transform_multirocket.fit_transform(X))
        self.classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)).fit(np.concatenate((Xh, Xm), axis=1), y)
        self.n_features_ = Xh.shape[1] + Xm.shape[1]
        return self

    def _predict(self, X):
        """Same predictions as aeon's, in batches of 512 series so the Hydra transform fits in memory."""
        return np.concatenate([super(MRHydra, self)._predict(X[s : s + 512]) for s in range(0, len(X), 512)])


METHODS = {
    "STCT+": lambda: STCTClassifier(random_state=bench.RNG, n_jobs=1),
    "MR-Hydra": lambda: MRHydra(random_state=bench.RNG, n_jobs=1),
    "MultiROCKET": lambda: MultiRocketClassifier(n_kernels=MR_KERNELS, random_state=bench.RNG, n_jobs=1),
    "RDST": lambda: RDSTClassifier(random_state=bench.RNG, n_jobs=1),
    "WEASEL-2.0": lambda: WEASEL_V2(random_state=bench.RNG, n_jobs=1),
    "MiniROCKET": lambda: MiniRocketClassifier(random_state=bench.RNG, n_jobs=1),
    "ROCKET": lambda: RocketClassifier(random_state=bench.RNG, n_jobs=1),
    "QUANT": lambda: QUANTClassifier(random_state=bench.RNG),
    "Catch22": lambda: Catch22Classifier(catch24=False, random_state=bench.RNG, n_jobs=1),
}


def n_features(model, X):
    """Width of the representation the model's final learner sees."""
    if isinstance(model, STCTClassifier):
        return model.n_features(X)
    if isinstance(model, MRHydra):
        return model.n_features_
    est = getattr(model, "estimator_", None) or getattr(model, "clf", None)
    if hasattr(est, "steps"):
        est = est.steps[-1][1]
    return int(est.n_features_in_) if hasattr(est, "n_features_in_") else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    a = ap.parse_args()
    known = set()
    if os.path.exists(OUT):
        p = pd.read_csv(OUT)
        known = set(zip(p.dataset, p.method))
    for i, name in enumerate(bench.UCR):
        if i % a.parts != a.part or all((name, m) in known for m in METHODS):
            continue
        try:
            Xtr, ytr, Xte, yte = bench.load("classification", name)
        except Exception as e:
            print(f"[failed] {name}: {str(e)[:120]}", flush=True)
            continue
        A, B = Xtr[:, None, :], Xte[:, None, :]
        for method, make in METHODS.items():
            if (name, method) in known:
                continue
            try:
                t = time.perf_counter()
                model = make()
                if isinstance(model, STCTClassifier):
                    pred = model.fit(Xtr, ytr).predict(Xte)
                else:
                    pred = model.fit(A, ytr).predict(B)
                seconds = time.perf_counter() - t
                row = dict(dataset=name, method=method, seconds=seconds, n_features=n_features(model, Xtr),
                           accuracy=float((pred == yte).mean()), n_train=len(Xtr), n_test=len(Xte), T=Xtr.shape[1])
            except Exception as e:
                print(f"[failed] {name} {method}: {type(e).__name__} {str(e)[:120]}", flush=True)
                continue
            pd.DataFrame([row]).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
            print(f"{name} {method} {seconds:.1f}s feats={row['n_features']} acc={row['accuracy']:.3f}", flush=True)


if __name__ == "__main__":
    main()
