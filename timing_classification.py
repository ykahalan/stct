"""Time classifiers and count their features on one thread, same machine, same pass.

    python timing_classification.py --part 0 --parts 4
    python timing_classification.py --methods STCT+,R-STSF,RIST,FreshPRINCE,1NN-DTW --out results/timing_classification_later.csv

For each dataset, STCT+ and the published methods that can run here (aeon
implementations, configured as published) are fitted on the training split and
predict the test split one after another in the same process, one thread each.
Test labels are read only to check accuracy against the published results.
Datasets are spread over --parts processes (run every --part from 0 to
--parts - 1).  Appends to --out and skips finished (dataset, method) pairs on resume.

R-STSF, RIST, FreshPRINCE and 1NN-DTW (full window) are slower and ran in a
later pass with STCT+ timed again in the same pass, so they are compared with
that STCT+ time.  HC2 and the deep networks are not run: HC2's published fit
time totals 290 hours, and the networks train on a GPU.
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
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.classification.feature_based import Catch22Classifier, FreshPRINCEClassifier
from aeon.classification.hybrid import RISTClassifier
from aeon.classification.interval_based import RSTSF, QUANTClassifier
from aeon.classification.shapelet_based import RDSTClassifier
from aeon.transformations.collection.convolution_based import MultiRocket
from aeon.transformations.collection.convolution_based._hydra import HydraTransformer
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler

import data
from stct_models import STCTClassifier

warnings.filterwarnings("ignore")
torch.set_num_threads(1)
OUT = "results/timing_classification.csv"
SEED = 42
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
    "STCT+": lambda: STCTClassifier(random_state=SEED, n_jobs=1),
    "MR-Hydra": lambda: MRHydra(random_state=SEED, n_jobs=1),
    "MultiROCKET": lambda: MultiRocketClassifier(n_kernels=MR_KERNELS, random_state=SEED, n_jobs=1),
    "RDST": lambda: RDSTClassifier(random_state=SEED, n_jobs=1),
    "WEASEL-2.0": lambda: WEASEL_V2(random_state=SEED, n_jobs=1),
    "MiniROCKET": lambda: MiniRocketClassifier(random_state=SEED, n_jobs=1),
    "ROCKET": lambda: RocketClassifier(random_state=SEED, n_jobs=1),
    "QUANT": lambda: QUANTClassifier(random_state=SEED),
    "Catch22": lambda: Catch22Classifier(catch24=False, random_state=SEED, n_jobs=1),
    "R-STSF": lambda: RSTSF(random_state=SEED, n_jobs=1),
    "RIST": lambda: RISTClassifier(random_state=SEED, n_jobs=1),
    "FreshPRINCE": lambda: FreshPRINCEClassifier(random_state=SEED, n_jobs=1),
    "1NN-DTW": lambda: KNeighborsTimeSeriesClassifier(distance="dtw", n_jobs=1),
}
SLOW = ("R-STSF", "RIST", "FreshPRINCE", "1NN-DTW")  # run only when named in --methods


def n_features(model, X):
    """Width of the representation the model's final learner sees."""
    if isinstance(model, STCTClassifier):
        return model.n_features(X)
    if isinstance(model, MRHydra):
        return model.n_features_
    if isinstance(model, KNeighborsTimeSeriesClassifier):
        return 0
    est = getattr(model, "estimator_", None) or getattr(model, "clf", None)
    if est is None:  # R-STSF, RIST and FreshPRINCE keep their final learner under a private name
        est = next((v for v in vars(model).values() if hasattr(v, "n_features_in_") or hasattr(v, "steps")), None)
    if hasattr(est, "steps"):
        est = est.steps[-1][1]
    return int(est.n_features_in_) if hasattr(est, "n_features_in_") else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--methods", default=",".join(m for m in METHODS if m not in SLOW))
    ap.add_argument("--datasets", default=",".join(data.UCR))
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    methods, out = a.methods.split(","), a.out
    known = set()
    if os.path.exists(out):
        p = pd.read_csv(out)
        known = set(zip(p.dataset, p.method))
    for i, name in enumerate(a.datasets.split(",")):
        if i % a.parts != a.part or all((name, m) in known for m in methods):
            continue
        try:
            Xtr, ytr, Xte, yte = data.load("classification", name)
        except Exception as e:
            print(f"[failed] {name}: {str(e)[:120]}", flush=True)
            continue
        A, B = Xtr[:, None, :], Xte[:, None, :]
        for method in methods:
            make = METHODS[method]
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
            pd.DataFrame([row]).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
            print(f"{name} {method} {seconds:.1f}s feats={row['n_features']} acc={row['accuracy']:.3f}", flush=True)


if __name__ == "__main__":
    main()
