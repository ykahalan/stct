"""Benchmark harness: STCT / STCT+ feature sets x learners on UCR and TSER.

    python bench.py --task classification --datasets dev --methods fixed:rf,plus:rotf
    python bench.py --task regression --datasets all --methods plus:rotf,minirocket:ridge

A method is ``<features>:<learner>``.  Results are appended to --out one row
per (dataset, method) as they finish, so a run can be stopped and resumed:
rows already present are skipped.  Features are computed once per dataset
and shared by every learner that uses them.
"""
from __future__ import annotations

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd

import stct_plus as SP

warnings.filterwarnings("ignore")
RNG = 42

# --------------------------------------------------------------------------
# dataset lists
# --------------------------------------------------------------------------

UCR = [
    "ACSF1", "Adiac", "ArrowHead", "Beef", "BeetleFly", "BirdChicken", "BME", "Car", "CBF",
    "Chinatown", "ChlorineConcentration", "CinCECGTorso", "Coffee", "Computers", "CricketX",
    "CricketY", "CricketZ", "Crop", "DiatomSizeReduction", "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxOutlineCorrect", "DistalPhalanxTW", "DodgerLoopDay", "DodgerLoopGame",
    "DodgerLoopWeekend", "Earthquakes", "ECG200", "ECG5000", "ECGFiveDays", "ElectricDevices",
    "EOGHorizontalSignal", "EOGVerticalSignal", "EthanolLevel", "FaceAll", "FaceFour",
    "FacesUCR", "FiftyWords", "Fish", "FordA", "FordB", "FreezerRegularTrain",
    "FreezerSmallTrain", "Fungi", "GunPoint", "GunPointAgeSpan", "GunPointMaleVersusFemale",
    "GunPointOldVersusYoung", "Ham", "HandOutlines", "Haptics", "Herring", "HouseTwenty",
    "InlineSkate", "InsectEPGRegularTrain", "InsectEPGSmallTrain", "InsectWingbeatSound",
    "ItalyPowerDemand", "LargeKitchenAppliances", "Lightning2", "Lightning7", "Mallat", "Meat",
    "MedicalImages", "MelbournePedestrian", "MiddlePhalanxOutlineCorrect",
    "MiddlePhalanxOutlineAgeGroup", "MiddlePhalanxTW", "MixedShapesRegularTrain",
    "MixedShapesSmallTrain", "MoteStrain", "NonInvasiveFetalECGThorax1",
    "NonInvasiveFetalECGThorax2", "OliveOil", "OSULeaf", "PhalangesOutlinesCorrect", "Phoneme",
    "PigAirwayPressure", "PigArtPressure", "PigCVP", "Plane", "PowerCons",
    "ProximalPhalanxOutlineCorrect", "ProximalPhalanxOutlineAgeGroup", "ProximalPhalanxTW",
    "RefrigerationDevices", "Rock", "ScreenType", "SemgHandGenderCh2", "SemgHandMovementCh2",
    "SemgHandSubjectCh2", "ShapeletSim", "ShapesAll", "SmallKitchenAppliances",
    "SmoothSubspace", "SonyAIBORobotSurface1", "SonyAIBORobotSurface2", "StarLightCurves",
    "Strawberry", "SwedishLeaf", "Symbols", "SyntheticControl", "ToeSegmentation1",
    "ToeSegmentation2", "Trace", "TwoLeadECG", "TwoPatterns", "UMD", "UWaveGestureLibraryAll",
    "UWaveGestureLibraryX", "UWaveGestureLibraryY", "UWaveGestureLibraryZ", "Wafer", "Wine",
    "WordSynonyms", "Worms", "WormsTwoClass", "Yoga",
]

TSER = [
    "AcousticContaminationMadrid", "AluminiumConcentration", "BoronConcentration",
    "CalciumConcentration", "CopperConcentration", "Covid19Andalusia", "Covid3Month",
    "DhakaHourlyAirQuality", "FloodModeling1", "FloodModeling2", "FloodModeling3",
    "GasSensorArrayAcetone", "GasSensorArrayEthanol", "IronConcentration",
    "LPGasMonitoringHomeActivity", "MagnesiumConcentration", "ManganeseConcentration",
    "MethaneMonitoringHomeActivity", "NaturalGasPricesSentiment", "PhosphorusConcentration",
    "PotassiumConcentration", "SodiumConcentration", "SulphurConcentration",
    "WaveDataTension", "WindTurbinePower", "ZincConcentration",
]

# small, fast, diverse subsets used only to choose the configuration
DEV_UCR = [
    "Adiac", "ArrowHead", "Beef", "BeetleFly", "BirdChicken", "BME", "Car", "CBF", "Chinatown",
    "Coffee", "CricketX", "DiatomSizeReduction", "DistalPhalanxOutlineAgeGroup", "ECG200",
    "ECGFiveDays", "FaceFour", "FacesUCR", "Fish", "GunPoint", "Ham", "Herring",
    "ItalyPowerDemand", "Lightning7", "Meat", "MedicalImages", "MoteStrain", "OliveOil",
    "OSULeaf", "Plane", "SonyAIBORobotSurface1", "SwedishLeaf", "Symbols", "SyntheticControl",
    "ToeSegmentation1", "Trace", "TwoLeadECG", "Wine", "WordSynonyms",
]
DEV_TSER = [
    "AcousticContaminationMadrid", "BoronConcentration", "CopperConcentration",
    "Covid19Andalusia", "Covid3Month", "DhakaHourlyAirQuality", "FloodModeling1",
    "FloodModeling2", "FloodModeling3", "IronConcentration", "ManganeseConcentration",
    "NaturalGasPricesSentiment", "SodiumConcentration", "WaveDataTension",
    "WindTurbinePower", "ZincConcentration",
]


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


def _prep(X):
    X = np.asarray(X)
    if X.ndim == 3:
        X = X[:, 0, :]
    if X.dtype == object or X.ndim != 2:
        raise ValueError("unequal-length series")
    X = X.astype(float)
    if np.isnan(X).any():  # DodgerLoop*, AcousticContaminationMadrid: linear interpolation
        for row in X:
            m = np.isnan(row)
            if m.all():
                row[:] = 0.0
            elif m.any():
                row[m] = np.interp(np.flatnonzero(m), np.flatnonzero(~m), row[~m])
    return X


def load(task, name):
    if task == "classification":
        from aeon.datasets import load_classification as loader
    else:
        from aeon.datasets import load_regression as loader
    Xtr, ytr = loader(name, split="train")
    Xte, yte = loader(name, split="test")
    Xtr, Xte = _prep(Xtr), _prep(Xte)
    if task == "regression":
        ytr, yte = np.asarray(ytr, float), np.asarray(yte, float)
    return Xtr, ytr, Xte, yte


# --------------------------------------------------------------------------
# feature sets: name -> fn(Xtr, Xte) -> (Ftr, Fte)
# --------------------------------------------------------------------------


def _both(fn):
    return lambda Xtr, Xte: (fn(Xtr), fn(Xte))


def feats_minirocket(Xtr, Xte):
    from aeon.transformations.collection.convolution_based import MiniRocket

    mr = MiniRocket(n_jobs=-1, random_state=RNG)
    return mr.fit_transform(Xtr[:, None, :]), mr.transform(Xte[:, None, :])


def feats_multirocket(Xtr, Xte):
    from aeon.transformations.collection.convolution_based import MultiRocket

    mr = MultiRocket(n_jobs=-1, random_state=RNG)
    return mr.fit_transform(Xtr[:, None, :]), mr.transform(Xte[:, None, :])


def _rppv(n_dir=32, n_biases=5, diff=False, pyramid=1, **kw):
    """Random polynomial kernel PPV features, fit on train; optional diff channel."""

    def fn(Xtr, Xte):
        pairs = [(Xtr, Xte)]
        if diff:
            pairs.append((np.diff(Xtr, axis=1), np.diff(Xte, axis=1)))
        ftr, fte = [], []
        for i, (A, B) in enumerate(pairs):
            P = SP.rppv_fit(A, n_dir=n_dir, n_biases=n_biases, seed=i, **kw)
            ftr.append(SP.stct_rppv(A, P, pyramid=pyramid, **kw))
            fte.append(SP.stct_rppv(B, P, pyramid=pyramid, **kw))
        return np.hstack(ftr), np.hstack(fte)

    return fn


PRIME_FRACS = (1 / 3, 1 / 5, 1 / 7, 1 / 11, 1 / 13, 1 / 19, 1 / 29)

STATS4 = ("mean", "std", "min", "max")

FEATURES = {
    # round 12: statistics per ladder level, and pooled statistics over prime segments
    "lstats": _both(lambda X: SP.stct_ladder_stats(X)),
    "lstats4": _both(lambda X: SP.stct_ladder_stats(X, stats=STATS4)),
    "msP7+lstats": _both(lambda X: np.hstack([SP.stct_pyramid(X, counts=SP.PRIMES[:7]),
                                              SP.stct_ladder_stats(X)])),
    "msP7+lstats4": _both(lambda X: np.hstack([SP.stct_pyramid(X, counts=SP.PRIMES[:7]),
                                               SP.stct_ladder_stats(X, stats=STATS4)])),
    "msP7+lstats4+pooled": _both(lambda X: np.hstack([SP.stct_pyramid(X, counts=SP.PRIMES[:7]),
                                                      SP.stct_ladder_stats(X, stats=STATS4),
                                                      SP.stct_pooled(X)])),
    "fixed+seg23": _both(lambda X: np.hstack([SP.stct_fixed(X), SP.stct_pooled(X, pyramid=(2, 3))])),
    "fixed+seg23s4": _both(lambda X: np.hstack([SP.stct_fixed(X),
                                                SP.stct_pooled(X, pyramid=(2, 3), stats=STATS4)])),
    "msP7+seg23s4": _both(lambda X: np.hstack([SP.stct_pyramid(X, counts=SP.PRIMES[:7]),
                                               SP.stct_pooled(X, pyramid=(2, 3), stats=STATS4)])),
    # round 11: non-dyadic ladders (prime, ternary, fives, sevens, dense) and prime pooled scales
    "msP7": _both(lambda X: SP.stct_pyramid(X, counts=SP.PRIMES[:7])),
    "msP10": _both(lambda X: SP.stct_pyramid(X, counts=SP.PRIMES[:10])),
    "msT4": _both(lambda X: SP.stct_pyramid(X, counts=(1, 3, 9, 27))),
    "msF3": _both(lambda X: SP.stct_pyramid(X, counts=(1, 5, 25))),
    "msS3": _both(lambda X: SP.stct_pyramid(X, counts=(1, 7, 49))),
    "msAll16": _both(lambda X: SP.stct_pyramid(X, counts=tuple(range(1, 17)))),
    "msP7+pooled": _both(lambda X: np.hstack([SP.stct_pyramid(X, counts=SP.PRIMES[:7]), SP.stct_pooled(X)])),
    "basePrimes": _both(lambda X: np.hstack([SP.stct_fixed(X), SP.stct_pooled(X, fractions=PRIME_FRACS)])),
    "pyr3Primes": _both(lambda X: np.hstack([SP.stct_fixed(X), SP.stct_pooled(X, fractions=PRIME_FRACS, pyramid=3)])),
    # round 10: pooled position pyramid + shallow dyadic multi-scale block
    "pyr4": _both(lambda X: SP.stct_plus(X, diff=False, pyramid=4)),
    "pyr3+ms5": _both(lambda X: np.hstack([SP.stct_plus(X, diff=False, pyramid=3),
                                           SP.stct_pyramid(X, levels=5)])),
    "pooledpyr3+ms5": _both(lambda X: np.hstack([SP.stct_pooled(X, pyramid=3),
                                                 SP.stct_pyramid(X, levels=5)])),
    "ms5+pooled": _both(lambda X: np.hstack([SP.stct_pyramid(X, levels=5), SP.stct_pooled(X)])),
    # round 9: overlapping windows in the pyramids
    "ms7o2": _both(lambda X: SP.stct_pyramid(X, levels=7, overlap=2)),
    "ms7o4": _both(lambda X: SP.stct_pyramid(X, levels=7, overlap=4)),
    "lap7o2": _both(lambda X: SP.stct_pyramid(X, levels=7, detail=True, overlap=2)),
    "lap7o4": _both(lambda X: SP.stct_pyramid(X, levels=7, detail=True, overlap=4)),
    "ms7o2+pooled": _both(lambda X: np.hstack([SP.stct_pyramid(X, levels=7, overlap=2), SP.stct_pooled(X)])),
    "lap7o2+pooled": _both(lambda X: np.hstack([
        SP.stct_pyramid(X, levels=7, detail=True, overlap=2), SP.stct_pooled(X)])),
    # round 8: multi-scale pyramids (fixed tiling windows at 1, 2, 4, ... windows)
    "ms5": _both(lambda X: SP.stct_pyramid(X, levels=5)),
    "ms7": _both(lambda X: SP.stct_pyramid(X, levels=7)),
    "lap5": _both(lambda X: SP.stct_pyramid(X, levels=5, detail=True)),
    "lap7": _both(lambda X: SP.stct_pyramid(X, levels=7, detail=True)),
    "ms7+pooled": _both(lambda X: np.hstack([SP.stct_pyramid(X, levels=7), SP.stct_pooled(X)])),
    "lap7+pooled": _both(lambda X: np.hstack([SP.stct_pyramid(X, levels=7, detail=True), SP.stct_pooled(X)])),
    "lap7+pooled+diff": _both(lambda X: np.hstack([
        SP.stct_pyramid(X, levels=7, detail=True), SP.stct_pooled(X),
        SP.stct_pyramid(np.diff(X, axis=1), levels=7, detail=True)])),
    # round 7: combinations at a 3k-9k feature budget
    "rppv128_diff": _rppv(128, diff=True),
    "rppv128_b9": _rppv(128, 9),
    "rppv256": _rppv(256),
    "rppv96_diff": _rppv(96, diff=True),
    "rppv128_p2": _rppv(128, pyramid=2),
    "rppv64_diff_p2": _rppv(64, diff=True, pyramid=2),
    # round 6: position pyramid on the PPV features
    "rppv32_p2": _rppv(32, pyramid=2),
    "rppv64_p2": _rppv(64, pyramid=2),
    "rppv32_p3": _rppv(32, pyramid=3),
    "rppv32_diff_p2": _rppv(32, diff=True, pyramid=2),
    # round 5: scaling the random-kernel PPV recipe
    "rppv128": _rppv(128),
    "rppv32_b9": _rppv(32, 9),
    "rppv32_s6": _rppv(32, fractions=(1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2)),
    "rppv32_d6": _rppv(32, degree=6),
    "rppv32_o16": _rppv(32, overlap=16),
    "rppv32_diff": _rppv(32, diff=True),
    "rppv64_diff": _rppv(64, diff=True),
    "rppv64_s6_diff": _rppv(64, diff=True, fractions=(1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2)),
    # paper
    "fixed": _both(lambda X: SP.stct_fixed(X)),
    # ablations
    "fixed4": _both(lambda X: SP.stct_fixed(X, window_counts=(4, 8, 16, 32))),
    "pooled": _both(lambda X: SP.stct_pooled(X)),
    "fixed+pooled": _both(lambda X: SP.stct_plus(X, diff=False)),
    "fixed+diff": _both(lambda X: SP.stct_plus(X, pooled=False)),
    "pooled+diff": _both(lambda X: SP.stct_plus(X, fixed=False)),
    # full STCT+
    "plus": _both(lambda X: SP.stct_plus(X)),
    "plus5": _both(lambda X: SP.stct_plus(X, fractions=(1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4))),
    "plus_d6": _both(lambda X: SP.stct_plus(X, degree=6)),
    "plus_d3": _both(lambda X: SP.stct_plus(X, degree=3)),
    "plus_q": _both(lambda X: SP.stct_plus(
        X, stats=SP.POOL_STATS + ("q10", "q90", "absmean", "rough"))),
    "plus_half": _both(lambda X: SP.stct_plus(
        X, fractions=(1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2))),
    "plus_f4": _both(lambda X: SP.stct_plus(X, window_counts=(4, 8, 16, 32))),
    "plus_all": _both(lambda X: SP.stct_plus(
        X, window_counts=(4, 8, 16, 32), fractions=(1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2),
        stats=SP.POOL_STATS + ("q10", "q90", "absmean", "rough"))),
    # round 3: position pyramid, PPV at training biases
    "base": _both(lambda X: SP.stct_plus(X, diff=False)),
    "pyr2": _both(lambda X: SP.stct_plus(X, diff=False, pyramid=2)),
    "pyr3": _both(lambda X: SP.stct_plus(X, diff=False, pyramid=3)),
    "pooledpyr2": _both(lambda X: SP.stct_pooled(X, pyramid=2)),
    "ppv": lambda Xtr, Xte: (lambda B: (SP.stct_ppv(Xtr, B), SP.stct_ppv(Xte, B)))(SP.ppv_biases(Xtr)),
    "base+ppv": lambda Xtr, Xte: (lambda B: (
        np.hstack([SP.stct_plus(Xtr, diff=False), SP.stct_ppv(Xtr, B)]),
        np.hstack([SP.stct_plus(Xte, diff=False), SP.stct_ppv(Xte, B)])))(SP.ppv_biases(Xtr)),
    "pyr2+ppv": lambda Xtr, Xte: (lambda B: (
        np.hstack([SP.stct_plus(Xtr, diff=False, pyramid=2), SP.stct_ppv(Xtr, B)]),
        np.hstack([SP.stct_plus(Xte, diff=False, pyramid=2), SP.stct_ppv(Xte, B)])))(SP.ppv_biases(Xtr)),
    # round 4: random polynomial kernels in coefficient space, PPV at training biases + max
    "rppv32": lambda Xtr, Xte: (lambda P: (SP.stct_rppv(Xtr, P), SP.stct_rppv(Xte, P)))(
        SP.rppv_fit(Xtr, n_dir=32)),
    "rppv64": lambda Xtr, Xte: (lambda P: (SP.stct_rppv(Xtr, P), SP.stct_rppv(Xte, P)))(
        SP.rppv_fit(Xtr, n_dir=64)),
    "base+rppv32": lambda Xtr, Xte: (lambda P: (
        np.hstack([SP.stct_plus(Xtr, diff=False), SP.stct_rppv(Xtr, P)]),
        np.hstack([SP.stct_plus(Xte, diff=False), SP.stct_rppv(Xte, P)])))(SP.rppv_fit(Xtr, n_dir=32)),
    "base+rppv64": lambda Xtr, Xte: (lambda P: (
        np.hstack([SP.stct_plus(Xtr, diff=False), SP.stct_rppv(Xtr, P)]),
        np.hstack([SP.stct_plus(Xte, diff=False), SP.stct_rppv(Xte, P)])))(SP.rppv_fit(Xtr, n_dir=64)),
    # references
    "raw": _both(lambda X: X),
    "minirocket": feats_minirocket,
    "multirocket": feats_multirocket,
}


# --------------------------------------------------------------------------
# learners
# --------------------------------------------------------------------------


class RandomRotation:
    """Append a fixed random orthonormal rotation of the features (oblique splits for trees)."""

    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, X, y=None):
        d = X.shape[1]
        G = np.random.default_rng(self.seed).standard_normal((d, d))
        self.R_, _ = np.linalg.qr(G)
        return self

    def transform(self, X):
        return np.hstack([X, X @ self.R_])

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_params(self, deep=True):
        return {"seed": self.seed}

    def set_params(self, **p):
        self.seed = p.get("seed", self.seed)
        return self


def make_learner(name, task):
    from sklearn.ensemble import (
        ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier,
        HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor,
    )
    from sklearn.linear_model import RidgeClassifierCV, RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cls = task == "classification"
    if name == "rf":  # the paper's learner
        est = (RandomForestClassifier if cls else RandomForestRegressor)(
            n_estimators=300, random_state=RNG, n_jobs=-1)
        return make_pipeline(StandardScaler(), est)
    if name in ("et", "et1k", "et_mf", "et_mf25"):
        mf = {"et_mf": 0.1, "et_mf25": 0.25}.get(name, "sqrt" if cls else 1.0)
        est = (ExtraTreesClassifier if cls else ExtraTreesRegressor)(
            n_estimators=1000 if name == "et1k" else 500, max_features=mf,
            random_state=RNG, n_jobs=-1)
        return make_pipeline(StandardScaler(), est)
    if name == "hgb":
        est = (HistGradientBoostingClassifier if cls else HistGradientBoostingRegressor)(
            random_state=RNG)
        return est
    if name == "ridge":
        alphas = np.logspace(-3, 3, 10)
        est = RidgeClassifierCV(alphas=alphas) if cls else RidgeCV(alphas=alphas)
        return make_pipeline(StandardScaler(), est)
    if name == "etrot":  # ExtraTrees on [F, F R] with a fixed random rotation R
        est = (ExtraTreesClassifier if cls else ExtraTreesRegressor)(
            n_estimators=500, random_state=RNG, n_jobs=-1)
        return make_pipeline(StandardScaler(), RandomRotation(RNG), est)
    if name == "vote":  # soft vote of ExtraTrees-500 and RandomForest-300
        from sklearn.ensemble import VotingClassifier, VotingRegressor

        et = (ExtraTreesClassifier if cls else ExtraTreesRegressor)(
            n_estimators=500, random_state=RNG, n_jobs=-1)
        rf = (RandomForestClassifier if cls else RandomForestRegressor)(
            n_estimators=300, random_state=RNG, n_jobs=-1)
        if cls:
            est = VotingClassifier([("et", et), ("rf", rf)], voting="soft")
        else:
            est = VotingRegressor([("et", et), ("rf", rf)])
        return make_pipeline(StandardScaler(), est)
    if name == "rotf":
        if cls:
            from aeon.classification.sklearn import RotationForestClassifier as RotF
        else:
            from aeon.regression.sklearn import RotationForestRegressor as RotF
        return RotF(n_estimators=200, n_jobs=-1, random_state=RNG)
    raise ValueError(name)


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------


def score(task, y_true, y_pred):
    if task == "classification":
        from sklearn.metrics import accuracy_score

        return {"accuracy": accuracy_score(y_true, y_pred)}
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


_BASE = ["dataset", "n_train", "n_test", "length", "method", "features", "learner"]
_TAIL = ["n_features", "extract_time_s", "fit_time_s", "predict_time_s", "error"]
COLUMNS = {"classification": _BASE + ["accuracy"] + _TAIL,
           "regression": _BASE + ["rmse", "mae", "r2"] + _TAIL}


def run(task, datasets, methods, out, max_train=None, max_len=None):
    done = set()
    if os.path.exists(out):
        prev = read_results(out, task)  # failed pairs are retried on resume
        done = set(zip(prev["dataset"], prev["method"]))
    t_all = time.time()
    for di, name in enumerate(datasets):
        todo = [m for m in methods if (name, m) not in done]
        if not todo:
            continue
        try:
            Xtr, ytr, Xte, yte = load(task, name)
        except Exception as e:  # ragged datasets
            print(f"[skip] {name}: {e}")
            continue
        if max_train and len(Xtr) > max_train or max_len and Xtr.shape[1] > max_len:
            print(f"[skip] {name}: too large for this run")
            continue
        cache = {}
        t_ds = time.time()
        for m in todo:
            fname, lname = m.split(":")
            try:
                if fname not in cache:
                    t0 = time.time()
                    cache[fname] = FEATURES[fname](Xtr, Xte) + (time.time() - t0,)
                Ftr, Fte, t_ext = cache[fname]
                Ftr = np.nan_to_num(Ftr); Fte = np.nan_to_num(Fte)
                est = make_learner(lname, task)
                t0 = time.time(); est.fit(Ftr, ytr); t_fit = time.time() - t0
                t0 = time.time(); pred = est.predict(Fte); t_pred = time.time() - t0
                res = score(task, yte, pred)
                row = {"dataset": name, "n_train": len(Xtr), "n_test": len(Xte),
                       "length": Xtr.shape[1], "method": m, "features": fname, "learner": lname,
                       **res, "n_features": Ftr.shape[1], "extract_time_s": t_ext,
                       "fit_time_s": t_fit, "predict_time_s": t_pred}
            except Exception as e:
                print(f"  [warn] {name} / {m} failed: {e}")
                res = {}
                row = {"dataset": name, "n_train": len(Xtr), "n_test": len(Xte),
                       "length": Xtr.shape[1], "method": m, "features": fname, "learner": lname,
                       "error": str(e)[:200]}
            cols = COLUMNS[task]
            pd.DataFrame([{c: row.get(c, np.nan) for c in cols}], columns=cols).to_csv(
                out, mode="a", header=not os.path.exists(out), index=False)
            main = res.get("accuracy", res.get("rmse")) if "error" not in row else float("nan")
            print(f"  {m:22s} {main:10.4f}  feats={row.get('n_features', '-')}", flush=True)
        print(f"[{di + 1}/{len(datasets)}] {name:28s} {time.time() - t_ds:6.1f}s "
              f"(elapsed {time.time() - t_all:7.1f}s)", flush=True)


def read_results(path, task):
    """Rows with a numeric score only (drops failed pairs, in any CSV layout)."""
    key = "accuracy" if task == "classification" else "rmse"
    df = pd.read_csv(path)
    df[key] = pd.to_numeric(df[key], errors="coerce")
    df = df[df[key].notna()].copy()
    for c in ["n_features", "extract_time_s", "fit_time_s", "predict_time_s"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def summarize(out, task):
    df = read_results(out, task)
    key = "accuracy" if task == "classification" else "rmse"
    piv = df.pivot_table(index="dataset", columns="method", values=key)
    piv = piv.dropna()
    ranks = piv.rank(axis=1, ascending=(task != "classification"))
    summ = pd.DataFrame({"mean": piv.mean(), "mean_rank": ranks.mean(),
                         "wins": (ranks == 1).sum()})
    if task == "regression":  # scale-free view
        rel = piv.div(piv.min(axis=1), axis=0)
        summ["mean_rel_rmse"] = rel.mean()
    feats = df.pivot_table(index="dataset", columns="method", values="n_features").mean()
    summ["n_features"] = feats
    tt = df.assign(total=df[["extract_time_s", "fit_time_s", "predict_time_s"]].sum(axis=1))
    summ["time_s"] = tt.pivot_table(index="dataset", columns="method", values="total").mean()
    print(f"\n{len(piv)} datasets with complete results\n")
    print(summ.sort_values("mean_rank").round(4).to_string())
    return summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["classification", "regression"], required=True)
    ap.add_argument("--datasets", default="dev")
    ap.add_argument("--methods", default="fixed:rf,plus:rotf")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    if a.datasets == "dev":
        ds = DEV_UCR if a.task == "classification" else DEV_TSER
    elif a.datasets == "all":
        ds = UCR if a.task == "classification" else TSER
    else:
        ds = a.datasets.split(",")
    if not a.summary_only:
        run(a.task, ds, a.methods.split(","), a.out, a.max_train, a.max_len)
    summarize(a.out, a.task)
