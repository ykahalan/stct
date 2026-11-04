"""Benchmark harness: STCT, STCT+ and baseline features x learners on UCR and TSER.

    python bench.py --task classification --datasets all --methods fixed:rf,msP7+pooled:et
    python bench.py --task regression --datasets all --methods fixed:et,minirocket:ridge

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


def wavelet_features(X, wavelet="db4", level=None):
    """Raw DWT coefficients, concatenated across levels (verbatim from the paper notebooks)."""
    import pywt

    feats = []
    for row in X:
        max_level = pywt.dwt_max_level(len(row), pywt.Wavelet(wavelet).dec_len)
        lvl = level or min(4, max_level)
        coeffs = pywt.wavedec(row, wavelet, level=lvl)
        feats.append(np.concatenate([np.asarray(c) for c in coeffs]))
    return np.vstack(feats)


def stft_features(X, nperseg=32, noverlap=None):
    """Raw STFT magnitude spectrogram, flattened (verbatim from the paper notebooks)."""
    from scipy.signal import stft

    noverlap = noverlap or nperseg // 2
    feats = []
    for row in X:
        seg = min(nperseg, len(row))
        _, _, Zxx = stft(row, nperseg=seg, noverlap=min(noverlap, seg - 1))
        feats.append(np.abs(Zxx).ravel())
    return np.vstack(feats)


FEATURES = {
    "fixed": _both(SP.stct_fixed),  # the paper's STCT
    "msP7+pooled": _both(SP.stct_plus),  # STCT+ before the algorithm selection
    "stct_plus": _both(lambda X: SP.stct_plus(X, orthonormal=True)),  # STCT+ as selected by algo_select.py
    "paper_pooled": _both(lambda X: SP.stct_plus(X, paper=True)),  # paper features + pooled statistics
    "paper_plus": _both(lambda X: SP.stct_plus(X, paper=True, resid_kind="peak", global_q=True)),  # as selected
    "paper_tuned": _both(lambda X: SP.stct_plus(X, paper=True, fractions=(1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4))),  # chosen by param_search.py
    "paper_views": _both(SP.stct_views),  # chosen by ablate.py
    "wavelet": _both(wavelet_features),
    "stft": _both(stft_features),
    "minirocket": feats_minirocket,
}


# --------------------------------------------------------------------------
# learners
# --------------------------------------------------------------------------


def make_learner(name, task):
    from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier,
                                  RandomForestRegressor)
    from sklearn.linear_model import RidgeClassifierCV, RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cls = task == "classification"
    if name == "rf":  # the paper's learner
        est = (RandomForestClassifier if cls else RandomForestRegressor)(
            n_estimators=300, random_state=RNG, n_jobs=-1)
        return make_pipeline(StandardScaler(), est)
    if name == "et":
        est = (ExtraTreesClassifier if cls else ExtraTreesRegressor)(
            n_estimators=500, random_state=RNG, n_jobs=-1)
        return make_pipeline(StandardScaler(), est)
    if name == "et_tuned":  # chosen by param_search.py
        est = (ExtraTreesClassifier if cls else ExtraTreesRegressor)(
            n_estimators=500, max_features=0.1 if cls else 1.0, random_state=RNG, n_jobs=-1)
        return make_pipeline(StandardScaler(), est)
    if name == "ridge":
        alphas = np.logspace(-3, 3, 10)
        est = RidgeClassifierCV(alphas=alphas) if cls else RidgeCV(alphas=alphas)
        return make_pipeline(StandardScaler(), est)
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
    ap.add_argument("--methods", default="fixed:rf,msP7+pooled:et")
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
