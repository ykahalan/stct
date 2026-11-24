"""Benchmark datasets: 116 UCR classification and 26 TSER regression datasets, default splits.

The UCR list is the 128 univariate archive datasets without its 11 variable-length
datasets and MelbournePedestrian, which the aeon loader cannot read as an
equal-length array.  Missing values are linearly interpolated.
"""
import numpy as np
import pandas as pd

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
    "MedicalImages", "MiddlePhalanxOutlineCorrect",
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

DATASETS = {"classification": UCR, "regression": TSER}


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
    """Default train and test splits of one dataset: Xtr, ytr, Xte, yte."""
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


def rmse(y, pred):
    return float(np.sqrt(np.mean((np.asarray(y, float) - pred) ** 2)))


def done_keys(path, keys):
    """Key tuples already in a results CSV, so a run can resume."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return set()
    return set(map(tuple, df[keys].astype(str).values))
