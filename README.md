# STCT+: Fast, Compact Time Series Classification and Regression from Local Chebyshev Fits

STCT+ is a time series classifier and regressor in which every feature is a statistic of degree-4 Chebyshev least-squares fits on short windows. It needs no training to extract features, and its kernels take their directions and thresholds from at most 32 training series, without labels.

- **Classification**, 112 UCR datasets: 86.72% mean accuracy with 3,547 features per series and 0.15 hours of single-thread fit plus predict time in total. Of 22 published classifiers only HIVE-COTE 2 is significantly more accurate.
- **Regression**, 26 TSER datasets: the lowest median relative RMSE of all methods that do not use a pretrained foundation model, in 118 seconds in total.
- **STCT-TabPFN**, the pretrained tabular model TabPFN 2.5 on the STCT+ views: the best mean rank of 24 classifiers and the lowest error of 20 regressors.

## Method

`STCTClassifier` and `STCTRegressor` in `stct_models.py`; the features are in `stct_features.py`.

- **Views (forest input).** STCT of the series: degree-4 Chebyshev coefficients plus residual RMS in 8 and 16 windows. The same for the sorted series and for its autocorrelation. Ten statistics (mean, standard deviation, minimum, maximum, quartiles, proportion positive, positions of the maximum and minimum) of sliding fits at up to six window lengths, 1/128 to 1/4 of the series, pooled over all positions.
- **Kernels (ridge input).** Random directions in each window's fit descriptor (coefficients and residual), thresholded at quantiles 1/6 to 5/6:
  - from length 128 up, **context kernels** on pairs of neighbouring windows, **dilated kernels** on fits to 9 samples spaced 1, 2, 4, ... apart, with no weight on the level coefficient, and **winner counts**, which count how often each of 8 competing directions scores highest;
  - below length 128, the dilated kernels alone, at half the forest's weight.
- **Classifier.** ExtraTrees of 200 trees (10% of features per split) on the views, averaged with one ridge classifier per kernel block.
- **Regressor.** ExtraTrees of 50 trees (half the features per split, 2 samples per leaf) on the views, averaged from length 128 up with a ridge on threshold kernels.

```python
from stct_models import STCTClassifier, STCTRegressor

acc = STCTClassifier().fit(X_train, y_train).score(X_test, y_test)  # X: (n_series, length)
pred = STCTRegressor().fit(X_train, y_train).predict(X_test)
```

## Results

Default train/test splits. Published accuracies and RMSEs come from timeseriesclassification.com and from the papers of the later methods (`results/published_recent/`). STCT+ (seed 42), STCT-TabPFN and all runtimes were measured here, on one Intel Core i9-10900K with 32 GB of RAM. Runtime is fit on the training split plus predict on the test split, on one thread, summed over datasets, with STCT+ and the references run on each dataset in the same process.

### Classification, 112 UCR datasets

Default train/test splits, sorted by mean accuracy (%). Rank: mean rank among 24 methods on the 111 datasets common to all. W/L: datasets where STCT+ is more/less accurate. p: two-sided Wilcoxon signed-rank test against STCT+, Holm-corrected over 23 comparisons. Features: per series. Time: fit plus predict on one thread, total hours; (h): published, other hardware; ‡: timed in a later pass in which STCT+ took 429 s; §: TabPFN on the CPU with 4 threads plus STCT feature extraction; ≈: estimated from the method's own paper; GPU: run on a GPU in its paper, not timed here. †: 111 datasets.

| Method | Accuracy | Rank | W/L | p | Features | Time (h) |
|---|---|---|---|---|---|---|
| HC2 | 87.64 | 8.36 | 28/66 | 0.007 | >100,000 | (290) |
| STCT-TabPFN | 87.37 | 8.28 | 28/68 | 0.015 | 685 | 237.66 § |
| SelF-Rocket | 87.06 | 10.21 | 38/59 | 0.84 | 9,996/19,992 | ≈2.9 |
| Hydra+SelF-Rocket | 87.04 | 9.91 | 41/52 | 1.00 | SR + Hydra | ≈3.5 |
| MR-Hydra | 86.80 | 10.08 | 39/54 | 1.00 | 55,643 | 1.60 |
| MultiROCKET-TabPFN | 86.79 † | 8.76 | 34/64 | 0.34 | 49,728 | GPU |
| **STCT+** | 86.72 | 11.30 |  |  | 3,547 | 0.15 |
| MultiROCKET | 86.57 | 10.59 | 40/51 | 1.00 | 49,728 | 0.66 |
| TS-CHIEF | 86.48 | 11.77 | 52/45 | 1.00 | - | - |
| RIST | 86.44 | 10.85 | 47/48 | 1.00 | 80,645 | 10.03 ‡ |
| RDST | 86.31 | 12.11 | 55/45 | 1.00 | 30,000 | 2.24 |
| WEASEL-2.0 | 86.05 | 12.98 | 61/37 | 0.39 | 32,665 | 2.63 |
| H-InceptionTime | 85.83 | 11.36 | 46/52 | 1.00 | 640 | GPU |
| MiniROCKET | 85.80 | 13.12 | 58/35 | 0.18 | 9,996 | 0.15 |
| InceptionTime | 85.78 | 12.23 | 52/45 | 1.00 | 640 | (87) |
| ROCKET | 85.58 | 12.52 | 56/43 | 0.84 | 20,000 | 4.06 |
| QUANT | 85.44 | 12.70 | 57/37 | 0.19 | 4,993 | 0.29 |
| LITETime | 85.26 | 12.41 | 51/47 | 1.00 | 160 | GPU |
| tsfresh-TabPFN | 84.98 † | 11.08 | 44/51 | 1.00 | 777 | GPU |
| R-STSF | 84.86 | 13.84 | 63/34 | 0.013 | 14,377 | 0.62 ‡ |
| FreshPRINCE | 84.16 | 14.91 | 71/31 | <0.001 | 783 | 29.11 ‡ |
| catch22-TabPFN | 80.46 | 18.43 | 87/15 | <0.001 | 22 | GPU |
| Catch22 | 78.14 | 20.67 | 101/4 | <0.001 | 22 | 0.03 |
| 1NN-DTW | 74.21 | 21.50 | 101/5 | <0.001 | - | 25.89 ‡ |

### Regression, 26 TSER datasets

Default train/test splits, sorted by median RMSE relative to the best of the 20 methods on each dataset. Rank: mean rank. W/L: datasets where STCT+ has lower/higher RMSE. p log, p raw: two-sided Wilcoxon signed-rank tests on log and raw RMSE, Holm-corrected over 19. Features: per series (raw: the series itself). Time: fit plus predict on one thread, total hours; (h): published, other hardware; ‡: timed in a later pass, with the STCT+ total of that pass in brackets; §: TabPFN on the CPU with 4 threads plus STCT feature extraction; GPU: run on a GPU in its paper.

| Method | RMSE/best | Rank | W/L | p log | p raw | Features | Time (h) |
|---|---|---|---|---|---|---|---|
| STCT-TabPFN | 1.008 | 2.46 | 2/24 | <0.001 | <0.001 | 673 | 2.45 § |
| tsfresh-TabPFN | 1.123 | 4.12 | 6/20 | 0.026 | 0.003 | 777 | GPU |
| MultiROCKET-TabPFN | 1.129 | 4.54 | 8/18 | 0.45 | 0.043 | 49,728 | GPU |
| **STCT+** | 1.277 | 6.81 |  |  |  | 2,096 | 0.03 |
| MultiROCKET | 1.326 | 8.85 | 15/11 | 0.45 | 1.00 | 79,968 | 0.38 ‡ (91 s) |
| DrCIF | 1.338 | 8.46 | 18/8 | 0.11 | 1.00 | 86,923 | (175.4) |
| FreshPRINCE | 1.346 | 9.00 | 18/8 | 0.11 | 1.00 | 783 | 38.85 ‡ (101 s) |
| MiniROCKET | 1.378 | 8.42 | 17/9 | 0.14 | 1.00 | 9,996 | 0.03 ‡ (91 s) |
| InceptionTime | 1.379 | 10.65 | 20/6 | 0.11 | 0.45 | 640 | (60.0) |
| catch22-TabPFN | 1.393 | 9.00 | 15/11 | 0.11 | 1.00 | 22 | GPU |
| ROCKET | 1.525 | 11.27 | 19/7 | 0.016 | 0.42 | 20,000 | 2.57 |
| Ridge | 1.622 | 13.50 | 22/4 | <0.001 | 0.055 | raw | 0.00 |
| RotF | 1.650 | 10.38 | 20/6 | 0.026 | 0.16 | 1,816 | 28.30 ‡ (101 s) |
| XGBoost | 1.686 | 13.92 | 24/2 | <0.001 | 0.002 | raw | (0.6) |
| GridSVR | 1.687 | 13.73 | 22/4 | 0.001 | 0.051 | raw | 0.65 |
| RandF | 1.690 | 12.92 | 23/3 | <0.001 | 0.002 | raw | 0.49 |
| 1NN-DTW | 1.724 | 15.50 | 23/3 | <0.001 | 0.016 | - | 7.99 |
| ResNet | 1.728 | 14.73 | 23/3 | <0.001 | 0.014 | 128 | (14.1) |
| FPCR | 1.878 | 14.96 | 24/2 | <0.001 | 0.001 | FPCA | (0.1) |
| FCN | 2.942 | 16.77 | 24/2 | <0.001 | <0.001 | 128 | (9.3) |

The full tables, with the groups of methods that do not differ significantly, are in `results/table_classification.md` and `results/table_regression.md`.

## Reproduce

```bash
pip install aeon numba scikit-learn scipy pandas numpy
```

`timing_classification.py` also needs `torch` (for aeon's MR-Hydra), FreshPRINCE needs `tsfresh`, and the TabPFN scripts need `tabpfn` 2.5 and `tabpfn-extensions`. Datasets download automatically through `aeon`.

```bash
python benchmark.py --task classification --seeds 42,1,2
python benchmark.py --task regression --seeds 42,1,2
python timing_classification.py --part 0 --parts 4
python timing_classification.py --methods STCT+,R-STSF,RIST,FreshPRINCE,1NN-DTW --out results/timing_classification_later.csv
python timing_regression.py --part 0 --parts 6
python timing_regression.py --methods STCT+,MiniROCKET,MultiROCKET --out results/timing_regression_rocket.csv
python timing_regression.py --methods STCT+,RotF,FreshPRINCE --out results/timing_regression_later.csv
python tabpfn_features.py --task classification --out feats_classification
python tabpfn_features.py --task regression --out feats_regression
python tabpfn_classification.py --feats feats_classification --out results/stct_tabpfn_classification.csv --device cpu
python tabpfn_regression.py --feats feats_regression --out results/stct_tabpfn_regression.csv --device cpu
python tables.py
```

Every script appends to its results file and skips finished rows, so a run can be stopped and resumed. The timing scripts spread datasets over `--parts` processes: run every `--part` from 0 to `--parts` minus 1. STCT-TabPFN ran on the CPU with four threads; TabPFN is designed for GPUs, on which it is considerably faster.

## Contents

- `stct_features.py`: STCT, the sliding fits and pooled statistics, and the kernel blocks (compiled with Numba)
- `stct_models.py`: `STCTClassifier` and `STCTRegressor`
- `data.py`: the 116 UCR and 26 TSER datasets and their loader
- `benchmark.py`: STCT+ scores on all datasets, per seed (`results/stct_plus_<task>.csv`)
- `timing_classification.py`, `timing_regression.py`: single-thread runtimes and feature counts of STCT+ and the references run here (`results/timing_*.csv`)
- `tabpfn_features.py`, `tabpfn_classification.py`, `tabpfn_regression.py`: STCT-TabPFN (`results/stct_tabpfn_<task>.csv`)
- `tables.py`: the comparison tables (`results/table_<task>.md`)
- `results/published_*`: published accuracies, RMSEs and fit and predict times of the references
- `stct_classification.ipynb`, `stct_regression.ipynb`: the original STCT benchmarks (wavelet, STFT and STCT under the same random forest), from the paper *Short-Time Chebyshev Transform: Efficient Local Polynomial Representations of Time Series*

## Contact

Yunis Carreon Kahalan (yuka34154@hbku.edu.qa), Samir Brahim Belhaouari (sbelhaouari@hbku.edu.qa)
