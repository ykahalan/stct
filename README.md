# STCT: Short-Time Chebyshev Transform

Code for the paper **"Short-Time Chebyshev Transform: Efficient Local Polynomial Representations of Time Series"** (submitted to *Neural Processing Letters*).

STCT describes each window of a time series by the coefficients of a low-degree Chebyshev least-squares fit plus the fit residual. It is closed-form, needs no training, works with any learner, and its feature size does not depend on series length.

## Results

Same 300-tree random forest for all three representations.

| Task | Metric | Wavelet | STFT | **STCT** |
|---|---|---|---|---|
| Classification (116 UCR) | Accuracy | 0.730 | 0.755 | **0.783** |
| | Features | 566 | 598 | **138** |
| | Total time (s) | 0.64 | 0.82 | **0.57** |
| Regression (26 TSER) | RMSE | 138.83 | 151.67 | **138.33** |
| | Features | 1839 | 1955 | **134** |
| | Total time (s) | 16.59 | 18.14 | **1.51** |

## STCT+: two minimal expansions (`stct_plus.py`)

STCT's features are tied to absolute window positions, so a pattern that shifts in time changes every feature. Both expansions below start from the same step: the Chebyshev fit is *slid* along the series (stride = window/8) at four window lengths (1/32, 1/16, 1/8, 1/4 of the series), giving six channels per scale (5 coefficients + residual RMS) whose values are then summarised in a shift-invariant way. Everything stays closed-form and length-independent; the only data-dependent quantities are channel scales and bias quantiles read off the training set.

1. **STCT+ pooled** (`stct_plus(X, diff=False)`): the original fixed-window block plus 10 statistics of each channel's time course (mean, std, min, max, quartiles, proportion positive, position of max/min). 384 features, paired with a 500-tree ExtraTrees. The compact, fastest option.
2. **STCT+ kernels** (`rppv_fit` / `stct_rppv`): every window's coefficient vector is projected onto 128 random unit directions in coefficient space, i.e. 128 random local polynomial shapes, which are random convolution kernels restricted to the Chebyshev subspace and cost no extra convolution. Each kernel is summarised by the proportion of positions above five training-quantile biases plus its max (the MiniRocket pooling), and a ridge classifier is fitted on standardised features. 2880 features. With `pyramid=2` the pooling is repeated over the two halves of the series (8640 features), which adds coarse position information.

Full-archive results, default train/test splits. Reference numbers are the published results from timeseriesclassification.com for the same splits (loaded with `aeon.benchmarking.results_loaders`); STCT variants and MiniRocket were run here. Times are mean seconds per dataset for extraction + fit + predict on one shared 20-thread machine, in the same run (the machine was also running other jobs, so ratios are more reliable than absolutes).

**Classification, 112 UCR datasets common to all methods** (`results/summary_classification.md`):

| Method | Mean acc. | Mean rank (of 18) | Features | Time (s) |
|---|---|---|---|---|
| HC2 | 0.876 | 5.4 | | |
| MR-Hydra | 0.868 | 6.4 | | |
| MultiROCKET | 0.866 | 6.9 | | |
| RDST | 0.863 | 8.2 | | |
| **STCT+ kernels, pyramid (ridge)** | **0.861** | **8.3** | **8640** | **6.1 (median 1.8)** |
| WEASEL-2.0 | 0.861 | 8.9 | | |
| **STCT+ kernels (ridge)** | **0.857** | **9.3** | **2880** | **2.8 (median 1.1)** |
| MiniRocket + ridge (run here) | 0.860 | 8.8 | 9996 | 3.3 (median 0.6, max 122) |
| MiniROCKET (published) | 0.858 | 9.0 | | |
| InceptionTime | 0.858 | 8.5 | | |
| ROCKET | 0.856 | 8.5 | | |
| QUANT | 0.854 | 8.6 | | |
| **STCT+ pooled (ExtraTrees)** | **0.845** | **10.4** | **363** | **1.9 (median 1.2)** |
| FreshPRINCE | 0.842 | 10.4 | | |
| STCT (paper, RF) | 0.782 | 14.1 | 138 | 0.85 |
| Catch22 | 0.781 | 15.1 | | |
| 1NN-DTW | 0.742 | 15.8 | | |

Paired Wilcoxon tests on the 112 datasets:

- STCT+ kernels (2880 features) is statistically indistinguishable from MiniROCKET (47 wins / 13 ties / 52 losses, p = 0.66), ROCKET, InceptionTime, QUANT and WEASEL-2.0, at 29 % of MiniRocket's feature count and the same total time as our MiniRocket run. It beats FreshPRINCE (59 / 8 / 45) and the paper's STCT on 88 of 112 datasets.
- The pyramid variant wins more often than it loses against MiniROCKET (62 / 10 / 40, p = 0.17) and FreshPRINCE (66 / 10 / 36, p = 0.005), and ties ROCKET, InceptionTime, QUANT, WEASEL-2.0 and RDST. It is still significantly behind HC2 (mean gap 1.6 points, p = 0.0001), MR-Hydra (0.7 points, p = 0.001) and MultiROCKET (0.5 points, p = 0.01); its worst case is ElectricDevices, where the ridge fit on 8640 features takes about three minutes.
- STCT+ pooled is the compact option: it ties FreshPRINCE (p = 0.78) and is 1.3 points behind MiniROCKET (p = 0.002) at 363 features.

**Regression, 26 TSER datasets** (`results/summary_regression.md`; mean rank over 20 methods, since raw RMSE is not comparable across datasets):

| Method | Mean rank | Features | Time (s) |
|---|---|---|---|
| **STCT+ pooled (ExtraTrees)** | **6.3** | 348 | 3.8 |
| **STCT+ pooled with diff (ExtraTrees)** | **6.3** | 696 | 7.3 |
| MiniRocket + ridge (run here) | 6.8 | 9996 | 1.9 |
| **STCT+ pooled + 32 kernels (ridge)** | **7.0** | 1035 | 2.5 |
| **STCT+ kernels, 128 (ridge)** | **7.0** | 2747 | 2.5 |
| **STCT+ kernels, 64 (ridge)** | **7.1** | 1374 | 2.0 |
| DrCIF | 7.1 | | |
| FreshPRINCE | 8.5 | | |
| InceptionTime | 9.8 | | |
| RotF | 10.2 | | |
| ROCKET | 10.4 | | |
| STCT (paper, RF) | 10.6 | 134 | 1.5 |
| RandF, Ridge, XGBoost, GridSVR, ResNet, FPCR, 1NN-DTW, FCN | 12.5 to 16.5 | | |

Every STCT+ variant ranks above the best published method (DrCIF). STCT+ pooled beats FreshPRINCE 19 / 7, RotF 20 / 6 (p = 0.03), InceptionTime 17 / 9 and ROCKET 18 / 8, and ties DrCIF (16 / 10, p = 0.92). The kernel variants have the best median relative RMSE (1.18 to 1.19 versus 1.25 for DrCIF) in 2 to 2.5 s per dataset.

The configurations were chosen on 38 small UCR and 16 TSER datasets (`results/dev_ablations.md`, seven rounds). For the pooled recipe, more scales, degree 3 or 6, extra statistics, more trees, larger per-split feature samples, random rotations and a two-model vote were all within noise, and RotationForest or gradient boosting cost 10 to 25x the time for no gain. For the kernel recipe, accuracy rises with the number of kernels (32: 0.900, 64: 0.901, 128: 0.907, 256: 0.909 on the dev set), with the first-difference channel and with the position pyramid; more biases, more scales, degree 6 and a denser stride did not help, and trees are clearly worse than ridge on these features.

## Contents

- `stct_classification.ipynb`, `stct_regression.ipynb`: the paper's UCR / TSER benchmarks (wavelet, STFT, STCT, Rocket)
- `stct_plus.py`: STCT (`stct_fixed`, verbatim from the notebooks), the sliding channels (`stct_channels`), STCT+ pooled (`stct_pooled`, `stct_plus`, sklearn-style `STCTPlus`) and STCT+ kernels (`rppv_fit`, `stct_rppv`)
- `bench.py`: resumable benchmark harness, `<features>:<learner>` methods, UCR and TSER lists
- `compare.py`: merges a benchmark CSV with the published reference results and writes `results/summary_<task>.md`
- `ablation.py`: tabulates the dev-set ablations
- `results/`: full-archive CSVs, published reference tables, summaries, per-dataset tables

## Usage

```bash
pip install aeon pywavelets scikit-learn scipy matplotlib pandas numpy tabulate
```

Notebooks: open in Colab or Jupyter and run all cells. Datasets download automatically via `aeon`. Results are saved as `results_classification.csv` and `results_regression.csv`.

STCT+ on your own series (`X` is `(n_series, length)`):

```python
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from stct_plus import rppv_fit, stct_plus, stct_rppv

# STCT+ pooled: 384 features, ExtraTrees
clf = ExtraTreesClassifier(n_estimators=500, n_jobs=-1).fit(stct_plus(X_train, diff=False), y_train)
acc = clf.score(stct_plus(X_test, diff=False), y_test)

# STCT+ kernels: 2880 features, ridge (pyramid=2 for 8640 features)
params = rppv_fit(X_train, n_dir=128)
ridge = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)))
ridge.fit(stct_rppv(X_train, params), y_train)
acc = ridge.score(stct_rppv(X_test, params), y_test)
```

Reproduce the tables above (a few hours for classification, half an hour for regression):

```bash
python bench.py --task classification --datasets all --methods fixed:rf,fixed+pooled:et,rppv128:ridge,rppv128_p2:ridge,minirocket:ridge --out results/full_classification.csv
python compare.py --task classification --results results/full_classification.csv --methods fixed:rf,fixed+pooled:et,rppv128:ridge,rppv128_diff:ridge,rppv128_p2:ridge,minirocket:ridge --refs HC2,MR-Hydra,InceptionTime,MultiROCKET,MiniROCKET,ROCKET,FreshPRINCE,RDST,QUANT,WEASEL-2.0,Catch22,1NN-DTW
```

Default STCT setting: degree 4, two scales (8 and 16 windows). Default STCT+ setting: the same degree, pooled scales 1/32, 1/16, 1/8, 1/4 with 8x window overlap; kernels: 128 directions, 5 biases.


## Contact

Samir Brahim Belhaouari (sbelhaouari@hbku.edu.qa), Yunis Carreon Kahalan (yuka34154@hbku.edu.qa)
