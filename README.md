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

## STCT+: learner and feature expansions (`stct_plus.py`)

**The simplest improvement is the learner.** Keeping the paper's exact features and replacing the 300-tree random forest with a 500-tree ExtraTrees is more accurate on both tasks and faster on both. On regression this alone is enough to tie the best published method.

Three closed-form, length-independent feature expansions were then tested. All keep the paper's building block (degree-4 Chebyshev least-squares coefficients plus residual RMS per window). None is trained; the only data-dependent quantities are channel scales and bias quantiles read off the training set.

1. **Pooled** (`stct_pooled`, `stct_plus`): the Chebyshev fit slid along the series (stride = window/8) at window lengths 1/32, 1/16, 1/8, 1/4 of the series, with each coefficient channel's time course collapsed into 10 shift-invariant statistics (240 features). Added to the paper's fixed-window block, this is the recommended compact classifier and gives the largest feature gain.
2. **Multi-scale ladder** (`stct_pyramid`): the paper's windows at more than two scales. The prime ladder 1, 2, 3, 5, 7, 11, 13 (252 features) beat the dyadic ladder. One possible reason is that prime counts never share a window boundary, whereas dyadic levels nest, but this was not tested directly. The gain is real but small, and it disappears once the pooled block is present.
3. **Kernels** (`rppv_fit` / `stct_rppv`): every sliding window's coefficient vector projected onto 128 random unit directions in coefficient space (random local polynomial shapes, with no extra convolution), pooled as proportion-above-bias at five training quantiles plus max, with a ridge classifier. 2880 features; `pyramid=2` repeats the pooling over the two halves (8640 features). The most accurate classifier here, but not compact.

All results use the default train/test splits and one seed. Reference numbers are the published results from timeseriesclassification.com for the same splits (loaded with `aeon.benchmarking.results_loaders`); STCT variants and MiniRocket were run here. Times are mean seconds per dataset for extraction + fit + predict on one shared 20-thread machine that was also running other jobs, so ratios are more reliable than absolutes.

**Held-out datasets.** Configurations were chosen on 38 UCR and 16 TSER datasets, so those datasets are optimistic for STCT. The tables therefore also report the datasets never used for tuning (`results/heldout_<task>.md`, produced by `heldout.py`). Each recipe there is ranked alone against the published methods, because ranking several similar STCT variants together pushes the references down the table.

### Classification

Mean accuracy on the 112 UCR datasets common to all methods, and on the 74 of those not used for tuning:

| Method | All (112) | Held out (74) | Features | Time (s) |
|---|---|---|---|---|
| HC2 | 0.876 | 0.857 | | |
| MR-Hydra | 0.868 | 0.842 | | |
| MultiROCKET | 0.866 | 0.841 | | |
| RDST | 0.863 | 0.839 | | |
| InceptionTime | 0.858 | 0.836 | | |
| QUANT | 0.854 | 0.833 | | |
| MiniRocket + ridge (run here) | 0.860 | 0.833 | 9996 | 3.3 |
| WEASEL-2.0 | 0.861 | 0.832 | | |
| **STCT+ kernels, pyramid (ridge)** | 0.861 | 0.832 | 8640 | 6.1 |
| MiniROCKET (published) | 0.858 | 0.831 | | |
| **STCT+ kernels (ridge)** | 0.857 | 0.830 | 2880 | 2.8 |
| ROCKET | 0.856 | 0.830 | | |
| FreshPRINCE | 0.842 | 0.824 | | |
| **STCT+ pooled + paper features (ExtraTrees)** | **0.845** | **0.823** | **363** | **1.9** |
| Catch22 | 0.781 | 0.775 | | |
| **STCT prime ladder (ExtraTrees)** | 0.807 | 0.773 | 242 | 1.1 |
| **STCT paper features (ExtraTrees)** | 0.797 | 0.767 | 138 | 0.64 |
| STCT paper features (paper, RF) | 0.782 | 0.756 | 138 | 0.85 |
| 1NN-DTW | 0.742 | 0.712 | | |

What each step contributes, paired over all 116 datasets (better / worse), with the held-out result in brackets:

| Change | Mean accuracy | Better / worse | Wilcoxon p | Held out (better / worse, p) |
|---|---|---|---|---|
| RF to ExtraTrees, same features | 0.783 to 0.799 | 85 / 18 | < 0.0001 | 55 / 15, < 0.0001 |
| Add ladder (paper to prime) | 0.799 to 0.809 | 71 / 22 | < 0.0001 | 47 / 18, 0.008 |
| Add pooled block to paper features | 0.799 to 0.846 | 80 / 20 | < 0.0001 | 55 / 14, < 0.0001 |
| Prime ladder vs dyadic ladder | 0.804 to 0.809 | 61 / 35 | 0.006 | 40 / 26, 0.07 |
| Prime ladder vs paper scales, with pooled block | 0.846 to 0.846 | 43 / 44 | 0.84 | 29 / 35, 0.27 |

STCT does not beat the state of the art on classification. Every recipe is significantly behind HC2 and MR-Hydra, on all datasets and on held-out ones (p <= 0.002). The kernel recipes are level with the ROCKET tier and the pooled recipe with FreshPRINCE, at far fewer features than MiniRocket. On held-out datasets MiniRocket remains marginally ahead of both kernel recipes.

### Regression

Each recipe ranked alone against 13 published regressors on the 26 TSER datasets. Better / worse counts are on RMSE; the held-out set has only 10 datasets.

| Method | Median RMSE vs best | Features | Time (s) | vs DrCIF | vs FreshPRINCE | Held out vs DrCIF | Held out vs FreshPRINCE |
|---|---|---|---|---|---|---|---|
| **STCT dyadic ladder + pooled (ExtraTrees)** | 1.13 | 378 | 4.3 | 17 / 9, p = 0.64 | 21 / 5, p = 0.01 | 5 / 5 | 9 / 1 |
| **STCT+ pooled + paper features (ExtraTrees)** | 1.18 | 348 | 3.8 | 16 / 10, p = 0.92 | 19 / 7, p = 0.09 | 5 / 5 | 10 / 0 |
| **STCT prime ladder (ExtraTrees)** | 1.23 | 230 | 1.9 | 12 / 14, p = 1.00 | 19 / 7, p = 0.02 | 4 / 6 | 10 / 0 |
| **STCT dyadic ladder (ExtraTrees)** | 1.25 | 163 | 1.5 | 11 / 15, p = 0.62 | 19 / 7, p = 0.01 | 4 / 6 | 10 / 0 |
| **STCT paper features (ExtraTrees)** | 1.25 | 134 | 1.3 | 11 / 15, p = 0.22 | 18 / 8, p = 0.03 | 5 / 5 | 10 / 0 |
| STCT+ kernels (ridge) | 1.18 | 2747 | 2.5 | 13 / 13, p = 0.44 | 17 / 9, p = 0.42 | 3 / 7 | 6 / 4 |
| MiniRocket + ridge (run here) | 1.23 | 9996 | 1.9 | 15 / 11, p = 0.60 | 15 / 11, p = 0.52 | 5 / 5 | 7 / 3 |
| STCT paper features (paper, RF) | 1.35 | 134 | 1.5 | 5 / 21, p = 0.001 | 8 / 18, p = 0.04 | 1 / 9 | 4 / 6 |

STCT with ExtraTrees is statistically tied with DrCIF, the strongest published regressor here, and ahead of FreshPRINCE, at 134 features and under 1.3 s per dataset. None of these comparisons is corrected for multiple testing, and no recipe is significantly better than DrCIF.

The feature expansions add no significant accuracy on regression over the paper's features with ExtraTrees (adding the pooled block: 14 / 12, p = 0.86; dyadic ladder: 16 / 10, p = 0.23). The published AcousticContaminationMadrid results use a version with missing values removed, while this code interpolates them, so that one dataset is not perfectly matched.

### What did not help

Tested on the tuning datasets over twelve rounds (`results/dev_ablations.md`): RotationForest and gradient boosting (10 to 25x the time), more trees, larger per-split feature samples, random rotations, a two-model vote, degree 3 or 6, more pooled scales or statistics, the first-difference channel, residual (detail) pyramids, overlapping windows, ternary / fives / sevens ladders, prime-length pooled scales, statistics across the windows of each ladder level, and pooled statistics over prime segments. A pooled position pyramid (1713 features) gained a point on the tuning datasets but not on the full archive (0.846).

## Contents

- `stct_classification.ipynb`, `stct_regression.ipynb`: the paper's UCR / TSER benchmarks (wavelet, STFT, STCT, Rocket)
- `stct_plus.py`: STCT (`stct_fixed`, verbatim from the notebooks), the multi-scale ladder (`stct_pyramid`, `PRIMES`, `stct_ladder_stats`), the sliding channels (`stct_channels`), the pooled features (`stct_pooled`, `stct_plus`, sklearn-style `STCTPlus`) and the kernels (`rppv_fit`, `stct_rppv`)
- `bench.py`: resumable benchmark harness, `<features>:<learner>` methods, UCR and TSER lists
- `compare.py`: merges a benchmark CSV with the published reference results and writes `results/summary_<task>.md`
- `heldout.py`: ranks one recipe at a time against the published methods, on all and on held-out datasets
- `ablation.py`: tabulates the tuning-set ablations
- `results/`: full-archive CSVs, published reference tables, summaries, per-dataset tables

## Usage

```bash
pip install aeon pywavelets scikit-learn scipy matplotlib pandas numpy tabulate
```

Notebooks: open in Colab or Jupyter and run all cells. Datasets download automatically via `aeon`. Results are saved as `results_classification.csv` and `results_regression.csv`.

STCT+ on your own series (`X` is `(n_series, length)`):

```python
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from stct_plus import rppv_fit, stct_fixed, stct_plus, stct_rppv

# regression: the paper's features with ExtraTrees (about 134 features)
reg = ExtraTreesRegressor(n_estimators=500, n_jobs=-1).fit(stct_fixed(X_train), y_train)
pred = reg.predict(stct_fixed(X_test))

# compact classification: paper features + pooled statistics (about 363 features)
clf = ExtraTreesClassifier(n_estimators=500, n_jobs=-1).fit(stct_plus(X_train, diff=False), y_train)
acc = clf.score(stct_plus(X_test, diff=False), y_test)

# most accurate classification: 128 random polynomial kernels (2880 features), ridge
params = rppv_fit(X_train, n_dir=128)
ridge = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)))
ridge.fit(stct_rppv(X_train, params), y_train)
acc = ridge.score(stct_rppv(X_test, params), y_test)
```

Reproduce the tables above (a few hours for classification, half an hour for regression):

```bash
python bench.py --task classification --datasets all --methods fixed:rf,fixed:et,msP7:et,fixed+pooled:et,rppv128:ridge,rppv128_p2:ridge,minirocket:ridge --out results/full_classification.csv
python bench.py --task regression --datasets all --methods fixed:rf,fixed:et,ms5:et,msP7:et,fixed+pooled:et,ms5+pooled:et,rppv128:ridge,minirocket:ridge --out results/full_regression.csv
python heldout.py
```

Default STCT setting: degree 4, two scales (8 and 16 windows). STCT+ defaults: the same degree; ladder counts 1, 2, 3, 5, 7, 11, 13; pooled scales 1/32, 1/16, 1/8, 1/4 with 8x window overlap; kernels: 128 directions, 5 biases.


## Contact

Samir Brahim Belhaouari (sbelhaouari@hbku.edu.qa), Yunis Carreon Kahalan (yuka34154@hbku.edu.qa)
