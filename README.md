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

## STCT+ (`stct_plus.py`)

STCT+ builds on the paper's features and uses a 500-tree ExtraTrees. It is closed-form and training-free, and its feature count does not depend on series length (at most 458, mean 434 on UCR).

- **Paper features:** degree-4 Chebyshev coefficients plus residual RMS in 8 and 16 windows.
- **Pooled statistics:** the same fit is slid along the series (stride = window/8) at window lengths 1/32, 1/16, 1/8 and 1/4 of the series, and each channel's time course is collapsed into 10 shift-invariant statistics.
- **Residual peak:** the largest residual relative to the residual RMS, added as one more channel in both blocks.
- **Global deciles:** deciles of the whole series and of its residual after a whole-series fit.

### How changes were chosen

`algo_select.py` scored each candidate by 5-fold cross-validation inside the training splits of all datasets, using a rule fixed in advance, and never loaded a test split. A candidate passes if it improves one task without hurting the other. Classification needs at least +0.2 accuracy points with more datasets better than worse, and a regression RMSE ratio of at most 1.002. Regression needs a ratio of at most 0.995 with more datasets better than worse, and an accuracy loss of at most 0.05 points.

First-pass results against paper features + pooled:

| Candidate | Accuracy change (points) | Better / worse | RMSE ratio | Passed |
|---|---|---|---|---|
| Residual sign changes | +0.28 | 45 / 35 | 1.004 | No |
| Residual peak | +0.21 | 44 / 35 | 0.996 | **Yes, adopted first** |
| Cross-channel correlations | +0.19 | 44 / 25 | 1.003 | No |
| Add a 16-sample scale | +0.17 | 34 / 31 | 1.003 | No |
| Global deciles | +0.14 | 37 / 34 | 0.976 | **Yes, adopted second** |
| Replace 1/32 and 1/16 with 8 and 32 samples | +0.14 | 41 / 40 | 1.010 | No |
| Orthonormal coefficients | +0.13 | 47 / 28 | 1.002 | No |
| Shape normalization | -0.01 | 41 / 44 | 1.036 | No |
| Degree grows with window length | -0.02 | 38 / 46 | 0.997 | No |
| Log-periodogram ladder | -0.07 | 33 / 36 | 1.006 | No |
| Energy per polynomial degree | -0.11 | 38 / 35 | 1.002 | No |
| Quartiles per window | -0.12 | 40 / 39 | 0.983 | No |
| Second-order STCT | -0.34 | 40 / 44 | 0.993 | No |
| Quartiles per ladder level | -0.34 | 30 / 51 | 0.990 | No |
| Z-normalization | -0.86 | 16 / 26 | 1.396 | No |

**Noise:** rescoring the same recipe with two other seeds changed training accuracy by -0.11 and -0.25 points, so gains of this size can be chance. Global deciles were adopted with accuracy unchanged and an RMSE ratio of 0.975 on top of residual peak.

**Limit of this protocol:** paper features + pooled was itself chosen after seeing test scores, including on 38 UCR and 16 TSER datasets used during development. The tables therefore also report the remaining datasets (`results/heldout_<task>.md`).

### Results

All results use the default train/test splits and one seed. Reference numbers are the published results from timeseriesclassification.com for the same splits; STCT variants and MiniRocket were run here.

**Classification** (mean accuracy on the 112 UCR datasets common to all methods, and on 74 of those outside the development set; features are the mean over 116 datasets):

| Method | All (112) | Excluding development (74) | Features |
|---|---|---|---|
| HC2 | 0.876 | 0.857 | |
| MR-Hydra | 0.868 | 0.842 | |
| MultiROCKET | 0.866 | 0.841 | |
| RDST | 0.863 | 0.839 | |
| WEASEL-2.0 | 0.861 | 0.832 | |
| MiniRocket + ridge (run here) | 0.860 | 0.833 | 9996 |
| MiniROCKET (published) | 0.858 | 0.831 | |
| InceptionTime | 0.858 | 0.836 | |
| ROCKET | 0.856 | 0.830 | |
| QUANT | 0.854 | 0.833 | |
| **STCT+ (ExtraTrees)** | **0.847** | **0.825** | **434** |
| Paper features + pooled (ExtraTrees) | 0.845 | 0.822 | 359 |
| FreshPRINCE | 0.842 | 0.824 | |
| Paper features (ExtraTrees) | 0.797 | 0.767 | 138 |
| Paper features (paper, RF) | 0.782 | 0.756 | 138 |
| Catch22 | 0.781 | 0.775 | |
| 1NN-DTW | 0.742 | 0.712 | |

STCT+ does not beat the state of the art on classification. It is statistically tied with FreshPRINCE (55 / 43, p = 0.86) and significantly behind every method above it across all datasets, including HC2 (15 / 79) and MR-Hydra (22 / 75). Its gain over paper features + pooled is not significant (51 / 38, p = 0.33). The pooled block is the main gain over paper features (80 / 22, p < 0.0001).

**Regression** (26 TSER datasets, each recipe ranked alone against 13 published regressors; better / worse on RMSE; the held-out set has 10 datasets):

| Method | Mean rank (of 14) | Features | vs DrCIF | vs FreshPRINCE | Held out vs DrCIF | Held out vs FreshPRINCE |
|---|---|---|---|---|---|---|
| **STCT+ (ExtraTrees)** | **3.4** | **417** | 15 / 11, p = 1.00 | 22 / 4, p = 0.001 | 3 / 7 | 10 / 0 |
| Paper features + pooled (ExtraTrees) | 3.7 | 345 | 15 / 11, p = 0.94 | 19 / 7, p = 0.07 | 4 / 6 | 10 / 0 |
| MiniRocket + ridge (run here) | 4.1 | 9996 | 15 / 11, p = 0.60 | 15 / 11, p = 0.52 | 5 / 5 | 7 / 3 |
| Paper features (ExtraTrees) | 4.5 | 134 | 11 / 15, p = 0.22 | 18 / 8, p = 0.03 | 5 / 5 | 10 / 0 |
| Paper features (paper, RF) | 6.0 | 134 | 5 / 21, p = 0.001 | 8 / 18, p = 0.04 | 1 / 9 | 4 / 6 |

STCT+ is statistically tied with DrCIF, the strongest published regressor here, and beats FreshPRINCE. Its gain over paper features + pooled is not significant (14 / 12, p = 0.39), and neither is the gain of the pooled block over paper features (14 / 12, p = 0.88). These tests are not corrected for multiple comparisons. The published AcousticContaminationMadrid results use a version with missing values removed, while this code interpolates them.

**Same-learner comparison with the paper's representations:** ExtraTrees helps wavelet and STFT as well, and STCT still beats both.

| Representation | Accuracy, RF | Accuracy, ExtraTrees | RMSE, RF | RMSE, ExtraTrees |
|---|---|---|---|---|
| Wavelet | 0.731 | 0.751 | 138.81 | 131.89 |
| STFT | 0.755 | 0.767 | 151.66 | 145.29 |
| Paper features | 0.783 | 0.799 | 138.32 | 113.55 |

The random forest columns reproduce the paper's table. Under ExtraTrees, paper features beat wavelet 89 / 16 and STFT 75 / 31 on classification, and 19 / 7 and 20 / 6 on regression (p < 0.02 for all four).

### Single-thread timing

`single_thread.py` measures extraction, fit and predict on one thread on this machine. The machine was also running other jobs, and STCT+ and paper features + pooled were timed in a later run than the other methods, so only large gaps are meaningful.

| Method | Task | Features | Mean (s) | Median (s) | Max (s) |
|---|---|---|---|---|---|
| **STCT+ (500 trees)** | Classification, 116 UCR | 434 | 1.6 | 0.8 | 12 |
| Paper features + pooled (500 trees) | Classification, 116 UCR | 359 | 1.2 | 0.7 | 11 |
| QUANT, aeon (200 trees) | Classification, 116 UCR | 4909 | 12.2 | 2.5 | 155 |
| MiniRocket + ridge | Classification, 116 UCR | 9996 | 8.2 | 0.9 | 404 |
| **STCT+ (500 trees)** | Regression, 26 TSER | 417 | 24.2 | 16.6 | 77 |
| Paper features + pooled (500 trees) | Regression, 26 TSER | 345 | 19.2 | 13.0 | 63 |
| MiniRocket + ridge | Regression, 26 TSER | 9996 | 5.0 | 3.3 | 19 |
| Paper features (500 trees) | Regression, 26 TSER | 134 | 11.3 | 7.7 | 39 |

Published QUANT totals are lower than this aeon run (9.6 minutes against 23.6 on 112 datasets, on different hardware), so the comparison with QUANT depends on the implementation. On regression, published mean fit times are about 17,209 s for DrCIF and 3,340 s for FreshPRINCE, far above STCT+ despite the hardware difference.

## Contents

- `stct_classification.ipynb`, `stct_regression.ipynb`: the paper's UCR / TSER benchmarks (wavelet, STFT, STCT, Rocket)
- `stct_plus.py`: STCT (`stct_fixed`, verbatim logic from the notebooks), `stct_pooled`, `stct_plus`, sklearn-style `STCTPlus`, and the tested candidate options
- `algo_select.py`: chooses changes using training splits only
- `single_thread.py`: times STCT+ and references on one thread
- `bench.py`: resumable benchmark harness, `<features>:<learner>` methods, UCR and TSER lists
- `compare.py`: merges a benchmark CSV with the published reference results and writes `results/summary_<task>.md`
- `heldout.py`: ranks one recipe at a time against the published methods, on all datasets and excluding the development set
- `results/`: full-archive CSVs, selection results, published reference tables, summaries, per-dataset tables

Variants removed during development, including random polynomial kernels, are in the git history at commit `101b981`.

## Usage

```bash
pip install aeon pywavelets scikit-learn scipy matplotlib pandas numpy tabulate
```

Notebooks: open in Colab or Jupyter and run all cells. Datasets download automatically via `aeon`. Results are saved as `results_classification.csv` and `results_regression.csv`.

STCT+ on your own series (`X` is `(n_series, length)`):

```python
from sklearn.ensemble import ExtraTreesClassifier
from stct_plus import stct_plus

feats = lambda X: stct_plus(X, paper=True, resid_kind="peak", global_q=True)
clf = ExtraTreesClassifier(n_estimators=500, n_jobs=-1).fit(feats(X_train), y_train)
acc = clf.score(feats(X_test), y_test)
```

For regression, use `ExtraTreesRegressor` the same way.

Reproduce the results above:

```bash
python algo_select.py --recipe paper --noise-seeds 1,2 --candidates orthonormal,shape,energy,cross,second,spectrum,global_q,window_q,level_q,adaptive_degree,fixed_swap,fixed_add,znorm,resid_sign,resid_peak --prefix results/paper
python bench.py --task classification --datasets all --methods fixed:rf,fixed:et,wavelet:rf,wavelet:et,stft:rf,stft:et,paper_pooled:et,paper_plus:et,minirocket:ridge --out results/full_classification.csv
python bench.py --task regression --datasets all --methods fixed:rf,fixed:et,wavelet:rf,wavelet:et,stft:rf,stft:et,paper_pooled:et,paper_plus:et,minirocket:ridge --out results/full_regression.csv
python heldout.py
python single_thread.py
```

Default STCT setting: degree 4, two scales (8 and 16 windows).


## Contact

Samir Brahim Belhaouari (sbelhaouari@hbku.edu.qa), Yunis Carreon Kahalan (yuka34154@hbku.edu.qa)
