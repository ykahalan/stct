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

STCT+ combines two closed-form, training-free blocks with a 500-tree ExtraTrees. Its feature count does not depend on series length and is at most 488.

- **Prime ladder:** the series is cut into 1, 2, 3, 5, 7, 11 and 13 tiling windows, and each window is described by its degree-4 polynomial coefficients plus the fit residual RMS (252 features).
- **Pooled statistics:** the same fit is slid along the series (stride = window/8) at window lengths 1/32, 1/16, 1/8 and 1/4 of the series, and each channel's time course is collapsed into 10 shift-invariant statistics (236 features).
- **Orthonormal coefficients:** both blocks report coefficients in an orthonormal basis of the Chebyshev polynomial space, which decorrelates them. The fit and residual are unchanged.

### How algorithm changes were chosen

`algo_select.py` scored each change by 5-fold cross-validation inside the training splits of all datasets, using a rule fixed in advance, and never loaded a test split. So decisions from this stage do not depend on any test score, and every dataset can be used as a test set.

| Candidate | Train CV accuracy gain | Better / worse | Regression RMSE ratio | Adopted |
|---|---|---|---|---|
| Orthonormal coefficients | +0.0064 | 48 / 32 | 0.999 | Yes |
| Cross-channel correlations | +0.0053 | 48 / 31 | 1.003 | No (hurt regression, rule limit 1.002) |
| Shape normalisation | +0.0015 | 36 / 44 | 1.042 | No |
| Energy per polynomial degree | -0.0011 | 41 / 38 | 0.996 | No |
| Second-order STCT | -0.0009 | 36 / 45 | 1.000 | No |
| Ladder over the log periodogram (on top of orthonormal) | -0.0019 | 39 / 40 | 1.005 | No |
| Global deciles of series and residual (on top of orthonormal) | -0.0029 | 35 / 40 | 0.974 | No (hurt classification) |
| Quartiles of raw values per ladder window (on top of orthonormal) | -0.0032 | 34 / 50 | 0.983 | No |
| Quartiles of coefficients per ladder level (on top of orthonormal) | -0.0039 | 31 / 42 | 0.999 | No |

Nothing else passed once orthonormal coefficients were in place, so the rejected candidates were never scored on test splits. On the test splits, the adopted change is not significantly better than the version before it (0.8458 to 0.8462 mean accuracy, 47 better / 39 worse, p = 0.33).

**Limit of this protocol:** the ladder + pooled structure itself was chosen earlier, after seeing test scores on 38 UCR and 16 TSER datasets. The tables therefore also report the remaining datasets (`results/heldout_<task>.md`).

### Results

All results use the default train/test splits and one seed. Reference numbers are the published results from timeseriesclassification.com for the same splits; STCT variants and MiniRocket were run here. Times are mean seconds per dataset for extraction + fit + predict on one shared 20-thread machine that was also running other jobs, so ratios are more reliable than absolutes. These times use all threads; see the single-thread table below.

**Classification** (mean accuracy on the 112 UCR datasets common to all methods, and on 74 of those outside the earlier 38):

| Method | All (112) | Excluding the 38 (74) | Features | Time (s) |
|---|---|---|---|---|
| HC2 | 0.876 | 0.856 | | |
| MR-Hydra | 0.868 | 0.842 | | |
| MultiROCKET | 0.866 | 0.841 | | |
| RDST | 0.863 | 0.839 | | |
| WEASEL-2.0 | 0.861 | 0.832 | | |
| MiniRocket + ridge (run here) | 0.860 | 0.832 | 9996 | 3.3 |
| MiniROCKET (published) | 0.858 | 0.831 | | |
| InceptionTime | 0.858 | 0.836 | | |
| ROCKET | 0.856 | 0.830 | | |
| QUANT | 0.854 | 0.833 | | |
| **STCT+ (ExtraTrees)** | **0.845** | **0.819** | **463** | **1.3** |
| FreshPRINCE | 0.842 | 0.824 | | |
| STCT paper features (ExtraTrees) | 0.797 | 0.767 | 138 | 0.64 |
| STCT paper features (paper, RF) | 0.782 | 0.756 | 138 | 0.85 |
| Catch22 | 0.781 | 0.774 | | |
| 1NN-DTW | 0.742 | 0.712 | | |

STCT+ does not beat the state of the art on classification. It is statistically tied with FreshPRINCE (51 / 49, p = 0.89) and significantly behind every other method above it, including HC2 (15 / 75) and MR-Hydra (22 / 71).

**Regression** (26 TSER datasets, each recipe ranked alone against 13 published regressors; better / worse on RMSE):

| Method | Mean rank (of 14) | Features | Time (s) | vs DrCIF | vs FreshPRINCE |
|---|---|---|---|---|---|
| **STCT+ (ExtraTrees)** | **3.3** | **441** | **5.0** | 16 / 10, p = 0.90 | 20 / 6, p = 0.03 |
| DrCIF (published, ranked with references only) | 3.5 | | | | |
| MiniRocket + ridge (run here) | 4.1 | 9996 | 1.9 | 15 / 11, p = 0.60 | 15 / 11, p = 0.52 |
| STCT paper features (ExtraTrees) | 4.5 | 134 | 1.3 | 11 / 15, p = 0.22 | 18 / 8, p = 0.03 |
| STCT paper features (paper, RF) | 6.0 | 134 | 1.5 | 5 / 21, p = 0.001 | 8 / 18, p = 0.04 |

STCT+ is statistically tied with DrCIF, the strongest published regressor here, and beats FreshPRINCE, RotF, ROCKET and every other reference except InceptionTime (18 / 8, p = 0.28). Its gain over the paper's features with ExtraTrees is not established. These tests are not corrected for multiple comparisons. The published AcousticContaminationMadrid results use a version with missing values removed, while this code interpolates them.

**Same-learner comparison with the paper's representations:** ExtraTrees helps wavelet and STFT as well, and STCT still beats both.

| Representation | Accuracy, RF | Accuracy, ExtraTrees | RMSE, RF | RMSE, ExtraTrees |
|---|---|---|---|---|
| Wavelet | 0.731 | 0.751 | 138.81 | 131.89 |
| STFT | 0.755 | 0.767 | 151.66 | 145.29 |
| STCT paper features | 0.783 | 0.799 | 138.32 | 113.55 |

The random forest columns reproduce the paper's table. Under ExtraTrees, STCT beats wavelet 89 / 16 and STFT 75 / 31 on classification, and 19 / 7 and 20 / 6 on regression (p < 0.02 for all four).

### Single-thread timing

Published baseline times come from other hardware, so `single_thread.py` ran STCT+ and two references on one thread on this machine, including extraction, fit and predict. The machine was also running other jobs.

| Method | Task | Mean score | Features | Mean (s) | Median (s) | Max (s) |
|---|---|---|---|---|---|---|
| **STCT+ (500 trees)** | Classification, 116 UCR | 0.846 | 463 | 3.1 | 1.4 | 39 |
| QUANT, aeon (200 trees) | Classification, 116 UCR | 0.853 | 4909 | 12.2 | 2.5 | 155 |
| MiniRocket + ridge | Classification, 116 UCR | 0.860 | 9996 | 8.2 | 0.9 | 404 |
| STCT paper features (500 trees) | Classification, 116 UCR | 0.799 | 138 | 1.5 | 0.8 | 22 |
| **STCT+ (500 trees)** | Regression, 26 TSER | RMSE 112.9 | 441 | 35.4 | 24.0 | 122 |
| MiniRocket + ridge | Regression, 26 TSER | RMSE 124.1 | 9996 | 5.0 | 3.3 | 19 |
| STCT paper features (500 trees) | Regression, 26 TSER | RMSE 114.4 | 134 | 11.3 | 7.7 | 39 |

STCT+ is faster than this QUANT run on 97 of 116 classification datasets, but slower than MiniRocket on most of them and on all 26 regression datasets. The published QUANT totals are lower than this aeon run (9.6 minutes against 23.6 on 112 datasets, on different hardware), so the QUANT comparison depends on the implementation. MiniRocket shows the opposite gap (128.8 against 15.7 minutes), so these runs do not give a single hardware correction. On regression, published mean fit times are about 17,209 s for DrCIF and 3,340 s for FreshPRINCE, far above STCT+ despite the hardware difference.

## Contents

- `stct_classification.ipynb`, `stct_regression.ipynb`: the paper's UCR / TSER benchmarks (wavelet, STFT, STCT, Rocket)
- `stct_plus.py`: STCT (`stct_fixed`, verbatim from the notebooks) and STCT+ (`stct_ladder`, `stct_pooled`, `stct_plus`, sklearn-style `STCTPlus`)
- `algo_select.py`: chooses algorithm changes using training splits only
- `single_thread.py`: times STCT+, MiniRocket and QUANT on one thread
- `bench.py`: resumable benchmark harness, `<features>:<learner>` methods, UCR and TSER lists
- `compare.py`: merges a benchmark CSV with the published reference results and writes `results/summary_<task>.md`
- `heldout.py`: ranks one recipe at a time against the published methods, on all datasets and excluding the earlier ones
- `results/`: full-archive CSVs, selection results, published reference tables, summaries, per-dataset tables

Other variants tested during development, including random polynomial kernels, are in the git history at commit `101b981`.

## Usage

```bash
pip install aeon pywavelets scikit-learn scipy matplotlib pandas numpy tabulate
```

Notebooks: open in Colab or Jupyter and run all cells. Datasets download automatically via `aeon`. Results are saved as `results_classification.csv` and `results_regression.csv`.

STCT+ on your own series (`X` is `(n_series, length)`):

```python
from sklearn.ensemble import ExtraTreesClassifier
from stct_plus import stct_plus

feats = lambda X: stct_plus(X, orthonormal=True)
clf = ExtraTreesClassifier(n_estimators=500, n_jobs=-1).fit(feats(X_train), y_train)
acc = clf.score(feats(X_test), y_test)
```

For regression, use `ExtraTreesRegressor` the same way.

Reproduce the results above:

```bash
python algo_select.py
python algo_select.py --base orthonormal --candidates spectrum --prefix results/algo_spectrum
python algo_select.py --base orthonormal --candidates global_q,window_q,level_q --prefix results/algo_quantile
python bench.py --task classification --datasets all --methods fixed:rf,fixed:et,wavelet:rf,wavelet:et,stft:rf,stft:et,msP7+pooled:et,stct_plus:et,minirocket:ridge --out results/full_classification.csv
python bench.py --task regression --datasets all --methods fixed:rf,fixed:et,wavelet:rf,wavelet:et,stft:rf,stft:et,msP7+pooled:et,stct_plus:et,minirocket:ridge --out results/full_regression.csv
python heldout.py
python single_thread.py
```

Default STCT setting: degree 4, two scales (8 and 16 windows).


## Contact

Samir Brahim Belhaouari (sbelhaouari@hbku.edu.qa), Yunis Carreon Kahalan (yuka34154@hbku.edu.qa)
