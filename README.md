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

STCT+ builds on the paper's features and uses a 500-tree ExtraTrees. It is closed-form and training-free, and its feature count does not depend on series length (at most 498, mean 423 on UCR).

- **Paper features:** degree-4 Chebyshev coefficients plus residual RMS in 8 and 16 windows.
- **Pooled statistics:** the same fit is slid along the series (stride = window/8) at six window lengths, 1/128 to 1/4 of the series, and each channel's time course is collapsed into 10 shift-invariant statistics. On short series the shortest fractions merge at a 6-sample minimum, so the number of scales grows with series length.
- **ExtraTrees:** 10% of features per split for classification, and all features for regression.

### How settings were chosen

`param_search.py` tried each setting in turn, scored by 5-fold cross-validation inside the training splits of all datasets, and never loaded a test split. The rule was fixed in advance and sits above the seed noise measured on this recipe (up to 0.25 accuracy points and a 0.6% RMSE change). A setting needed at least +0.3 accuracy points or a 1% lower RMSE, with more datasets better than worse and no harm to the other task beyond noise.

| Setting | Tried | Result on training splits |
|---|---|---|
| Pooled window lengths | 3 to 6 scales, 1/128 to 1/2 | **Adopted 1/128 to 1/4: +0.43 points, 36 better / 21 worse, RMSE ratio 0.996** |
| ExtraTrees features per split, classification | 10%, 25%, 50% | **Adopted 10%: +0.35 points, 46 better / 33 worse** |
| Paper window counts | 8 alternatives from 1 to 4 scales | No gain |
| Paper fit degree | 2, 3, 5, 6 | Within noise |
| Pooled stride | window / 2, 4, 16, 32 | Coarser is clearly worse; window/32 gained 0.31 points but raised RMSE 0.7% and fails the rule |
| Pooled fit degree | 1, 2, 3, 5 (6 was over the 600-feature limit) | No gain |
| Pooled statistics | 4, 6, 8 and 12 statistics | Fewer is worse; 12 gained 0.25 points, below the threshold |
| Residual measure | RMS, mean absolute, maximum | Within noise |
| Split criterion and leaf size | Entropy; leaf size 2, 3 | No gain |

With two other seeds, the final settings gained +1.01 and +0.96 training accuracy points over the starting recipe. Results are in `results/param_trials.csv` and `results/param_selected.json`. Fifteen extra feature blocks were also tested on training splits and are not in this recipe (`results/paper_select.csv`).

**Limit of this protocol:** the paper + pooled structure was chosen after seeing test scores, including on 38 UCR and 16 TSER datasets used during development. The settings above were chosen without test scores. The tables also report the datasets outside the development set (`results/heldout_<task>.md`).

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
| **STCT+ (ExtraTrees)** | **0.853** | **0.832** | **423** |
| Paper features + pooled, before tuning | 0.845 | 0.822 | 359 |
| FreshPRINCE | 0.842 | 0.824 | |
| Paper features (ExtraTrees) | 0.797 | 0.767 | 138 |
| Paper features (paper, RF) | 0.782 | 0.756 | 138 |
| Catch22 | 0.781 | 0.775 | |
| 1NN-DTW | 0.742 | 0.712 | |

Tuning improved the recipe on the test splits: 63 better / 25 worse on all 116 datasets (p = 0.0001), and 47 better / 18 worse on the 78 outside the development set (p = 0.0003). The gain grows with series length, from none up to length 100 to 1.71 points above length 1000.

STCT+ does not beat the state of the art on classification. HC2 (19 / 78), MR-Hydra (25 / 71), MultiROCKET (33 / 63), RDST (41 / 62) and WEASEL-2.0 (40 / 59) remain significantly ahead across all datasets (p < 0.04), and the first three also outside the development set. STCT+ is statistically tied with QUANT (39 / 55, p = 0.17), MiniROCKET (39 / 55, p = 0.11), InceptionTime (47 / 53, p = 0.17), ROCKET (42 / 55, p = 0.07) and FreshPRINCE (60 / 39, p = 0.11). It reaches that tier with about a tenth of QUANT's features and a twentieth of MiniRocket's.

**Regression** (26 TSER datasets, each recipe ranked alone against 13 published regressors; better / worse on RMSE; the held-out set has 10 datasets):

| Method | Mean rank (of 14) | Features | vs DrCIF | vs FreshPRINCE | Held out vs DrCIF | Held out vs FreshPRINCE |
|---|---|---|---|---|---|---|
| **STCT+ (ExtraTrees)** | **3.7** | **417** | 15 / 11, p = 0.92 | 19 / 7, p = 0.08 | 3 / 7 | 10 / 0 |
| Paper features + pooled, before tuning | 3.7 | 345 | 15 / 11, p = 0.94 | 19 / 7, p = 0.07 | 4 / 6 | 10 / 0 |
| MiniRocket + ridge (run here) | 4.1 | 9996 | 15 / 11, p = 0.60 | 15 / 11, p = 0.52 | 5 / 5 | 7 / 3 |
| Paper features (ExtraTrees) | 4.5 | 134 | 11 / 15, p = 0.22 | 18 / 8, p = 0.03 | 5 / 5 | 10 / 0 |
| Paper features (paper, RF) | 6.0 | 134 | 5 / 21, p = 0.001 | 8 / 18, p = 0.04 | 1 / 9 | 4 / 6 |

STCT+ is statistically tied with DrCIF, the strongest published regressor here. Tuning did not change regression accuracy (11 better / 9 worse, p = 0.97), and the pooled block's gain over the paper's features is not significant (14 / 12, p = 0.88), so for regression alone the paper's features with ExtraTrees are the leaner choice. These tests are not corrected for multiple comparisons. The published AcousticContaminationMadrid results use a version with missing values removed, while this code interpolates them.

**Same-learner comparison with the paper's representations:** ExtraTrees helps wavelet and STFT as well, and STCT still beats both.

| Representation | Accuracy, RF | Accuracy, ExtraTrees | RMSE, RF | RMSE, ExtraTrees |
|---|---|---|---|---|
| Wavelet | 0.731 | 0.751 | 138.81 | 131.89 |
| STFT | 0.755 | 0.767 | 151.66 | 145.29 |
| Paper features | 0.783 | 0.799 | 138.32 | 113.55 |

The random forest columns reproduce the paper's table. Under ExtraTrees, paper features beat wavelet 89 / 16 and STFT 75 / 31 on classification, and 19 / 7 and 20 / 6 on regression (p < 0.02 for all four).

### Single-thread timing

`single_thread.py` measures extraction, fit and predict on one thread on this machine. The machine was also running other jobs, and the methods were timed in separate runs, so only large gaps are meaningful.

| Method | Task | Features | Mean (s) | Median (s) | Max (s) |
|---|---|---|---|---|---|
| **STCT+** | Classification, 116 UCR | 423 | 1.8 | 1.0 | 17 |
| QUANT, aeon (200 trees) | Classification, 116 UCR | 4909 | 12.2 | 2.5 | 155 |
| MiniRocket + ridge | Classification, 116 UCR | 9996 | 8.2 | 0.9 | 404 |
| **STCT+** | Regression, 26 TSER | 417 | 25.2 | 18.3 | 82 |
| Paper features (500 trees) | Regression, 26 TSER | 134 | 11.3 | 7.7 | 39 |
| MiniRocket + ridge | Regression, 26 TSER | 9996 | 5.0 | 3.3 | 19 |

Published QUANT totals are lower than this aeon run (9.6 minutes against 23.6 on 112 datasets, on different hardware), so the comparison with QUANT depends on the implementation. On regression, published mean fit times are about 17,209 s for DrCIF and 3,340 s for FreshPRINCE, far above STCT+ despite the hardware difference.

## Contents

- `stct_classification.ipynb`, `stct_regression.ipynb`: the paper's UCR / TSER benchmarks (wavelet, STFT, STCT, Rocket)
- `stct_plus.py`: STCT (`stct_fixed`, verbatim logic from the notebooks), `stct_pooled`, `stct_plus`, sklearn-style `STCTPlus`, and options for the tested candidates
- `param_search.py`: searches settings using training splits only
- `algo_select.py`: tests extra feature blocks using training splits only
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

feats = lambda X: stct_plus(X, paper=True, fractions=(1/128, 1/64, 1/32, 1/16, 1/8, 1/4))
clf = ExtraTreesClassifier(n_estimators=500, max_features=0.1, n_jobs=-1).fit(feats(X_train), y_train)
acc = clf.score(feats(X_test), y_test)
```

For regression, use `ExtraTreesRegressor(n_estimators=500, n_jobs=-1)` with the same features.

Reproduce the results above:

```bash
python param_search.py
python bench.py --task classification --datasets all --methods fixed:rf,fixed:et,wavelet:rf,wavelet:et,stft:rf,stft:et,paper_pooled:et,paper_tuned:et_tuned,minirocket:ridge --out results/full_classification.csv
python bench.py --task regression --datasets all --methods fixed:rf,fixed:et,wavelet:rf,wavelet:et,stft:rf,stft:et,paper_pooled:et,paper_tuned:et_tuned,minirocket:ridge --out results/full_regression.csv
python heldout.py
python single_thread.py
```

Default STCT setting: degree 4, two scales (8 and 16 windows).


## Contact

Samir Brahim Belhaouari (sbelhaouari@hbku.edu.qa), Yunis Carreon Kahalan (yuka34154@hbku.edu.qa)
