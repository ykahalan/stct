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

STCT+ builds on the paper's features and uses a 500-tree ExtraTrees. It is closed-form and training-free, and its feature count does not depend on series length (at most 786, mean 686 on UCR).

- **Paper features:** degree-4 Chebyshev coefficients plus residual RMS in 8 and 16 windows.
- **Pooled statistics:** the same fit is slid along the series (stride = window/8) at six window lengths, 1/128 to 1/4 of the series, and each channel's time course is collapsed into 10 shift-invariant statistics. On short series the shortest fractions merge at a 6-sample minimum, so the number of scales grows with series length.
- **Extra views:** the paper features of the sorted series (its value distribution) and of its autocorrelation (its periodicity).
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

### How blocks were chosen

`ablate.py` scored 90 combinations of blocks on training splits only, with one seed: paper windows, pooled statistics, a pooled position pyramid, random coefficient kernels, coefficient correlations, first and second differences, cumulative sums, shape histograms, path signatures, DWT and STFT summaries, and raw DWT and STFT. The leaders were rechecked with two other seeds.

| Added to paper + pooled | Mean features | Accuracy change by seed (points) | Better / worse | RMSE ratio | Result |
|---|---|---|---|---|---|
| Sorted + autocorrelation | 682 | +0.48, +0.59, +0.41 | 53 / 39 | 0.974 | **Adopted** |
| Sorted + autocorrelation + difference | 819 | +0.36, +0.52, +0.51 | 61 / 36 | 0.973 | Passes, larger |
| Kernels + correlations | 698 | +0.30, +0.21, +0.65 | 53 / 41 | 0.991 | Passes, less stable |
| Shapes + signatures + DWT + STFT | 654 | +0.48, +0.05, +0.15 | 42 / 56 | 0.985 | Failed |
| DWT summary | 467 | -0.14, -0.18, -0.09 | 31 / 59 | 0.986 | Failed |

Pooled statistics on the difference or cumulative sum, the full triple view, and raw DWT or STFT did not help. Length rules for pooled windows (`length_rule.py`) also failed. Flat and nested prime pyramids in place of the paper windows (`pyramid_splits.py`) passed only through regression and were not adopted. All results are in `results/`.

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
| **STCT+ (ExtraTrees)** | **0.856** | **0.836** | **686** |
| STCT+ without the extra views | 0.853 | 0.832 | 423 |
| Paper features + pooled, before tuning | 0.845 | 0.822 | 359 |
| FreshPRINCE | 0.842 | 0.824 | |
| Paper features (ExtraTrees) | 0.797 | 0.767 | 138 |
| Paper features (paper, RF) | 0.782 | 0.756 | 138 |
| Catch22 | 0.781 | 0.775 | |
| 1NN-DTW | 0.742 | 0.712 | |

Tuning improved the recipe on the test splits: 63 better / 25 worse on all 116 datasets (p = 0.0001), and 47 better / 18 worse on the 78 outside the development set (p = 0.0003). The extra views added 0.35 points, which is not significant (50 better / 42 worse, p = 0.18).

STCT+ does not beat the state of the art on classification. HC2 (19 / 77), MR-Hydra (27 / 71), MultiROCKET (35 / 66) and RDST (39 / 62) remain significantly ahead across all datasets (p < 0.04). It is statistically tied with QUANT (42 / 53, p = 0.40), MiniROCKET (36 / 60, p = 0.16), InceptionTime (43 / 57, p = 0.11), ROCKET (42 / 58, p = 0.12) and WEASEL-2.0 (44 / 58, p = 0.06), and it beats FreshPRINCE (63 / 35, p = 0.009). It uses about 14% of QUANT's mean features and 7% of MiniRocket's.

**Regression** (26 TSER datasets, each recipe ranked alone against 13 published regressors; better / worse on RMSE; the held-out set has 10 datasets):

| Method | Mean rank (of 14) | Features | vs DrCIF | vs FreshPRINCE | Held out vs DrCIF | Held out vs FreshPRINCE |
|---|---|---|---|---|---|---|
| **STCT+ (ExtraTrees)** | **3.5** | **673** | 16 / 10, p = 0.60 | 20 / 6, p = 0.03 | 4 / 6 | 9 / 1 |
| STCT+ without the extra views | 3.7 | 417 | 15 / 11, p = 0.92 | 19 / 7, p = 0.08 | 3 / 7 | 10 / 0 |
| Paper features + pooled, before tuning | 3.7 | 345 | 15 / 11, p = 0.94 | 19 / 7, p = 0.07 | 4 / 6 | 10 / 0 |
| MiniRocket + ridge (run here) | 4.1 | 9996 | 15 / 11, p = 0.60 | 15 / 11, p = 0.52 | 5 / 5 | 7 / 3 |
| Paper features (ExtraTrees) | 4.5 | 134 | 11 / 15, p = 0.22 | 18 / 8, p = 0.03 | 5 / 5 | 10 / 0 |
| Paper features (paper, RF) | 6.0 | 134 | 5 / 21, p = 0.001 | 8 / 18, p = 0.04 | 1 / 9 | 4 / 6 |

STCT+ is statistically tied with DrCIF, the strongest published regressor here. The extra views lowered RMSE about 4% without reaching significance (16 / 10, p = 0.17), and the paper's features with ExtraTrees are the leaner choice for regression alone. These tests are not corrected for multiple comparisons. The published AcousticContaminationMadrid results use a version with missing values removed, while this code interpolates them.

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
| **STCT+** | Classification, 116 UCR | 686 | 3.4 | 1.7 | 41 |
| STCT+ without the extra views | Classification, 116 UCR | 423 | 1.8 | 1.0 | 17 |
| QUANT, aeon (200 trees) | Classification, 116 UCR | 4909 | 12.2 | 2.5 | 155 |
| MiniRocket + ridge | Classification, 116 UCR | 9996 | 8.2 | 0.9 | 404 |
| **STCT+** | Regression, 26 TSER | 673 | 48.6 | 34.4 | 163 |
| STCT+ without the extra views | Regression, 26 TSER | 417 | 25.2 | 18.3 | 82 |
| Paper features (500 trees) | Regression, 26 TSER | 134 | 11.3 | 7.7 | 39 |
| MiniRocket + ridge | Regression, 26 TSER | 9996 | 5.0 | 3.3 | 19 |

Published QUANT totals are lower than this aeon run (9.6 minutes against 23.6 on 112 datasets, on different hardware), so the comparison with QUANT depends on the implementation. On regression, published mean fit times are about 17,209 s for DrCIF and 3,340 s for FreshPRINCE, far above STCT+ despite the hardware difference.

## Contents

- `stct_classification.ipynb`, `stct_regression.ipynb`: the paper's UCR / TSER benchmarks (wavelet, STFT, STCT, Rocket)
- `stct_plus.py`: STCT (`stct_fixed`, verbatim logic from the notebooks), `stct_pooled`, `stct_plus`, `stct_views`, sklearn-style `STCTPlus`, and options for the tested candidates
- `param_search.py`: searches settings using training splits only
- `ablate.py`: ablates blocks and combinations using training splits only
- `length_rule.py`, `pyramid_splits.py`: test pooled window rules and split pyramids on training splits
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
from stct_plus import stct_views as feats

clf = ExtraTreesClassifier(n_estimators=500, max_features=0.1, n_jobs=-1).fit(feats(X_train), y_train)
acc = clf.score(feats(X_test), y_test)
```

For regression, use `ExtraTreesRegressor(n_estimators=500, n_jobs=-1)` with the same features.

Reproduce the results above:

```bash
python param_search.py
python ablate.py
python ablate.py --configs new
python bench.py --task classification --datasets all --methods fixed:rf,fixed:et,wavelet:rf,wavelet:et,stft:rf,stft:et,paper_pooled:et,paper_tuned:et_tuned,paper_views:et_tuned,minirocket:ridge --out results/full_classification.csv
python bench.py --task regression --datasets all --methods fixed:rf,fixed:et,wavelet:rf,wavelet:et,stft:rf,stft:et,paper_pooled:et,paper_tuned:et_tuned,paper_views:et_tuned,minirocket:ridge --out results/full_regression.csv
python heldout.py
python single_thread.py
```

Default STCT setting: degree 4, two scales (8 and 16 windows).


## Contact

Samir Brahim Belhaouari (sbelhaouari@hbku.edu.qa), Yunis Carreon Kahalan (yuka34154@hbku.edu.qa)
