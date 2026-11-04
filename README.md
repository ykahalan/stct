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

## STCT+ classifier (`stct_models.py`)

`STCTClassifier()` is STCT+. Every feature it computes comes from Chebyshev least-squares fits; it copies no shapelets, and its kernel weightings and thresholds come from at most 32 training series without labels. Extraction is compiled in `stct_fast.py`.

- **Forest on the STCT views:** ExtraTrees of 200 trees, 10% of features per split, on degree-4 Chebyshev coefficients and residual RMS in 8 and 16 windows, pooled statistics of sliding fits at six window lengths, and the same window features for the sorted series and for its autocorrelation (see `stct_plus.py` below).
- **Ridges on Chebyshev kernels.** Kernels get one random direction per 8 samples of series length, shared equally among the kernel types in use, at most 64 per type:
  - From length 128 up, two kernel types get one direction per 16 samples each. **Context kernels** threshold random projections of two neighbouring windows' coefficients, so they respond to ordered pairs of local shapes. **Dilated kernels** fit the degree-4 polynomial to 9 samples spaced 1, 2, 4, ... apart at every position, with no weight on the level coefficient, so they respond to local shape and not to local level. **Winner counts** are added: in groups of 8 random directions over each window's coefficients, how often each direction scores highest. Each block has its own ridge, and class probabilities are averaged with the forest's.
  - Below length 128 the dilated kernels alone get one direction per 8 samples and enter at half the forest's weight.

**Paper table** (112 UCR datasets common to all methods, default splits, sorted by mean accuracy; published accuracies from timeseriesclassification.com, STCT+ run here with one seed; mean rank among the 13 methods; `paper_table.py` writes `results/paper_table_classification.md`):

| # | Method | Accuracy (%) | Held out (%) | Mean rank | STCT+ better / worse | Features | Runtime here, total (h) | Median per dataset (s) |
|---|---|---|---|---|---|---|---|---|
| 1 | HC2 | 87.64 | 85.65 | 4.31 | 28 / 66 (p < 0.001) | over 100,000 (its Arsenal alone: 25 x 4,000) | not run (published 290.2) | |
| 2 | MR-Hydra | 86.80 | 84.23 | 5.08 | 39 / 54 (p = 0.44) | 55,643 (50,752-58,944) | 1.60 | 15.9 |
| **3** | **STCT+** | **86.72** | **84.50** | 5.81 | | **3,547 (125-8,082)** | **0.15** | 1.6 |
| 4 | MultiROCKET | 86.57 | 84.08 | 5.42 | 40 / 51 (p = 0.67) | 49,728 | 0.66 | 5.7 |
| 5 | RDST | 86.31 | 83.91 | 6.45 | 55 / 45 (p = 0.17) | 30,000 | 2.24 | 22.2 |
| 6 | WEASEL-2.0 | 86.05 | 83.22 | 6.91 | 61 / 37 (p = 0.030) | 32,665 (6,862-56,760) | 2.63 | 16.4 |
| 7 | MiniROCKET | 85.80 | 83.11 | 6.87 | 58 / 35 (p = 0.011) | 9,996 | 0.15 | 0.8 |
| 8 | InceptionTime | 85.78 | 83.63 | 6.43 | 52 / 45 (p = 0.31) | learned: 5 networks x 128 | not run (published 86.8) | |
| 9 | ROCKET | 85.58 | 82.95 | 6.60 | 56 / 43 (p = 0.07) | 20,000 | 4.06 | 41.1 |
| 10 | QUANT | 85.44 | 83.27 | 6.61 | 57 / 37 (p = 0.013) | 4,993 (116-25,203) | 0.29 | 2.8 |
| 11 | FreshPRINCE | 84.16 | 82.38 | 7.89 | 71 / 31 (p < 0.001) | 783 | not run (published 39.0) | |
| 12 | Catch22 | 78.14 | 77.45 | 11.02 | 101 / 4 | 22 | 0.03 | 0.6 |
| 13 | 1NN-DTW | 74.21 | 71.17 | 11.60 | 101 / 5 | none (raw series) | not run (published 81.7) | |

Features are means over the 112 datasets (range in brackets) where they depend on the data or series length. Runtime is fit on the training split plus predict on the test split on one thread, summed over the 112 datasets, with all methods on a dataset run in the same process on the same machine (`paper_time.py`, four processes over disjoint datasets; the five pairs that ran out of memory with four processes, four MR-Hydra and one WEASEL-2.0, were rerun alone, MR-Hydra on StarLightCurves predicting in batches of 512 series with identical predictions). The references are aeon implementations configured as published: MultiROCKET and the MultiROCKET part of MR-Hydra use 6,250 kernels (49,728 features; aeon 1.6 defaults to 10,000), and Catch22 uses the 22 catch22 features. Their accuracies here are within 0.43 points of the published ones. HC2, InceptionTime, FreshPRINCE and 1NN-DTW were not run: their published fit + predict times total 39 to 290 hours on other hardware, which is not comparable with the runtime here (on that hardware MiniROCKET took 2.1 hours and MultiROCKET 9.0, against 0.15 and 0.66 here).

Sorted by accuracy STCT+ is third, behind HC2 and MR-Hydra, with about 6% of MR-Hydra's features and a tenth of its runtime. It ran faster than MR-Hydra, RDST, ROCKET and WEASEL-2.0 on all 112 datasets, than MultiROCKET on 111 and QUANT on 98; MiniROCKET (faster on 104) and Catch22 (105) were faster. Its total runtime about equals MiniROCKET's (557 against 548 seconds), but its median is twice MiniROCKET's.

By mean rank STCT+ is fourth (5.81, behind MultiROCKET's 5.42) on all datasets and on the 74 outside the development set (5.62; HC2 4.11, MR-Hydra 5.09, MultiROCKET 5.39), where it is second by mean accuracy (84.50), tied with MR-Hydra (28 / 35) and MultiROCKET (28 / 32). STCT+ uses fewer features than QUANT on 113 of 116 datasets; the exceptions are the three shortest, Chinatown and ItalyPowerDemand (length 24: 215 against 212) and SmoothSubspace (length 15: 125 against 116). It beats MiniRocket + ridge run here 57 / 40 (p = 0.022) and the three-model recipe below 53 / 27 (p = 0.028). On series shorter than 128 it is 7 / 13 against MultiROCKET; the rank gap to third place is there.

**How it was chosen.** All choices were made on training splits by 5-fold cross-validation with three seeds. The dilated kernels came from a diagnosis on the development datasets furthest behind HC2: every convolution method is about 3 points ahead there and QUANT is not, so the missing piece was the dilated kernel. A ridge on level-free dilated STCT kernels alone reaches 89.7% mean training-split accuracy on the 38 development datasets against 90.2% for MiniRocket + ridge, with about 1,300 features against 9,996 (`fast_probe.py`). Added to the three-model recipe with 200 trees, at the sizes of the earlier rule described below, they gained +0.19, +0.31 and +0.37 points on series of length 128 and up (`fast_quant.py`). On short training splits the forest alone trailed QUANT by 0.4 points; the half-weight dilated ridge gained +0.50, +0.22 and +0.78 points there, while Chebyshev fits of locally sorted values, difference and periodogram views, full-weight kernels and 500 trees were weaker or unstable (`fast_short.py`).

The size rule replaced an earlier one that shrank the kernels to stay under QUANT's feature count at every length (`quant_sized.py`, scored as `paper_quant` and `paper_quant2`). Over all 117 training sets the two rules differ by +0.02, -0.04 and -0.00 points (`fast_native.py`), and on test STCT+ is 21 / 17 against the QUANT-sized version (p = 0.79) with 165 fewer features on average. **Limit:** these recipes were scored on the test splits several times during development (once per recipe), the decision to work on short series came from an earlier test scoring, and all scores use one seed without multiple-comparison correction.

**Earlier recipes** (`STCTClassifier(recipe=...)`, each ranked alone against the 12 published methods):

| Recipe | Accuracy, all / held out | Place, all / held out | Mean / max features |
|---|---|---|---|
| `"four"`: three-model + full-size dilated kernels | 86.69 / 84.37 | 4 / 4 | 4,262 / 8,082 |
| `"dilated"`: three-model budget, context and dilated kernels in one ridge | 86.49 / 84.25 | 4 / 5 | 2,856 / 5,202 |
| `"trio"`: 100 trees, context kernels, winner counts | 86.47 / 84.20 | 4 / 4 | 2,746 / 4,626 |
| `"fast"`: one ridge on threshold kernels (also `STCTRegressor()`) | 86.20 / 84.15 | 8 / 6 | 1,968 |
| MiniRocket + ridge (run here) | 85.97 / 83.25 | 8 / 7 | 9,996 |

For regression, `STCTRegressor()` averages ExtraTrees of 50 trees (half the features per split, 2 samples per leaf) with a ridge on threshold kernels from length 128 up. Its mean rank on 26 TSER datasets is 3.2 of 14 (held out 3.9 on 10), first place, tied with DrCIF (18 / 8, p = 0.30), against 4.1 for MiniRocket + ridge run here.

**Regression paper table** (26 TSER datasets, default splits, sorted by median RMSE relative to the best of the 14 methods on each dataset, lowest first; published RMSE from timeseriesclassification.com, the STCT+ regressor run here with one seed; `paper_table_reg.py` writes `results/paper_table_regression.md`):

| # | Method | RMSE vs best | Held out | Mean rank | STCT+ better / worse | Features | Runtime here, total (h) | Median per dataset (s) |
|---|---|---|---|---|---|---|---|---|
| **1** | **STCT+ regressor** | **1.131** | **1.187** | **3.23** | | **2,096 (143-3,090)** | **0.03** | 2.9 |
| 2 | DrCIF | 1.251 | 1.248 | 4.19 | 18 / 8 (p = 0.30) | 200 trees x 150-890 interval features (86,923 on average) | not run (published 175.4) | |
| 3 | FreshPRINCE | 1.287 | 1.335 | 4.85 | 18 / 8 (p = 0.27) | 783 | not run (published 14.5) | |
| 4 | ROCKET | 1.336 | 1.284 | 6.58 | 19 / 7 (p = 0.059) | 20,000 | 2.57 | 199.1 |
| 5 | InceptionTime | 1.349 | 1.178 | 6.19 | 20 / 6 (p = 0.075) | learned: 5 networks x 128 | not run (published 60.0) | |
| 6 | RotF | 1.367 | 1.226 | 6.35 | 20 / 6 (p = 0.020) | raw series (1,816 values on average) | not run (published 15.6) | |
| 7 | Ridge | 1.466 | 1.346 | 8.38 | 22 / 4 (p = 0.006) | raw series | 0.00 | 0.0 |
| 8 | GridSVR | 1.555 | 1.555 | 8.77 | 22 / 4 (p = 0.005) | raw series | 0.65 | 11.6 |
| 9 | ResNet | 1.557 | 1.386 | 9.23 | 23 / 3 (p = 0.001) | learned: 128 | not run (published 14.1) | |
| 10 | XGBoost | 1.592 | 1.680 | 8.58 | 24 / 2 (p < 0.001) | raw series | not run, not installed (published 0.6) | |
| 11 | RandF | 1.612 | 1.679 | 7.81 | 23 / 3 (p < 0.001) | raw series | 0.49 | 31.8 |
| 12 | 1NN-DTW | 1.658 | 1.677 | 9.92 | 23 / 3 (p = 0.001) | none (raw series) | 7.99 | 131.5 |
| 13 | FPCR | 1.759 | 1.710 | 9.65 | 24 / 2 (p < 0.001) | functional principal components of the raw series | not run, not installed (published 0.1) | |
| 14 | FCN | 2.547 | 2.448 | 11.27 | 24 / 2 (p < 0.001) | learned: 128 | not run (published 9.3) | |

"Held out" is the same error measure on the 10 datasets outside the development set. Runtime is fit plus predict on one thread, summed over the 26 datasets, with all methods on a dataset run in the same process on the same machine (`paper_time_reg.py`, six processes over disjoint datasets). The references that ran here use the configurations of the published benchmark code (tsml-eval): aeon's RocketRegressor and 1-NN DTW with a 10% window, and scikit-learn's RidgeCV, 100-tree random forest and grid-searched SVR on the raw series; their RMSE here is within 0.5% of the published RMSE (geometric mean ratios 0.997 to 1.005). DrCIF, FreshPRINCE, RotF and the three networks were not run: their published fit + predict times total 9.3 to 175 hours on other hardware (25 of the 26 datasets), and on one dataset here RotF and FreshPRINCE took 1.8 and 3.0 times their published time. XGBoost and FPCR were not run because their packages are not installed here. The published 1NN-DTW times contain negative values and are not usable.

The STCT+ regressor has the lowest error and the best mean rank; no published regressor is significantly better, and it is significantly better than RotF, Ridge, GridSVR, ResNet, XGBoost, RandF, 1NN-DTW, FPCR and FCN (p < 0.05). It ran faster than ROCKET on all 26 datasets, than RandF on 25, 1NN-DTW on 23 and GridSVR on 20; only ridge regression on the raw series was faster on every dataset (118 seconds in total for STCT+, against 7 for Ridge and 9,250 for ROCKET).

**Earlier timing runs**, one thread, MiniRocket + ridge in the same pass (`dilated_time.py`, `fast_time.py`); the paper table above is the main comparison. The machine ran other jobs, so compare each recipe with the MiniRocket time from its own run:

| Recipe | Mean (s) | Median (s) | Max (s) | Faster than MiniRocket on | MiniRocket in the same run, mean / median / max |
|---|---|---|---|---|---|
| **STCT+** | 6.3 | 2.1 | 56 | 4 of 116 | 7.4 / 0.9 / 361 (three-model in the same run: 4.2 / 1.3 / 54) |
| Three-model | 5.0 | 1.6 | 61 | 22 of 116 | 8.4 / 1.0 / 383 |
| Four-model | 7.6 | 2.4 | 105 | 19 of 116 | 8.4 / 1.0 / 383 |
| Fast, classification | 2.9 | 1.0 | 43 | 50 of 116 | 8.2 / 0.9 / 415 |
| Fast, regression | 4.5 | 3.3 | | 13 of 26 | 5.2 / 3.5 |

STCT+ is faster than MiniRocket on mean and maximum time but about 2.3 times slower on median time. The median gap is structural: every window is processed in up to four separate passes (pooled statistics, context kernels, winner counts, dilated kernels), windows shorter than 16 samples slide one sample at a time, and the forest adds a fixed cost on small datasets. Computing the projections as one matrix multiply or counting thresholds by rank was slower than the current compiled loops. The shared sliding fit cut extraction by 17% with identical values (`trio_shared_time.py`).

Also rejected on training splits (`fast_search.py`, `fast_context.py`, `fast_trio.py`, `fast_refine.py`, `fast_words.py`, `fast_calib.py`, `fast_dlf*.py`, `fast_stack.py`, `fast_probe*.py`, `fast_distinct.py`): ridge or LDA in place of the forest, kernel window caps, contexts of 3 or more windows, winner counts for regression or for series shorter than 128, HC2-style accuracy weights, a heavier forest weight, a ridge on the forest's own features, context kernels below length 128, a ridge on Chebyshev words (quantized coefficients 1 to 3 counted per scale, like SFA words), sharper ridge probabilities whether fixed or fitted per ridge to leave-one-out outputs, forest weights chosen by an internal 2-fold split, degree 6 or 7 for the dilated fits, dilated kernels on the first-difference series, dilated kernels for regression, and computing collapsed paper windows once on short series.

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
- `stct_models.py`: `STCTClassifier` (STCT+ by default, earlier recipes by name) and `STCTRegressor` (fast recipe)
- `stct_fast.py`: compiled extraction of the views, threshold kernels, context kernels, winner counts and dilated kernels
- `dilated_test.py`, `trio_test.py`, `fast_test.py`: score STCT+ and the earlier recipes once on test splits; `fast_*.py` test their choices on training splits; `dilated_time.py` times them against MiniRocket on one thread
- `paper_time.py`, `paper_table.py`: time STCT+ and eight published methods on one thread on the same machine, count their features, and write the paper table (`results/paper_time.csv`, `results/paper_table_classification.md`); `results/published_times/` holds the published fit and predict times
- `paper_time_reg.py`, `paper_table_reg.py`: the same for regression (`results/paper_time_reg.csv`, `results/paper_table_regression.md`, published times in `results/published_times_regression/`)
- `quant_sized.py`: QUANT's feature count, used only to compare feature counts, and the earlier QUANT-sized variant with its scoring scripts (`quant2_test.py`); not part of STCT+
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

STCT+ models on your own series (`X` is `(n_series, length)`):

```python
from stct_models import STCTClassifier, STCTRegressor

acc = STCTClassifier().fit(X_train, y_train).score(X_test, y_test)  # STCT+; recipe="trio" or "fast" for faster ones
pred = STCTRegressor().fit(X_train, y_train).predict(X_test)
```

STCT+ views with a plain forest:

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
