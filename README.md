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

## STCT+: expansions of the transform (`stct_plus.py`)

Three closed-form, length-independent expansions, from most to least compact. All keep the paper's building block (degree-4 Chebyshev least-squares coefficients plus residual RMS per window); none is trained, the only data-dependent quantities are channel scales and bias quantiles read off the training set.

1. **Multi-scale ladder** (`stct_pyramid`): the paper's windows at more than two scales. Windows tile the series at a ladder of window counts; the prime ladder 1, 2, 3, 5, 7, 11, 13 (42 windows, 252 features) works best per feature because prime counts never share a window boundary, whereas dyadic levels nest. Variants tested and rejected: fitting each level to the residual of the coarser level (a polynomial wavelet-style detail pyramid; worse), overlapping windows (half a point for double the features), ternary / fives / sevens ladders (too few scales).
2. **Pooled** (`stct_pooled`, `stct_plus`): the Chebyshev fit slid along the series (stride = window/8) at window lengths 1/32, 1/16, 1/8, 1/4 of the series, each coefficient channel's time course collapsed into 10 shift-invariant statistics (240 features). This is what makes the representation tolerant to patterns that move in time. Combined with a positional block (the paper's two scales or the prime ladder) it is the recommended compact recipe, with a 500-tree ExtraTrees.
3. **Kernels** (`rppv_fit` / `stct_rppv`): every sliding window's coefficient vector projected onto 128 random unit directions in coefficient space (random local polynomial shapes, i.e. random convolution kernels restricted to the Chebyshev subspace, no extra convolution), pooled as proportion-above-bias at five training quantiles plus max, ridge classifier. 2880 features; `pyramid=2` repeats the pooling over the two halves (8640). The most accurate but no longer compact.

Full-archive results, default train/test splits. Reference numbers are the published results from timeseriesclassification.com for the same splits (loaded with `aeon.benchmarking.results_loaders`); STCT variants and MiniRocket were run here. Times are mean seconds per dataset for extraction + fit + predict on one shared 20-thread machine, in the same run (the machine was also running other jobs, so ratios are more reliable than absolutes).

**Classification, 112 UCR datasets common to all methods** (`results/summary_classification.md`):

| Method | Mean acc. | Features | Time (s) |
|---|---|---|---|
| HC2 | 0.876 | | |
| MR-Hydra | 0.868 | | |
| MultiROCKET | 0.866 | | |
| RDST | 0.863 | | |
| **STCT+ kernels, pyramid (ridge)** | 0.861 | 8640 | 6.1 |
| WEASEL-2.0 | 0.861 | | |
| MiniRocket + ridge (run here) | 0.860 | 9996 | 3.3 |
| MiniROCKET (published) | 0.858 | | |
| InceptionTime | 0.858 | | |
| **STCT+ kernels (ridge)** | 0.857 | 2880 | 2.8 |
| ROCKET | 0.856 | | |
| QUANT | 0.854 | | |
| **STCT+ pooled + paper scales (ExtraTrees)** | **0.845** | **363** | **1.9** |
| **STCT+ pooled + prime ladder (ExtraTrees)** | 0.845 | 467 | 1.9 |
| FreshPRINCE | 0.842 | | |
| **STCT prime ladder alone (ExtraTrees)** | 0.807 | 242 | 1.1 |
| **STCT dyadic ladder, 5 levels (ExtraTrees)** | 0.802 | 168 | 0.95 |
| STCT (paper, RF) | 0.782 | 138 | 0.85 |
| Catch22 | 0.781 | | |
| 1NN-DTW | 0.742 | | |

Paired Wilcoxon tests on the 112 datasets:

- The prime ladder alone adds 2.5 points to the paper's representation at the paper's cost (beats it on 85 datasets, loses on 18) and beats the dyadic ladder 59 / 18 / 35 (p = 0.01).
- The pooled recipe (363 or 467 features) ties FreshPRINCE (p = 0.85) and is 1.3 points behind MiniROCKET (p = 0.002); a pooled position pyramid (1713 features) gained a point on the dev subset but not on the full archive (0.846).
- The kernel recipe (2880 features) is statistically indistinguishable from MiniROCKET (47 / 13 / 52, p = 0.66), ROCKET, InceptionTime, QUANT and WEASEL-2.0; with the pyramid it wins 62 / 10 / 40 against MiniROCKET (p = 0.17). Both remain significantly behind HC2 (1.6 points) and MR-Hydra (0.7 points), which are ensembles of several representation types.

**Regression, 26 TSER datasets** (`results/summary_regression.md`; mean rank over 23 methods, since raw RMSE is not comparable across datasets):

| Method | Mean rank | Features | Time (s) |
|---|---|---|---|
| **STCT+ pooled + dyadic ladder (ExtraTrees)** | **7.0** | **378** | 4.3 |
| **STCT+ pooled + prime ladder (ExtraTrees)** | 7.5 | 444 | 4.4 |
| **STCT+ pooled + paper scales (ExtraTrees)** | 7.6 | 348 | 3.8 |
| **STCT dyadic ladder alone (ExtraTrees)** | 8.0 | 163 | 1.5 |
| **STCT prime ladder alone (ExtraTrees)** | 8.4 | 230 | 1.9 |
| MiniRocket + ridge (run here) | 8.5 | 9996 | 1.9 |
| **STCT+ kernels (ridge)** | 8.7 | 2747 | 2.5 |
| DrCIF | 8.8 | | |
| FreshPRINCE | 10.8 | | |
| InceptionTime | 12.1 | | |
| RotF | 12.4 | | |
| ROCKET | 12.7 | | |
| STCT (paper, RF) | 13.3 | 134 | 1.5 |
| Ridge, RandF, GridSVR, XGBoost, ResNet, FPCR, 1NN-DTW, FCN | 15.1 to 19.3 | | |

Every STCT+ variant ranks above the best published regressor (DrCIF). The dyadic ladder + pooled recipe beats FreshPRINCE 21 / 5 (p = 0.01), RotF 20 / 6 (p = 0.02), InceptionTime 18 / 8 and ties DrCIF (17 / 9, p = 0.63); the dyadic ladder alone, at 163 features and the paper's time, beats FreshPRINCE 19 / 7 (p = 0.014).

The configurations were chosen on 38 small UCR and 16 TSER datasets over eleven rounds (`results/dev_ablations.md`). Things that did not help: RotationForest or gradient boosting (10 to 25x the time), more trees, larger per-split feature samples, random rotations, a two-model vote, degree 3 or 6, more pooled scales or statistics, the first-difference channel, residual (detail) pyramids, overlapping windows, prime-length pooled scales.

## Contents

- `stct_classification.ipynb`, `stct_regression.ipynb`: the paper's UCR / TSER benchmarks (wavelet, STFT, STCT, Rocket)
- `stct_plus.py`: STCT (`stct_fixed`, verbatim from the notebooks), the multi-scale ladder (`stct_pyramid`, `PRIMES`), the sliding channels (`stct_channels`), the pooled features (`stct_pooled`, `stct_plus`, sklearn-style `STCTPlus`) and the kernels (`rppv_fit`, `stct_rppv`)
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
from stct_plus import PRIMES, rppv_fit, stct_plus, stct_pooled, stct_pyramid, stct_rppv

# compact: prime ladder + pooled statistics (about 470 features), ExtraTrees
feats = lambda X: np.hstack([stct_pyramid(X, counts=PRIMES[:7]), stct_pooled(X)])
clf = ExtraTreesClassifier(n_estimators=500, n_jobs=-1).fit(feats(X_train), y_train)
acc = clf.score(feats(X_test), y_test)

# most compact: prime ladder alone (about 250 features)
clf = ExtraTreesClassifier(n_estimators=500, n_jobs=-1).fit(stct_pyramid(X_train, counts=PRIMES[:7]), y_train)

# most accurate: 128 random polynomial kernels (2880 features), ridge
params = rppv_fit(X_train, n_dir=128)
ridge = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)))
ridge.fit(stct_rppv(X_train, params), y_train)
acc = ridge.score(stct_rppv(X_test, params), y_test)
```

Reproduce the tables above (a few hours for classification, half an hour for regression):

```bash
python bench.py --task classification --datasets all --methods fixed:rf,ms5:et,msP7:et,fixed+pooled:et,msP7+pooled:et,rppv128:ridge,minirocket:ridge --out results/full_classification.csv
python compare.py --task classification --results results/full_classification.csv --methods fixed:rf,ms5:et,msP7:et,fixed+pooled:et,msP7+pooled:et,rppv128:ridge,minirocket:ridge --refs HC2,MR-Hydra,InceptionTime,MultiROCKET,MiniROCKET,ROCKET,FreshPRINCE,RDST,QUANT,WEASEL-2.0,Catch22,1NN-DTW
```

Default STCT setting: degree 4, two scales (8 and 16 windows). STCT+ defaults: the same degree; ladder counts 1, 2, 3, 5, 7, 11, 13; pooled scales 1/32, 1/16, 1/8, 1/4 with 8x window overlap; kernels: 128 directions, 5 biases.


## Contact

Samir Brahim Belhaouari (sbelhaouari@hbku.edu.qa), Yunis Carreon Kahalan (yuka34154@hbku.edu.qa)
