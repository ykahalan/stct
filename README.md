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

## STCT+: a minimal expansion (`stct_plus.py`)

STCT's features are tied to absolute window positions, so a pattern that shifts in time changes every feature. STCT+ keeps the original fixed-window block and adds one thing: the same Chebyshev fit is slid along the series (stride = window/8) at four window lengths (1/32, 1/16, 1/8, 1/4 of the series), and each coefficient channel's time course (5 coefficients + residual) is summarised by 10 translation-invariant statistics (mean, std, min, max, quartiles, proportion positive, position of max/min). This is still closed-form, training-free and length-independent: 144 + 240 = 384 features. The learner is a 500-tree ExtraTrees instead of the random forest. An optional first-difference channel (`diff=True`) doubles the features for no measurable gain on classification.

Full-archive results, default train/test splits. Reference numbers are the published results from timeseriesclassification.com for the same splits (loaded with `aeon.benchmarking.results_loaders`); STCT variants and MiniRocket were run here. Times are mean seconds per dataset for extraction + fit + predict on one shared 20-thread machine, in the same run.

**Classification, 112 UCR datasets common to all methods** (`results/summary_classification.md`):

| Method | Mean acc. | Mean rank (of 17) | Features | Time (s) |
|---|---|---|---|---|
| HC2 | 0.876 | 5.0 | | |
| MR-Hydra | 0.868 | 5.9 | | |
| MultiROCKET | 0.866 | 6.3 | | |
| RDST | 0.863 | 7.5 | | |
| WEASEL-2.0 | 0.861 | 8.1 | | |
| MiniRocket + ridge (run here) | 0.860 | 7.9 | 9996 | 3.34 (median 0.61, max 122) |
| InceptionTime | 0.858 | 7.9 | | |
| ROCKET | 0.856 | 7.8 | | |
| QUANT | 0.854 | 7.7 | | |
| **STCT+ (ExtraTrees)** | **0.845** | **9.2** | **363** | **1.88 (median 1.21)** |
| FreshPRINCE | 0.842 | 9.5 | | |
| STCT+ with the paper's RF | 0.834 | 10.7 | 726 | 2.71 |
| STCT (paper, RF) | 0.782 | 13.2 | 138 | 0.85 |
| Catch22 | 0.781 | 14.2 | | |
| 1NN-DTW | 0.742 | 14.8 | | |

Paired Wilcoxon on the 112 datasets: STCT+ beats the paper's STCT on 88 datasets, ties on 9 and loses on 15; it is indistinguishable from FreshPRINCE (50 wins / 15 ties / 47 losses, p = 0.78) and still significantly behind MiniRocket (36 / 15 / 61, p = 0.002, mean gap 1.3 points) and HC2 (mean gap 3.1 points). The gap to MiniRocket went from 7.6 points to 1.3 at 4 % of its feature count and roughly half its mean time; the feature change alone (STCT+ with the original random forest) accounts for 5.2 of the 6.4 points gained.

**Regression, 26 TSER datasets** (`results/summary_regression.md`; mean rank over 18 methods, since raw RMSE is not comparable across datasets):

| Method | Mean rank | Features | Time (s) |
|---|---|---|---|
| **STCT+ with diff (ExtraTrees)** | **4.7** | 696 | 7.30 |
| **STCT+ (ExtraTrees)** | **4.8** | **348** | **3.78** |
| DrCIF | 5.8 | | |
| MiniRocket + ridge (run here) | 5.9 | 9996 | 1.89 |
| FreshPRINCE | 7.1 | | |
| STCT+ with the paper's RF | 7.2 | 696 | 8.79 |
| InceptionTime | 8.6 | | |
| RotF | 8.9 | | |
| ROCKET | 9.1 | | |
| STCT (paper, RF) | 9.3 | 134 | 1.53 |
| Ridge, RandF, XGBoost, GridSVR, ResNet, FPCR, 1NN-DTW, FCN | 11.1 to 14.8 | | |

Pairwise: STCT+ beats FreshPRINCE 19 / 7, RotF 20 / 6 (p = 0.03), InceptionTime 17 / 9 and ROCKET 18 / 8, and ties the best published method DrCIF (16 / 10, p = 0.92).

The configuration was chosen on 38 small UCR and 16 TSER datasets (`results/dev_ablations.md`); more scales, degree 3 or 6, extra statistics, more trees and larger per-split feature samples were all within noise, and RotationForest or gradient boosting cost 10 to 25x the time for no gain.

## Contents

- `stct_classification.ipynb`, `stct_regression.ipynb`: the paper's UCR / TSER benchmarks (wavelet, STFT, STCT, Rocket)
- `stct_plus.py`: STCT (`stct_fixed`, verbatim from the notebooks) and STCT+ (`stct_pooled`, `stct_plus`, sklearn-style `STCTPlus`)
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
from sklearn.ensemble import ExtraTreesClassifier
from stct_plus import stct_plus

clf = ExtraTreesClassifier(n_estimators=500, n_jobs=-1).fit(stct_plus(X_train, diff=False), y_train)
acc = clf.score(stct_plus(X_test, diff=False), y_test)
```

Reproduce the tables above (about an hour for classification, 15 minutes for regression):

```bash
python bench.py --task classification --datasets all --methods fixed:rf,plus:rf,fixed+pooled:et,plus:et,minirocket:ridge --out results/full_classification.csv
python compare.py --task classification --results results/full_classification.csv --methods fixed:rf,plus:rf,fixed+pooled:et,plus:et,minirocket:ridge --refs HC2,MR-Hydra,InceptionTime,MultiROCKET,MiniROCKET,ROCKET,FreshPRINCE,RDST,QUANT,WEASEL-2.0,Catch22,1NN-DTW
```

Default STCT setting: degree 4, two scales (8 and 16 windows). Default STCT+ setting: the same fixed block plus pooled scales 1/32, 1/16, 1/8, 1/4 with 8x window overlap.


## Contact

Samir Brahim Belhaouari (sbelhaouari@hbku.edu.qa), Yunis Carreon Kahalan (yuka34154@hbku.edu.qa)
