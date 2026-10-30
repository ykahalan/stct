# regression: STCT variants vs published results

26 datasets common to all methods (default train/test split).

## Overall

|                  |     mean |   mean_rank |   best_on |   mean_rel_rmse |   median_rel_rmse |
|:-----------------|---------:|------------:|----------:|----------------:|------------------:|
| plus:et          |  109.453 |      4.6923 |         1 |          1.4187 |            1.1971 |
| fixed+pooled:et  |  110.044 |      4.7692 |         4 |          1.4374 |            1.1836 |
| DrCIF            |  105.584 |      5.8077 |         1 |          1.4367 |            1.2509 |
| minirocket:ridge |  124.082 |      5.8846 |         2 |          2.4633 |            1.2359 |
| FreshPRINCE      |  123.212 |      7.0769 |         3 |          1.4786 |            1.2244 |
| plus:rf          |  132.988 |      7.1923 |         1 |          1.7051 |            1.2653 |
| InceptionTime    |  128.912 |      8.5769 |         3 |          1.733  |            1.3262 |
| RotF             |  161.025 |      8.9231 |         5 |          2.4981 |            1.4139 |
| ROCKET           |  157.998 |      9.1154 |         2 |          3.2429 |            1.3356 |
| fixed:rf         |  138.324 |      9.2692 |         0 |          2.0753 |            1.3459 |
| Ridge            | 2220.62  |     11.0769 |         1 |         49.4858 |            1.4664 |
| RandF            |  159.715 |     11.1538 |         0 |          2.0347 |            1.6122 |
| XGBoost          |  179.227 |     11.8462 |         0 |          2.3301 |            1.5915 |
| GridSVR          |  219.129 |     11.9615 |         2 |          4.7648 |            1.5688 |
| ResNet           |  242.422 |     12.6538 |         0 |          3.8042 |            1.5035 |
| FPCR             |  208.702 |     12.8846 |         0 |          3.0499 |            1.7593 |
| 1NN-DTW          |  193.866 |     13.3462 |         0 |          2.2082 |            1.6577 |
| FCN              |  290.396 |     14.7692 |         1 |          4.3603 |            2.547  |

## Pairwise: each STCT variant vs each reference (wins/ties/losses on RMSE, Wilcoxon p)

| variant          | reference     |   wins |   ties |   losses |   wilcoxon_p |   mean_diff |
|:-----------------|:--------------|-------:|-------:|---------:|-------------:|------------:|
| fixed:rf         | FreshPRINCE   |      8 |      0 |       18 |       0.0357 |     15.1119 |
| fixed:rf         | DrCIF         |      5 |      0 |       21 |       0.0006 |     32.7403 |
| fixed:rf         | InceptionTime |     13 |      0 |       13 |       0.6171 |      9.4116 |
| fixed:rf         | ROCKET        |     14 |      0 |       12 |       0.8028 |    -19.6738 |
| fixed:rf         | RotF          |     13 |      0 |       13 |       0.5153 |    -22.701  |
| fixed:rf         | Ridge         |     17 |      0 |        9 |       0.3941 |  -2082.3    |
| fixed:rf         | XGBoost       |     19 |      0 |        7 |       0.0751 |    -40.9029 |
| fixed:rf         | RandF         |     19 |      0 |        7 |       0.0559 |    -21.3914 |
| fixed:rf         | 1NN-DTW       |     21 |      0 |        5 |       0.0032 |    -55.5421 |
| fixed:rf         | FPCR          |     21 |      0 |        5 |       0.0102 |    -70.3784 |
| fixed:rf         | GridSVR       |     19 |      0 |        7 |       0.0151 |    -80.805  |
| fixed:rf         | ResNet        |     18 |      0 |        8 |       0.013  |   -104.099  |
| fixed:rf         | FCN           |     21 |      0 |        5 |       0.0002 |   -152.072  |
| plus:rf          | FreshPRINCE   |     12 |      0 |       14 |       0.227  |      9.7766 |
| plus:rf          | DrCIF         |     10 |      0 |       16 |       0.0751 |     27.4049 |
| plus:rf          | InceptionTime |     15 |      0 |       11 |       0.8028 |      4.0763 |
| plus:rf          | ROCKET        |     16 |      0 |       10 |       0.7644 |    -25.0091 |
| plus:rf          | RotF          |     18 |      0 |        8 |       0.2079 |    -28.0364 |
| plus:rf          | Ridge         |     17 |      0 |        9 |       0.4992 |  -2087.63   |
| plus:rf          | XGBoost       |     22 |      0 |        4 |       0.0014 |    -46.2383 |
| plus:rf          | RandF         |     23 |      0 |        3 |       0.0005 |    -26.7267 |
| plus:rf          | 1NN-DTW       |     22 |      0 |        4 |       0.0043 |    -60.8775 |
| plus:rf          | FPCR          |     22 |      0 |        4 |       0.0051 |    -75.7138 |
| plus:rf          | GridSVR       |     22 |      0 |        4 |       0.0027 |    -86.1404 |
| plus:rf          | ResNet        |     21 |      0 |        5 |       0.0056 |   -109.434  |
| plus:rf          | FCN           |     24 |      0 |        2 |       0      |   -157.408  |
| fixed+pooled:et  | FreshPRINCE   |     19 |      0 |        7 |       0.089  |    -13.1681 |
| fixed+pooled:et  | DrCIF         |     16 |      0 |       10 |       0.9205 |      4.4603 |
| fixed+pooled:et  | InceptionTime |     17 |      0 |        9 |       0.5317 |    -18.8684 |
| fixed+pooled:et  | ROCKET        |     18 |      0 |        8 |       0.1574 |    -47.9538 |
| fixed+pooled:et  | RotF          |     20 |      0 |        6 |       0.0272 |    -50.981  |
| fixed+pooled:et  | Ridge         |     18 |      0 |        8 |       0.3941 |  -2110.58   |
| fixed+pooled:et  | XGBoost       |     23 |      0 |        3 |       0.0005 |    -69.1829 |
| fixed+pooled:et  | RandF         |     24 |      0 |        2 |       0.0003 |    -49.6714 |
| fixed+pooled:et  | 1NN-DTW       |     24 |      0 |        2 |       0.0004 |    -83.8221 |
| fixed+pooled:et  | FPCR          |     22 |      0 |        4 |       0.0043 |    -98.6584 |
| fixed+pooled:et  | GridSVR       |     22 |      0 |        4 |       0.0061 |   -109.085  |
| fixed+pooled:et  | ResNet        |     24 |      0 |        2 |       0      |   -132.379  |
| fixed+pooled:et  | FCN           |     24 |      0 |        2 |       0      |   -180.352  |
| plus:et          | FreshPRINCE   |     22 |      0 |        4 |       0.0056 |    -13.759  |
| plus:et          | DrCIF         |     14 |      0 |       12 |       0.7644 |      3.8693 |
| plus:et          | InceptionTime |     18 |      0 |        8 |       0.4227 |    -19.4593 |
| plus:et          | ROCKET        |     18 |      0 |        8 |       0.0941 |    -48.5447 |
| plus:et          | RotF          |     20 |      0 |        6 |       0.0312 |    -51.572  |
| plus:et          | Ridge         |     18 |      0 |        8 |       0.3666 |  -2111.17   |
| plus:et          | XGBoost       |     23 |      0 |        3 |       0.0005 |    -69.7739 |
| plus:et          | RandF         |     24 |      0 |        2 |       0.0003 |    -50.2623 |
| plus:et          | 1NN-DTW       |     23 |      0 |        3 |       0.001  |    -84.4131 |
| plus:et          | FPCR          |     22 |      0 |        4 |       0.0036 |    -99.2494 |
| plus:et          | GridSVR       |     22 |      0 |        4 |       0.0039 |   -109.676  |
| plus:et          | ResNet        |     25 |      0 |        1 |       0      |   -132.969  |
| plus:et          | FCN           |     24 |      0 |        2 |       0      |   -180.943  |
| minirocket:ridge | FreshPRINCE   |     15 |      0 |       11 |       0.5153 |      0.8698 |
| minirocket:ridge | DrCIF         |     15 |      0 |       11 |       0.5995 |     18.4981 |
| minirocket:ridge | InceptionTime |     19 |      0 |        7 |       0.3153 |     -4.8305 |
| minirocket:ridge | ROCKET        |     19 |      0 |        7 |       0.0272 |    -33.9159 |
| minirocket:ridge | RotF          |     16 |      0 |       10 |       0.2173 |    -36.9431 |
| minirocket:ridge | Ridge         |     22 |      0 |        4 |       0.0022 |  -2096.54   |
| minirocket:ridge | XGBoost       |     22 |      0 |        4 |       0.0022 |    -55.145  |
| minirocket:ridge | RandF         |     20 |      0 |        6 |       0.012  |    -35.6335 |
| minirocket:ridge | 1NN-DTW       |     22 |      0 |        4 |       0.0032 |    -69.7842 |
| minirocket:ridge | FPCR          |     21 |      0 |        5 |       0.013  |    -84.6205 |
| minirocket:ridge | GridSVR       |     20 |      0 |        6 |       0.0176 |    -95.0471 |
| minirocket:ridge | ResNet        |     24 |      0 |        2 |       0.0001 |   -118.341  |
| minirocket:ridge | FCN           |     22 |      0 |        4 |       0.0001 |   -166.315  |

## Pairwise among STCT variants

| a               | b                |   a_wins |   ties |   b_wins |   wilcoxon_p |
|:----------------|:-----------------|---------:|-------:|---------:|-------------:|
| fixed:rf        | plus:rf          |        7 |      0 |       19 |       0.1651 |
| fixed:rf        | fixed+pooled:et  |        3 |      0 |       23 |       0.0001 |
| fixed:rf        | plus:et          |        2 |      0 |       24 |       0.0001 |
| fixed:rf        | minirocket:ridge |        7 |      0 |       19 |       0.0151 |
| plus:rf         | fixed+pooled:et  |        4 |      0 |       22 |       0.0007 |
| plus:rf         | plus:et          |        4 |      0 |       22 |       0.0005 |
| plus:rf         | minirocket:ridge |       10 |      0 |       16 |       0.1227 |
| fixed+pooled:et | plus:et          |       13 |      0 |       13 |       0.5153 |
| fixed+pooled:et | minirocket:ridge |       15 |      0 |       11 |       0.6893 |
| plus:et         | minirocket:ridge |       14 |      0 |       12 |       0.9801 |

## Mean rank is the headline for regression: raw RMSE is not comparable across datasets.
