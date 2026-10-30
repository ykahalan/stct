# regression: STCT variants vs published results

26 datasets common to all methods (default train/test split).

## Overall

|                  |     mean |   mean_rank |   best_on |   mean_rel_rmse |   median_rel_rmse |
|:-----------------|---------:|------------:|----------:|----------------:|------------------:|
| stct_plus:et     |  112.798 |      4.1538 |         4 |          1.3499 |            1.1392 |
| DrCIF            |  105.584 |      5.3077 |         1 |          1.4326 |            1.2509 |
| minirocket:ridge |  124.082 |      5.3462 |         2 |          2.4591 |            1.237  |
| fixed:et         |  113.553 |      5.9231 |         1 |          1.8039 |            1.2513 |
| FreshPRINCE      |  123.212 |      6.5    |         3 |          1.4748 |            1.2161 |
| InceptionTime    |  128.912 |      7.9231 |         3 |          1.723  |            1.3262 |
| RotF             |  161.025 |      8.0769 |         5 |          2.4911 |            1.4151 |
| ROCKET           |  157.998 |      8.3846 |         3 |          3.2345 |            1.3356 |
| fixed:rf         |  138.324 |      8.4615 |         0 |          2.0686 |            1.3459 |
| RandF            |  159.715 |     10.1154 |         0 |          2.0271 |            1.6122 |
| Ridge            | 2220.62  |     10.6154 |         1 |         49.4748 |            1.4664 |
| XGBoost          |  179.227 |     11      |         0 |          2.3224 |            1.5915 |
| GridSVR          |  219.129 |     11.1154 |         2 |          4.7482 |            1.5554 |
| ResNet           |  242.422 |     11.7692 |         0 |          3.7926 |            1.5035 |
| FPCR             |  208.702 |     12.0385 |         0 |          3.0404 |            1.7593 |
| 1NN-DTW          |  193.866 |     12.4231 |         0 |          2.2013 |            1.6577 |
| FCN              |  290.396 |     13.8462 |         1 |          4.3462 |            2.547  |

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
| fixed:et         | FreshPRINCE   |     18 |      0 |        8 |       0.0292 |     -9.6587 |
| fixed:et         | DrCIF         |     11 |      0 |       15 |       0.2173 |      7.9696 |
| fixed:et         | InceptionTime |     15 |      0 |       11 |       0.7835 |    -15.359  |
| fixed:et         | ROCKET        |     14 |      0 |       12 |       0.5317 |    -44.4444 |
| fixed:et         | RotF          |     16 |      0 |       10 |       0.2914 |    -47.4717 |
| fixed:et         | Ridge         |     20 |      0 |        6 |       0.0669 |  -2107.07   |
| fixed:et         | XGBoost       |     23 |      0 |        3 |       0.0029 |    -65.6736 |
| fixed:et         | RandF         |     20 |      0 |        6 |       0.0163 |    -46.162  |
| fixed:et         | 1NN-DTW       |     21 |      0 |        5 |       0.0007 |    -80.3128 |
| fixed:et         | FPCR          |     22 |      0 |        4 |       0.0039 |    -95.1491 |
| fixed:et         | GridSVR       |     22 |      0 |        4 |       0.0056 |   -105.576  |
| fixed:et         | ResNet        |     22 |      0 |        4 |       0      |   -128.869  |
| fixed:et         | FCN           |     24 |      0 |        2 |       0      |   -176.843  |
| stct_plus:et     | FreshPRINCE   |     20 |      0 |        6 |       0.0312 |    -10.4138 |
| stct_plus:et     | DrCIF         |     16 |      0 |       10 |       0.9007 |      7.2145 |
| stct_plus:et     | InceptionTime |     18 |      0 |        8 |       0.2799 |    -16.1141 |
| stct_plus:et     | ROCKET        |     19 |      0 |        7 |       0.0163 |    -45.1995 |
| stct_plus:et     | RotF          |     20 |      0 |        6 |       0.0204 |    -48.2267 |
| stct_plus:et     | Ridge         |     21 |      0 |        5 |       0.0236 |  -2107.82   |
| stct_plus:et     | XGBoost       |     23 |      0 |        3 |       0.0008 |    -66.4286 |
| stct_plus:et     | RandF         |     24 |      0 |        2 |       0.0003 |    -46.9171 |
| stct_plus:et     | 1NN-DTW       |     24 |      0 |        2 |       0.0003 |    -81.0678 |
| stct_plus:et     | FPCR          |     22 |      0 |        4 |       0.0043 |    -95.9041 |
| stct_plus:et     | GridSVR       |     22 |      0 |        4 |       0.0061 |   -106.331  |
| stct_plus:et     | ResNet        |     25 |      0 |        1 |       0      |   -129.624  |
| stct_plus:et     | FCN           |     24 |      0 |        2 |       0      |   -177.598  |
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

| a            | b                |   a_wins |   ties |   b_wins |   wilcoxon_p |
|:-------------|:-----------------|---------:|-------:|---------:|-------------:|
| fixed:rf     | fixed:et         |        4 |      0 |       22 |       0.0005 |
| fixed:rf     | stct_plus:et     |        3 |      0 |       23 |       0.0013 |
| fixed:rf     | minirocket:ridge |        7 |      0 |       19 |       0.0151 |
| fixed:et     | stct_plus:et     |        9 |      0 |       17 |       0.3533 |
| fixed:et     | minirocket:ridge |        9 |      0 |       17 |       0.1651 |
| stct_plus:et | minirocket:ridge |       16 |      0 |       10 |       0.4227 |

## Mean rank is the headline for regression: raw RMSE is not comparable across datasets.
