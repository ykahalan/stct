# regression: STCT variants vs published results

26 datasets common to all methods (default train/test split).

## Overall

|                   |     mean |   mean_rank |   best_on |   mean_rel_rmse |   median_rel_rmse |
|:------------------|---------:|------------:|----------:|----------------:|------------------:|
| fixed+pooled:et   |  110.044 |      6.3077 |         2 |          1.4966 |            1.2279 |
| plus:et           |  109.453 |      6.3077 |         1 |          1.4807 |            1.2304 |
| minirocket:ridge  |  124.082 |      6.8462 |         2 |          2.542  |            1.2504 |
| base+rppv32:ridge |  141.272 |      6.9615 |         0 |          1.419  |            1.1892 |
| rppv128:ridge     |  133.027 |      7.0385 |         1 |          1.4437 |            1.1838 |
| rppv64:ridge      |  141.549 |      7.1154 |         2 |          1.4415 |            1.2228 |
| DrCIF             |  105.584 |      7.1154 |         1 |          1.528  |            1.2509 |
| FreshPRINCE       |  123.212 |      8.5    |         2 |          1.5425 |            1.2869 |
| InceptionTime     |  128.912 |      9.7692 |         3 |          1.8557 |            1.3491 |
| RotF              |  161.025 |     10.1538 |         5 |          2.6388 |            1.4139 |
| ROCKET            |  157.998 |     10.3846 |         3 |          3.5885 |            1.3356 |
| fixed:rf          |  138.324 |     10.5769 |         0 |          2.2348 |            1.3459 |
| RandF             |  159.715 |     12.5    |         0 |          2.2186 |            1.6122 |
| Ridge             | 2220.62  |     12.6923 |         1 |         67.2496 |            1.4664 |
| XGBoost           |  179.227 |     13.5385 |         0 |          2.515  |            1.5915 |
| GridSVR           |  219.129 |     13.6154 |         2 |          5.3399 |            1.5688 |
| ResNet            |  242.422 |     14.3846 |         0 |          4.2199 |            1.5575 |
| FPCR              |  208.702 |     14.6154 |         0 |          3.5095 |            1.7593 |
| 1NN-DTW           |  193.866 |     15.0385 |         0 |          2.399  |            1.6577 |
| FCN               |  290.396 |     16.5385 |         1 |          5.0002 |            2.547  |

## Pairwise: each STCT variant vs each reference (wins/ties/losses on RMSE, Wilcoxon p)

| variant           | reference     |   wins |   ties |   losses |   wilcoxon_p |   mean_diff |
|:------------------|:--------------|-------:|-------:|---------:|-------------:|------------:|
| fixed:rf          | FreshPRINCE   |      8 |      0 |       18 |       0.0357 |     15.1119 |
| fixed:rf          | DrCIF         |      5 |      0 |       21 |       0.0006 |     32.7403 |
| fixed:rf          | InceptionTime |     13 |      0 |       13 |       0.6171 |      9.4116 |
| fixed:rf          | ROCKET        |     14 |      0 |       12 |       0.8028 |    -19.6738 |
| fixed:rf          | RotF          |     13 |      0 |       13 |       0.5153 |    -22.701  |
| fixed:rf          | Ridge         |     17 |      0 |        9 |       0.3941 |  -2082.3    |
| fixed:rf          | XGBoost       |     19 |      0 |        7 |       0.0751 |    -40.9029 |
| fixed:rf          | RandF         |     19 |      0 |        7 |       0.0559 |    -21.3914 |
| fixed:rf          | 1NN-DTW       |     21 |      0 |        5 |       0.0032 |    -55.5421 |
| fixed:rf          | FPCR          |     21 |      0 |        5 |       0.0102 |    -70.3784 |
| fixed:rf          | GridSVR       |     19 |      0 |        7 |       0.0151 |    -80.805  |
| fixed:rf          | ResNet        |     18 |      0 |        8 |       0.013  |   -104.099  |
| fixed:rf          | FCN           |     21 |      0 |        5 |       0.0002 |   -152.072  |
| fixed+pooled:et   | FreshPRINCE   |     19 |      0 |        7 |       0.089  |    -13.1681 |
| fixed+pooled:et   | DrCIF         |     16 |      0 |       10 |       0.9205 |      4.4603 |
| fixed+pooled:et   | InceptionTime |     17 |      0 |        9 |       0.5317 |    -18.8684 |
| fixed+pooled:et   | ROCKET        |     18 |      0 |        8 |       0.1574 |    -47.9538 |
| fixed+pooled:et   | RotF          |     20 |      0 |        6 |       0.0272 |    -50.981  |
| fixed+pooled:et   | Ridge         |     18 |      0 |        8 |       0.3941 |  -2110.58   |
| fixed+pooled:et   | XGBoost       |     23 |      0 |        3 |       0.0005 |    -69.1829 |
| fixed+pooled:et   | RandF         |     24 |      0 |        2 |       0.0003 |    -49.6714 |
| fixed+pooled:et   | 1NN-DTW       |     24 |      0 |        2 |       0.0004 |    -83.8221 |
| fixed+pooled:et   | FPCR          |     22 |      0 |        4 |       0.0043 |    -98.6584 |
| fixed+pooled:et   | GridSVR       |     22 |      0 |        4 |       0.0061 |   -109.085  |
| fixed+pooled:et   | ResNet        |     24 |      0 |        2 |       0      |   -132.379  |
| fixed+pooled:et   | FCN           |     24 |      0 |        2 |       0      |   -180.352  |
| plus:et           | FreshPRINCE   |     22 |      0 |        4 |       0.0056 |    -13.759  |
| plus:et           | DrCIF         |     14 |      0 |       12 |       0.7644 |      3.8693 |
| plus:et           | InceptionTime |     18 |      0 |        8 |       0.4227 |    -19.4593 |
| plus:et           | ROCKET        |     18 |      0 |        8 |       0.0941 |    -48.5447 |
| plus:et           | RotF          |     20 |      0 |        6 |       0.0312 |    -51.572  |
| plus:et           | Ridge         |     18 |      0 |        8 |       0.3666 |  -2111.17   |
| plus:et           | XGBoost       |     23 |      0 |        3 |       0.0005 |    -69.7739 |
| plus:et           | RandF         |     24 |      0 |        2 |       0.0003 |    -50.2623 |
| plus:et           | 1NN-DTW       |     23 |      0 |        3 |       0.001  |    -84.4131 |
| plus:et           | FPCR          |     22 |      0 |        4 |       0.0036 |    -99.2494 |
| plus:et           | GridSVR       |     22 |      0 |        4 |       0.0039 |   -109.676  |
| plus:et           | ResNet        |     25 |      0 |        1 |       0      |   -132.969  |
| plus:et           | FCN           |     24 |      0 |        2 |       0      |   -180.943  |
| rppv128:ridge     | FreshPRINCE   |     17 |      0 |        9 |       0.4227 |      9.815  |
| rppv128:ridge     | DrCIF         |     13 |      0 |       13 |       0.4374 |     27.4433 |
| rppv128:ridge     | InceptionTime |     14 |      0 |       12 |       0.7835 |      4.1147 |
| rppv128:ridge     | ROCKET        |     17 |      0 |        9 |       0.3941 |    -24.9707 |
| rppv128:ridge     | RotF          |     17 |      0 |        9 |       0.3666 |    -27.998  |
| rppv128:ridge     | Ridge         |     19 |      0 |        7 |       0.1899 |  -2087.59   |
| rppv128:ridge     | XGBoost       |     21 |      0 |        5 |       0.0014 |    -46.1999 |
| rppv128:ridge     | RandF         |     20 |      0 |        6 |       0.0029 |    -26.6883 |
| rppv128:ridge     | 1NN-DTW       |     22 |      0 |        4 |       0.0024 |    -60.8391 |
| rppv128:ridge     | FPCR          |     22 |      0 |        4 |       0.0027 |    -75.6754 |
| rppv128:ridge     | GridSVR       |     22 |      0 |        4 |       0.0018 |    -86.102  |
| rppv128:ridge     | ResNet        |     23 |      0 |        3 |       0.0018 |   -109.395  |
| rppv128:ridge     | FCN           |     23 |      0 |        3 |       0      |   -157.369  |
| base+rppv32:ridge | FreshPRINCE   |     15 |      0 |       11 |       0.4227 |     18.0601 |
| base+rppv32:ridge | DrCIF         |     16 |      0 |       10 |       0.671  |     35.6884 |
| base+rppv32:ridge | InceptionTime |     18 |      0 |        8 |       0.3533 |     12.3598 |
| base+rppv32:ridge | ROCKET        |     17 |      0 |        9 |       0.2578 |    -16.7256 |
| base+rppv32:ridge | RotF          |     17 |      0 |        9 |       0.3153 |    -19.7528 |
| base+rppv32:ridge | Ridge         |     20 |      0 |        6 |       0.0236 |  -2079.35   |
| base+rppv32:ridge | XGBoost       |     22 |      0 |        4 |       0.0003 |    -37.9547 |
| base+rppv32:ridge | RandF         |     19 |      0 |        7 |       0.0047 |    -18.4432 |
| base+rppv32:ridge | 1NN-DTW       |     22 |      0 |        4 |       0.0079 |    -52.5939 |
| base+rppv32:ridge | FPCR          |     22 |      0 |        4 |       0.0003 |    -67.4302 |
| base+rppv32:ridge | GridSVR       |     21 |      0 |        5 |       0.0029 |    -77.8568 |
| base+rppv32:ridge | ResNet        |     20 |      0 |        6 |       0.0067 |   -101.15   |
| base+rppv32:ridge | FCN           |     23 |      0 |        3 |       0      |   -149.124  |
| rppv64:ridge      | FreshPRINCE   |     17 |      0 |        9 |       0.3533 |     18.3366 |
| rppv64:ridge      | DrCIF         |     15 |      0 |       11 |       1      |     35.9649 |
| rppv64:ridge      | InceptionTime |     14 |      0 |       12 |       0.7644 |     12.6363 |
| rppv64:ridge      | ROCKET        |     15 |      0 |       11 |       0.5317 |    -16.4491 |
| rppv64:ridge      | RotF          |     16 |      0 |       10 |       0.4227 |    -19.4763 |
| rppv64:ridge      | Ridge         |     20 |      0 |        6 |       0.1291 |  -2079.07   |
| rppv64:ridge      | XGBoost       |     23 |      0 |        3 |       0.0001 |    -37.6782 |
| rppv64:ridge      | RandF         |     19 |      0 |        7 |       0.0029 |    -18.1667 |
| rppv64:ridge      | 1NN-DTW       |     22 |      0 |        4 |       0.0086 |    -52.3175 |
| rppv64:ridge      | FPCR          |     23 |      0 |        3 |       0.0001 |    -67.1537 |
| rppv64:ridge      | GridSVR       |     22 |      0 |        4 |       0.0016 |    -77.5804 |
| rppv64:ridge      | ResNet        |     23 |      0 |        3 |       0.0024 |   -100.874  |
| rppv64:ridge      | FCN           |     24 |      0 |        2 |       0      |   -148.848  |
| minirocket:ridge  | FreshPRINCE   |     15 |      0 |       11 |       0.5153 |      0.8698 |
| minirocket:ridge  | DrCIF         |     15 |      0 |       11 |       0.5995 |     18.4981 |
| minirocket:ridge  | InceptionTime |     19 |      0 |        7 |       0.3153 |     -4.8305 |
| minirocket:ridge  | ROCKET        |     19 |      0 |        7 |       0.0272 |    -33.9159 |
| minirocket:ridge  | RotF          |     16 |      0 |       10 |       0.2173 |    -36.9431 |
| minirocket:ridge  | Ridge         |     22 |      0 |        4 |       0.0022 |  -2096.54   |
| minirocket:ridge  | XGBoost       |     22 |      0 |        4 |       0.0022 |    -55.145  |
| minirocket:ridge  | RandF         |     20 |      0 |        6 |       0.012  |    -35.6335 |
| minirocket:ridge  | 1NN-DTW       |     22 |      0 |        4 |       0.0032 |    -69.7842 |
| minirocket:ridge  | FPCR          |     21 |      0 |        5 |       0.013  |    -84.6205 |
| minirocket:ridge  | GridSVR       |     20 |      0 |        6 |       0.0176 |    -95.0471 |
| minirocket:ridge  | ResNet        |     24 |      0 |        2 |       0.0001 |   -118.341  |
| minirocket:ridge  | FCN           |     22 |      0 |        4 |       0.0001 |   -166.315  |

## Pairwise among STCT variants

| a                 | b                 |   a_wins |   ties |   b_wins |   wilcoxon_p |
|:------------------|:------------------|---------:|-------:|---------:|-------------:|
| fixed:rf          | fixed+pooled:et   |        3 |      0 |       23 |       0.0001 |
| fixed:rf          | plus:et           |        2 |      0 |       24 |       0.0001 |
| fixed:rf          | rppv128:ridge     |        8 |      0 |       18 |       0.0463 |
| fixed:rf          | base+rppv32:ridge |        8 |      0 |       18 |       0.0842 |
| fixed:rf          | rppv64:ridge      |        9 |      0 |       17 |       0.1651 |
| fixed:rf          | minirocket:ridge  |        7 |      0 |       19 |       0.0151 |
| fixed+pooled:et   | plus:et           |       13 |      0 |       13 |       0.5153 |
| fixed+pooled:et   | rppv128:ridge     |       13 |      0 |       13 |       0.8613 |
| fixed+pooled:et   | base+rppv32:ridge |       10 |      0 |       16 |       0.2799 |
| fixed+pooled:et   | rppv64:ridge      |       11 |      0 |       15 |       0.5995 |
| fixed+pooled:et   | minirocket:ridge  |       15 |      0 |       11 |       0.6893 |
| plus:et           | rppv128:ridge     |       11 |      0 |       15 |       0.8028 |
| plus:et           | base+rppv32:ridge |       10 |      0 |       16 |       0.2914 |
| plus:et           | rppv64:ridge      |       11 |      0 |       15 |       0.8028 |
| plus:et           | minirocket:ridge  |       14 |      0 |       12 |       0.9801 |
| rppv128:ridge     | base+rppv32:ridge |       14 |      0 |       12 |       0.9205 |
| rppv128:ridge     | rppv64:ridge      |       15 |      0 |       11 |       0.4374 |
| rppv128:ridge     | minirocket:ridge  |       12 |      0 |       14 |       0.7835 |
| base+rppv32:ridge | rppv64:ridge      |       13 |      0 |       13 |       0.9205 |
| base+rppv32:ridge | minirocket:ridge  |       12 |      0 |       14 |       0.6528 |
| rppv64:ridge      | minirocket:ridge  |       11 |      0 |       15 |       0.4834 |

## Mean rank is the headline for regression: raw RMSE is not comparable across datasets.
