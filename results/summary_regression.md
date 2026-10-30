# regression: STCT variants vs published results

26 datasets common to all methods (default train/test split).

## Overall

|                   |     mean |   mean_rank |   best_on |   mean_rel_rmse |   median_rel_rmse |
|:------------------|---------:|------------:|----------:|----------------:|------------------:|
| ms5+pooled:et     |  109.867 |      7.3077 |         1 |          1.4299 |            1.1951 |
| msP7+pooled:et    |  108.57  |      7.8462 |         1 |          1.4158 |            1.2132 |
| fixed+pooled:et   |  110.044 |      8.0385 |         1 |          1.494  |            1.2279 |
| pyr3:et           |  106.554 |      8.3846 |         2 |          1.5028 |            1.2174 |
| ms5:et            |  112.756 |      8.4231 |         2 |          1.6616 |            1.2451 |
| msP7:et           |  107.91  |      8.7308 |         0 |          1.652  |            1.2251 |
| base+rppv32:ridge |  141.272 |      8.8077 |         0 |          1.4194 |            1.1892 |
| minirocket:ridge  |  124.082 |      8.8462 |         2 |          2.5395 |            1.255  |
| rppv128:ridge     |  133.027 |      9.1154 |         3 |          1.4468 |            1.1838 |
| DrCIF             |  105.584 |      9.1923 |         0 |          1.5224 |            1.2541 |
| fixed:et          |  113.553 |     10.1154 |         0 |          1.9549 |            1.2513 |
| FreshPRINCE       |  123.212 |     11.5385 |         1 |          1.5377 |            1.2983 |
| InceptionTime     |  128.912 |     12.6538 |         2 |          1.8489 |            1.3491 |
| RotF              |  161.025 |     13.0385 |         5 |          2.6278 |            1.4156 |
| ROCKET            |  157.998 |     13.1923 |         2 |          3.6059 |            1.3356 |
| fixed:rf          |  138.324 |     14.1538 |         0 |          2.2228 |            1.3459 |
| Ridge             | 2220.62  |     15.8846 |         1 |         66.7658 |            1.4816 |
| RandF             |  159.715 |     16      |         0 |          2.2021 |            1.6122 |
| GridSVR           |  219.129 |     16.9615 |         2 |          5.3391 |            1.557  |
| XGBoost           |  179.227 |     17.1154 |         0 |          2.4985 |            1.5915 |
| ResNet            |  242.422 |     17.8077 |         0 |          4.2127 |            1.5575 |
| FPCR              |  208.702 |     17.9615 |         0 |          3.4908 |            1.7593 |
| 1NN-DTW           |  193.866 |     18.6923 |         0 |          2.3911 |            1.6577 |
| FCN               |  290.396 |     20.1923 |         1 |          4.9795 |            2.5695 |

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
| fixed:et          | FreshPRINCE   |     18 |      0 |        8 |       0.0292 |     -9.6587 |
| fixed:et          | DrCIF         |     11 |      0 |       15 |       0.2173 |      7.9696 |
| fixed:et          | InceptionTime |     15 |      0 |       11 |       0.7835 |    -15.359  |
| fixed:et          | ROCKET        |     14 |      0 |       12 |       0.5317 |    -44.4444 |
| fixed:et          | RotF          |     16 |      0 |       10 |       0.2914 |    -47.4717 |
| fixed:et          | Ridge         |     20 |      0 |        6 |       0.0669 |  -2107.07   |
| fixed:et          | XGBoost       |     23 |      0 |        3 |       0.0029 |    -65.6736 |
| fixed:et          | RandF         |     20 |      0 |        6 |       0.0163 |    -46.162  |
| fixed:et          | 1NN-DTW       |     21 |      0 |        5 |       0.0007 |    -80.3128 |
| fixed:et          | FPCR          |     22 |      0 |        4 |       0.0039 |    -95.1491 |
| fixed:et          | GridSVR       |     22 |      0 |        4 |       0.0056 |   -105.576  |
| fixed:et          | ResNet        |     22 |      0 |        4 |       0      |   -128.869  |
| fixed:et          | FCN           |     24 |      0 |        2 |       0      |   -176.843  |
| ms5:et            | FreshPRINCE   |     19 |      0 |        7 |       0.014  |    -10.4562 |
| ms5:et            | DrCIF         |     11 |      0 |       15 |       0.6171 |      7.1721 |
| ms5:et            | InceptionTime |     17 |      0 |        9 |       0.237  |    -16.1565 |
| ms5:et            | ROCKET        |     18 |      0 |        8 |       0.1814 |    -45.2419 |
| ms5:et            | RotF          |     18 |      0 |        8 |       0.1048 |    -48.2692 |
| ms5:et            | Ridge         |     21 |      0 |        5 |       0.0254 |  -2107.86   |
| ms5:et            | XGBoost       |     23 |      0 |        3 |       0.0029 |    -66.4711 |
| ms5:et            | RandF         |     22 |      0 |        4 |       0.0073 |    -46.9595 |
| ms5:et            | 1NN-DTW       |     24 |      0 |        2 |       0.0004 |    -81.1103 |
| ms5:et            | FPCR          |     22 |      0 |        4 |       0.0016 |    -95.9466 |
| ms5:et            | GridSVR       |     22 |      0 |        4 |       0.0051 |   -106.373  |
| ms5:et            | ResNet        |     22 |      0 |        4 |       0      |   -129.667  |
| ms5:et            | FCN           |     24 |      0 |        2 |       0      |   -177.641  |
| msP7:et           | FreshPRINCE   |     19 |      0 |        7 |       0.0163 |    -15.3019 |
| msP7:et           | DrCIF         |     12 |      0 |       14 |       1      |      2.3264 |
| msP7:et           | InceptionTime |     18 |      0 |        8 |       0.1048 |    -21.0022 |
| msP7:et           | ROCKET        |     18 |      0 |        8 |       0.0594 |    -50.0876 |
| msP7:et           | RotF          |     18 |      0 |        8 |       0.1048 |    -53.1149 |
| msP7:et           | Ridge         |     22 |      0 |        4 |       0.0036 |  -2112.71   |
| msP7:et           | XGBoost       |     23 |      0 |        3 |       0.0027 |    -71.3168 |
| msP7:et           | RandF         |     21 |      0 |        5 |       0.0073 |    -51.8052 |
| msP7:et           | 1NN-DTW       |     24 |      0 |        2 |       0.0003 |    -85.956  |
| msP7:et           | FPCR          |     22 |      0 |        4 |       0.0036 |   -100.792  |
| msP7:et           | GridSVR       |     22 |      0 |        4 |       0.0056 |   -111.219  |
| msP7:et           | ResNet        |     21 |      0 |        5 |       0      |   -134.512  |
| msP7:et           | FCN           |     24 |      0 |        2 |       0      |   -182.486  |
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
| ms5+pooled:et     | FreshPRINCE   |     21 |      0 |        5 |       0.0102 |    -13.3449 |
| ms5+pooled:et     | DrCIF         |     17 |      0 |        9 |       0.6348 |      4.2834 |
| ms5+pooled:et     | InceptionTime |     18 |      0 |        8 |       0.3941 |    -19.0452 |
| ms5+pooled:et     | ROCKET        |     19 |      0 |        7 |       0.0796 |    -48.1306 |
| ms5+pooled:et     | RotF          |     20 |      0 |        6 |       0.0204 |    -51.1578 |
| ms5+pooled:et     | Ridge         |     18 |      0 |        8 |       0.3533 |  -2110.75   |
| ms5+pooled:et     | XGBoost       |     24 |      0 |        2 |       0.0004 |    -69.3597 |
| ms5+pooled:et     | RandF         |     24 |      0 |        2 |       0.0003 |    -49.8482 |
| ms5+pooled:et     | 1NN-DTW       |     24 |      0 |        2 |       0.0003 |    -83.9989 |
| ms5+pooled:et     | FPCR          |     22 |      0 |        4 |       0.0036 |    -98.8352 |
| ms5+pooled:et     | GridSVR       |     22 |      0 |        4 |       0.0061 |   -109.262  |
| ms5+pooled:et     | ResNet        |     25 |      0 |        1 |       0      |   -132.555  |
| ms5+pooled:et     | FCN           |     24 |      0 |        2 |       0      |   -180.529  |
| msP7+pooled:et    | FreshPRINCE   |     21 |      0 |        5 |       0.0029 |    -14.6415 |
| msP7+pooled:et    | DrCIF         |     15 |      0 |       11 |       0.8417 |      2.9869 |
| msP7+pooled:et    | InceptionTime |     19 |      0 |        7 |       0.0993 |    -20.3418 |
| msP7+pooled:et    | ROCKET        |     18 |      0 |        8 |       0.1165 |    -49.4272 |
| msP7+pooled:et    | RotF          |     20 |      0 |        6 |       0.022  |    -52.4544 |
| msP7+pooled:et    | Ridge         |     21 |      0 |        5 |       0.0357 |  -2112.05   |
| msP7+pooled:et    | XGBoost       |     23 |      0 |        3 |       0.0005 |    -70.6563 |
| msP7+pooled:et    | RandF         |     24 |      0 |        2 |       0.0003 |    -51.1448 |
| msP7+pooled:et    | 1NN-DTW       |     24 |      0 |        2 |       0.0003 |    -85.2955 |
| msP7+pooled:et    | FPCR          |     22 |      0 |        4 |       0.0036 |   -100.132  |
| msP7+pooled:et    | GridSVR       |     22 |      0 |        4 |       0.0061 |   -110.558  |
| msP7+pooled:et    | ResNet        |     24 |      0 |        2 |       0      |   -133.852  |
| msP7+pooled:et    | FCN           |     24 |      0 |        2 |       0      |   -181.826  |
| pyr3:et           | FreshPRINCE   |     20 |      0 |        6 |       0.0204 |    -16.6581 |
| pyr3:et           | DrCIF         |     17 |      0 |        9 |       0.1651 |      0.9702 |
| pyr3:et           | InceptionTime |     20 |      0 |        6 |       0.0993 |    -22.3584 |
| pyr3:et           | ROCKET        |     19 |      0 |        7 |       0.0941 |    -51.4438 |
| pyr3:et           | RotF          |     19 |      0 |        7 |       0.0594 |    -54.4711 |
| pyr3:et           | Ridge         |     19 |      0 |        7 |       0.1814 |  -2114.07   |
| pyr3:et           | XGBoost       |     23 |      0 |        3 |       0.0005 |    -72.673  |
| pyr3:et           | RandF         |     23 |      0 |        3 |       0.0007 |    -53.1614 |
| pyr3:et           | 1NN-DTW       |     23 |      0 |        3 |       0.0014 |    -87.3122 |
| pyr3:et           | FPCR          |     22 |      0 |        4 |       0.0036 |   -102.148  |
| pyr3:et           | GridSVR       |     21 |      0 |        5 |       0.0067 |   -112.575  |
| pyr3:et           | ResNet        |     23 |      0 |        3 |       0.0001 |   -135.869  |
| pyr3:et           | FCN           |     23 |      0 |        3 |       0      |   -183.843  |
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
| fixed:rf          | fixed:et          |        4 |      0 |       22 |       0.0005 |
| fixed:rf          | ms5:et            |        3 |      0 |       23 |       0.0004 |
| fixed:rf          | msP7:et           |        4 |      0 |       22 |       0.0032 |
| fixed:rf          | fixed+pooled:et   |        3 |      0 |       23 |       0.0001 |
| fixed:rf          | ms5+pooled:et     |        4 |      0 |       22 |       0.0002 |
| fixed:rf          | msP7+pooled:et    |        4 |      0 |       22 |       0.0009 |
| fixed:rf          | pyr3:et           |        3 |      0 |       23 |       0.0001 |
| fixed:rf          | rppv128:ridge     |        8 |      0 |       18 |       0.0463 |
| fixed:rf          | base+rppv32:ridge |        8 |      0 |       18 |       0.0842 |
| fixed:rf          | minirocket:ridge  |        7 |      0 |       19 |       0.0151 |
| fixed:et          | ms5:et            |       10 |      0 |       16 |       0.227  |
| fixed:et          | msP7:et           |        9 |      0 |       17 |       0.1427 |
| fixed:et          | fixed+pooled:et   |       12 |      0 |       14 |       0.8613 |
| fixed:et          | ms5+pooled:et     |        9 |      0 |       17 |       0.3153 |
| fixed:et          | msP7+pooled:et    |       10 |      0 |       16 |       0.237  |
| fixed:et          | pyr3:et           |       10 |      0 |       16 |       0.7644 |
| fixed:et          | rppv128:ridge     |       12 |      0 |       14 |       0.9007 |
| fixed:et          | base+rppv32:ridge |       10 |      0 |       16 |       0.2472 |
| fixed:et          | minirocket:ridge  |        9 |      0 |       17 |       0.1651 |
| ms5:et            | msP7:et           |       14 |      0 |       12 |       0.5483 |
| ms5:et            | fixed+pooled:et   |       14 |      0 |       12 |       0.3276 |
| ms5:et            | ms5+pooled:et     |       13 |      0 |       13 |       0.5483 |
| ms5:et            | msP7+pooled:et    |       10 |      0 |       16 |       0.3032 |
| ms5:et            | pyr3:et           |       13 |      0 |       13 |       0.5483 |
| ms5:et            | rppv128:ridge     |       13 |      0 |       13 |       0.7835 |
| ms5:et            | base+rppv32:ridge |       14 |      0 |       12 |       0.9403 |
| ms5:et            | minirocket:ridge  |       12 |      0 |       14 |       0.7644 |
| msP7:et           | fixed+pooled:et   |       12 |      0 |       14 |       0.4227 |
| msP7:et           | ms5+pooled:et     |       10 |      0 |       16 |       0.9403 |
| msP7:et           | msP7+pooled:et    |       12 |      0 |       14 |       0.4992 |
| msP7:et           | pyr3:et           |       12 |      0 |       14 |       0.6348 |
| msP7:et           | rppv128:ridge     |       12 |      0 |       14 |       0.9007 |
| msP7:et           | base+rppv32:ridge |       12 |      0 |       14 |       0.9602 |
| msP7:et           | minirocket:ridge  |       12 |      0 |       14 |       0.9602 |
| fixed+pooled:et   | ms5+pooled:et     |       12 |      0 |       14 |       0.6528 |
| fixed+pooled:et   | msP7+pooled:et    |       16 |      0 |       10 |       1      |
| fixed+pooled:et   | pyr3:et           |       15 |      0 |       11 |       0.8613 |
| fixed+pooled:et   | rppv128:ridge     |       13 |      0 |       13 |       0.8613 |
| fixed+pooled:et   | base+rppv32:ridge |       10 |      0 |       16 |       0.2799 |
| fixed+pooled:et   | minirocket:ridge  |       15 |      0 |       11 |       0.6893 |
| ms5+pooled:et     | msP7+pooled:et    |       17 |      0 |        9 |       0.6171 |
| ms5+pooled:et     | pyr3:et           |       17 |      0 |        9 |       0.5317 |
| ms5+pooled:et     | rppv128:ridge     |       13 |      0 |       13 |       0.7644 |
| ms5+pooled:et     | base+rppv32:ridge |       12 |      0 |       14 |       0.6348 |
| ms5+pooled:et     | minirocket:ridge  |       15 |      0 |       11 |       0.5995 |
| msP7+pooled:et    | pyr3:et           |       16 |      0 |       10 |       0.8222 |
| msP7+pooled:et    | rppv128:ridge     |       12 |      0 |       14 |       0.9403 |
| msP7+pooled:et    | base+rppv32:ridge |       12 |      0 |       14 |       0.8028 |
| msP7+pooled:et    | minirocket:ridge  |       16 |      0 |       10 |       0.3802 |
| pyr3:et           | rppv128:ridge     |       14 |      0 |       12 |       0.6893 |
| pyr3:et           | base+rppv32:ridge |       11 |      0 |       15 |       0.4082 |
| pyr3:et           | minirocket:ridge  |       13 |      0 |       13 |       0.9007 |
| rppv128:ridge     | base+rppv32:ridge |       14 |      0 |       12 |       0.9205 |
| rppv128:ridge     | minirocket:ridge  |       12 |      0 |       14 |       0.7835 |
| base+rppv32:ridge | minirocket:ridge  |       12 |      0 |       14 |       0.6528 |

## Mean rank is the headline for regression: raw RMSE is not comparable across datasets.
