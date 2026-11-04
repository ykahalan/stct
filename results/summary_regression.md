# regression: STCT variants vs published results

26 datasets common to all methods (default train/test split).

## Overall

|                      |     mean |   mean_rank |   best_on |   mean_rel_rmse |   median_rel_rmse |
|:---------------------|---------:|------------:|----------:|----------------:|------------------:|
| paper_views:et_tuned |  105.089 |      5.2692 |         4 |          1.359  |            1.1512 |
| paper_tuned:et_tuned |  112.212 |      5.7692 |         1 |          1.4724 |            1.208  |
| paper_pooled:et      |  110.895 |      5.8846 |         0 |          1.4496 |            1.1932 |
| minirocket:ridge     |  124.082 |      6.4615 |         2 |          2.4827 |            1.2486 |
| DrCIF                |  105.584 |      6.4615 |         1 |          1.4575 |            1.2509 |
| fixed:et             |  113.553 |      6.9615 |         0 |          1.8449 |            1.2513 |
| FreshPRINCE          |  123.212 |      7.9615 |         2 |          1.4915 |            1.2507 |
| InceptionTime        |  128.912 |      9.2308 |         3 |          1.7633 |            1.3369 |
| RotF                 |  161.025 |      9.5385 |         5 |          2.529  |            1.4283 |
| ROCKET               |  157.998 |      9.7692 |         3 |          3.3394 |            1.3356 |
| fixed:rf             |  138.324 |     10.1154 |         0 |          2.1092 |            1.3492 |
| Ridge                | 2220.62  |     11.8846 |         1 |         56.0684 |            1.4814 |
| RandF                |  159.715 |     11.9231 |         0 |          2.0711 |            1.6122 |
| GridSVR              |  219.129 |     12.8077 |         2 |          4.925  |            1.5554 |
| XGBoost              |  179.227 |     12.8462 |         0 |          2.367  |            1.5915 |
| ResNet               |  242.422 |     13.6154 |         0 |          3.9432 |            1.5035 |
| FPCR                 |  208.702 |     13.6923 |         0 |          3.1871 |            1.7593 |
| 1NN-DTW              |  193.866 |     14.1923 |         0 |          2.2579 |            1.6577 |
| FCN                  |  290.396 |     15.6154 |         1 |          4.572  |            2.5766 |

## Pairwise: each STCT variant vs each reference (wins/ties/losses on RMSE, Wilcoxon p)

| variant              | reference     |   wins |   ties |   losses |   wilcoxon_p |   mean_diff |
|:---------------------|:--------------|-------:|-------:|---------:|-------------:|------------:|
| fixed:rf             | FreshPRINCE   |      8 |      0 |       18 |       0.0357 |     15.1119 |
| fixed:rf             | DrCIF         |      5 |      0 |       21 |       0.0006 |     32.7403 |
| fixed:rf             | InceptionTime |     13 |      0 |       13 |       0.6171 |      9.4116 |
| fixed:rf             | ROCKET        |     14 |      0 |       12 |       0.8028 |    -19.6738 |
| fixed:rf             | RotF          |     13 |      0 |       13 |       0.5153 |    -22.701  |
| fixed:rf             | Ridge         |     17 |      0 |        9 |       0.3941 |  -2082.3    |
| fixed:rf             | XGBoost       |     19 |      0 |        7 |       0.0751 |    -40.9029 |
| fixed:rf             | RandF         |     19 |      0 |        7 |       0.0559 |    -21.3914 |
| fixed:rf             | 1NN-DTW       |     21 |      0 |        5 |       0.0032 |    -55.5421 |
| fixed:rf             | FPCR          |     21 |      0 |        5 |       0.0102 |    -70.3784 |
| fixed:rf             | GridSVR       |     19 |      0 |        7 |       0.0151 |    -80.805  |
| fixed:rf             | ResNet        |     18 |      0 |        8 |       0.013  |   -104.099  |
| fixed:rf             | FCN           |     21 |      0 |        5 |       0.0002 |   -152.072  |
| fixed:et             | FreshPRINCE   |     18 |      0 |        8 |       0.0292 |     -9.6587 |
| fixed:et             | DrCIF         |     11 |      0 |       15 |       0.2173 |      7.9696 |
| fixed:et             | InceptionTime |     15 |      0 |       11 |       0.7835 |    -15.359  |
| fixed:et             | ROCKET        |     14 |      0 |       12 |       0.5317 |    -44.4444 |
| fixed:et             | RotF          |     16 |      0 |       10 |       0.2914 |    -47.4717 |
| fixed:et             | Ridge         |     20 |      0 |        6 |       0.0669 |  -2107.07   |
| fixed:et             | XGBoost       |     23 |      0 |        3 |       0.0029 |    -65.6736 |
| fixed:et             | RandF         |     20 |      0 |        6 |       0.0163 |    -46.162  |
| fixed:et             | 1NN-DTW       |     21 |      0 |        5 |       0.0007 |    -80.3128 |
| fixed:et             | FPCR          |     22 |      0 |        4 |       0.0039 |    -95.1491 |
| fixed:et             | GridSVR       |     22 |      0 |        4 |       0.0056 |   -105.576  |
| fixed:et             | ResNet        |     22 |      0 |        4 |       0      |   -128.869  |
| fixed:et             | FCN           |     24 |      0 |        2 |       0      |   -176.843  |
| paper_pooled:et      | FreshPRINCE   |     19 |      0 |        7 |       0.0709 |    -12.3171 |
| paper_pooled:et      | DrCIF         |     15 |      0 |       11 |       0.9403 |      5.3113 |
| paper_pooled:et      | InceptionTime |     17 |      0 |        9 |       0.5822 |    -18.0174 |
| paper_pooled:et      | ROCKET        |     18 |      0 |        8 |       0.1651 |    -47.1028 |
| paper_pooled:et      | RotF          |     19 |      0 |        7 |       0.0292 |    -50.13   |
| paper_pooled:et      | Ridge         |     18 |      0 |        8 |       0.4227 |  -2109.72   |
| paper_pooled:et      | XGBoost       |     23 |      0 |        3 |       0.0006 |    -68.3319 |
| paper_pooled:et      | RandF         |     24 |      0 |        2 |       0.0004 |    -48.8204 |
| paper_pooled:et      | 1NN-DTW       |     23 |      0 |        3 |       0.0006 |    -82.9711 |
| paper_pooled:et      | FPCR          |     22 |      0 |        4 |       0.0043 |    -97.8074 |
| paper_pooled:et      | GridSVR       |     22 |      0 |        4 |       0.0061 |   -108.234  |
| paper_pooled:et      | ResNet        |     24 |      0 |        2 |       0      |   -131.528  |
| paper_pooled:et      | FCN           |     24 |      0 |        2 |       0      |   -179.501  |
| paper_tuned:et_tuned | FreshPRINCE   |     19 |      0 |        7 |       0.0796 |    -11.0003 |
| paper_tuned:et_tuned | DrCIF         |     15 |      0 |       11 |       0.9205 |      6.6281 |
| paper_tuned:et_tuned | InceptionTime |     17 |      0 |        9 |       0.5822 |    -16.7006 |
| paper_tuned:et_tuned | ROCKET        |     17 |      0 |        9 |       0.3032 |    -45.7859 |
| paper_tuned:et_tuned | RotF          |     19 |      0 |        7 |       0.0292 |    -48.8132 |
| paper_tuned:et_tuned | Ridge         |     18 |      0 |        8 |       0.4082 |  -2108.41   |
| paper_tuned:et_tuned | XGBoost       |     24 |      0 |        2 |       0.0004 |    -67.0151 |
| paper_tuned:et_tuned | RandF         |     24 |      0 |        2 |       0.0004 |    -47.5036 |
| paper_tuned:et_tuned | 1NN-DTW       |     23 |      0 |        3 |       0.0016 |    -81.6543 |
| paper_tuned:et_tuned | FPCR          |     22 |      0 |        4 |       0.0043 |    -96.4906 |
| paper_tuned:et_tuned | GridSVR       |     22 |      0 |        4 |       0.0061 |   -106.917  |
| paper_tuned:et_tuned | ResNet        |     24 |      0 |        2 |       0.0001 |   -130.211  |
| paper_tuned:et_tuned | FCN           |     24 |      0 |        2 |       0      |   -178.185  |
| paper_views:et_tuned | FreshPRINCE   |     20 |      0 |        6 |       0.0292 |    -18.1229 |
| paper_views:et_tuned | DrCIF         |     16 |      0 |       10 |       0.5995 |     -0.4946 |
| paper_views:et_tuned | InceptionTime |     18 |      0 |        8 |       0.4374 |    -23.8232 |
| paper_views:et_tuned | ROCKET        |     20 |      0 |        6 |       0.0594 |    -52.9086 |
| paper_views:et_tuned | RotF          |     20 |      0 |        6 |       0.0236 |    -55.9358 |
| paper_views:et_tuned | Ridge         |     18 |      0 |        8 |       0.3032 |  -2115.53   |
| paper_views:et_tuned | XGBoost       |     24 |      0 |        2 |       0.0004 |    -74.1377 |
| paper_views:et_tuned | RandF         |     23 |      0 |        3 |       0.0007 |    -54.6262 |
| paper_views:et_tuned | 1NN-DTW       |     24 |      0 |        2 |       0.0004 |    -88.777  |
| paper_views:et_tuned | FPCR          |     21 |      0 |        5 |       0.0047 |   -103.613  |
| paper_views:et_tuned | GridSVR       |     22 |      0 |        4 |       0.0047 |   -114.04   |
| paper_views:et_tuned | ResNet        |     25 |      0 |        1 |       0      |   -137.333  |
| paper_views:et_tuned | FCN           |     22 |      0 |        4 |       0.0002 |   -185.307  |
| minirocket:ridge     | FreshPRINCE   |     15 |      0 |       11 |       0.5153 |      0.8698 |
| minirocket:ridge     | DrCIF         |     15 |      0 |       11 |       0.5995 |     18.4981 |
| minirocket:ridge     | InceptionTime |     19 |      0 |        7 |       0.3153 |     -4.8305 |
| minirocket:ridge     | ROCKET        |     19 |      0 |        7 |       0.0272 |    -33.9159 |
| minirocket:ridge     | RotF          |     16 |      0 |       10 |       0.2173 |    -36.9431 |
| minirocket:ridge     | Ridge         |     22 |      0 |        4 |       0.0022 |  -2096.54   |
| minirocket:ridge     | XGBoost       |     22 |      0 |        4 |       0.0022 |    -55.145  |
| minirocket:ridge     | RandF         |     20 |      0 |        6 |       0.012  |    -35.6335 |
| minirocket:ridge     | 1NN-DTW       |     22 |      0 |        4 |       0.0032 |    -69.7842 |
| minirocket:ridge     | FPCR          |     21 |      0 |        5 |       0.013  |    -84.6205 |
| minirocket:ridge     | GridSVR       |     20 |      0 |        6 |       0.0176 |    -95.0471 |
| minirocket:ridge     | ResNet        |     24 |      0 |        2 |       0.0001 |   -118.341  |
| minirocket:ridge     | FCN           |     22 |      0 |        4 |       0.0001 |   -166.315  |

## Pairwise among STCT variants

| a                    | b                    |   a_wins |   ties |   b_wins |   wilcoxon_p |
|:---------------------|:---------------------|---------:|-------:|---------:|-------------:|
| fixed:rf             | fixed:et             |        4 |      0 |       22 |       0.0005 |
| fixed:rf             | paper_pooled:et      |        5 |      0 |       21 |       0.0024 |
| fixed:rf             | paper_tuned:et_tuned |        4 |      0 |       22 |       0.0009 |
| fixed:rf             | paper_views:et_tuned |        3 |      0 |       23 |       0.0004 |
| fixed:rf             | minirocket:ridge     |        7 |      0 |       19 |       0.0151 |
| fixed:et             | paper_pooled:et      |       12 |      0 |       14 |       0.8809 |
| fixed:et             | paper_tuned:et_tuned |       11 |      0 |       15 |       0.9602 |
| fixed:et             | paper_views:et_tuned |       11 |      0 |       15 |       0.6348 |
| fixed:et             | minirocket:ridge     |        9 |      0 |       17 |       0.1651 |
| paper_pooled:et      | paper_tuned:et_tuned |        8 |      8 |       10 |       0.9702 |
| paper_pooled:et      | paper_views:et_tuned |       11 |      0 |       15 |       0.4227 |
| paper_pooled:et      | minirocket:ridge     |       15 |      0 |       11 |       0.7454 |
| paper_tuned:et_tuned | paper_views:et_tuned |       10 |      0 |       16 |       0.1651 |
| paper_tuned:et_tuned | minirocket:ridge     |       15 |      0 |       11 |       0.8417 |
| paper_views:et_tuned | minirocket:ridge     |       15 |      0 |       11 |       0.7265 |

## Mean rank is the headline for regression: raw RMSE is not comparable across datasets.
