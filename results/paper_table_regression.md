# Regression: error, features and runtime

26 TSER datasets, 10 of them outside the development set; sorted by median RMSE relative to the best method on each dataset.

|   # | Method        |   RMSE vs best |   Held out |   Mean rank | STCT+ better / worse   | Features                                                  | Runtime here, total (h)               |   Median per dataset (s) | Published runtime (h)      |
|----:|:--------------|---------------:|-----------:|------------:|:-----------------------|:----------------------------------------------------------|:--------------------------------------|-------------------------:|:---------------------------|
|   1 | STCT+         |          1.131 |      1.187 |        3.23 |                        | 2,096 (143-3,090)                                         | 0.03                                  |                      2.9 |                            |
|   2 | DrCIF         |          1.251 |      1.248 |        4.19 | 18 / 8 (p = 0.303)     | 200 trees x 150-890 interval features (86,923 on average) | not run (published time over 9 hours) |                          | 175.4 on 25 datasets       |
|   3 | FreshPRINCE   |          1.287 |      1.335 |        4.85 | 18 / 8 (p = 0.269)     | 783                                                       | not run (published time over 9 hours) |                          | 14.5 on 25 datasets        |
|   4 | ROCKET        |          1.336 |      1.284 |        6.58 | 19 / 7 (p = 0.059)     | 20,000                                                    | 2.57                                  |                    199.1 | 2.8 on 25 datasets         |
|   5 | InceptionTime |          1.349 |      1.178 |        6.19 | 20 / 6 (p = 0.075)     | learned: 5 networks x 128                                 | not run (published time over 9 hours) |                          | 60.0 on 25 datasets        |
|   6 | RotF          |          1.367 |      1.226 |        6.35 | 20 / 6 (p = 0.020)     | raw series (1,816 values on average)                      | not run (published time over 9 hours) |                          | 15.6 on 25 datasets        |
|   7 | Ridge         |          1.466 |      1.346 |        8.38 | 22 / 4 (p = 0.006)     | raw series (1,816 values on average)                      | 0.00                                  |                      0   | 0.1 on 25 datasets         |
|   8 | GridSVR       |          1.555 |      1.555 |        8.77 | 22 / 4 (p = 0.005)     | raw series (1,816 values on average)                      | 0.65                                  |                     11.6 | 0.8 on 25 datasets         |
|   9 | ResNet        |          1.557 |      1.386 |        9.23 | 23 / 3 (p = 0.001)     | learned: 128                                              | not run (published time over 9 hours) |                          | 14.1 on 25 datasets        |
|  10 | XGBoost       |          1.592 |      1.68  |        8.58 | 24 / 2 (p = 0.000)     | raw series (1,816 values on average)                      | not run (not installed)               |                          | 0.6 on 25 datasets         |
|  11 | RandF         |          1.612 |      1.679 |        7.81 | 23 / 3 (p = 0.000)     | raw series (1,816 values on average)                      | 0.49                                  |                     31.8 | 1.7 on 25 datasets         |
|  12 | 1NN-DTW       |          1.658 |      1.677 |        9.92 | 23 / 3 (p = 0.001)     | none (raw series)                                         | 7.99                                  |                    131.5 | corrupted (negative times) |
|  13 | FPCR          |          1.759 |      1.71  |        9.65 | 24 / 2 (p = 0.000)     | functional principal components of the raw series         | not run (not installed)               |                          | 0.1 on 25 datasets         |
|  14 | FCN           |          2.547 |      2.448 |       11.27 | 24 / 2 (p = 0.000)     | learned: 128                                              | not run (published time over 9 hours) |                          | 9.3 on 25 datasets         |

Runtime here: fit on the training split plus predict on the test split, one thread, summed over the datasets; all methods on a dataset ran in the same process on the same machine (paper_time_reg.py). Published runtime: fit + predict times from timeseriesclassification.com (other hardware; not comparable with the runtime here).

RMSE measured here divided by published RMSE, geometric mean per method (configuration check): 1NN-DTW 1.001, GridSVR 1.000, ROCKET 0.997, RandF 1.005, Ridge 1.000
