# regression: one recipe at a time vs published methods

## all datasets (score = median RMSE relative to best; place out of 14)

| recipe           |   datasets |   score |   mean_rank |   place | vs DrCIF      | vs FreshPRINCE   |
|:-----------------|-----------:|--------:|------------:|--------:|:--------------|:-----------------|
| fixed:rf         |         26 |  1.3459 |      6      |       4 | 5/21 p=0.001  | 8/18 p=0.036     |
| fixed:et         |         26 |  1.2513 |      4.4615 |       2 | 11/15 p=0.217 | 18/8 p=0.029     |
| ms5:et           |         26 |  1.2451 |      3.8846 |       1 | 11/15 p=0.617 | 19/7 p=0.014     |
| msP7:et          |         26 |  1.2251 |      3.8462 |       1 | 12/14 p=1.000 | 19/7 p=0.016     |
| fixed+pooled:et  |         26 |  1.1836 |      3.5769 |       1 | 16/10 p=0.920 | 19/7 p=0.089     |
| ms5+pooled:et    |         26 |  1.1276 |      3.3077 |       1 | 17/9 p=0.635  | 21/5 p=0.010     |
| rppv128:ridge    |         26 |  1.1802 |      4.3846 |       2 | 13/13 p=0.437 | 17/9 p=0.423     |
| minirocket:ridge |         26 |  1.2256 |      4.1154 |       2 | 15/11 p=0.600 | 15/11 p=0.515    |

Published methods alone, mean rank: DrCIF 3.5, FreshPRINCE 4.2, InceptionTime 5.4, RotF 5.6, ROCKET 5.8, RandF 6.9, Ridge 7.5, XGBoost 7.7, GridSVR 7.9, ResNet 8.3, FPCR 8.7, 1NN-DTW 9.0, FCN 10.3

## held-out datasets (score = median RMSE relative to best; place out of 14)

| recipe           |   datasets |   score |   mean_rank |   place | vs DrCIF    | vs FreshPRINCE   |
|:-----------------|-----------:|--------:|------------:|--------:|:------------|:-----------------|
| fixed:rf         |         10 |  1.3459 |         6.1 |       4 | 1/9 p=0.037 | 4/6 p=0.432      |
| fixed:et         |         10 |  1.0819 |         3.3 |       1 | 5/5 p=0.770 | 10/0 p=0.002     |
| ms5:et           |         10 |  1.09   |         3.3 |       1 | 4/6 p=0.846 | 10/0 p=0.002     |
| msP7:et          |         10 |  1.1186 |         3.3 |       1 | 4/6 p=0.846 | 10/0 p=0.002     |
| fixed+pooled:et  |         10 |  1.1836 |         3.3 |       1 | 5/5 p=0.770 | 10/0 p=0.002     |
| ms5+pooled:et    |         10 |  1.1666 |         3.4 |       1 | 5/5 p=0.846 | 9/1 p=0.037      |
| rppv128:ridge    |         10 |  1.286  |         5.2 |       2 | 3/7 p=0.193 | 6/4 p=0.846      |
| minirocket:ridge |         10 |  1.2062 |         4.8 |       2 | 5/5 p=0.922 | 7/3 p=0.695      |

Published methods alone, mean rank: DrCIF 3.4, FreshPRINCE 4.9, RotF 5.2, InceptionTime 5.8, Ridge 5.9, ROCKET 6.4, GridSVR 6.5, RandF 7.2, ResNet 7.6, XGBoost 8.5, FPCR 9.0, 1NN-DTW 9.5, FCN 11.1

## cost

| method           |   features |   mean_s |   median_s |
|:-----------------|-----------:|---------:|-----------:|
| fixed+pooled:et  |     348.46 |     3.77 |       2.71 |
| fixed:et         |     133.85 |     1.26 |       0.95 |
| fixed:rf         |     133.85 |     1.53 |       0.92 |
| minirocket:ridge |    9996    |     1.89 |       1.08 |
| ms5+pooled:et    |     377.54 |     4.28 |       3.27 |
| ms5:et           |     162.92 |     1.52 |       1.1  |
| msP7:et          |     229.85 |     1.88 |       1.37 |
| rppv128:ridge    |    2747.08 |     2.49 |       2.03 |
