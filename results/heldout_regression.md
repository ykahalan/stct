# regression: one recipe at a time vs published methods

## all datasets (score = median RMSE relative to best; place out of 14)

| recipe           |   datasets |   score |   mean_rank |   place | vs DrCIF      | vs FreshPRINCE   |
|:-----------------|-----------:|--------:|------------:|--------:|:--------------|:-----------------|
| fixed:rf         |         26 |  1.3459 |      6      |       4 | 5/21 p=0.001  | 8/18 p=0.036     |
| fixed:et         |         26 |  1.2513 |      4.4615 |       2 | 11/15 p=0.217 | 18/8 p=0.029     |
| msP7+pooled:et   |         26 |  1.1164 |      3.3462 |       1 | 15/11 p=0.842 | 21/5 p=0.003     |
| stct_plus:et     |         26 |  1.1392 |      3.3077 |       1 | 16/10 p=0.901 | 20/6 p=0.031     |
| minirocket:ridge |         26 |  1.2256 |      4.1154 |       2 | 15/11 p=0.600 | 15/11 p=0.515    |

Published methods alone, mean rank: DrCIF 3.5, FreshPRINCE 4.2, InceptionTime 5.4, RotF 5.6, ROCKET 5.8, RandF 6.9, Ridge 7.5, XGBoost 7.7, GridSVR 7.9, ResNet 8.3, FPCR 8.7, 1NN-DTW 9.0, FCN 10.3

## held-out datasets (score = median RMSE relative to best; place out of 14)

| recipe           |   datasets |   score |   mean_rank |   place | vs DrCIF    | vs FreshPRINCE   |
|:-----------------|-----------:|--------:|------------:|--------:|:------------|:-----------------|
| fixed:rf         |         10 |  1.3459 |         6.1 |       4 | 1/9 p=0.037 | 4/6 p=0.432      |
| fixed:et         |         10 |  1.0819 |         3.3 |       1 | 5/5 p=0.770 | 10/0 p=0.002     |
| msP7+pooled:et   |         10 |  1.1788 |         3.3 |       1 | 4/6 p=0.375 | 10/0 p=0.002     |
| stct_plus:et     |         10 |  1.1974 |         3.4 |       1 | 5/5 p=0.625 | 9/1 p=0.027      |
| minirocket:ridge |         10 |  1.2062 |         4.8 |       2 | 5/5 p=0.922 | 7/3 p=0.695      |

Published methods alone, mean rank: DrCIF 3.4, FreshPRINCE 4.9, RotF 5.2, InceptionTime 5.8, Ridge 5.9, ROCKET 6.4, GridSVR 6.5, RandF 7.2, ResNet 7.6, XGBoost 8.5, FPCR 9.0, 1NN-DTW 9.5, FCN 11.1

## cost

| method           |   features |   mean_s |   median_s |
|:-----------------|-----------:|---------:|-----------:|
| fixed:et         |     133.85 |     1.26 |       0.95 |
| fixed:rf         |     133.85 |     1.53 |       0.92 |
| minirocket:ridge |    9996    |     1.89 |       1.08 |
| msP7+pooled:et   |     444.46 |     4.35 |       3.16 |
| stct_plus:et     |     440.88 |     4.96 |       3.84 |
