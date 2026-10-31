# regression: one recipe at a time vs published methods

## all datasets (score = median RMSE relative to best; place out of 14)

| recipe           |   datasets |   score |   mean_rank |   place | vs DrCIF      | vs FreshPRINCE   |
|:-----------------|-----------:|--------:|------------:|--------:|:--------------|:-----------------|
| fixed:rf         |         26 |  1.3459 |      6      |       4 | 5/21 p=0.001  | 8/18 p=0.036     |
| fixed:et         |         26 |  1.2513 |      4.4615 |       2 | 11/15 p=0.217 | 18/8 p=0.029     |
| paper_pooled:et  |         26 |  1.1932 |      3.6923 |       1 | 15/11 p=0.940 | 19/7 p=0.071     |
| paper_plus:et    |         26 |  1.1583 |      3.4231 |       1 | 15/11 p=1.000 | 22/4 p=0.001     |
| minirocket:ridge |         26 |  1.2256 |      4.1154 |       2 | 15/11 p=0.600 | 15/11 p=0.515    |

Published methods alone, mean rank: DrCIF 3.5, FreshPRINCE 4.2, InceptionTime 5.4, RotF 5.6, ROCKET 5.8, RandF 6.9, Ridge 7.5, XGBoost 7.7, GridSVR 7.9, ResNet 8.3, FPCR 8.7, 1NN-DTW 9.0, FCN 10.3

## held-out datasets (score = median RMSE relative to best; place out of 14)

| recipe           |   datasets |   score |   mean_rank |   place | vs DrCIF    | vs FreshPRINCE   |
|:-----------------|-----------:|--------:|------------:|--------:|:------------|:-----------------|
| fixed:rf         |         10 |  1.3459 |         6.1 |       4 | 1/9 p=0.037 | 4/6 p=0.432      |
| fixed:et         |         10 |  1.0819 |         3.3 |       1 | 5/5 p=0.770 | 10/0 p=0.002     |
| paper_pooled:et  |         10 |  1.1932 |         3.5 |       1 | 4/6 p=0.557 | 10/0 p=0.002     |
| paper_plus:et    |         10 |  1.2099 |         3.7 |       1 | 3/7 p=0.322 | 10/0 p=0.002     |
| minirocket:ridge |         10 |  1.2062 |         4.8 |       2 | 5/5 p=0.922 | 7/3 p=0.695      |

Published methods alone, mean rank: DrCIF 3.4, FreshPRINCE 4.9, RotF 5.2, InceptionTime 5.8, Ridge 5.9, ROCKET 6.4, GridSVR 6.5, RandF 7.2, ResNet 7.6, XGBoost 8.5, FPCR 9.0, 1NN-DTW 9.5, FCN 11.1

## cost

| method           |   features |   mean_s |   median_s |
|:-----------------|-----------:|---------:|-----------:|
| fixed:et         |     133.85 |     1.26 |       0.95 |
| fixed:rf         |     133.85 |     1.53 |       0.92 |
| minirocket:ridge |    9996    |     1.89 |       1.08 |
| paper_plus:et    |     417.38 |     3.93 |       2.99 |
| paper_pooled:et  |     344.88 |     2.56 |       1.87 |
