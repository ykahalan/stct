# classification: one recipe at a time vs published methods

## all datasets (score = mean accuracy; place out of 13)

| recipe                |   datasets |   score |   mean_rank |   place | vs HC2        | vs MR-Hydra   |
|:----------------------|-----------:|--------:|------------:|--------:|:--------------|:--------------|
| fixed:rf              |        112 |  0.7816 |      9.933  |      11 | 10/93 p=0.000 | 11/93 p=0.000 |
| fixed:et              |        112 |  0.7967 |      9.3125 |      11 | 13/88 p=0.000 | 15/89 p=0.000 |
| paper_pooled:et       |        112 |  0.8448 |      7.8304 |      11 | 13/81 p=0.000 | 21/78 p=0.000 |
| paper_tuned:et_tuned  |        112 |  0.853  |      7.1339 |      10 | 19/78 p=0.000 | 25/71 p=0.000 |
| paper_views:et_tuned  |        112 |  0.8563 |      7.1384 |      10 | 19/77 p=0.000 | 27/71 p=0.000 |
| paper_vote:et_ridge   |        112 |  0.8646 |      6.192  |       4 | 25/66 p=0.000 | 35/59 p=0.040 |
| paper_fast:et_ridge   |        112 |  0.862  |      6.6295 |       8 | 22/72 p=0.000 | 30/65 p=0.002 |
| paper_trio:et_ridge   |        112 |  0.8647 |      6.2321 |       4 | 29/67 p=0.000 | 36/60 p=0.055 |
| paper_dil:et_ridge    |        112 |  0.8649 |      6.2545 |       4 | 24/68 p=0.000 | 36/60 p=0.020 |
| paper_four:et_ridge   |        112 |  0.8669 |      5.9643 |       4 | 29/66 p=0.000 | 37/56 p=0.192 |
| paper_quant:et_ridge  |        112 |  0.8665 |      6.0134 |       4 | 28/65 p=0.000 | 37/55 p=0.167 |
| paper_quant2:et_ridge |        112 |  0.8673 |      5.8036 |       4 | 29/65 p=0.000 | 39/53 p=0.401 |
| stct_plus:et_ridge    |        112 |  0.8672 |      5.8125 |       4 | 28/66 p=0.000 | 39/54 p=0.441 |
| minirocket:ridge      |        112 |  0.8597 |      6.6652 |       8 | 20/76 p=0.000 | 26/67 p=0.000 |

Published methods alone, mean rank: HC2 4.0, MR-Hydra 4.7, MultiROCKET 5.0, InceptionTime 5.9, RDST 5.9, QUANT 6.0, ROCKET 6.0, MiniROCKET 6.3, WEASEL-2.0 6.3, FreshPRINCE 7.2, Catch22 10.1, 1NN-DTW 10.7

## held-out datasets (score = mean accuracy; place out of 13)

| recipe                |   datasets |   score |   mean_rank |   place | vs HC2        | vs MR-Hydra   |
|:----------------------|-----------:|--------:|------------:|--------:|:--------------|:--------------|
| fixed:rf              |         74 |  0.7559 |      9.8514 |      11 | 7/62 p=0.000  | 9/61 p=0.000  |
| fixed:et              |         74 |  0.7665 |      9.3243 |      11 | 8/61 p=0.000  | 11/59 p=0.000 |
| paper_pooled:et       |         74 |  0.8223 |      7.7973 |      11 | 7/57 p=0.000  | 16/52 p=0.000 |
| paper_tuned:et_tuned  |         74 |  0.8323 |      6.8311 |       7 | 13/52 p=0.000 | 21/46 p=0.008 |
| paper_views:et_tuned  |         74 |  0.836  |      6.8784 |       7 | 13/52 p=0.000 | 22/46 p=0.013 |
| paper_vote:et_ridge   |         74 |  0.8435 |      5.9662 |       4 | 16/48 p=0.000 | 27/38 p=0.423 |
| paper_fast:et_ridge   |         74 |  0.8415 |      6.4054 |       6 | 14/51 p=0.000 | 23/41 p=0.117 |
| paper_trio:et_ridge   |         74 |  0.842  |      6.0811 |       4 | 17/49 p=0.000 | 25/40 p=0.299 |
| paper_dil:et_ridge    |         74 |  0.8425 |      6.1554 |       5 | 13/50 p=0.000 | 24/41 p=0.249 |
| paper_four:et_ridge   |         74 |  0.8437 |      5.8378 |       4 | 17/49 p=0.000 | 25/37 p=0.634 |
| paper_quant:et_ridge  |         74 |  0.8443 |      5.8514 |       4 | 15/48 p=0.000 | 26/37 p=0.707 |
| paper_quant2:et_ridge |         74 |  0.8453 |      5.5405 |       4 | 16/47 p=0.000 | 30/33 p=0.771 |
| stct_plus:et_ridge    |         74 |  0.845  |      5.6216 |       4 | 16/48 p=0.000 | 28/35 p=0.926 |
| minirocket:ridge      |         74 |  0.8325 |      6.8851 |       7 | 12/56 p=0.000 | 18/47 p=0.001 |

Published methods alone, mean rank: HC2 3.8, MR-Hydra 4.6, MultiROCKET 4.9, QUANT 5.6, RDST 5.7, InceptionTime 6.1, ROCKET 6.5, MiniROCKET 6.5, WEASEL-2.0 6.6, FreshPRINCE 7.0, Catch22 9.7, 1NN-DTW 10.9

## cost

| method                |   features |   mean_s |   median_s |
|:----------------------|-----------:|---------:|-----------:|
| fixed:et              |     138.1  |     0.64 |       0.58 |
| fixed:rf              |     138.1  |     0.85 |       0.75 |
| minirocket:ridge      |    9996    |     3.34 |       0.61 |
| paper_dil:et_ridge    |    2855.74 |     4.19 |       1.6  |
| paper_fast:et_ridge   |    1968.45 |     2.07 |       0.79 |
| paper_four:et_ridge   |    4262.22 |    26.48 |       4.13 |
| paper_pooled:et       |     359.35 |     0.83 |       0.59 |
| paper_quant2:et_ridge |    3670.4  |     4.55 |       1.6  |
| paper_quant:et_ridge  |    3632.48 |     4.47 |       1.6  |
| paper_trio:et_ridge   |    2745.83 |     3.71 |       1.21 |
| paper_tuned:et_tuned  |     422.93 |     1.12 |       0.67 |
| paper_views:et_tuned  |     686.31 |     1.86 |       0.95 |
| paper_vote:et_ridge   |    1968.45 |     7.3  |       2.61 |
| stct_plus:et_ridge    |    3504.62 |     4.57 |       1.62 |
