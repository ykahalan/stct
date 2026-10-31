# classification: one recipe at a time vs published methods

## all datasets (score = mean accuracy; place out of 13)

| recipe               |   datasets |   score |   mean_rank |   place | vs HC2        | vs MR-Hydra   |
|:---------------------|-----------:|--------:|------------:|--------:|:--------------|:--------------|
| fixed:rf             |        112 |  0.7816 |      9.933  |      11 | 10/93 p=0.000 | 11/93 p=0.000 |
| fixed:et             |        112 |  0.7967 |      9.3125 |      11 | 13/88 p=0.000 | 15/89 p=0.000 |
| paper_pooled:et      |        112 |  0.8448 |      7.8304 |      11 | 13/81 p=0.000 | 21/78 p=0.000 |
| paper_tuned:et_tuned |        112 |  0.853  |      7.1339 |      10 | 19/78 p=0.000 | 25/71 p=0.000 |
| minirocket:ridge     |        112 |  0.8597 |      6.6652 |       8 | 20/76 p=0.000 | 26/67 p=0.000 |

Published methods alone, mean rank: HC2 4.0, MR-Hydra 4.7, MultiROCKET 5.0, InceptionTime 5.9, RDST 5.9, QUANT 6.0, ROCKET 6.0, MiniROCKET 6.3, WEASEL-2.0 6.3, FreshPRINCE 7.2, Catch22 10.1, 1NN-DTW 10.7

## held-out datasets (score = mean accuracy; place out of 13)

| recipe               |   datasets |   score |   mean_rank |   place | vs HC2        | vs MR-Hydra   |
|:---------------------|-----------:|--------:|------------:|--------:|:--------------|:--------------|
| fixed:rf             |         74 |  0.7559 |      9.8514 |      11 | 7/62 p=0.000  | 9/61 p=0.000  |
| fixed:et             |         74 |  0.7665 |      9.3243 |      11 | 8/61 p=0.000  | 11/59 p=0.000 |
| paper_pooled:et      |         74 |  0.8223 |      7.7973 |      11 | 7/57 p=0.000  | 16/52 p=0.000 |
| paper_tuned:et_tuned |         74 |  0.8323 |      6.8311 |       7 | 13/52 p=0.000 | 21/46 p=0.008 |
| minirocket:ridge     |         74 |  0.8325 |      6.8851 |       7 | 12/56 p=0.000 | 18/47 p=0.001 |

Published methods alone, mean rank: HC2 3.8, MR-Hydra 4.6, MultiROCKET 4.9, QUANT 5.6, RDST 5.7, InceptionTime 6.1, ROCKET 6.5, MiniROCKET 6.5, WEASEL-2.0 6.6, FreshPRINCE 7.0, Catch22 9.7, 1NN-DTW 10.9

## cost

| method               |   features |   mean_s |   median_s |
|:---------------------|-----------:|---------:|-----------:|
| fixed:et             |     138.1  |     0.64 |       0.58 |
| fixed:rf             |     138.1  |     0.85 |       0.75 |
| minirocket:ridge     |    9996    |     3.34 |       0.61 |
| paper_pooled:et      |     359.35 |     0.83 |       0.59 |
| paper_tuned:et_tuned |     422.93 |     1.12 |       0.67 |
