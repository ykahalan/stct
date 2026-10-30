# classification: one recipe at a time vs published methods

## all datasets (score = mean accuracy; place out of 13)

| recipe           |   datasets |   score |   mean_rank |   place | vs HC2        | vs MR-Hydra   |
|:-----------------|-----------:|--------:|------------:|--------:|:--------------|:--------------|
| fixed:rf         |        112 |  0.7816 |      9.933  |      11 | 10/93 p=0.000 | 11/93 p=0.000 |
| fixed:et         |        112 |  0.7967 |      9.3125 |      11 | 13/88 p=0.000 | 15/89 p=0.000 |
| msP7:et          |        112 |  0.807  |      8.7411 |      11 | 14/83 p=0.000 | 21/80 p=0.000 |
| fixed+pooled:et  |        112 |  0.8453 |      7.6384 |      10 | 16/78 p=0.000 | 23/77 p=0.000 |
| msP7+pooled:et   |        112 |  0.8447 |      7.7991 |      11 | 16/81 p=0.000 | 21/72 p=0.000 |
| rppv128:ridge    |        112 |  0.8572 |      7.0045 |      10 | 24/75 p=0.000 | 31/68 p=0.000 |
| rppv128_p2:ridge |        112 |  0.8607 |      6.4241 |       6 | 26/68 p=0.000 | 31/60 p=0.001 |
| minirocket:ridge |        112 |  0.8597 |      6.6652 |       8 | 20/76 p=0.000 | 26/67 p=0.000 |

Published methods alone, mean rank: HC2 4.0, MR-Hydra 4.7, MultiROCKET 5.0, InceptionTime 5.9, RDST 5.9, QUANT 6.0, ROCKET 6.0, MiniROCKET 6.3, WEASEL-2.0 6.3, FreshPRINCE 7.2, Catch22 10.1, 1NN-DTW 10.7

## held-out datasets (score = mean accuracy; place out of 13)

| recipe           |   datasets |   score |   mean_rank |   place | vs HC2        | vs MR-Hydra   |
|:-----------------|-----------:|--------:|------------:|--------:|:--------------|:--------------|
| fixed:rf         |         74 |  0.7559 |      9.8514 |      11 | 7/62 p=0.000  | 9/61 p=0.000  |
| fixed:et         |         74 |  0.7665 |      9.3243 |      11 | 8/61 p=0.000  | 11/59 p=0.000 |
| msP7:et          |         74 |  0.7734 |      8.7838 |      11 | 9/58 p=0.000  | 16/52 p=0.000 |
| fixed+pooled:et  |         74 |  0.8228 |      7.6014 |      11 | 11/53 p=0.000 | 17/51 p=0.000 |
| msP7+pooled:et   |         74 |  0.8195 |      7.9595 |      11 | 10/56 p=0.000 | 15/50 p=0.000 |
| rppv128:ridge    |         74 |  0.8304 |      7.2905 |      10 | 12/55 p=0.000 | 19/49 p=0.002 |
| rppv128_p2:ridge |         74 |  0.8316 |      6.8716 |       7 | 14/51 p=0.000 | 19/48 p=0.000 |
| minirocket:ridge |         74 |  0.8325 |      6.8851 |       7 | 12/56 p=0.000 | 18/47 p=0.001 |

Published methods alone, mean rank: HC2 3.8, MR-Hydra 4.6, MultiROCKET 4.9, QUANT 5.6, RDST 5.7, InceptionTime 6.1, ROCKET 6.5, MiniROCKET 6.5, WEASEL-2.0 6.6, FreshPRINCE 7.0, Catch22 9.7, 1NN-DTW 10.9

## cost

| method           |   features |   mean_s |   median_s |
|:-----------------|-----------:|---------:|-----------:|
| fixed+pooled:et  |     363.1  |     1.88 |       1.21 |
| fixed:et         |     138.1  |     0.64 |       0.58 |
| fixed:rf         |     138.1  |     0.85 |       0.75 |
| minirocket:ridge |    9996    |     3.34 |       0.61 |
| msP7+pooled:et   |     466.86 |     1.85 |       1.22 |
| msP7:et          |     241.86 |     1.07 |       0.82 |
| rppv128:ridge    |    2880    |     2.84 |       1.11 |
| rppv128_p2:ridge |    8640    |     6.1  |       1.8  |
