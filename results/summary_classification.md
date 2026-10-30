# classification: STCT variants vs published results

112 datasets common to all methods (default train/test split).

## Overall

|                  |   mean |   mean_rank |   best_on |
|:-----------------|-------:|------------:|----------:|
| HC2              | 0.8764 |      4.9777 |        10 |
| MR-Hydra         | 0.868  |      5.8527 |         9 |
| MultiROCKET      | 0.8657 |      6.2768 |         2 |
| RDST             | 0.8631 |      7.5312 |         7 |
| QUANT            | 0.8544 |      7.7054 |         7 |
| ROCKET           | 0.8558 |      7.8214 |         3 |
| InceptionTime    | 0.8578 |      7.8616 |        15 |
| minirocket:ridge | 0.8597 |      7.8973 |         0 |
| MiniROCKET       | 0.858  |      8.1027 |         0 |
| WEASEL-2.0       | 0.8605 |      8.1205 |         7 |
| plus:et          | 0.8456 |      9.192  |         1 |
| fixed+pooled:et  | 0.8453 |      9.2411 |         1 |
| FreshPRINCE      | 0.8416 |      9.5    |         9 |
| plus:rf          | 0.8336 |     10.7143 |         3 |
| fixed:rf         | 0.7816 |     13.1741 |         0 |
| Catch22          | 0.7814 |     14.192  |         1 |
| 1NN-DTW          | 0.7421 |     14.8393 |         1 |

## Pairwise: each STCT variant vs each reference (wins/ties/losses on accuracy, Wilcoxon p)

| variant          | reference     |   wins |   ties |   losses |   wilcoxon_p |   mean_diff |
|:-----------------|:--------------|-------:|-------:|---------:|-------------:|------------:|
| fixed:rf         | HC2           |     10 |      9 |       93 |       0      |     -0.0948 |
| fixed:rf         | MR-Hydra      |     11 |      8 |       93 |       0      |     -0.0864 |
| fixed:rf         | InceptionTime |     22 |      6 |       84 |       0      |     -0.0762 |
| fixed:rf         | MultiROCKET   |     11 |     11 |       90 |       0      |     -0.0841 |
| fixed:rf         | MiniROCKET    |     16 |     11 |       85 |       0      |     -0.0764 |
| fixed:rf         | ROCKET        |     17 |      6 |       89 |       0      |     -0.0742 |
| fixed:rf         | FreshPRINCE   |     23 |      8 |       81 |       0      |     -0.06   |
| fixed:rf         | RDST          |     15 |      8 |       89 |       0      |     -0.0815 |
| fixed:rf         | QUANT         |      8 |      9 |       95 |       0      |     -0.0728 |
| fixed:rf         | WEASEL-2.0    |     21 |      6 |       85 |       0      |     -0.0789 |
| fixed:rf         | Catch22       |     67 |      7 |       38 |       0.103  |      0.0002 |
| fixed:rf         | 1NN-DTW       |     74 |      8 |       30 |       0.0001 |      0.0395 |
| plus:rf          | HC2           |     15 |     13 |       84 |       0      |     -0.0428 |
| plus:rf          | MR-Hydra      |     18 |     12 |       82 |       0      |     -0.0344 |
| plus:rf          | InceptionTime |     37 |     11 |       64 |       0.0013 |     -0.0242 |
| plus:rf          | MultiROCKET   |     21 |     14 |       77 |       0      |     -0.0321 |
| plus:rf          | MiniROCKET    |     29 |     12 |       71 |       0      |     -0.0244 |
| plus:rf          | ROCKET        |     30 |     10 |       72 |       0      |     -0.0222 |
| plus:rf          | FreshPRINCE   |     36 |     13 |       63 |       0.0201 |     -0.008  |
| plus:rf          | RDST          |     27 |     10 |       75 |       0      |     -0.0295 |
| plus:rf          | QUANT         |     19 |     16 |       77 |       0      |     -0.0208 |
| plus:rf          | WEASEL-2.0    |     33 |      9 |       70 |       0      |     -0.0269 |
| plus:rf          | Catch22       |     93 |      5 |       14 |       0      |      0.0522 |
| plus:rf          | 1NN-DTW       |     89 |      6 |       17 |       0      |      0.0915 |
| fixed+pooled:et  | HC2           |     16 |     18 |       78 |       0      |     -0.0312 |
| fixed+pooled:et  | MR-Hydra      |     23 |     12 |       77 |       0      |     -0.0228 |
| fixed+pooled:et  | InceptionTime |     42 |     10 |       60 |       0.0051 |     -0.0125 |
| fixed+pooled:et  | MultiROCKET   |     28 |     13 |       71 |       0      |     -0.0204 |
| fixed+pooled:et  | MiniROCKET    |     36 |     15 |       61 |       0.0018 |     -0.0128 |
| fixed+pooled:et  | ROCKET        |     35 |     15 |       62 |       0.0017 |     -0.0105 |
| fixed+pooled:et  | FreshPRINCE   |     50 |     15 |       47 |       0.7817 |      0.0037 |
| fixed+pooled:et  | RDST          |     31 |     13 |       68 |       0      |     -0.0179 |
| fixed+pooled:et  | QUANT         |     34 |     21 |       57 |       0.0038 |     -0.0091 |
| fixed+pooled:et  | WEASEL-2.0    |     38 |     10 |       64 |       0.0005 |     -0.0152 |
| fixed+pooled:et  | Catch22       |     94 |      5 |       13 |       0      |      0.0639 |
| fixed+pooled:et  | 1NN-DTW       |     97 |      6 |        9 |       0      |      0.1032 |
| plus:et          | HC2           |     14 |     17 |       81 |       0      |     -0.0309 |
| plus:et          | MR-Hydra      |     24 |     14 |       74 |       0      |     -0.0225 |
| plus:et          | InceptionTime |     43 |      9 |       60 |       0.0149 |     -0.0122 |
| plus:et          | MultiROCKET   |     26 |     14 |       72 |       0      |     -0.0201 |
| plus:et          | MiniROCKET    |     39 |     13 |       60 |       0.0068 |     -0.0125 |
| plus:et          | ROCKET        |     39 |     13 |       60 |       0.0072 |     -0.0102 |
| plus:et          | FreshPRINCE   |     55 |     12 |       45 |       0.9165 |      0.004  |
| plus:et          | RDST          |     36 |      9 |       67 |       0.0005 |     -0.0176 |
| plus:et          | QUANT         |     33 |     20 |       59 |       0.0022 |     -0.0088 |
| plus:et          | WEASEL-2.0    |     38 |     10 |       64 |       0.001  |     -0.0149 |
| plus:et          | Catch22       |     96 |      7 |        9 |       0      |      0.0642 |
| plus:et          | 1NN-DTW       |     94 |      7 |       11 |       0      |      0.1035 |
| minirocket:ridge | HC2           |     20 |     16 |       76 |       0      |     -0.0167 |
| minirocket:ridge | MR-Hydra      |     26 |     19 |       67 |       0      |     -0.0083 |
| minirocket:ridge | InceptionTime |     50 |     16 |       46 |       0.9592 |      0.0019 |
| minirocket:ridge | MultiROCKET   |     25 |     20 |       67 |       0.0002 |     -0.0059 |
| minirocket:ridge | MiniROCKET    |     44 |     33 |       35 |       0.1225 |      0.0017 |
| minirocket:ridge | ROCKET        |     49 |     14 |       49 |       0.8165 |      0.0039 |
| minirocket:ridge | FreshPRINCE   |     62 |     12 |       38 |       0.0054 |      0.0181 |
| minirocket:ridge | RDST          |     46 |     14 |       52 |       0.3921 |     -0.0034 |
| minirocket:ridge | QUANT         |     53 |     18 |       41 |       0.4153 |      0.0053 |
| minirocket:ridge | WEASEL-2.0    |     52 |      8 |       52 |       0.9625 |     -0.0008 |
| minirocket:ridge | Catch22       |     95 |      6 |       11 |       0      |      0.0784 |
| minirocket:ridge | 1NN-DTW       |     97 |      5 |       10 |       0      |      0.1177 |

## Pairwise among STCT variants

| a               | b                |   a_wins |   ties |   b_wins |   wilcoxon_p |
|:----------------|:-----------------|---------:|-------:|---------:|-------------:|
| fixed:rf        | plus:rf          |       24 |      9 |       79 |       0      |
| fixed:rf        | fixed+pooled:et  |       15 |      9 |       88 |       0      |
| fixed:rf        | plus:et          |       12 |      9 |       91 |       0      |
| fixed:rf        | minirocket:ridge |       15 |     11 |       86 |       0      |
| plus:rf         | fixed+pooled:et  |       30 |     18 |       64 |       0.0007 |
| plus:rf         | plus:et          |       25 |     20 |       67 |       0      |
| plus:rf         | minirocket:ridge |       27 |     14 |       71 |       0      |
| fixed+pooled:et | plus:et          |       51 |     24 |       37 |       0.5271 |
| fixed+pooled:et | minirocket:ridge |       32 |     16 |       64 |       0.0003 |
| plus:et         | minirocket:ridge |       34 |     19 |       59 |       0.0016 |
