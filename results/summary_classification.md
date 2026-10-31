# classification: STCT variants vs published results

112 datasets common to all methods (default train/test split).

## Overall

|                  |   mean |   mean_rank |   best_on |
|:-----------------|-------:|------------:|----------:|
| HC2              | 0.8764 |      4.9375 |        11 |
| MR-Hydra         | 0.868  |      5.7812 |         9 |
| MultiROCKET      | 0.8657 |      6.183  |         2 |
| RDST             | 0.8631 |      7.4955 |         7 |
| QUANT            | 0.8544 |      7.5759 |         7 |
| InceptionTime    | 0.8578 |      7.75   |        15 |
| ROCKET           | 0.8558 |      7.7634 |         3 |
| minirocket:ridge | 0.8597 |      7.817  |         0 |
| WEASEL-2.0       | 0.8605 |      7.9643 |         7 |
| MiniROCKET       | 0.858  |      7.9777 |         0 |
| paper_plus:et    | 0.8471 |      9.0848 |         2 |
| FreshPRINCE      | 0.8416 |      9.442  |         9 |
| paper_pooled:et  | 0.8448 |      9.4554 |         2 |
| fixed:et         | 0.7967 |     11.808  |         0 |
| fixed:rf         | 0.7816 |     13.2232 |         0 |
| Catch22          | 0.7814 |     13.9598 |         1 |
| 1NN-DTW          | 0.7421 |     14.7812 |         1 |

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
| fixed:et         | HC2           |     13 |     11 |       88 |       0      |     -0.0797 |
| fixed:et         | MR-Hydra      |     15 |      8 |       89 |       0      |     -0.0713 |
| fixed:et         | InceptionTime |     28 |      8 |       76 |       0      |     -0.0611 |
| fixed:et         | MultiROCKET   |     17 |     10 |       85 |       0      |     -0.069  |
| fixed:et         | MiniROCKET    |     21 |     14 |       77 |       0      |     -0.0613 |
| fixed:et         | ROCKET        |     22 |     10 |       80 |       0      |     -0.0591 |
| fixed:et         | FreshPRINCE   |     31 |      8 |       73 |       0      |     -0.0449 |
| fixed:et         | RDST          |     25 |      8 |       79 |       0      |     -0.0664 |
| fixed:et         | QUANT         |     13 |     11 |       88 |       0      |     -0.0577 |
| fixed:et         | WEASEL-2.0    |     22 |      7 |       83 |       0      |     -0.0638 |
| fixed:et         | Catch22       |     71 |      5 |       36 |       0.0017 |      0.0153 |
| fixed:et         | 1NN-DTW       |     81 |      8 |       23 |       0      |      0.0547 |
| paper_pooled:et  | HC2           |     13 |     18 |       81 |       0      |     -0.0316 |
| paper_pooled:et  | MR-Hydra      |     21 |     13 |       78 |       0      |     -0.0232 |
| paper_pooled:et  | InceptionTime |     41 |     11 |       60 |       0.0038 |     -0.013  |
| paper_pooled:et  | MultiROCKET   |     22 |     18 |       72 |       0      |     -0.0209 |
| paper_pooled:et  | MiniROCKET    |     34 |     14 |       64 |       0.0007 |     -0.0132 |
| paper_pooled:et  | ROCKET        |     35 |     13 |       64 |       0.0014 |     -0.011  |
| paper_pooled:et  | FreshPRINCE   |     51 |     13 |       48 |       0.7841 |      0.0032 |
| paper_pooled:et  | RDST          |     31 |     12 |       69 |       0      |     -0.0183 |
| paper_pooled:et  | QUANT         |     32 |     16 |       64 |       0.0017 |     -0.0096 |
| paper_pooled:et  | WEASEL-2.0    |     34 |     12 |       66 |       0.0005 |     -0.0157 |
| paper_pooled:et  | Catch22       |     93 |      6 |       13 |       0      |      0.0634 |
| paper_pooled:et  | 1NN-DTW       |     96 |      6 |       10 |       0      |      0.1027 |
| paper_plus:et    | HC2           |     15 |     18 |       79 |       0      |     -0.0294 |
| paper_plus:et    | MR-Hydra      |     22 |     15 |       75 |       0      |     -0.021  |
| paper_plus:et    | InceptionTime |     41 |     10 |       61 |       0.0094 |     -0.0107 |
| paper_plus:et    | MultiROCKET   |     25 |     14 |       73 |       0      |     -0.0186 |
| paper_plus:et    | MiniROCKET    |     33 |     16 |       63 |       0.0024 |     -0.011  |
| paper_plus:et    | ROCKET        |     41 |     14 |       57 |       0.0077 |     -0.0088 |
| paper_plus:et    | FreshPRINCE   |     55 |     14 |       43 |       0.8622 |      0.0055 |
| paper_plus:et    | RDST          |     33 |     14 |       65 |       0.0003 |     -0.0161 |
| paper_plus:et    | QUANT         |     32 |     19 |       61 |       0.0058 |     -0.0073 |
| paper_plus:et    | WEASEL-2.0    |     33 |     15 |       64 |       0.0005 |     -0.0134 |
| paper_plus:et    | Catch22       |     93 |      6 |       13 |       0      |      0.0657 |
| paper_plus:et    | 1NN-DTW       |     96 |      6 |       10 |       0      |      0.105  |
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
| fixed:rf        | fixed:et         |       18 |     12 |       82 |       0      |
| fixed:rf        | paper_pooled:et  |       15 |      8 |       89 |       0      |
| fixed:rf        | paper_plus:et    |       12 |      8 |       92 |       0      |
| fixed:rf        | minirocket:ridge |       15 |     11 |       86 |       0      |
| fixed:et        | paper_pooled:et  |       21 |     13 |       78 |       0      |
| fixed:et        | paper_plus:et    |       18 |     15 |       79 |       0      |
| fixed:et        | minirocket:ridge |       23 |      9 |       80 |       0      |
| paper_pooled:et | paper_plus:et    |       36 |     26 |       50 |       0.2189 |
| paper_pooled:et | minirocket:ridge |       30 |     19 |       63 |       0.0003 |
| paper_plus:et   | minirocket:ridge |       34 |     15 |       63 |       0.0002 |
