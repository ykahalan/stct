# classification: STCT variants vs published results

112 datasets common to all methods (default train/test split).

## Overall

|                  |   mean |   mean_rank |   best_on |
|:-----------------|-------:|------------:|----------:|
| HC2              | 0.8764 |      4.7589 |        13 |
| MR-Hydra         | 0.868  |      5.5536 |         9 |
| MultiROCKET      | 0.8657 |      5.9107 |         3 |
| RDST             | 0.8631 |      7.1518 |         7 |
| QUANT            | 0.8544 |      7.2321 |         8 |
| ROCKET           | 0.8558 |      7.3393 |         3 |
| InceptionTime    | 0.8578 |      7.3527 |        15 |
| minirocket:ridge | 0.8597 |      7.4688 |         0 |
| WEASEL-2.0       | 0.8605 |      7.6295 |         7 |
| MiniROCKET       | 0.858  |      7.6339 |         0 |
| stct_plus:et     | 0.8447 |      8.6429 |         0 |
| FreshPRINCE      | 0.8416 |      8.8839 |         9 |
| fixed:et         | 0.7967 |     11.058  |         0 |
| fixed:rf         | 0.7816 |     12.3884 |         0 |
| Catch22          | 0.7814 |     13.0938 |         1 |
| 1NN-DTW          | 0.7421 |     13.9018 |         1 |

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
| stct_plus:et     | HC2           |     15 |     22 |       75 |       0      |     -0.0317 |
| stct_plus:et     | MR-Hydra      |     22 |     19 |       71 |       0      |     -0.0233 |
| stct_plus:et     | InceptionTime |     42 |     12 |       58 |       0.0099 |     -0.0131 |
| stct_plus:et     | MultiROCKET   |     24 |     17 |       71 |       0      |     -0.021  |
| stct_plus:et     | MiniROCKET    |     35 |     17 |       60 |       0.0011 |     -0.0133 |
| stct_plus:et     | ROCKET        |     35 |     14 |       63 |       0.0019 |     -0.0111 |
| stct_plus:et     | FreshPRINCE   |     51 |     12 |       49 |       0.8852 |      0.0031 |
| stct_plus:et     | RDST          |     31 |     15 |       66 |       0.0001 |     -0.0184 |
| stct_plus:et     | QUANT         |     36 |     14 |       62 |       0.0036 |     -0.0097 |
| stct_plus:et     | WEASEL-2.0    |     37 |     12 |       63 |       0.0005 |     -0.0158 |
| stct_plus:et     | Catch22       |     91 |      8 |       13 |       0      |      0.0633 |
| stct_plus:et     | 1NN-DTW       |     96 |      7 |        9 |       0      |      0.1027 |
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

| a            | b                |   a_wins |   ties |   b_wins |   wilcoxon_p |
|:-------------|:-----------------|---------:|-------:|---------:|-------------:|
| fixed:rf     | fixed:et         |       18 |     12 |       82 |       0      |
| fixed:rf     | stct_plus:et     |       11 |     11 |       90 |       0      |
| fixed:rf     | minirocket:ridge |       15 |     11 |       86 |       0      |
| fixed:et     | stct_plus:et     |       18 |     14 |       80 |       0      |
| fixed:et     | minirocket:ridge |       23 |      9 |       80 |       0      |
| stct_plus:et | minirocket:ridge |       34 |     16 |       62 |       0.0005 |
