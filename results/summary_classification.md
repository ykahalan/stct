# classification: STCT variants vs published results

112 datasets common to all methods (default train/test split).

## Overall

|                      |   mean |   mean_rank |   best_on |
|:---------------------|-------:|------------:|----------:|
| HC2                  | 0.8764 |      4.9598 |        10 |
| MR-Hydra             | 0.868  |      5.8125 |         9 |
| MultiROCKET          | 0.8657 |      6.2634 |         2 |
| RDST                 | 0.8631 |      7.5446 |         7 |
| QUANT                | 0.8544 |      7.6339 |         7 |
| ROCKET               | 0.8558 |      7.7768 |         3 |
| InceptionTime        | 0.8578 |      7.8125 |        15 |
| minirocket:ridge     | 0.8597 |      7.8839 |         0 |
| WEASEL-2.0           | 0.8605 |      8.0179 |         7 |
| MiniROCKET           | 0.858  |      8.0402 |         0 |
| paper_tuned:et_tuned | 0.853  |      8.4107 |         4 |
| FreshPRINCE          | 0.8416 |      9.4821 |         9 |
| paper_pooled:et      | 0.8448 |      9.558  |         0 |
| fixed:et             | 0.7967 |     11.8125 |         0 |
| fixed:rf             | 0.7816 |     13.2098 |         0 |
| Catch22              | 0.7814 |     13.9821 |         0 |
| 1NN-DTW              | 0.7421 |     14.7991 |         1 |

## Pairwise: each STCT variant vs each reference (wins/ties/losses on accuracy, Wilcoxon p)

| variant              | reference     |   wins |   ties |   losses |   wilcoxon_p |   mean_diff |
|:---------------------|:--------------|-------:|-------:|---------:|-------------:|------------:|
| fixed:rf             | HC2           |     10 |      9 |       93 |       0      |     -0.0948 |
| fixed:rf             | MR-Hydra      |     11 |      8 |       93 |       0      |     -0.0864 |
| fixed:rf             | InceptionTime |     22 |      6 |       84 |       0      |     -0.0762 |
| fixed:rf             | MultiROCKET   |     11 |     11 |       90 |       0      |     -0.0841 |
| fixed:rf             | MiniROCKET    |     16 |     11 |       85 |       0      |     -0.0764 |
| fixed:rf             | ROCKET        |     17 |      6 |       89 |       0      |     -0.0742 |
| fixed:rf             | FreshPRINCE   |     23 |      8 |       81 |       0      |     -0.06   |
| fixed:rf             | RDST          |     15 |      8 |       89 |       0      |     -0.0815 |
| fixed:rf             | QUANT         |      8 |      9 |       95 |       0      |     -0.0728 |
| fixed:rf             | WEASEL-2.0    |     21 |      6 |       85 |       0      |     -0.0789 |
| fixed:rf             | Catch22       |     67 |      7 |       38 |       0.103  |      0.0002 |
| fixed:rf             | 1NN-DTW       |     74 |      8 |       30 |       0.0001 |      0.0395 |
| fixed:et             | HC2           |     13 |     11 |       88 |       0      |     -0.0797 |
| fixed:et             | MR-Hydra      |     15 |      8 |       89 |       0      |     -0.0713 |
| fixed:et             | InceptionTime |     28 |      8 |       76 |       0      |     -0.0611 |
| fixed:et             | MultiROCKET   |     17 |     10 |       85 |       0      |     -0.069  |
| fixed:et             | MiniROCKET    |     21 |     14 |       77 |       0      |     -0.0613 |
| fixed:et             | ROCKET        |     22 |     10 |       80 |       0      |     -0.0591 |
| fixed:et             | FreshPRINCE   |     31 |      8 |       73 |       0      |     -0.0449 |
| fixed:et             | RDST          |     25 |      8 |       79 |       0      |     -0.0664 |
| fixed:et             | QUANT         |     13 |     11 |       88 |       0      |     -0.0577 |
| fixed:et             | WEASEL-2.0    |     22 |      7 |       83 |       0      |     -0.0638 |
| fixed:et             | Catch22       |     71 |      5 |       36 |       0.0017 |      0.0153 |
| fixed:et             | 1NN-DTW       |     81 |      8 |       23 |       0      |      0.0547 |
| paper_pooled:et      | HC2           |     13 |     18 |       81 |       0      |     -0.0316 |
| paper_pooled:et      | MR-Hydra      |     21 |     13 |       78 |       0      |     -0.0232 |
| paper_pooled:et      | InceptionTime |     41 |     11 |       60 |       0.0038 |     -0.013  |
| paper_pooled:et      | MultiROCKET   |     22 |     18 |       72 |       0      |     -0.0209 |
| paper_pooled:et      | MiniROCKET    |     34 |     14 |       64 |       0.0007 |     -0.0132 |
| paper_pooled:et      | ROCKET        |     35 |     13 |       64 |       0.0014 |     -0.011  |
| paper_pooled:et      | FreshPRINCE   |     51 |     13 |       48 |       0.7841 |      0.0032 |
| paper_pooled:et      | RDST          |     31 |     12 |       69 |       0      |     -0.0183 |
| paper_pooled:et      | QUANT         |     32 |     16 |       64 |       0.0017 |     -0.0096 |
| paper_pooled:et      | WEASEL-2.0    |     34 |     12 |       66 |       0.0005 |     -0.0157 |
| paper_pooled:et      | Catch22       |     93 |      6 |       13 |       0      |      0.0634 |
| paper_pooled:et      | 1NN-DTW       |     96 |      6 |       10 |       0      |      0.1027 |
| paper_tuned:et_tuned | HC2           |     19 |     15 |       78 |       0      |     -0.0234 |
| paper_tuned:et_tuned | MR-Hydra      |     25 |     16 |       71 |       0      |     -0.015  |
| paper_tuned:et_tuned | InceptionTime |     47 |     12 |       53 |       0.1696 |     -0.0048 |
| paper_tuned:et_tuned | MultiROCKET   |     33 |     16 |       63 |       0.0004 |     -0.0127 |
| paper_tuned:et_tuned | MiniROCKET    |     39 |     18 |       55 |       0.1098 |     -0.005  |
| paper_tuned:et_tuned | ROCKET        |     42 |     15 |       55 |       0.0749 |     -0.0028 |
| paper_tuned:et_tuned | FreshPRINCE   |     60 |     13 |       39 |       0.1099 |      0.0114 |
| paper_tuned:et_tuned | RDST          |     41 |      9 |       62 |       0.0162 |     -0.0101 |
| paper_tuned:et_tuned | QUANT         |     39 |     18 |       55 |       0.1652 |     -0.0014 |
| paper_tuned:et_tuned | WEASEL-2.0    |     40 |     13 |       59 |       0.038  |     -0.0075 |
| paper_tuned:et_tuned | Catch22       |     94 |      9 |        9 |       0      |      0.0716 |
| paper_tuned:et_tuned | 1NN-DTW       |     98 |      6 |        8 |       0      |      0.1109 |
| minirocket:ridge     | HC2           |     20 |     16 |       76 |       0      |     -0.0167 |
| minirocket:ridge     | MR-Hydra      |     26 |     19 |       67 |       0      |     -0.0083 |
| minirocket:ridge     | InceptionTime |     50 |     16 |       46 |       0.9592 |      0.0019 |
| minirocket:ridge     | MultiROCKET   |     25 |     20 |       67 |       0.0002 |     -0.0059 |
| minirocket:ridge     | MiniROCKET    |     44 |     33 |       35 |       0.1225 |      0.0017 |
| minirocket:ridge     | ROCKET        |     49 |     14 |       49 |       0.8165 |      0.0039 |
| minirocket:ridge     | FreshPRINCE   |     62 |     12 |       38 |       0.0054 |      0.0181 |
| minirocket:ridge     | RDST          |     46 |     14 |       52 |       0.3921 |     -0.0034 |
| minirocket:ridge     | QUANT         |     53 |     18 |       41 |       0.4153 |      0.0053 |
| minirocket:ridge     | WEASEL-2.0    |     52 |      8 |       52 |       0.9625 |     -0.0008 |
| minirocket:ridge     | Catch22       |     95 |      6 |       11 |       0      |      0.0784 |
| minirocket:ridge     | 1NN-DTW       |     97 |      5 |       10 |       0      |      0.1177 |

## Pairwise among STCT variants

| a                    | b                    |   a_wins |   ties |   b_wins |   wilcoxon_p |
|:---------------------|:---------------------|---------:|-------:|---------:|-------------:|
| fixed:rf             | fixed:et             |       18 |     12 |       82 |       0      |
| fixed:rf             | paper_pooled:et      |       15 |      8 |       89 |       0      |
| fixed:rf             | paper_tuned:et_tuned |       13 |      9 |       90 |       0      |
| fixed:rf             | minirocket:ridge     |       15 |     11 |       86 |       0      |
| fixed:et             | paper_pooled:et      |       21 |     13 |       78 |       0      |
| fixed:et             | paper_tuned:et_tuned |       18 |     14 |       80 |       0      |
| fixed:et             | minirocket:ridge     |       23 |      9 |       80 |       0      |
| paper_pooled:et      | paper_tuned:et_tuned |       24 |     27 |       61 |       0.0001 |
| paper_pooled:et      | minirocket:ridge     |       30 |     19 |       63 |       0.0003 |
| paper_tuned:et_tuned | minirocket:ridge     |       41 |     16 |       55 |       0.0344 |
