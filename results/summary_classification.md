# classification: STCT variants vs published results

112 datasets common to all methods (default train/test split).

## Overall

|                    |   mean |   mean_rank |   best_on |
|:-------------------|-------:|------------:|----------:|
| HC2                | 0.8764 |      5.4464 |         8 |
| MR-Hydra           | 0.868  |      6.4286 |         9 |
| MultiROCKET        | 0.8657 |      6.875  |         2 |
| RDST               | 0.8631 |      8.1696 |         5 |
| rppv128_p2:ridge   | 0.8607 |      8.25   |         4 |
| InceptionTime      | 0.8578 |      8.4911 |        13 |
| ROCKET             | 0.8558 |      8.5357 |         1 |
| rppv128_diff:ridge | 0.8597 |      8.558  |         7 |
| QUANT              | 0.8544 |      8.5848 |         7 |
| minirocket:ridge   | 0.8597 |      8.75   |         0 |
| WEASEL-2.0         | 0.8605 |      8.8571 |         4 |
| MiniROCKET         | 0.858  |      8.9821 |         0 |
| rppv128:ridge      | 0.8572 |      9.3125 |         2 |
| fixed+pooled:et    | 0.8453 |     10.3571 |         3 |
| FreshPRINCE        | 0.8416 |     10.3571 |         9 |
| fixed:rf           | 0.7816 |     14.0982 |         0 |
| Catch22            | 0.7814 |     15.0982 |         1 |
| 1NN-DTW            | 0.7421 |     15.8482 |         1 |

## Pairwise: each STCT variant vs each reference (wins/ties/losses on accuracy, Wilcoxon p)

| variant            | reference     |   wins |   ties |   losses |   wilcoxon_p |   mean_diff |
|:-------------------|:--------------|-------:|-------:|---------:|-------------:|------------:|
| fixed:rf           | HC2           |     10 |      9 |       93 |       0      |     -0.0948 |
| fixed:rf           | MR-Hydra      |     11 |      8 |       93 |       0      |     -0.0864 |
| fixed:rf           | InceptionTime |     22 |      6 |       84 |       0      |     -0.0762 |
| fixed:rf           | MultiROCKET   |     11 |     11 |       90 |       0      |     -0.0841 |
| fixed:rf           | MiniROCKET    |     16 |     11 |       85 |       0      |     -0.0764 |
| fixed:rf           | ROCKET        |     17 |      6 |       89 |       0      |     -0.0742 |
| fixed:rf           | FreshPRINCE   |     23 |      8 |       81 |       0      |     -0.06   |
| fixed:rf           | RDST          |     15 |      8 |       89 |       0      |     -0.0815 |
| fixed:rf           | QUANT         |      8 |      9 |       95 |       0      |     -0.0728 |
| fixed:rf           | WEASEL-2.0    |     21 |      6 |       85 |       0      |     -0.0789 |
| fixed:rf           | Catch22       |     67 |      7 |       38 |       0.103  |      0.0002 |
| fixed:rf           | 1NN-DTW       |     74 |      8 |       30 |       0.0001 |      0.0395 |
| fixed+pooled:et    | HC2           |     16 |     18 |       78 |       0      |     -0.0312 |
| fixed+pooled:et    | MR-Hydra      |     23 |     12 |       77 |       0      |     -0.0228 |
| fixed+pooled:et    | InceptionTime |     42 |     10 |       60 |       0.0051 |     -0.0125 |
| fixed+pooled:et    | MultiROCKET   |     28 |     13 |       71 |       0      |     -0.0204 |
| fixed+pooled:et    | MiniROCKET    |     36 |     15 |       61 |       0.0018 |     -0.0128 |
| fixed+pooled:et    | ROCKET        |     35 |     15 |       62 |       0.0017 |     -0.0105 |
| fixed+pooled:et    | FreshPRINCE   |     50 |     15 |       47 |       0.7817 |      0.0037 |
| fixed+pooled:et    | RDST          |     31 |     13 |       68 |       0      |     -0.0179 |
| fixed+pooled:et    | QUANT         |     34 |     21 |       57 |       0.0038 |     -0.0091 |
| fixed+pooled:et    | WEASEL-2.0    |     38 |     10 |       64 |       0.0005 |     -0.0152 |
| fixed+pooled:et    | Catch22       |     94 |      5 |       13 |       0      |      0.0639 |
| fixed+pooled:et    | 1NN-DTW       |     97 |      6 |        9 |       0      |      0.1032 |
| rppv128:ridge      | HC2           |     24 |     13 |       75 |       0      |     -0.0193 |
| rppv128:ridge      | MR-Hydra      |     31 |     13 |       68 |       0.0005 |     -0.0109 |
| rppv128:ridge      | InceptionTime |     46 |     10 |       56 |       0.2515 |     -0.0006 |
| rppv128:ridge      | MultiROCKET   |     32 |     13 |       67 |       0.0032 |     -0.0085 |
| rppv128:ridge      | MiniROCKET    |     47 |     13 |       52 |       0.6601 |     -0.0009 |
| rppv128:ridge      | ROCKET        |     43 |     16 |       53 |       0.2737 |      0.0013 |
| rppv128:ridge      | FreshPRINCE   |     59 |      8 |       45 |       0.0616 |      0.0156 |
| rppv128:ridge      | RDST          |     42 |     10 |       60 |       0.0497 |     -0.006  |
| rppv128:ridge      | QUANT         |     44 |     16 |       52 |       0.6141 |      0.0028 |
| rppv128:ridge      | WEASEL-2.0    |     44 |     12 |       56 |       0.2865 |     -0.0033 |
| rppv128:ridge      | Catch22       |     94 |      6 |       12 |       0      |      0.0758 |
| rppv128:ridge      | 1NN-DTW       |     97 |      7 |        8 |       0      |      0.1151 |
| rppv128_diff:ridge | HC2           |     24 |     14 |       74 |       0      |     -0.0167 |
| rppv128_diff:ridge | MR-Hydra      |     30 |     21 |       61 |       0.0017 |     -0.0083 |
| rppv128_diff:ridge | InceptionTime |     47 |     12 |       53 |       0.5225 |      0.0019 |
| rppv128_diff:ridge | MultiROCKET   |     34 |     24 |       54 |       0.021  |     -0.006  |
| rppv128_diff:ridge | MiniROCKET    |     53 |     11 |       48 |       0.7734 |      0.0017 |
| rppv128_diff:ridge | ROCKET        |     46 |     14 |       52 |       0.5865 |      0.0039 |
| rppv128_diff:ridge | FreshPRINCE   |     61 |      9 |       42 |       0.0108 |      0.0181 |
| rppv128_diff:ridge | RDST          |     44 |     14 |       54 |       0.1959 |     -0.0034 |
| rppv128_diff:ridge | QUANT         |     50 |     12 |       50 |       0.6561 |      0.0053 |
| rppv128_diff:ridge | WEASEL-2.0    |     48 |     11 |       53 |       0.7476 |     -0.0008 |
| rppv128_diff:ridge | Catch22       |     98 |      7 |        7 |       0      |      0.0783 |
| rppv128_diff:ridge | 1NN-DTW       |     97 |      7 |        8 |       0      |      0.1176 |
| rppv128_p2:ridge   | HC2           |     26 |     18 |       68 |       0.0001 |     -0.0157 |
| rppv128_p2:ridge   | MR-Hydra      |     31 |     21 |       60 |       0.0012 |     -0.0073 |
| rppv128_p2:ridge   | InceptionTime |     50 |     13 |       49 |       0.603  |      0.003  |
| rppv128_p2:ridge   | MultiROCKET   |     34 |     19 |       59 |       0.0112 |     -0.0049 |
| rppv128_p2:ridge   | MiniROCKET    |     62 |     10 |       40 |       0.1695 |      0.0027 |
| rppv128_p2:ridge   | ROCKET        |     48 |     17 |       47 |       0.3841 |      0.0049 |
| rppv128_p2:ridge   | FreshPRINCE   |     66 |     10 |       36 |       0.0051 |      0.0191 |
| rppv128_p2:ridge   | RDST          |     40 |     12 |       60 |       0.2638 |     -0.0024 |
| rppv128_p2:ridge   | QUANT         |     53 |     15 |       44 |       0.243  |      0.0063 |
| rppv128_p2:ridge   | WEASEL-2.0    |     53 |     13 |       46 |       0.3763 |      0.0002 |
| rppv128_p2:ridge   | Catch22       |     95 |      6 |       11 |       0      |      0.0794 |
| rppv128_p2:ridge   | 1NN-DTW       |     98 |      7 |        7 |       0      |      0.1187 |
| minirocket:ridge   | HC2           |     20 |     16 |       76 |       0      |     -0.0167 |
| minirocket:ridge   | MR-Hydra      |     26 |     19 |       67 |       0      |     -0.0083 |
| minirocket:ridge   | InceptionTime |     50 |     16 |       46 |       0.9592 |      0.0019 |
| minirocket:ridge   | MultiROCKET   |     25 |     20 |       67 |       0.0002 |     -0.0059 |
| minirocket:ridge   | MiniROCKET    |     44 |     33 |       35 |       0.1225 |      0.0017 |
| minirocket:ridge   | ROCKET        |     49 |     14 |       49 |       0.8165 |      0.0039 |
| minirocket:ridge   | FreshPRINCE   |     62 |     12 |       38 |       0.0054 |      0.0181 |
| minirocket:ridge   | RDST          |     46 |     14 |       52 |       0.3921 |     -0.0034 |
| minirocket:ridge   | QUANT         |     53 |     18 |       41 |       0.4153 |      0.0053 |
| minirocket:ridge   | WEASEL-2.0    |     52 |      8 |       52 |       0.9625 |     -0.0008 |
| minirocket:ridge   | Catch22       |     95 |      6 |       11 |       0      |      0.0784 |
| minirocket:ridge   | 1NN-DTW       |     97 |      5 |       10 |       0      |      0.1177 |

## Pairwise among STCT variants

| a                  | b                  |   a_wins |   ties |   b_wins |   wilcoxon_p |
|:-------------------|:-------------------|---------:|-------:|---------:|-------------:|
| fixed:rf           | fixed+pooled:et    |       15 |      9 |       88 |       0      |
| fixed:rf           | rppv128:ridge      |       14 |     10 |       88 |       0      |
| fixed:rf           | rppv128_diff:ridge |       14 |      6 |       92 |       0      |
| fixed:rf           | rppv128_p2:ridge   |       13 |      9 |       90 |       0      |
| fixed:rf           | minirocket:ridge   |       15 |     11 |       86 |       0      |
| fixed+pooled:et    | rppv128:ridge      |       40 |     12 |       60 |       0.0047 |
| fixed+pooled:et    | rppv128_diff:ridge |       34 |     10 |       68 |       0.0003 |
| fixed+pooled:et    | rppv128_p2:ridge   |       30 |     16 |       66 |       0.0003 |
| fixed+pooled:et    | minirocket:ridge   |       32 |     16 |       64 |       0.0003 |
| rppv128:ridge      | rppv128_diff:ridge |       36 |     17 |       59 |       0.0604 |
| rppv128:ridge      | rppv128_p2:ridge   |       35 |     18 |       59 |       0.0148 |
| rppv128:ridge      | minirocket:ridge   |       44 |     20 |       48 |       0.4281 |
| rppv128_diff:ridge | rppv128_p2:ridge   |       48 |     16 |       48 |       0.6115 |
| rppv128_diff:ridge | minirocket:ridge   |       50 |     12 |       50 |       0.911  |
| rppv128_p2:ridge   | minirocket:ridge   |       55 |     16 |       41 |       0.2934 |
