# classification: STCT variants vs published results

112 datasets common to all methods (default train/test split).

## Overall

|                    |   mean |   mean_rank |   best_on |
|:-------------------|-------:|------------:|----------:|
| HC2                | 0.8764 |      6.2634 |         6 |
| MR-Hydra           | 0.868  |      7.4286 |         9 |
| MultiROCKET        | 0.8657 |      7.942  |         2 |
| RDST               | 0.8631 |      9.442  |         4 |
| rppv128_p2:ridge   | 0.8607 |      9.5045 |         4 |
| QUANT              | 0.8544 |      9.7902 |         7 |
| rppv128_diff:ridge | 0.8597 |      9.8348 |         6 |
| ROCKET             | 0.8558 |      9.8839 |         1 |
| InceptionTime      | 0.8578 |      9.9688 |        13 |
| minirocket:ridge   | 0.8597 |     10.0714 |         0 |
| WEASEL-2.0         | 0.8605 |     10.1518 |         4 |
| MiniROCKET         | 0.858  |     10.2857 |         0 |
| rppv128:ridge      | 0.8572 |     10.75   |         2 |
| pyr3:et            | 0.846  |     11.1116 |         1 |
| fixed+pooled:et    | 0.8453 |     12.0179 |         2 |
| FreshPRINCE        | 0.8416 |     12.0848 |         9 |
| msP7+pooled:et     | 0.8447 |     12.1384 |         2 |
| msP7:et            | 0.807  |     14.3438 |         0 |
| ms5:et             | 0.8024 |     15.308  |         0 |
| fixed:rf           | 0.7816 |     17.3705 |         0 |
| Catch22            | 0.7814 |     18.1384 |         1 |
| 1NN-DTW            | 0.7421 |     19.1696 |         1 |

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
| ms5:et             | HC2           |     13 |     11 |       88 |       0      |     -0.074  |
| ms5:et             | MR-Hydra      |     15 |     10 |       87 |       0      |     -0.0656 |
| ms5:et             | InceptionTime |     27 |     12 |       73 |       0      |     -0.0554 |
| ms5:et             | MultiROCKET   |     18 |     10 |       84 |       0      |     -0.0633 |
| ms5:et             | MiniROCKET    |     22 |     11 |       79 |       0      |     -0.0557 |
| ms5:et             | ROCKET        |     25 |      8 |       79 |       0      |     -0.0534 |
| ms5:et             | FreshPRINCE   |     27 |     11 |       74 |       0      |     -0.0392 |
| ms5:et             | RDST          |     23 |      8 |       81 |       0      |     -0.0607 |
| ms5:et             | QUANT         |     20 |      9 |       83 |       0      |     -0.052  |
| ms5:et             | WEASEL-2.0    |     25 |      8 |       79 |       0      |     -0.0581 |
| ms5:et             | Catch22       |     74 |      4 |       34 |       0.0003 |      0.021  |
| ms5:et             | 1NN-DTW       |     82 |      8 |       22 |       0      |      0.0603 |
| msP7:et            | HC2           |     14 |     15 |       83 |       0      |     -0.0694 |
| msP7:et            | MR-Hydra      |     21 |     11 |       80 |       0      |     -0.061  |
| msP7:et            | InceptionTime |     31 |      9 |       72 |       0      |     -0.0507 |
| msP7:et            | MultiROCKET   |     22 |     10 |       80 |       0      |     -0.0586 |
| msP7:et            | MiniROCKET    |     25 |     13 |       74 |       0      |     -0.051  |
| msP7:et            | ROCKET        |     30 |      9 |       73 |       0      |     -0.0488 |
| msP7:et            | FreshPRINCE   |     38 |     11 |       63 |       0.0011 |     -0.0346 |
| msP7:et            | RDST          |     26 |     13 |       73 |       0      |     -0.0561 |
| msP7:et            | QUANT         |     18 |     17 |       77 |       0      |     -0.0474 |
| msP7:et            | WEASEL-2.0    |     29 |      9 |       74 |       0      |     -0.0535 |
| msP7:et            | Catch22       |     76 |      7 |       29 |       0      |      0.0257 |
| msP7:et            | 1NN-DTW       |     81 |      8 |       23 |       0      |      0.065  |
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
| msP7+pooled:et     | HC2           |     16 |     15 |       81 |       0      |     -0.0317 |
| msP7+pooled:et     | MR-Hydra      |     21 |     19 |       72 |       0      |     -0.0233 |
| msP7+pooled:et     | InceptionTime |     42 |     12 |       58 |       0.0161 |     -0.0131 |
| msP7+pooled:et     | MultiROCKET   |     25 |     15 |       72 |       0      |     -0.021  |
| msP7+pooled:et     | MiniROCKET    |     35 |     14 |       63 |       0.0022 |     -0.0133 |
| msP7+pooled:et     | ROCKET        |     34 |     15 |       63 |       0.0024 |     -0.0111 |
| msP7+pooled:et     | FreshPRINCE   |     48 |     12 |       52 |       0.85   |      0.0031 |
| msP7+pooled:et     | RDST          |     34 |     10 |       68 |       0.0001 |     -0.0184 |
| msP7+pooled:et     | QUANT         |     31 |     19 |       62 |       0.0047 |     -0.0097 |
| msP7+pooled:et     | WEASEL-2.0    |     33 |     11 |       68 |       0.0004 |     -0.0158 |
| msP7+pooled:et     | Catch22       |     88 |      9 |       15 |       0      |      0.0634 |
| msP7+pooled:et     | 1NN-DTW       |     97 |      6 |        9 |       0      |      0.1027 |
| pyr3:et            | HC2           |     20 |     16 |       76 |       0      |     -0.0304 |
| pyr3:et            | MR-Hydra      |     29 |     12 |       71 |       0      |     -0.022  |
| pyr3:et            | InceptionTime |     43 |     12 |       57 |       0.0473 |     -0.0117 |
| pyr3:et            | MultiROCKET   |     29 |     16 |       67 |       0      |     -0.0196 |
| pyr3:et            | MiniROCKET    |     36 |     18 |       58 |       0.0195 |     -0.012  |
| pyr3:et            | ROCKET        |     40 |     12 |       60 |       0.0288 |     -0.0098 |
| pyr3:et            | FreshPRINCE   |     57 |     13 |       42 |       0.4668 |      0.0044 |
| pyr3:et            | RDST          |     39 |     10 |       63 |       0.0024 |     -0.0171 |
| pyr3:et            | QUANT         |     35 |     17 |       60 |       0.0144 |     -0.0084 |
| pyr3:et            | WEASEL-2.0    |     39 |     10 |       63 |       0.0054 |     -0.0145 |
| pyr3:et            | Catch22       |     88 |      9 |       15 |       0      |      0.0647 |
| pyr3:et            | 1NN-DTW       |     98 |      6 |        8 |       0      |      0.104  |
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
| fixed:rf           | ms5:et             |       21 |     11 |       80 |       0      |
| fixed:rf           | msP7:et            |       18 |      9 |       85 |       0      |
| fixed:rf           | fixed+pooled:et    |       15 |      9 |       88 |       0      |
| fixed:rf           | msP7+pooled:et     |       15 |      7 |       90 |       0      |
| fixed:rf           | pyr3:et            |       10 |      8 |       94 |       0      |
| fixed:rf           | rppv128:ridge      |       14 |     10 |       88 |       0      |
| fixed:rf           | rppv128_diff:ridge |       14 |      6 |       92 |       0      |
| fixed:rf           | rppv128_p2:ridge   |       13 |      9 |       90 |       0      |
| fixed:rf           | minirocket:ridge   |       15 |     11 |       86 |       0      |
| ms5:et             | msP7:et            |       35 |     18 |       59 |       0.0114 |
| ms5:et             | fixed+pooled:et    |       22 |     16 |       74 |       0      |
| ms5:et             | msP7+pooled:et     |       22 |     14 |       76 |       0      |
| ms5:et             | pyr3:et            |       15 |     13 |       84 |       0      |
| ms5:et             | rppv128:ridge      |       25 |     11 |       76 |       0      |
| ms5:et             | rppv128_diff:ridge |       24 |      9 |       79 |       0      |
| ms5:et             | rppv128_p2:ridge   |       19 |     11 |       82 |       0      |
| ms5:et             | minirocket:ridge   |       23 |     14 |       75 |       0      |
| msP7:et            | fixed+pooled:et    |       29 |     16 |       67 |       0      |
| msP7:et            | msP7+pooled:et     |       24 |     19 |       69 |       0      |
| msP7:et            | pyr3:et            |       15 |     19 |       78 |       0      |
| msP7:et            | rppv128:ridge      |       28 |     14 |       70 |       0      |
| msP7:et            | rppv128_diff:ridge |       28 |     11 |       73 |       0      |
| msP7:et            | rppv128_p2:ridge   |       23 |     11 |       78 |       0      |
| msP7:et            | minirocket:ridge   |       26 |     10 |       76 |       0      |
| fixed+pooled:et    | msP7+pooled:et     |       43 |     27 |       42 |       0.8523 |
| fixed+pooled:et    | pyr3:et            |       37 |     23 |       52 |       0.0955 |
| fixed+pooled:et    | rppv128:ridge      |       40 |     12 |       60 |       0.0047 |
| fixed+pooled:et    | rppv128_diff:ridge |       34 |     10 |       68 |       0.0003 |
| fixed+pooled:et    | rppv128_p2:ridge   |       30 |     16 |       66 |       0.0003 |
| fixed+pooled:et    | minirocket:ridge   |       32 |     16 |       64 |       0.0003 |
| msP7+pooled:et     | pyr3:et            |       32 |     23 |       57 |       0.0782 |
| msP7+pooled:et     | rppv128:ridge      |       43 |      9 |       60 |       0.0088 |
| msP7+pooled:et     | rppv128_diff:ridge |       34 |     10 |       68 |       0.0005 |
| msP7+pooled:et     | rppv128_p2:ridge   |       32 |     17 |       63 |       0.0004 |
| msP7+pooled:et     | minirocket:ridge   |       32 |     18 |       62 |       0.0004 |
| pyr3:et            | rppv128:ridge      |       43 |     10 |       59 |       0.0659 |
| pyr3:et            | rppv128_diff:ridge |       36 |     12 |       64 |       0.007  |
| pyr3:et            | rppv128_p2:ridge   |       39 |     16 |       57 |       0.0012 |
| pyr3:et            | minirocket:ridge   |       37 |     18 |       57 |       0.0074 |
| rppv128:ridge      | rppv128_diff:ridge |       36 |     17 |       59 |       0.0604 |
| rppv128:ridge      | rppv128_p2:ridge   |       35 |     18 |       59 |       0.0148 |
| rppv128:ridge      | minirocket:ridge   |       44 |     20 |       48 |       0.4281 |
| rppv128_diff:ridge | rppv128_p2:ridge   |       48 |     16 |       48 |       0.6115 |
| rppv128_diff:ridge | minirocket:ridge   |       50 |     12 |       50 |       0.911  |
| rppv128_p2:ridge   | minirocket:ridge   |       55 |     16 |       41 |       0.2934 |
