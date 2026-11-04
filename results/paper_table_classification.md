# Classification: accuracy, features and runtime

112 UCR datasets, 74 of them outside the development set; sorted by mean accuracy.

|   # | Method        |   Accuracy (%) |   Held out (%) |   Mean rank | Features                                     | Runtime here (h)   |   Median per dataset (s) |   Published runtime (h) |
|----:|:--------------|---------------:|---------------:|------------:|:---------------------------------------------|:-------------------|-------------------------:|------------------------:|
|   1 | HC2           |          87.64 |          85.65 |        4.31 | over 100,000 (its Arsenal alone: 25 x 4,000) | not run            |                          |                   290.2 |
|   2 | MR-Hydra      |          86.8  |          84.23 |        5.08 | 55,643 (mean; 50,752-58,944)                 | 1.60               |                     15.9 |                     1.6 |
|   3 | STCT+         |          86.72 |          84.5  |        5.81 | 3,547 (mean; 125-8,082)                      | 0.15               |                      1.6 |                         |
|   4 | MultiROCKET   |          86.57 |          84.08 |        5.42 | 49,728                                       | 0.66               |                      5.7 |                     9   |
|   5 | RDST          |          86.31 |          83.91 |        6.45 | 30,000                                       | 2.24               |                     22.2 |                     4.9 |
|   6 | WEASEL-2.0    |          86.05 |          83.22 |        6.91 | 32,665 (mean; 6,862-56,760)                  | 2.63               |                     16.4 |                     2.4 |
|   7 | MiniROCKET    |          85.8  |          83.11 |        6.87 | 9,996                                        | 0.15               |                      0.8 |                     2.1 |
|   8 | InceptionTime |          85.78 |          83.63 |        6.43 | learned: 5 networks x 128                    | not run            |                          |                    86.8 |
|   9 | ROCKET        |          85.58 |          82.95 |        6.6  | 20,000                                       | 4.06               |                     41.1 |                     9.7 |
|  10 | QUANT         |          85.44 |          83.27 |        6.61 | 4,993 (mean; 116-25,203)                     | 0.29               |                      2.8 |                     0.2 |
|  11 | FreshPRINCE   |          84.16 |          82.38 |        7.89 | 783                                          | not run            |                          |                    39   |
|  12 | Catch22       |          78.14 |          77.45 |       11.02 | 22                                           | 0.03               |                      0.6 |                     0.9 |
|  13 | 1NN-DTW       |          74.21 |          71.17 |       11.6  | none (raw series)                            | not run            |                          |                    81.7 |

Runtime here: fit on the training split plus predict on the test split, one thread, summed over the datasets; all methods on a dataset ran in the same process on the same machine (paper_time.py). Published runtime: fit + predict times from timeseriesclassification.com (other hardware; not comparable with the runtime here).

Accuracy measured here minus published accuracy, per method (configuration check): Catch22 -0.43, MR-Hydra +0.04, MiniROCKET +0.17, MultiROCKET +0.10, QUANT -0.21, RDST +0.02, ROCKET +0.12, WEASEL-2.0 +0.02
