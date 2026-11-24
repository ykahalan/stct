# Classification on 112 UCR datasets

Default train/test splits, sorted by mean accuracy (%). Rank: mean rank among 24 methods on the 111 datasets common to all. W/L: datasets where STCT+ is more/less accurate. p: two-sided Wilcoxon signed-rank test against STCT+, Holm-corrected over 23 comparisons. Features: per series. Time: fit plus predict on one thread, total hours; (h): published, other hardware; ‡: timed in a later pass in which STCT+ took 429 s; §: TabPFN on the CPU with 4 threads plus STCT feature extraction; ≈: estimated from the method's own paper; GPU: run on a GPU in its paper, not timed here. †: 111 datasets.

| Method | Accuracy | Rank | W/L | p | Features | Time (h) |
|---|---|---|---|---|---|---|
| HC2 | 87.64 | 8.36 | 28/66 | 0.007 | >100,000 | (290) |
| STCT-TabPFN | 87.37 | 8.28 | 28/68 | 0.015 | 685 | 237.66 § |
| SelF-Rocket | 87.06 | 10.21 | 38/59 | 0.84 | 9,996/19,992 | ≈2.9 |
| Hydra+SelF-Rocket | 87.04 | 9.91 | 41/52 | 1.00 | SR + Hydra | ≈3.5 |
| MR-Hydra | 86.80 | 10.08 | 39/54 | 1.00 | 55,643 | 1.60 |
| MultiROCKET-TabPFN | 86.79 † | 8.76 | 34/64 | 0.34 | 49,728 | GPU |
| **STCT+** | 86.72 | 11.30 |  |  | 3,547 | 0.15 |
| MultiROCKET | 86.57 | 10.59 | 40/51 | 1.00 | 49,728 | 0.66 |
| TS-CHIEF | 86.48 | 11.77 | 52/45 | 1.00 | - | - |
| RIST | 86.44 | 10.85 | 47/48 | 1.00 | 80,645 | 10.03 ‡ |
| RDST | 86.31 | 12.11 | 55/45 | 1.00 | 30,000 | 2.24 |
| WEASEL-2.0 | 86.05 | 12.98 | 61/37 | 0.39 | 32,665 | 2.63 |
| H-InceptionTime | 85.83 | 11.36 | 46/52 | 1.00 | 640 | GPU |
| MiniROCKET | 85.80 | 13.12 | 58/35 | 0.18 | 9,996 | 0.15 |
| InceptionTime | 85.78 | 12.23 | 52/45 | 1.00 | 640 | (87) |
| ROCKET | 85.58 | 12.52 | 56/43 | 0.84 | 20,000 | 4.06 |
| QUANT | 85.44 | 12.70 | 57/37 | 0.19 | 4,993 | 0.29 |
| LITETime | 85.26 | 12.41 | 51/47 | 1.00 | 160 | GPU |
| tsfresh-TabPFN | 84.98 † | 11.08 | 44/51 | 1.00 | 777 | GPU |
| R-STSF | 84.86 | 13.84 | 63/34 | 0.013 | 14,377 | 0.62 ‡ |
| FreshPRINCE | 84.16 | 14.91 | 71/31 | <0.001 | 783 | 29.11 ‡ |
| catch22-TabPFN | 80.46 | 18.43 | 87/15 | <0.001 | 22 | GPU |
| Catch22 | 78.14 | 20.67 | 101/4 | <0.001 | 22 | 0.03 |
| 1NN-DTW | 74.21 | 21.50 | 101/5 | <0.001 | - | 25.89 ‡ |

Groups of consecutive methods with no significant pairwise difference (Wilcoxon, Holm over all 276 pairs, alpha 0.05):

- HC2, STCT-TabPFN, SelF-Rocket, Hydra+SelF-Rocket, MR-Hydra, MultiROCKET-TabPFN, STCT+, MultiROCKET
- STCT-TabPFN, SelF-Rocket, Hydra+SelF-Rocket, MR-Hydra, MultiROCKET-TabPFN, STCT+, MultiROCKET, TS-CHIEF, RIST, RDST
- MR-Hydra, MultiROCKET-TabPFN, STCT+, MultiROCKET, TS-CHIEF, RIST, RDST, WEASEL-2.0, H-InceptionTime
- TS-CHIEF, RIST, RDST, WEASEL-2.0, H-InceptionTime, MiniROCKET, InceptionTime, ROCKET, QUANT, LITETime, tsfresh-TabPFN
- RDST, WEASEL-2.0, H-InceptionTime, MiniROCKET, InceptionTime, ROCKET, QUANT, LITETime, tsfresh-TabPFN, R-STSF
- R-STSF, FreshPRINCE

## STCT+ over random seeds

Against the 12 benchmark methods (13 with STCT+).

| Seed | Accuracy | Place | Significantly more accurate (Holm) |
|---|---|---|---|
| 42 | 86.72 | 3 of 13 | HC2 |
| 1 | 86.68 | 3 of 13 | HC2 |
| 2 | 86.51 | 4 of 13 | HC2 |
