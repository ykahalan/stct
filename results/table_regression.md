# Regression on 26 TSER datasets

Default train/test splits, sorted by median RMSE relative to the best of the 20 methods on each dataset. Rank: mean rank. W/L: datasets where STCT+ has lower/higher RMSE. p log, p raw: two-sided Wilcoxon signed-rank tests on log and raw RMSE, Holm-corrected over 19. Features: per series (raw: the series itself). Time: fit plus predict on one thread, total hours; (h): published, other hardware; ‡: timed in a later pass, with the STCT+ total of that pass in brackets; §: TabPFN on the CPU with 4 threads plus STCT feature extraction; GPU: run on a GPU in its paper.

| Method | RMSE/best | Rank | W/L | p log | p raw | Features | Time (h) |
|---|---|---|---|---|---|---|---|
| STCT-TabPFN | 1.008 | 2.46 | 2/24 | <0.001 | <0.001 | 673 | 2.45 § |
| tsfresh-TabPFN | 1.123 | 4.12 | 6/20 | 0.026 | 0.003 | 777 | GPU |
| MultiROCKET-TabPFN | 1.129 | 4.54 | 8/18 | 0.45 | 0.043 | 49,728 | GPU |
| **STCT+** | 1.277 | 6.81 |  |  |  | 2,096 | 0.03 |
| MultiROCKET | 1.326 | 8.85 | 15/11 | 0.45 | 1.00 | 79,968 | 0.38 ‡ (91 s) |
| DrCIF | 1.338 | 8.46 | 18/8 | 0.11 | 1.00 | 86,923 | (175.4) |
| FreshPRINCE | 1.346 | 9.00 | 18/8 | 0.11 | 1.00 | 783 | 38.85 ‡ (101 s) |
| MiniROCKET | 1.378 | 8.42 | 17/9 | 0.14 | 1.00 | 9,996 | 0.03 ‡ (91 s) |
| InceptionTime | 1.379 | 10.65 | 20/6 | 0.11 | 0.45 | 640 | (60.0) |
| catch22-TabPFN | 1.393 | 9.00 | 15/11 | 0.11 | 1.00 | 22 | GPU |
| ROCKET | 1.525 | 11.27 | 19/7 | 0.016 | 0.42 | 20,000 | 2.57 |
| Ridge | 1.622 | 13.50 | 22/4 | <0.001 | 0.055 | raw | 0.00 |
| RotF | 1.650 | 10.38 | 20/6 | 0.026 | 0.16 | 1,816 | 28.30 ‡ (101 s) |
| XGBoost | 1.686 | 13.92 | 24/2 | <0.001 | 0.002 | raw | (0.6) |
| GridSVR | 1.687 | 13.73 | 22/4 | 0.001 | 0.051 | raw | 0.65 |
| RandF | 1.690 | 12.92 | 23/3 | <0.001 | 0.002 | raw | 0.49 |
| 1NN-DTW | 1.724 | 15.50 | 23/3 | <0.001 | 0.016 | - | 7.99 |
| ResNet | 1.728 | 14.73 | 23/3 | <0.001 | 0.014 | 128 | (14.1) |
| FPCR | 1.878 | 14.96 | 24/2 | <0.001 | 0.001 | FPCA | (0.1) |
| FCN | 2.942 | 16.77 | 24/2 | <0.001 | <0.001 | 128 | (9.3) |

Groups of consecutive methods with no significant pairwise difference (Wilcoxon on log RMSE, Holm over all 190 pairs, alpha 0.05):

- STCT-TabPFN, tsfresh-TabPFN, MultiROCKET-TabPFN
- tsfresh-TabPFN, MultiROCKET-TabPFN, STCT+
- STCT+, MultiROCKET, DrCIF, FreshPRINCE, MiniROCKET, InceptionTime, catch22-TabPFN, ROCKET
- FreshPRINCE, MiniROCKET, InceptionTime, catch22-TabPFN, ROCKET, Ridge, RotF
- MiniROCKET, InceptionTime, catch22-TabPFN, ROCKET, Ridge, RotF, XGBoost, GridSVR, RandF
- ROCKET, Ridge, RotF, XGBoost, GridSVR, RandF, 1NN-DTW, ResNet, FPCR
- 1NN-DTW, ResNet, FPCR, FCN

## STCT+ over random seeds

Against the 13 benchmark methods (14 with STCT+).

| Seed | RMSE/best | Place |
|---|---|---|
| 42 | 1.131 | 1 of 14 |
| 1 | 1.144 | 1 of 14 |
| 2 | 1.130 | 1 of 14 |
