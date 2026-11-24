"""Comparison tables of STCT+ and STCT-TabPFN against published classifiers and regressors.

    python tables.py

Reads results/ and writes results/table_classification.md and
results/table_regression.md.  Reference accuracies and RMSEs are the published
results on the default splits (timeseriesclassification.com for the benchmark
methods, results/published_recent/ for later methods); STCT+ (seed 42),
STCT-TabPFN and every runtime were measured here.  Seeds 1 and 2 give the
spread over random seeds.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

R = "results"
CLS_REFS = ["HC2", "MR-Hydra", "MultiROCKET", "RDST", "WEASEL-2.0", "MiniROCKET", "InceptionTime", "ROCKET", "QUANT",
            "FreshPRINCE", "Catch22", "1NN-DTW"]
REG_REFS = ["DrCIF", "FreshPRINCE", "InceptionTime", "RotF", "ROCKET", "Ridge", "RandF", "XGBoost", "GridSVR", "ResNet",
            "FPCR", "1NN-DTW", "FCN"]
RECENT_C = {  # label -> per-resample result file; column "0" is the default split
    "Hydra+SelF-Rocket": "Perf_rsmpl_HSR_UCR112_fpc5000_nr10.csv",
    "SelF-Rocket": "Perf_rsmpl_SR_UCR112_fpc5000_nr10.csv",
    "MultiROCKET-TabPFN": "tsc_results_Tab_multi.csv",
    "tsfresh-TabPFN": "tsc_results_Tab_tsfresh.csv",
    "catch22-TabPFN": "tsc_results_Tab_c22.csv",
    "TS-CHIEF": "uni_TS-CHIEF.csv",
    "RIST": "uni_RIST.csv",
    "H-InceptionTime": "uni_H-InceptionTime.csv",
    "LITETime": "uni_LITETime.csv",
    "R-STSF": "uni_R-STSF.csv",
}
RECENT_R = {
    "tsfresh-TabPFN": "tser_results_Tab_tsfresh.csv",
    "MultiROCKET-TabPFN": "tser_results_Tab_multi.csv",
    "catch22-TabPFN": "tser_results_Tab_c22.csv",
    "MultiROCKET": "tser_results_MultiROCKET.csv",
    "MiniROCKET": "tser_results_MiniROCKET.csv",
}
# features of methods not run here (deep networks: width fed to the output layer, summed over ensemble members)
FEATS_C = {"HC2": ">100,000", "InceptionTime": "640", "Hydra+SelF-Rocket": "SR + Hydra", "SelF-Rocket": "9,996/19,992",
           "MultiROCKET-TabPFN": "49,728", "tsfresh-TabPFN": "777", "catch22-TabPFN": "22", "TS-CHIEF": "-",
           "H-InceptionTime": "640", "LITETime": "160"}
FEATS_R = {"DrCIF": "86,923", "InceptionTime": "640", "XGBoost": "raw", "ResNet": "128", "FPCR": "FPCA", "FCN": "128",
           "tsfresh-TabPFN": "777", "MultiROCKET-TabPFN": "49,728", "catch22-TabPFN": "22"}
RAW_R = ("Ridge", "RandF", "GridSVR")
GPU = {"MultiROCKET-TabPFN", "tsfresh-TabPFN", "catch22-TabPFN", "H-InceptionTime", "LITETime"}
# SelF-Rocket: total single-thread hours from its paper's own timings relative to MiniRocket and MultiRocket
SELF_ROCKET = {"SelF-Rocket": (292.27 / 15.55, 292.27 / 65.20), "Hydra+SelF-Rocket": (347.80 / 15.55, 347.80 / 65.20)}
TP = "STCT-TabPFN"


def holm(p):
    p = np.asarray(p, float)
    order, m, out, running = np.argsort(p), len(p), np.empty(len(p)), 0.0
    for k, i in enumerate(order):
        running = max(running, (m - k) * p[i])
        out[i] = min(1.0, running)
    return out


def wilcoxon(a, b):
    x = np.asarray(a) - np.asarray(b)
    x = x[x != 0]
    return stats.wilcoxon(x).pvalue if len(x) else 1.0


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def fmt_int(v):
    return f"{v:,.0f}"


def default_split(fname):
    s = pd.read_csv(os.path.join(R, "published_recent", fname), index_col=0)["0"]
    s.index = s.index.str.replace(r"_(eq|nmv)$", "", regex=True)
    return s[~s.index.duplicated()]


def published_hours(task, method, datasets):
    path = os.path.join(R, f"published_times{'' if task == 'classification' else '_regression'}", method)
    try:
        fit = pd.read_csv(f"{path}_fittime.csv", index_col=0)["0"]
        pred = pd.read_csv(f"{path}_predicttime.csv", index_col=0)["0"]
    except FileNotFoundError:
        return np.nan
    t = ((fit + pred) / 1000).reindex(datasets)
    return np.nan if (t < 0).any() else t.sum() / 3600


def later_pass(paths, datasets):
    """Methods timed in a later pass: (rows, total seconds of STCT+ in that pass)."""
    out = {}
    for path in paths:
        s = pd.read_csv(path).drop_duplicates(["dataset", "method"])
        s = s[s.dataset.isin(datasets)]
        full = {m: g for m, g in s.groupby("method") if len(g) == len(datasets)}
        st = full.pop("STCT+")
        out.update({m: (g, st.seconds.sum()) for m, g in full.items()})
    return out


def groups(D, order):
    """Runs of consecutive rows with no significant pairwise difference (Wilcoxon, Holm over all pairs, 0.05)."""
    k = len(order)
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    p = holm([wilcoxon(*D[[order[i], order[j]]].dropna().T.values) for i, j in pairs])
    same = np.eye(k, dtype=bool)
    for (i, j), q in zip(pairs, p):
        same[i, j] = same[j, i] = q > 0.05
    runs = []
    for i in range(k):
        j = i
        while j + 1 < k and same[i:j + 2, i:j + 2].all():
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in runs):
            runs.append((i, j))
    return runs, len(pairs)


def markdown(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows]) + "\n"


def tabpfn(path, score, index):
    t = pd.read_csv(path).query("method == @TP").set_index("dataset").reindex(index)
    hours = (t.model_seconds + t.cpu_feature_seconds).sum() / 3600
    return t[score], fmt_int(t.n_views.mean()), f"{hours:.2f} §"


def group_lines(runs, order):
    return "".join(f"- {', '.join(order[a:b + 1])}\n" for a, b in runs)


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------

ours_c = pd.read_csv(os.path.join(R, "stct_plus_classification.csv"))
seeds_c = ours_c.pivot(index="dataset", columns="seed", values="accuracy")
DC = pd.read_csv(os.path.join(R, "published_classification_accuracy.csv"), index_col=0)[CLS_REFS]
DC = DC.join(seeds_c[42].rename("STCT+"), how="inner").dropna()
time_c = pd.read_csv(os.path.join(R, "timing_classification.csv"))
time_c = time_c[time_c.dataset.isin(DC.index)]
t_wide = time_c.pivot(index="dataset", columns="method", values="seconds")
later_c = later_pass([os.path.join(R, "timing_classification_later.csv")], DC.index)
AC = DC.copy()
for label, f in RECENT_C.items():
    AC[label] = default_split(f).reindex(DC.index)
tp_c = tabpfn(os.path.join(R, "stct_tabpfn_classification.csv"), "accuracy", DC.index)
AC[TP] = tp_c[0]
common = AC.dropna().index
rank = AC.loc[common].rank(axis=1, ascending=False).mean()
others = [m for m in AC.columns if m != "STCT+"]
p_raw = {m: wilcoxon(AC[["STCT+", m]].dropna()["STCT+"], AC[m].dropna()) for m in others}
p_holm = dict(zip(others, holm([p_raw[m] for m in others])))
order_c = sorted(AC.columns, key=lambda m: -AC[m].mean())
runs_c, npairs_c = groups(AC, order_c)
rows = []
for m in order_c:
    a = AC[m].dropna()
    s = AC.loc[a.index, "STCT+"]
    if m == TP:
        feats, hours = tp_c[1:]
    elif m in t_wide.columns:
        n = time_c[time_c.method == m].n_features
        feats = fmt_int(n.iloc[0]) if n.nunique() == 1 else fmt_int(n.mean())
        hours = f"{t_wide[m].sum() / 3600:.2f}"
    elif m in later_c:
        g, st = later_c[m]
        feats = "-" if m == "1NN-DTW" else fmt_int(g.n_features.mean())
        hours = f"{g.seconds.sum() / 3600:.2f} ‡"
    elif m in SELF_ROCKET:
        lo, hi = sorted(r * x for r, x in zip(SELF_ROCKET[m], (t_wide["MiniROCKET"].sum(), t_wide["MultiROCKET"].sum())))
        feats, hours = FEATS_C[m], f"≈{(lo + hi) / 2 / 3600:.1f}"
    else:
        ph = published_hours("classification", m, DC.index)
        feats, hours = FEATS_C[m], "GPU" if m in GPU else f"({ph:.0f})" if not np.isnan(ph) else "-"
    name = f"**{m}**" if m == "STCT+" else m
    acc = f"{a.mean() * 100:.2f}" + (" †" if len(a) < len(DC) else "")
    wl = "" if m == "STCT+" else f"{int((s > a).sum())}/{int((s < a).sum())}"
    rows.append([name, acc, f"{rank[m]:.2f}", wl, "" if m == "STCT+" else fmt_p(p_holm[m]), feats, hours])

seed_rows = []
for seed in sorted(seeds_c.columns, key=lambda s: (s != 42, s)):
    a = seeds_c[seed].reindex(DC.index)
    allc = DC[CLS_REFS].join(a.rename("x"))
    place = int((allc.mean() > a.mean()).sum()) + 1
    q = dict(zip(CLS_REFS, holm([wilcoxon(a, DC[m]) for m in CLS_REFS])))
    better = ", ".join(m for m in CLS_REFS if q[m] < 0.05 and DC[m].mean() > a.mean()) or "none"
    seed_rows.append([str(seed), f"{a.mean() * 100:.2f}", f"{place} of 13", better])

with open(os.path.join(R, "table_classification.md"), "w", encoding="utf-8") as f:
    f.write(f"# Classification on {len(DC)} UCR datasets\n\n"
            f"Default train/test splits, sorted by mean accuracy (%). Rank: mean rank among {len(AC.columns)} methods "
            f"on the {len(common)} datasets common to all. W/L: datasets where STCT+ is more/less accurate. "
            f"p: two-sided Wilcoxon signed-rank test against STCT+, Holm-corrected over {len(others)} comparisons. "
            "Features: per series. Time: fit plus predict on one thread, total hours; (h): published, other hardware; "
            "‡: timed in a later pass in which STCT+ took "
            f"{list(later_c.values())[0][1]:.0f} s; §: TabPFN on the CPU with 4 threads plus STCT feature extraction; "
            "≈: estimated from the method's own paper; GPU: run on a GPU in its paper, not timed here. "
            f"†: {len(common)} datasets.\n\n")
    f.write(markdown(["Method", "Accuracy", "Rank", "W/L", "p", "Features", "Time (h)"], rows))
    f.write(f"\nGroups of consecutive methods with no significant pairwise difference "
            f"(Wilcoxon, Holm over all {npairs_c} pairs, alpha 0.05):\n\n" + group_lines(runs_c, order_c))
    f.write("\n## STCT+ over random seeds\n\nAgainst the 12 benchmark methods (13 with STCT+).\n\n")
    f.write(markdown(["Seed", "Accuracy", "Place", "Significantly more accurate (Holm)"], seed_rows))

# --------------------------------------------------------------------------
# regression
# --------------------------------------------------------------------------

ours_r = pd.read_csv(os.path.join(R, "stct_plus_regression.csv"))
seeds_r = ours_r.pivot(index="dataset", columns="seed", values="rmse")
DR = pd.read_csv(os.path.join(R, "published_regression_rmse.csv"), index_col=0)[REG_REFS]
DR = DR.join(seeds_r[42].rename("STCT+"), how="inner").dropna()
time_r = pd.read_csv(os.path.join(R, "timing_regression.csv"))
time_r = time_r[time_r.dataset.isin(DR.index)]
tr_wide = time_r.pivot(index="dataset", columns="method", values="seconds")
later_r = later_pass([os.path.join(R, "timing_regression_rocket.csv"),
                      os.path.join(R, "timing_regression_later.csv")], DR.index)
AR = DR.copy()
for label, f in RECENT_R.items():
    AR[label] = default_split(f).reindex(DR.index)
tp_r = tabpfn(os.path.join(R, "stct_tabpfn_regression.csv"), "rmse", DR.index)
AR[TP] = tp_r[0]
AR = AR.dropna()
rel = AR.div(AR.min(axis=1), axis=0).median()
rank_r = AR.rank(axis=1).mean()
others = [m for m in AR.columns if m != "STCT+"]
pl = dict(zip(others, holm([wilcoxon(np.log(AR["STCT+"]), np.log(AR[m])) for m in others])))
pr = dict(zip(others, holm([wilcoxon(AR["STCT+"], AR[m]) for m in others])))
order_r = sorted(AR.columns, key=lambda m: rel[m])
runs_r, npairs_r = groups(np.log(AR), order_r)
rows = []
for m in order_r:
    if m == TP:
        feats, hours = tp_r[1:]
    elif m in tr_wide.columns:
        n = time_r[time_r.method == m].n_features
        feats = "raw" if m in RAW_R else "-" if m == "1NN-DTW" else fmt_int(n.iloc[0]) if n.nunique() == 1 else fmt_int(n.mean())
        hours = f"{tr_wide[m].sum() / 3600:.2f}"
    elif m in later_r:
        g, st = later_r[m]
        feats = "raw" if m in RAW_R else fmt_int(g.n_features.mean())
        hours = f"{g.seconds.sum() / 3600:.2f} ‡ ({st:.0f} s)"
    else:
        ph = published_hours("regression", m, DR.index)
        feats, hours = FEATS_R[m], "GPU" if m in GPU else "-" if np.isnan(ph) else f"({ph:.1f})"
    name = f"**{m}**" if m == "STCT+" else m
    wl = "" if m == "STCT+" else f"{int((AR['STCT+'] < AR[m]).sum())}/{int((AR['STCT+'] > AR[m]).sum())}"
    rows.append([name, f"{rel[m]:.3f}", f"{rank_r[m]:.2f}", wl, "" if m == "STCT+" else fmt_p(pl[m]),
                 "" if m == "STCT+" else fmt_p(pr[m]), feats, hours])

seed_rows = []
for seed in sorted(seeds_r.columns, key=lambda s: (s != 42, s)):
    r = seeds_r[seed].reindex(DR.index)
    allr = DR[REG_REFS].join(r.rename("x"))
    rr = allr.div(allr.min(axis=1), axis=0).median()
    place = int((rr < rr["x"]).sum()) + 1
    seed_rows.append([str(seed), f"{rr['x']:.3f}", f"{place} of 14"])

with open(os.path.join(R, "table_regression.md"), "w", encoding="utf-8") as f:
    f.write(f"# Regression on {len(AR)} TSER datasets\n\n"
            f"Default train/test splits, sorted by median RMSE relative to the best of the {len(AR.columns)} methods "
            "on each dataset. Rank: mean rank. W/L: datasets where STCT+ has lower/higher RMSE. "
            f"p log, p raw: two-sided Wilcoxon signed-rank tests on log and raw RMSE, Holm-corrected over {len(others)}. "
            "Features: per series (raw: the series itself). Time: fit plus predict on one thread, total hours; "
            "(h): published, other hardware; ‡: timed in a later pass, with the STCT+ total of that pass in brackets; "
            "§: TabPFN on the CPU with 4 threads plus STCT feature extraction; GPU: run on a GPU in its paper.\n\n")
    f.write(markdown(["Method", "RMSE/best", "Rank", "W/L", "p log", "p raw", "Features", "Time (h)"], rows))
    f.write(f"\nGroups of consecutive methods with no significant pairwise difference "
            f"(Wilcoxon on log RMSE, Holm over all {npairs_r} pairs, alpha 0.05):\n\n" + group_lines(runs_r, order_r))
    f.write("\n## STCT+ over random seeds\n\nAgainst the 13 benchmark methods (14 with STCT+).\n\n")
    f.write(markdown(["Seed", "RMSE/best", "Place"], seed_rows))

print("wrote", os.path.join(R, "table_classification.md"), "and", os.path.join(R, "table_regression.md"))
