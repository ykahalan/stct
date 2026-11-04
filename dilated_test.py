"""Score classification recipes on the default test splits, one seed each, resumable.

    python dilated_test.py
    python dilated_test.py --methods stct_plus:et_ridge

stct_plus:et_ridge is STCT+, STCTClassifier().  The others are earlier recipes:
paper_dil:et_ridge is STCTClassifier(recipe="dilated"), paper_four:et_ridge is
STCTClassifier(recipe="four"), and paper_quant:et_ridge is the QUANT-sized variant
without its short-series ridge (quant_sized.QuantSizedClassifier).  Rows are
appended to results/full_classification.csv, on the datasets scored for
paper_trio:et_ridge.
"""
import argparse
import time

import numpy as np
import pandas as pd

import bench
from quant_sized import QuantSizedClassifier
from stct_models import STCTClassifier

METHODS = {"stct_plus:et_ridge": lambda: STCTClassifier(random_state=bench.RNG),
           "paper_dil:et_ridge": lambda: STCTClassifier("dilated", random_state=bench.RNG),
           "paper_four:et_ridge": lambda: STCTClassifier("four", random_state=bench.RNG),
           "paper_quant:et_ridge": lambda: QuantSizedClassifier(random_state=bench.RNG, short_ridge=False)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=",".join(METHODS))
    a = ap.parse_args()
    out = "results/full_classification.csv"
    prev = bench.read_results(out, "classification")
    names = prev.loc[prev.method == "paper_trio:et_ridge", "dataset"].drop_duplicates().tolist()
    cols = pd.read_csv(out, nrows=0).columns.tolist()
    for method in a.methods.split(","):
        done = set(prev.loc[prev.method == method, "dataset"])
        for i, name in enumerate(names):
            if name in done:
                continue
            Xtr, ytr, Xte, yte = bench.load("classification", name)
            t0 = time.time()
            model = METHODS[method]().fit(Xtr, ytr)
            t_fit = time.time() - t0
            t0 = time.time()
            res = bench.score("classification", yte, model.predict(Xte))
            row = {"dataset": name, "n_train": len(Xtr), "n_test": len(Xte), "length": Xtr.shape[1], "method": method,
                   "features": method.split(":")[0], "learner": "et_ridge", **res, "n_features": model.n_features(Xtr),
                   "extract_time_s": np.nan, "fit_time_s": t_fit, "predict_time_s": time.time() - t0}
            pd.DataFrame([{c: row.get(c, np.nan) for c in cols}], columns=cols).to_csv(out, mode="a", header=False, index=False)
            print(f"{method} [{i + 1}/{len(names)}] {name} {res['accuracy']:.4f} feats={row['n_features']}", flush=True)


if __name__ == "__main__":
    main()
