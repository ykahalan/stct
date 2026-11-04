"""Score the QUANT-budget recipe with its short-series ridge on the default test splits, one seed.

    python quant2_test.py

quant_sized.QuantSizedClassifier() adds, below length 128, a half-weight ridge on
level-free dilated kernels to the QUANT-sized variant.  From length 128 up it is
unchanged, so those rows are carried over from paper_quant:et_ridge; series
shorter than 128 are scored with the classifier.  Rows are appended to results/full_classification.csv
as paper_quant2:et_ridge.
"""
import time

import numpy as np
import pandas as pd

import bench
from quant_sized import QuantSizedClassifier
from stct_models import MIN_LENGTH

METHOD, OLD = "paper_quant2:et_ridge", "paper_quant:et_ridge"


def main():
    out = "results/full_classification.csv"
    prev = bench.read_results(out, "classification")
    old = prev[prev.method == OLD].drop_duplicates("dataset").set_index("dataset")
    done = set(prev.loc[prev.method == METHOD, "dataset"])
    cols = pd.read_csv(out, nrows=0).columns.tolist()
    for i, name in enumerate(old.index):
        if name in done:
            continue
        if old.loc[name, "length"] >= MIN_LENGTH:
            row = old.loc[name].to_dict() | {"dataset": name, "method": METHOD, "features": "paper_quant2"}
        else:
            Xtr, ytr, Xte, yte = bench.load("classification", name)
            t0 = time.time()
            model = QuantSizedClassifier(random_state=bench.RNG).fit(Xtr, ytr)
            t_fit = time.time() - t0
            t0 = time.time()
            res = bench.score("classification", yte, model.predict(Xte))
            row = {"dataset": name, "n_train": len(Xtr), "n_test": len(Xte), "length": Xtr.shape[1], "method": METHOD,
                   "features": "paper_quant2", "learner": "et_ridge", **res,
                   "n_features": model.n_features(Xtr), "extract_time_s": np.nan,
                   "fit_time_s": t_fit, "predict_time_s": time.time() - t0}
        pd.DataFrame([{c: row.get(c, np.nan) for c in cols}], columns=cols).to_csv(out, mode="a", header=False, index=False)
        print(f"[{i + 1}/{len(old)}] {name} {row['accuracy']:.4f} feats={row['n_features']}", flush=True)


if __name__ == "__main__":
    main()
