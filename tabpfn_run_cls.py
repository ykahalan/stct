"""TabPFN on the STCT+ classifier's features (run in an environment with tabpfn installed).

    python tabpfn_run_cls.py --feats E:/stct-tmp/stct_feats_cls --out results/stct_tabpfn_classification.csv --device cpu

Reads the files written by tabpfn_export.py --task classification.  One variant, fixed before any test
score was seen, set up as TS2TabPFN classifies (its code, utils_train.py):
  STCT-TabPFN  TabPFN 2.5 classifier at its version defaults (seed 42, ignore_pretraining_limits) on the
               STCT views, wrapped in ManyClassClassifier(alphabet_size=10) from tabpfn-extensions 0.2.2,
               which passes datasets with at most 10 classes straight to TabPFN and splits larger
               label sets into a codebook of 10-symbol subproblems (seeded here, unseeded in TS2TabPFN).
tabpfn_extensions/__init__ starts telemetry, so the many_class module is loaded on its own and the
telemetry hook it imports is replaced by a no-op (as is its sklearn shim, by sklearn's own validate_data).  Datasets run
cheapest first by the estimate in cost(); each is scored once on its test split.  Appends to --out and skips finished datasets; a dataset that fails (for example out of
memory) is reported and left for the next run.  --chunk n predicts the test split n series at a time, which
lowers peak memory; TabPFN's test series attend only to the training set, so the predictions do not change
(checked by --check-chunk on a finished dataset).
"""
import argparse
import csv
import glob
import importlib.util
import os
import sys
import time
import types

import numpy as np
import torch
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion


def load_many_class():
    telemetry = types.ModuleType("tabpfn_common_utils.telemetry")
    telemetry.set_extension = lambda name: (lambda cls: cls)
    sys.modules.setdefault("tabpfn_common_utils", types.ModuleType("tabpfn_common_utils"))
    sys.modules["tabpfn_common_utils.telemetry"] = telemetry
    spec = importlib.util.find_spec("tabpfn_extensions")
    root = os.path.dirname(spec.origin)
    pkg = types.ModuleType("tabpfn_extensions")
    pkg.__path__ = [root]
    sys.modules["tabpfn_extensions"] = pkg  # the package without its __init__
    # its sklearn shim imports a private name that sklearn 1.8 dropped; sklearn >= 1.6 exports validate_data
    from sklearn.utils.validation import validate_data
    compat = types.ModuleType("tabpfn_extensions.misc.sklearn_compat")
    compat.validate_data = validate_data
    sys.modules["tabpfn_extensions.misc"] = types.ModuleType("tabpfn_extensions.misc")
    sys.modules["tabpfn_extensions.misc.sklearn_compat"] = compat
    from tabpfn_extensions.many_class import ManyClassClassifier
    return ManyClassClassifier


def cost(path):
    """Rough relative TabPFN time: per subproblem, feature attention ~ N G^2 and item attention ~ 2 N G n_train
    (the ratio fitted on the first 72 datasets), times 8 subproblems above 10 classes."""
    d = np.load(path, allow_pickle=True)
    (n_train, g), n = d["Vtr"].shape, len(d["ytr"]) + len(d["yte"])
    sub = 1 if len(np.unique(d["ytr"])) <= 10 else 8
    return sub * n * g * (g + 2 * n_train)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu", choices=("cuda", "cpu"))
    ap.add_argument("--chunk", type=int, default=0, help="predict this many test series at a time (0: all at once)")
    ap.add_argument("--only", help="run just this dataset")
    ap.add_argument("--check-chunk", help="fit this dataset once, compare whole and chunked predictions, write nothing")
    a = ap.parse_args()
    ManyClassClassifier = load_many_class()

    def fit(d):
        base = TabPFNClassifier.create_default_for_version(ModelVersion.V2_5, random_state=42, device=a.device,
                                                           ignore_pretraining_limits=True)
        return ManyClassClassifier(estimator=base, alphabet_size=10, random_state=42).fit(d["Vtr"], d["ytr"])

    def predict(model, X):
        if not a.chunk:
            return model.predict(X)
        return np.concatenate([model.predict(X[i:i + a.chunk]) for i in range(0, len(X), a.chunk)])

    if a.check_chunk:
        d = np.load(os.path.join(a.feats, a.check_chunk + ".npz"), allow_pickle=True)
        model = fit(d)
        whole, chunked = model.predict(d["Vte"]), predict(model, d["Vte"])
        print(f"{a.check_chunk} chunk={a.chunk}: {int((whole != chunked).sum())} of {len(whole)} predictions differ; "
              f"acc whole {(whole == d['yte']).mean():.4f} chunked {(chunked == d['yte']).mean():.4f}", flush=True)
        return
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            done = {r["dataset"] for r in csv.DictReader(f)}
    todo = [p for p in glob.glob(os.path.join(a.feats, "*.npz")) if os.path.splitext(os.path.basename(p))[0] not in done
            and a.only in (None, os.path.splitext(os.path.basename(p))[0])]
    todo.sort(key=cost)
    print("order:", ", ".join(os.path.splitext(os.path.basename(p))[0] for p in todo), flush=True)
    for path in todo:
        name = os.path.splitext(os.path.basename(path))[0]
        d = np.load(path, allow_pickle=True)
        try:
            t = time.perf_counter()
            model = fit(d)
            pred = predict(model, d["Vte"])
            if a.device == "cuda":
                torch.cuda.synchronize()
            model_s = time.perf_counter() - t
        except Exception as e:
            print(f"[failed] {name}: {type(e).__name__} {str(e)[:160]}", flush=True)
            continue
        row = dict(dataset=name, method="STCT-TabPFN", accuracy=float((pred == d["yte"]).mean()), device=a.device,
                   threads=torch.get_num_threads(), model_seconds=model_s, cpu_feature_seconds=float(d["feat_s"]),
                   n_views=d["Vtr"].shape[1], n_kernel=int(d["n_kernel"]), n_train=len(d["ytr"]),
                   n_test=len(d["yte"]), n_classes=len(np.unique(d["ytr"])),
                   n_subproblems=1 if model.no_mapping_needed_ else model.code_book_.shape[0])
        new = not os.path.exists(a.out)
        with open(a.out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"{name} views={row['n_views']} classes={row['n_classes']} sub={row['n_subproblems']} {a.device} "
              f"{model_s:.1f}s acc {row['accuracy']:.4f}", flush=True)


if __name__ == "__main__":
    main()
