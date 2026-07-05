"""
reproduce.py
============
Regenerates the result tables reported in the paper under the exact preliminary
configuration described there (Table 5): feature-token Transformer with batch
size 4096, 12 pretraining + 12 fine-tuning epochs, learning rate 1e-3, on
stratified 20,000-row samples, with 3 random seeds for UNSW-NB15 and 1 for
CIC-IDS-2017. All six models are trained on identical splits and feature sets.

This reduced budget is intended for fast, fully reproducible results; for best
performance, train with the larger schedule in `config.py` on a GPU
(`python run_all.py --all`).

Usage:
    python reproduce.py            # all datasets + transfer + latency + SHAP
    python reproduce.py --datasets unsw
"""
from __future__ import annotations
import argparse, json
from dataclasses import replace
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config as C
import data
import pipeline as P
from models import build_sklearn_baseline, count_parameters
from run_all import write_table, _fmt

MODELS = ["SST-IDS", "Random Forest", "XGBoost", "CNN", "LSTM", "TabTransformer"]
METRICS = ["accuracy", "precision", "recall", "f1", "auc"]
SAMPLE = 20_000
SEEDS = {"unsw": [0, 1, 2], "cic": [0]}
PRELIM = dict(pretrain_epochs=12, finetune_epochs=12, batch_size=4096)


def prepared(dataset, seed):
    X, y = (data.load_unsw("train", random_state=seed) if dataset == "unsw"
            else data.load_cic_pooled(sample_per_file=12000, random_state=seed))
    if SAMPLE < len(X):
        X, _, y, _ = train_test_split(X, y, train_size=SAMPLE, stratify=y, random_state=0)
    cats, top_k = data.cats_and_topk(dataset)
    # strict_shap: the paper states feature selection is RF-SHAP, so the
    # reproduction path refuses to fall back to impurity importance.
    return data.prepare_split(X.reset_index(drop=True), y.reset_index(drop=True),
                              cats, top_k, seed=seed, strict_shap=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["unsw", "cic"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="the reported results were obtained on CPU (default)")
    args = ap.parse_args()

    # Fail fast, with exact expected paths, if the datasets are not in place.
    data.check_datasets(args.datasets)

    C.RUN.device = args.device
    cfg = replace(C.MODEL, **PRELIM)
    dev = P.get_device(cfg)
    out = C.OUTPUT_DIR; out.mkdir(parents=True, exist_ok=True)

    # Table 5
    write_table(pd.DataFrame(C.hyperparameter_table_rows(cfg, C.RUN, SEEDS, SAMPLE),
                             columns=["Hyperparameter", "Value"]),
                str(out / "table5_hyperparameters"),
                "SST-IDS hyperparameter settings.", "tab:hyperparameters")

    rows6, rows8, sig = [], [], []
    for ds in args.datasets:
        seeds = SEEDS[ds]
        raw = {m: {k: [] for k in METRICS} for m in MODELS}
        feats = None
        for s in seeds:
            prep = prepared(ds, s)
            feats = feats or prep["features"]
            for m in MODELS:
                try:
                    met, _ = (P.run_sstids_once(prep, cfg, s, dev) if m == "SST-IDS"
                              else P.run_baseline_once(m, prep, cfg, s, dev))
                except Exception as e:
                    print(f"  {ds}/{m}/seed{s} failed: {e}")
                    met = {k: float("nan") for k in METRICS}
                for k in METRICS:
                    raw[m][k].append(met[k])
        pd.Series(feats).to_csv(out / f"selected_features_{ds}.csv", index=False)
        for m in MODELS:
            r = {"Dataset": ds.upper(), "Model": m,
                 **{k.capitalize(): _fmt(np.nanmean(raw[m][k]), np.nanstd(raw[m][k]))
                    for k in METRICS}}
            rows8.append(r)
            if m == "SST-IDS":
                rows6.append(r)
        if len(seeds) >= 2:
            for c in P.paired_ttests(raw, "SST-IDS"):
                sig.append({"Dataset": ds.upper(), "Comparison": c[0], "Metric": c[1],
                            "t": "--" if np.isnan(c[2]) else f"{c[2]:.3f}",
                            "p": "--" if np.isnan(c[3]) else f"{c[3]:.4f}", "Result": c[4]})

    write_table(pd.DataFrame(rows6), str(out / "table6_within_dataset"),
                "Within-dataset performance of SST-IDS (mean $\\pm$ std).", "tab:within_dataset_results")
    write_table(pd.DataFrame(rows8), str(out / "table8_baseline_comparison"),
                "Baseline comparison (mean $\\pm$ std over seeds).", "tab:baseline_comparison")
    if sig:
        write_table(pd.DataFrame(sig), str(out / "table9_significance"),
                    "Paired t-tests: SST-IDS vs. baselines.", "tab:statistical_testing")

    if "cic" in args.datasets:
        # transfer (behavioral features only) + deployment cost + global SHAP
        met, fmap = P.transfer_cic_to_unsw(cfg, seed=0, sample_per_file=12000)
        write_table(pd.DataFrame([{"Train": "CIC-IDS-2017",
                                   "Test": "UNSW-NB15 (official test split)",
                                   **{k.capitalize() if k != "auc" else "AUC":
                                      f"{met[k]:.4f}" for k in METRICS}}]),
                    str(out / "table7_transfer"),
                    "Zero-shot cross-dataset transfer of SST-IDS "
                    "(behavioral features only).",
                    "tab:cross_dataset_results")
        (out / "transfer_feature_map.json").write_text(json.dumps(fmap, indent=2))
        prep = prepared("cic", 0)
        _, model = P.run_sstids_once(prep, cfg, 0, dev)
        b = P.benchmark_latency_memory(model, prep["X_test"], dev)
        rows = [{"Model": "SST-IDS",
                 "Per-flow batched (ms)": round(b["batched_ms"], 4),
                 "Per-flow batch-1 (ms)": round(b["batch1_ms"], 4),
                 "Params (MB)": round(count_parameters(model) * 4 / 1024 ** 2, 3)}]
        for name in ("Random Forest", "XGBoost"):
            clf = build_sklearn_baseline(name, 0); clf.fit(prep["X_train"], prep["y_train"])
            b = P.benchmark_latency_memory(clf, prep["X_test"], dev, is_torch=False)
            rows.append({"Model": name,
                         "Per-flow batched (ms)": round(b["batched_ms"], 4),
                         "Per-flow batch-1 (ms)": round(b["batch1_ms"], 4),
                         "Params (MB)": "--"})
        write_table(pd.DataFrame(rows), str(out / "deployment_cost"),
                    "Per-flow inference latency (amortized over a 1000-flow batch, "
                    "and strict single-flow calls) and model size.",
                    "tab:deployment_cost")
        imp = P.shap_analysis(model, prep["X_train"], prep["X_test"], prep["features"],
                              dev, str(out / "cic"), y_true=prep["y_test"])
        if imp is None:
            raise RuntimeError(
                "SHAP analysis could not run (is shap installed?). The paper's "
                "global/local SHAP table and figures require it; refusing to "
                "finish a reproduction without them.")
        write_table(pd.DataFrame({"feature": prep["features"], "mean_abs_shap": imp})
                    .sort_values("mean_abs_shap", ascending=False),
                    str(out / "table10_shap_global"),
                    "Global SHAP feature importances over the selected behavioral features.",
                    "tab:shap_results")

    print(f"\nDone. Tables and figures in {out.resolve()}")


if __name__ == "__main__":
    main()
