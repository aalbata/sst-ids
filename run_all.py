"""
run_all.py
==========
Orchestrates the full SST-IDS experiment suite on the datasets under DATA_DIR
and writes paper-ready artifacts into OUTPUT_DIR:

  table5_hyperparameters.{csv,tex}      <- fills Table 5
  table6_within_dataset.{csv,tex}       <- within-dataset metrics, mean+/-std
  table7_transfer.{csv,tex}             <- CIC->UNSW zero-shot (behavioral feats)
  table8_baseline_comparison.{csv,tex}  <- all models, mean+/-std
  table9_significance.{csv,tex}         <- paired t-tests, sig + non-sig
  table10_shap_global.csv + *.png       <- SHAP on selected behavioral feats
  shap_local_{TP,FP,FN}.png             <- local explanations
  deployment_cost.{csv,tex}             <- per-flow latency + memory
  selected_features_<dataset>.csv       <- per-dataset feature lists

Usage
-----
  python run_all.py --all                 # everything (use a GPU; slow on CPU)
  python run_all.py --quick               # small sample / few epochs, smoke level
  python run_all.py --datasets unsw cic   # choose datasets
  python run_all.py --skip-shap           # skip the (slow) KernelSHAP stage

All numbers are computed from the data on disk; none are hard-coded.
"""
from __future__ import annotations
import argparse, json
from dataclasses import replace, asdict
from pathlib import Path
import numpy as np
import pandas as pd

import config as C
import data
import pipeline as P
from models import build_torch_model, build_sklearn_baseline, count_parameters

MODELS = ["SST-IDS", "Random Forest", "XGBoost", "CNN", "LSTM", "TabTransformer"]


# --------------------------------------------------------------------------- #
# tiny LaTeX helpers (so output drops straight into a LaTeX document)
# --------------------------------------------------------------------------- #
def _fmt(mean, std):
    if np.isnan(mean):
        return "--"
    return f"{mean:.4f} $\\pm$ {std:.4f}" if not np.isnan(std) else f"{mean:.4f}"


def write_table(df, path_stub, caption, label):
    df.to_csv(f"{path_stub}.csv", index=False)
    with open(f"{path_stub}.tex", "w") as fh:
        cols = " & ".join(df.columns)
        fh.write("\\begin{table}[!t]\n\\centering\n\\caption{%s}\n\\label{%s}\n"
                 % (caption, label))
        fh.write("\\begin{tabular}{l%s}\n\\hline\n" % ("c" * (len(df.columns) - 1)))
        fh.write(cols + " \\\\\n\\hline\n")
        for _, r in df.iterrows():
            fh.write(" & ".join(str(x) for x in r.values) + " \\\\\n")
        fh.write("\\hline\n\\end{tabular}\n\\end{table}\n")


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
def stage_hyperparameters(out: Path):
    rows = C.hyperparameter_table_rows()
    df = pd.DataFrame(rows, columns=["Hyperparameter", "Value"])
    write_table(df, str(out / "table5_hyperparameters"),
                "SST-IDS hyperparameter settings.", "tab:hyperparameters")
    (out / "config_dump.json").write_text(json.dumps(
        {"model": asdict(C.MODEL), "run": asdict(C.RUN)}, indent=2, default=str))
    print("[table5] hyperparameters written")


def stage_within_and_baselines(out, datasets, cfg, sample_per_file, seeds):
    rows6, rows8, sig_rows = [], [], []
    for ds in datasets:
        prepare = lambda s, ds=ds: data.prepare_within_dataset(ds, seed=s,
                                                               sample_per_file=sample_per_file)
        # record selected features once from seed 0
        prep0 = prepare(seeds[0])
        prep0["ranking"].to_csv(out / f"selected_features_{ds}.csv", index=False)
        pd.Series(prep0["features"]).to_csv(out / f"selected_{ds}_topk.csv", index=False)
        print(f"[{ds}] features: {', '.join(prep0['features'])}")
        print(f"[{ds}] train +/-: pos={prep0['n_pos_train']} neg={prep0['n_neg_train']} | "
              f"test pos={prep0['n_pos_test']} neg={prep0['n_neg_test']}")

        summary, raw = P.repeated_runs(prepare, MODELS, cfg, seeds=seeds)
        for model in MODELS:
            s = summary[model]
            row = {"Dataset": ds.upper(), "Model": model,
                   "Accuracy": _fmt(*s["accuracy"]), "Precision": _fmt(*s["precision"]),
                   "Recall": _fmt(*s["recall"]), "F1": _fmt(*s["f1"]), "AUC": _fmt(*s["auc"])}
            rows8.append(row)
            if model == "SST-IDS":
                rows6.append(row)
        for r in P.paired_ttests(raw, reference="SST-IDS"):
            sig_rows.append({"Dataset": ds.upper(), "Comparison": r[0], "Metric": r[1],
                             "t": "--" if np.isnan(r[2]) else f"{r[2]:.3f}",
                             "p": "--" if np.isnan(r[3]) else f"{r[3]:.4f}", "Result": r[4]})

    write_table(pd.DataFrame(rows6), str(out / "table6_within_dataset"),
                "Within-dataset performance of SST-IDS (mean $\\pm$ std over seeds).",
                "tab:within_dataset_results")
    write_table(pd.DataFrame(rows8), str(out / "table8_baseline_comparison"),
                "Baseline comparison (mean $\\pm$ std over repeated runs).",
                "tab:baseline_comparison")
    write_table(pd.DataFrame(sig_rows), str(out / "table9_significance"),
                "Paired $t$-tests: SST-IDS vs. each baseline (significant and "
                "non-significant).", "tab:statistical_testing")
    print("[table6/8/9] within-dataset, baselines, significance written")


def stage_transfer(out, cfg, sample_per_file, seeds):
    # The transfer representation is the fixed behavioral alignment map
    # pipeline.CIC_TO_UNSW, which contains no identity/port features by
    # construction, so there is exactly one setting to report. (An earlier
    # revision toggled RUN.drop_identity_features here, but that flag cannot
    # affect the transfer experiment: the aligned columns never include
    # identity features.)
    metrics = ("accuracy", "precision", "recall", "f1", "auc")
    accs = []
    for s in seeds:
        met, fmap = P.transfer_cic_to_unsw(cfg, seed=s,
                                           sample_per_file=sample_per_file or 20000)
        accs.append(met)
    mean = {k: np.nanmean([m[k] for m in accs]) for k in metrics}
    std = {k: np.nanstd([m[k] for m in accs]) for k in metrics}
    row = {"Train": "CIC-IDS-2017", "Test": "UNSW-NB15 (official test split)",
           **{(k.capitalize() if k != "auc" else "AUC"): _fmt(mean[k], std[k])
              for k in metrics}}
    write_table(pd.DataFrame([row]), str(out / "table7_transfer"),
                "Zero-shot cross-dataset transfer (train CIC-IDS-2017, test "
                "UNSW-NB15) over aligned behavioral features only.",
                "tab:cross_dataset_results")
    (out / "transfer_feature_map.json").write_text(json.dumps(P.CIC_TO_UNSW, indent=2))
    print("[table7] zero-shot transfer (behavioral features) written")


def stage_explain_and_cost(out, cfg, sample_per_file, seeds, do_shap):
    prep = data.prepare_within_dataset("cic", seed=seeds[0], sample_per_file=sample_per_file)
    dev = P.get_device(cfg)
    met, model = P.run_sstids_once(prep, cfg, seeds[0], dev)
    print(f"[explain] SST-IDS test metrics for SHAP/cost host: {met}")

    if do_shap:
        imp = P.shap_analysis(model, prep["X_train"], prep["X_test"], prep["features"],
                              dev, str(out / "cic"), y_true=prep["y_test"])
        if imp is not None:
            pd.DataFrame({"feature": prep["features"], "mean_abs_shap": imp}) \
                .sort_values("mean_abs_shap", ascending=False) \
                .to_csv(out / "table10_shap_global.csv", index=False)
            print("[table10] real SHAP global + local TP/FP/FN figures written")

    # deployment cost: SST-IDS vs RF vs XGBoost (identical measurement for all)
    rows = []
    b = P.benchmark_latency_memory(model, prep["X_test"], dev, is_torch=True)
    rows.append({"Model": "SST-IDS",
                 "Per-flow batched (ms)": f"{b['batched_ms']:.4f}",
                 "Per-flow batch-1 (ms)": f"{b['batch1_ms']:.4f}",
                 "Params (MB)": f"{b['param_size_mb']:.3f}"})
    for name in ("Random Forest", "XGBoost"):
        try:
            clf = build_sklearn_baseline(name, random_state=seeds[0])
            clf.fit(prep["X_train"], prep["y_train"])
            bb = P.benchmark_latency_memory(clf, prep["X_test"], dev, is_torch=False)
            rows.append({"Model": name,
                         "Per-flow batched (ms)": f"{bb['batched_ms']:.4f}",
                         "Per-flow batch-1 (ms)": f"{bb['batch1_ms']:.4f}",
                         "Params (MB)": "--"})
        except Exception as e:
            print(f"  ({name} skipped: {e})")
    write_table(pd.DataFrame(rows), str(out / "deployment_cost"),
                "Per-flow inference latency (amortized over a 1000-flow batch, and "
                "strict single-flow calls) and model size.", "tab:deployment_cost")
    print("[deployment] latency + memory written")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--quick", action="store_true", help="small sample + few epochs")
    ap.add_argument("--datasets", nargs="+", default=["unsw", "cic"])
    ap.add_argument("--skip-shap", action="store_true")
    ap.add_argument("--seeds", type=int, default=None, help="override number of seeds")
    args = ap.parse_args()

    data.check_datasets(args.datasets)   # fail fast with the expected file paths

    out = C.OUTPUT_DIR; out.mkdir(parents=True, exist_ok=True)
    cfg = C.MODEL
    sample_per_file = None
    seeds = list(C.RUN.seeds)
    if args.quick:
        cfg = replace(C.MODEL, pretrain_epochs=3, finetune_epochs=3)
        sample_per_file = 20000
        seeds = [0, 1]
    if args.seeds:
        seeds = list(range(args.seeds))

    print(f"Device: {P.get_device(cfg)} | datasets={args.datasets} | seeds={seeds} | "
          f"sample_per_file={sample_per_file}")
    stage_hyperparameters(out)
    stage_within_and_baselines(out, args.datasets, cfg, sample_per_file, seeds)
    if "cic" in args.datasets:
        stage_transfer(out, cfg, sample_per_file, seeds)
        stage_explain_and_cost(out, cfg, sample_per_file, seeds, do_shap=not args.skip_shap)
    print(f"\nDone. Artifacts in {out.resolve()}")


if __name__ == "__main__":
    main()
