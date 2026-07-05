# SST-IDS: A Self-Supervised Transformer Framework for Explainable Intrusion Detection

Reference implementation of SST-IDS, a two-stage intrusion-detection framework
for tabular network-flow data:

1. **Self-supervised pretraining** — a feature-token Transformer encoder learns
   representations from unlabeled flows by reconstructing randomly masked input
   features (masked-feature modelling, MSE objective).
2. **Supervised fine-tuning** — the pretrained encoder is fine-tuned for binary
   benign/malicious classification (sigmoid head, binary cross-entropy).

Each input feature is treated as a token with a learned feature-index embedding
(the inputs are tabular, not sequential). The repository also provides the
baselines, cross-dataset transfer, SHAP-based explainability, statistical
significance testing, and latency/memory benchmarking used in the paper.

## Installation

```bash
git clone https://github.com/aalbata/sst-ids.git
cd sst-ids
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended for the full-budget schedule; the reduced
reproduction budget runs on CPU (the reported results were obtained on CPU).

## Datasets

The experiments use two public, heterogeneous flow datasets. Download them from
their official providers and place them in the folder pointed to by `DATA_DIR`
in `config.py` (default `./Datasets`) with these exact filenames:

- **CIC-IDS-2017** (MachineLearningCVE flow CSVs, Canadian Institute for
  Cybersecurity):
  - `Monday-WorkingHours.pcap_ISCX.csv`
  - `Tuesday-WorkingHours.pcap_ISCX.csv`
  - `Wednesday-workingHours.pcap_ISCX.csv`
  - `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`
  - `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`
  - `Friday-WorkingHours-Morning.pcap_ISCX.csv`
  - `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`
  - `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
- **UNSW-NB15**:
  - `UNSW_NB15_training-set.csv`
  - `UNSW_NB15_testing-set.csv`

If any file is missing, `reproduce.py` and `run_all.py` stop immediately and
print the full list of expected paths. The loaders match the native column
layout of each dataset (CIC-IDS-2017 flow features; UNSW-NB15 with
`proto`/`service`/`state` categoricals). Labels are binarized to benign (0) /
malicious (1).

## Usage

```bash
python config.py            # print the hyperparameter configuration
python reproduce.py         # regenerate the reported tables/figures (paper settings, CPU)
python run_all.py --quick   # quick check on a small sample
python run_all.py --all     # full-budget experiment suite (recommended, GPU)
python run_all.py --datasets unsw cic --seeds 5
```

Hyperparameters live in `config.py` (`ModelConfig`, `RunConfig`); edit them
there. `reproduce.py` fixes the reduced training budget reported in the paper
(batch size 4096, 12 + 12 epochs, 20,000-row stratified samples, 3 seeds for
UNSW-NB15 and 1 for CIC-IDS-2017) for exact, fully reproducible results, and
defaults to CPU because the reported numbers were obtained on CPU (pass
`--device cuda` to use a GPU). The default configuration in `config.py` uses a
smaller batch and more epochs and is recommended for best results on a GPU.

## What the scripts regenerate

`reproduce.py` writes CSV/LaTeX tables and SHAP figures to `./outputs`, mapping
to the paper as follows:

| Output | Paper artifact |
|---|---|
| `table5_hyperparameters.*` | Hyperparameter settings table |
| `table6_within_dataset.*` | Within-dataset SST-IDS performance |
| `table8_baseline_comparison.*` | SST-IDS vs. RF/XGBoost/CNN/LSTM/TabTransformer |
| `table9_significance.*` | Paired t-tests (UNSW-NB15, 3 seeds) |
| `table7_transfer.*` | Zero-shot CIC-IDS-2017 → UNSW-NB15 transfer (all five metrics) |
| `table10_shap_global.*`, `cic_shap_global_bar.png` | Global SHAP table/figure |
| `cic_shap_local_{TP,FP,FN}.png` | Local SHAP waterfall figures |
| `deployment_cost.*` | Per-flow latency and model size |
| `selected_features_<dataset>.csv` | Selected feature lists |

All table and figure numbers are computed from the data on disk; none are
stored in the repository. The architecture diagram in the paper (Fig. 1) is a
drawn illustration and is not produced by code. Exact numeric values depend on
hardware-level nondeterminism in PyTorch; results should match the paper to
within the reported seed-to-seed variation.

## Methodological notes

- **Feature selection is RF-SHAP, strictly.** In the reproduction path
  (`reproduce.py`), feature ranking uses SHAP values from a Random Forest and
  the script stops with an error if SHAP is unavailable or fails, so the method
  used always matches the paper. Exploratory runs via `run_all.py` may fall
  back to Random Forest impurity importance with a loud warning; the
  `selected_features_<dataset>.csv` output records which method was used.
- **Transfer uses the official UNSW-NB15 testing split.** The zero-shot
  transfer experiment trains on CIC-IDS-2017 only and evaluates on
  `UNSW_NB15_testing-set.csv`. No UNSW-NB15 data is used for training, so the
  transfer is strictly zero-shot; the official testing split keeps the transfer
  evaluation set disjoint from the UNSW-NB15 training subset used in the
  within-dataset experiments.
- **Latency is measured identically for every model, two ways.** The benchmark
  reports (i) amortized per-flow latency of one predict call over a 1000-flow
  batch, the throughput-oriented figure, and (ii) strict single-flow latency
  (one predict call per flow), where per-call framework overhead dominates for
  the scikit-learn/XGBoost Python APIs. Both columns appear in
  `deployment_cost.*` so the comparison is like-for-like under either
  definition.
- Metrics are computed only where they are well defined. For an evaluation set
  containing a single class (e.g., the benign-only CIC-IDS-2017 Monday and
  Friday-morning captures), precision/recall/F1/AUC are reported as `NaN`, and
  the pooled setup (`load_cic_pooled`) is recommended for within-dataset
  evaluation.
- Feature selection and preprocessing are fit on the training split only and
  reused for validation/test/transfer to prevent leakage.

## Repository layout

| File | Purpose |
|---|---|
| `config.py` | Hyperparameters, dataset paths, run settings |
| `data.py` | Loading, cleaning, preprocessing, RF-SHAP feature selection |
| `models.py` | SST-IDS Transformer + baselines (RF, XGBoost, CNN, LSTM, TabTransformer) |
| `pipeline.py` | Pretraining, fine-tuning, metrics, transfer, SHAP, latency, statistics |
| `run_all.py` | Orchestrates the full suite and writes result tables/figures |
| `reproduce.py` | Regenerates the reported tables/figures under the paper's settings |

## Citation

```bibtex
@article{vamsi2026sslvstrees,
  title   = {Self-Supervised Transformers versus Gradient-Boosted Trees for
             Flow-Based Intrusion Detection: A Reproducible Empirical Study},
  author  = {Vamsi, Bandi and Al Bataineh, Elvira and Ahmed, Awder and
             Al Bataineh, Ali},
  note    = {Manuscript under review},
  year    = {2026}
}
```

## License

Released under the MIT License. See [LICENSE](LICENSE).
