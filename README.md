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

A CUDA-capable GPU is recommended for training; the code falls back to CPU.

## Datasets

The reported experiments use two public, heterogeneous flow datasets, which
should be downloaded from their official sources and placed in a folder pointed
to by `DATA_DIR` in `config.py` (default `./Datsets`):

- **CIC-IDS-2017** — flow CSVs (`*.pcap_ISCX.csv`), Canadian Institute for
  Cybersecurity.
- **UNSW-NB15** — `UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv`.

The loaders match the native column layout of each dataset (CIC-IDS-2017 flow
features; UNSW-NB15 with `proto`/`service`/`state` categoricals). Labels are
binarized to benign (0) / malicious (1).

## Usage

```bash
python config.py            # print the hyperparameter configuration
python reproduce.py         # regenerate the reported tables/figures (paper settings)
python run_all.py --quick   # quick check on a small sample
python run_all.py --all     # full-budget experiment suite (recommended, GPU)
python run_all.py --datasets unsw cic --seeds 5
```

Hyperparameters live in `config.py` (`ModelConfig`, `RunConfig`); edit them there.
`reproduce.py` fixes the reduced training budget reported in the paper (batch size
4096, 12 + 12 epochs, 20,000-row stratified samples, 3 seeds for UNSW-NB15 and 1
for CIC-IDS-2017) for exact, fully reproducible results. A GPU is recommended;
the reduced budget also runs on CPU, but a full multi-seed reproduction can take
a while there. The default configuration in
`config.py` uses a smaller batch and more epochs and is recommended for best
results on a GPU.

## Repository layout

| File | Purpose |
|---|---|
| `config.py` | Hyperparameters, dataset paths, run settings |
| `data.py` | Loading, cleaning, preprocessing, RF/SHAP feature selection |
| `models.py` | SST-IDS Transformer + baselines (RF, XGBoost, CNN, LSTM, TabTransformer) |
| `pipeline.py` | Pretraining, fine-tuning, metrics, transfer, SHAP, latency, statistics |
| `run_all.py` | Orchestrates the full suite and writes result tables/figures |
| `reproduce.py` | Regenerates the reported tables/figures under the paper's settings |

`run_all.py` (and `reproduce.py`) write CSV/LaTeX tables and SHAP figures to
`./outputs`: hyperparameter settings, within-dataset and baseline comparisons
(mean ± standard deviation over seeds), paired-t-test significance, cross-dataset
transfer, global and local SHAP attributions over the selected behavioral
features, per-dataset feature lists, and per-flow inference latency / model size.

## Notes

- Metrics are computed only where they are well defined. For an evaluation set
  containing a single class (e.g., the benign-only CIC-IDS-2017 Monday and
  Friday-morning captures), precision/recall/F1/AUC are reported as `NaN`, and the
  pooled setup (`load_cic_pooled`) is recommended for within-dataset evaluation.
- Feature selection and preprocessing are fit on the training split only and
  reused for validation/test/transfer to prevent leakage.

## Citation

```bibtex
@article{vamsi2026sslvstrees,
  title   = {Self-Supervised Transformers versus Gradient-Boosted Trees for
             Flow-Based Intrusion Detection: A Reproducible Empirical Study},
  author  = {Vamsi, Bandi and Ahmed, Awder and Al Bataineh, Ali},
  note    = {Manuscript under review},
  year    = {2026}
}
```

## License

Released under the MIT License. See [LICENSE](LICENSE).
