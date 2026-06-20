"""
config.py
=========
Central configuration for SST-IDS hyperparameters, dataset paths, and run
settings. The defaults below define the larger default training schedule. The
paper's Table 5 reports a reduced reproduction budget, which ``reproduce.py``
applies and regenerates Table 5 from for fast, exact reproduction; the values
used for any given run are whatever ``model_cfg``/``run_cfg`` are passed.

The defaults below are starting points for a feature-token Transformer on tabular
network-flow data and are light enough to train on a 4 GB GPU (or CPU). Tune them
for your hardware and data.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths -- point DATA_DIR at the folder that contains the CSV files you shared.
# ----------------------------------------------------------------------------
DATA_DIR = Path("./Datsets")          # folder with the CIC/UNSW CSVs
OUTPUT_DIR = Path("./outputs")        # metrics, figures, saved models land here

# CIC-IDS-2017 day-wise files (MachineLearningCVE / "ISCX" flow CSVs).
CIC_FILES = {
    "Monday-benign":        "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-bruteforce":   "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-dos":        "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-webattack":   "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-infiltration":"Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-morning":       "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-portscan":      "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-ddos":          "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
}
UNSW_TRAIN = "UNSW_NB15_training-set.csv"
UNSW_TEST  = "UNSW_NB15_testing-set.csv"

# Per-file sample sizes (None = use all rows). Mirrors the per-dataset sampling described in the paper
# but is fully explicit and reproducible. Lower these for quick smoke runs.
CIC_SAMPLE_SIZES = {
    "Monday-benign":         290_000,
    "Tuesday-bruteforce":    290_000,
    "Wednesday-dos":         100_000,
    "Thursday-webattack":    100_000,
    "Thursday-infiltration": 288_600,
    "Friday-morning":        100_000,
    "Friday-portscan":       100_000,
    "Friday-ddos":           100_000,
}


@dataclass
class ModelConfig:
    """Transformer backbone + training schedule. -> Table 5."""
    input_representation: str = "feature_tokens"   # each feature is one token
    embed_dim: int            = 64                 # d_model
    n_heads: int              = 4
    n_encoder_blocks: int     = 2
    ff_hidden_dim: int        = 128                # feed-forward hidden dim
    dropout: float            = 0.1
    mask_rate: float          = 0.30               # masked-feature prediction
    optimizer: str            = "adam"
    learning_rate: float      = 1e-3
    batch_size: int           = 256
    pretrain_epochs: int      = 20
    finetune_epochs: int      = 20
    freeze_encoder_in_finetune: bool = False       # fine-tune end-to-end
    classification_activation: str   = "sigmoid"
    ssl_loss: str                    = "mse"       # masked reconstruction
    supervised_loss: str             = "bce"       # binary cross-entropy
    weight_decay: float       = 1e-5
    grad_clip: float          = 1.0


@dataclass
class RunConfig:
    """Experiment-level knobs (splits, seeds, feature counts)."""
    test_size: float   = 0.20
    val_size: float    = 0.10           # carved out of the train split
    n_seeds: int       = 5              # repeated runs -> mean +/- std
    seeds: tuple       = (0, 1, 2, 3, 4)
    threshold: float   = 0.5
    # top-k features kept after RF-SHAP ranking, per dataset.
    top_k_cic: int     = 15
    top_k_unsw: int    = 12
    # categorical encoding: "label" (single numeric token) or "onehot".
    categorical_encoding: str = "label"
    # exclude identifier/port-like columns.
    drop_identity_features: bool = False
    device: str = "cuda"               # falls back to cpu automatically


MODEL = ModelConfig()
RUN = RunConfig()

# Identity / non-behavioral columns to drop when drop_identity_features=True.
# NOTE: the flow CSVs used here have NO source/destination IP columns; CIC keeps
# only "Destination Port" and UNSW has none.
IDENTITY_COLUMNS = {
    "cic":  ["Destination Port"],
    "unsw": [],
}


def hyperparameter_table_rows(model_cfg=None, run_cfg=None, seeds=None, sample=None):
    """Return (name, value) rows describing the hyperparameter configuration.

    Pass the model_cfg / run_cfg actually used for a run (e.g. the reduced
    budget in reproduce.py) so the generated table matches what was trained.
    `seeds` may be an int, a sequence, or a {dataset: [seeds]} mapping; `sample`
    is the per-run stratified sample size (None -> "full")."""
    m = asdict(model_cfg if model_cfg is not None else MODEL)
    r = asdict(run_cfg if run_cfg is not None else RUN)

    if seeds is None:
        seed_str = str(r["n_seeds"])
    elif isinstance(seeds, dict):
        _names = {"unsw": "UNSW-NB15", "cic": "CIC-IDS-2017"}
        seed_str = ", ".join(f"{len(v)} ({_names.get(k, k.upper())})" for k, v in seeds.items())
    else:
        seed_str = str(len(seeds))

    tr = round((1 - r["test_size"]) * (1 - r["val_size"]) * 100)
    va = round((1 - r["test_size"]) * r["val_size"] * 100)
    te = round(r["test_size"] * 100)

    return [
        ("Input representation", m["input_representation"]),
        ("Embedding dimension (d_model)", m["embed_dim"]),
        ("Number of attention heads", m["n_heads"]),
        ("Number of Transformer encoder blocks", m["n_encoder_blocks"]),
        ("Feed-forward hidden dimension", m["ff_hidden_dim"]),
        ("Dropout rate", m["dropout"]),
        ("Masking rate", m["mask_rate"]),
        ("Optimizer", f'{m["optimizer"].capitalize()} (weight decay {m["weight_decay"]})'),
        ("Learning rate", m["learning_rate"]),
        ("Gradient clipping (max norm)", m["grad_clip"]),
        ("Batch size", m["batch_size"]),
        ("Number of pretraining epochs", m["pretrain_epochs"]),
        ("Number of fine-tuning epochs", m["finetune_epochs"]),
        ("Train/validation/test split", f"{tr}%/{va}%/{te}%"),
        ("Repeated runs (random seeds)", seed_str),
        ("Samples per run (stratified)", sample if sample is not None else "full"),
        ("Classification activation", m["classification_activation"]),
        ("Self-supervised loss", m["ssl_loss"]),
        ("Supervised classification loss", m["supervised_loss"]),
    ]


if __name__ == "__main__":
    print("SST-IDS hyperparameters:")
    for name, val in hyperparameter_table_rows():
        print(f"  {name:42s} {val}")
