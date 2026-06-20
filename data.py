"""
data.py
=======
Loading, cleaning, preprocessing, and RF-SHAP feature selection for the three
IDS datasets, matched to the EXACT column layout of the CSVs in Datsets.zip:

  * CIC-IDS-2017  : "*.pcap_ISCX.csv" MachineLearningCVE flow files. Columns have
                    leading spaces; the label column is "Label" with values
                    BENIGN / <attack name>. There are NO IP columns and no
                    protocol column; only "Destination Port" is port-like.
  * UNSW-NB15     : *_training-set.csv / *_testing-set.csv with proto/service/
                    state categoricals, attack_cat, and a 0/1 "label".

Everything is reproducible: preprocessing is fit on the training split only and
re-applied to validation/test/transfer to avoid leakage.
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import config as C

try:                       # SHAP is preferred for feature ranking (as in the paper)
    import shap
    _HAVE_SHAP = True
except Exception:
    _HAVE_SHAP = False


# ---------------------------------------------------------------------------
# Low-level CSV readers (one file at a time, memory-bounded via float32 cast)
# ---------------------------------------------------------------------------
def _read_csv_clean(path, label_col_candidates):
    """Read a CSV, strip column names, locate the label column, drop inf/NaN."""
    df = pd.read_csv(path, skipinitialspace=True, low_memory=False)
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    label_col = next((c for c in label_col_candidates if c in df.columns), None)
    if label_col is None:
        raise ValueError(f"No label column among {label_col_candidates} in {path}. "
                         f"Got columns: {list(df.columns)[:8]}...")
    df = df.replace([np.inf, -np.inf], np.nan)
    return df, label_col


def _binarize_cic(series):
    return (series.astype(str).str.upper().str.strip() != "BENIGN").astype(int)


# ---------------------------------------------------------------------------
# CIC-IDS-2017
# ---------------------------------------------------------------------------
def load_cic_day(day_key, sample_size=None, random_state=0):
    """Load one CIC day file -> (X_df, y) with a clean binary label.

    Sampling: stratified when both classes are present, else head/random
    (a benign-only file such as Monday cannot be stratified by class).
    """
    path = C.DATA_DIR / C.CIC_FILES[day_key]
    df, label_col = _read_csv_clean(path, ["Label"])
    y = _binarize_cic(df[label_col])
    X = df.drop(columns=[label_col])

    # keep numeric features only; coerce, then drop rows that became NaN
    X = X.apply(pd.to_numeric, errors="coerce")
    keep = X.notna().all(axis=1)
    X, y = X[keep], y[keep]
    X = X.astype(np.float32)

    if C.RUN.drop_identity_features:
        X = X.drop(columns=[c for c in C.IDENTITY_COLUMNS["cic"] if c in X.columns],
                   errors="ignore")

    if sample_size is not None and sample_size < len(X):
        n_classes = y.nunique()
        if n_classes > 1:
            X, _, y, _ = train_test_split(
                X, y, train_size=sample_size, stratify=y, random_state=random_state)
        else:
            X = X.sample(n=sample_size, random_state=random_state)
            y = y.loc[X.index]
    return X.reset_index(drop=True), y.reset_index(drop=True)


def load_cic_pooled(sample_per_file=None, random_state=0):
    """Pool all CIC day files into one labelled set (benign + every attack).

    This is the RECOMMENDED within-dataset setup: it guarantees both classes
    are present so precision/recall/F1/AUC are well defined -- unlike evaluating
    a benign-only day (Monday / Friday-morning) in isolation.
    """
    sizes = C.CIC_SAMPLE_SIZES if sample_per_file is None else \
            {k: sample_per_file for k in C.CIC_FILES}
    parts_X, parts_y = [], []
    for day in C.CIC_FILES:
        Xi, yi = load_cic_day(day, sizes.get(day), random_state)
        parts_X.append(Xi); parts_y.append(yi)
    # align on the common feature set (all CIC files share the same schema)
    common = sorted(set.intersection(*[set(p.columns) for p in parts_X]))
    X = pd.concat([p[common] for p in parts_X], ignore_index=True)
    y = pd.concat(parts_y, ignore_index=True)
    return X, y


# ---------------------------------------------------------------------------
# UNSW-NB15
# ---------------------------------------------------------------------------
_UNSW_DROP = ["id", "attack_cat"]
_UNSW_CATEGORICAL = ["proto", "service", "state"]


def load_unsw(which="train", random_state=0):
    path = C.DATA_DIR / (C.UNSW_TRAIN if which == "train" else C.UNSW_TEST)
    df, label_col = _read_csv_clean(path, ["label", "Label"])
    y = df[label_col].astype(int)
    X = df.drop(columns=[c for c in _UNSW_DROP + [label_col] if c in df.columns],
                errors="ignore")
    if C.RUN.drop_identity_features:
        X = X.drop(columns=[c for c in C.IDENTITY_COLUMNS["unsw"] if c in X.columns],
                   errors="ignore")
    return X.reset_index(drop=True), y.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Preprocessing: encode categoricals + z-score scale (fit on TRAIN only)
# ---------------------------------------------------------------------------
class Preprocessor:
    """Encodes categoricals and standardizes numerics. Fit on train, reuse."""

    def __init__(self, categorical_cols=None, encoding=None):
        self.categorical_cols = categorical_cols or []
        self.encoding = encoding or C.RUN.categorical_encoding
        self.encoders: dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.onehot_cols_ = None
        self.feature_names_: list[str] = []

    def _encode(self, X, fit):
        X = X.copy()
        cats = [c for c in self.categorical_cols if c in X.columns]
        if self.encoding == "onehot" and cats:
            X = pd.get_dummies(X, columns=cats, dummy_na=False)
            if fit:
                self.onehot_cols_ = X.columns
            else:
                X = X.reindex(columns=self.onehot_cols_, fill_value=0)
        else:                                   # label-encode each categorical
            for c in cats:
                s = X[c].astype(str).fillna("NA")
                if fit:
                    le = LabelEncoder().fit(s)
                    self.encoders[c] = le
                else:
                    le = self.encoders[c]
                    s = s.where(s.isin(le.classes_), le.classes_[0])  # unseen->first
                X[c] = le.transform(s)
        # any remaining non-numeric columns -> coerce
        for c in X.columns:
            if not np.issubdtype(X[c].dtype, np.number):
                X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
        return X.astype(np.float32)

    def fit_transform(self, X):
        X = self._encode(X, fit=True)
        self.feature_names_ = list(X.columns)
        Xs = self.scaler.fit_transform(X.values)
        return Xs.astype(np.float32)

    def transform(self, X):
        X = self._encode(X, fit=False)
        X = X.reindex(columns=self.feature_names_, fill_value=0.0)
        return self.scaler.transform(X.values).astype(np.float32)


# ---------------------------------------------------------------------------
# Feature selection: Random Forest + SHAP ranking (impurity fallback)
# ---------------------------------------------------------------------------
def select_features_rf_shap(X_df, y, top_k, random_state=0, max_shap_rows=1000):
    """Rank features by mean |SHAP| from a Random Forest; fall back to impurity
    importance if SHAP is unavailable. Returns (selected_names, ranking_df).

    NOTE: the RF here is depth-capped purely to keep TreeSHAP tractable; this is
    the feature-ranking model, not the RF *baseline* (which is full-depth)."""
    rf = RandomForestClassifier(
        n_estimators=150, max_depth=16, n_jobs=-1,
        class_weight="balanced", random_state=random_state)
    rf.fit(X_df.values, y.values)

    method = "rf_impurity"
    importances = rf.feature_importances_
    if _HAVE_SHAP and X_df.shape[0] > 0:
        try:
            bg = X_df.sample(min(max_shap_rows, len(X_df)), random_state=random_state)
            explainer = shap.TreeExplainer(rf)
            sv = explainer.shap_values(bg.values, check_additivity=False)
            sv = sv[1] if isinstance(sv, list) else sv          # positive class
            if sv.ndim == 3:                                    # (n, feat, classes)
                sv = sv[:, :, -1]
            importances = np.abs(sv).mean(axis=0)
            method = "rf_shap"
        except Exception as e:
            warnings.warn(f"SHAP ranking failed ({e}); using impurity importance.")

    ranking = (pd.DataFrame({"feature": X_df.columns, "importance": importances})
               .sort_values("importance", ascending=False).reset_index(drop=True))
    ranking["method"] = method
    selected = ranking["feature"].head(top_k).tolist()
    return selected, ranking


# ---------------------------------------------------------------------------
# Convenience: build a fully-prepared within-dataset split
# ---------------------------------------------------------------------------
def cats_and_topk(dataset):
    return {"cic": ([], C.RUN.top_k_cic),
            "unsw": (_UNSW_CATEGORICAL, C.RUN.top_k_unsw)}[dataset]


def prepare_split(X, y, cats, top_k, seed=0):
    """Split -> preprocess (fit on train) -> RF-SHAP feature selection. No leakage.
    Works on any in-memory (X, y), which lets callers cache the raw frame once."""
    strat = y if y.nunique() > 1 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=C.RUN.test_size, stratify=strat, random_state=seed)
    strat_tr = y_tr if y_tr.nunique() > 1 else None
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=C.RUN.val_size, stratify=strat_tr, random_state=seed)

    pp = Preprocessor(cats)
    full_tr = pd.DataFrame(pp.fit_transform(X_tr), columns=pp.feature_names_)
    full_val = pd.DataFrame(pp.transform(X_val), columns=pp.feature_names_)
    full_te = pd.DataFrame(pp.transform(X_te), columns=pp.feature_names_)

    selected, ranking = select_features_rf_shap(
        full_tr, y_tr.reset_index(drop=True), top_k, random_state=seed)
    sel = [c for c in selected if c in full_tr.columns]

    return {
        "X_train": full_tr[sel].values.astype(np.float32),
        "y_train": y_tr.to_numpy(),
        "X_val":   full_val[sel].values.astype(np.float32),
        "y_val":   y_val.to_numpy(),
        "X_test":  full_te[sel].values.astype(np.float32),
        "y_test":  y_te.to_numpy(),
        "features": sel,
        "ranking": ranking,
        "preprocessor": pp,
        "n_pos_train": int(y_tr.sum()), "n_neg_train": int((y_tr == 0).sum()),
        "n_pos_test": int(y_te.sum()),  "n_neg_test": int((y_te == 0).sum()),
    }


def prepare_within_dataset(dataset, seed=0, sample_per_file=None):
    """Returns dict with scaled train/val/test arrays + selected feature names."""
    if dataset == "cic":
        X, y = load_cic_pooled(sample_per_file, random_state=seed)
    elif dataset == "unsw":
        X, y = load_unsw("train", random_state=seed)
    else:
        raise ValueError(dataset)
    cats, top_k = cats_and_topk(dataset)
    return prepare_split(X, y, cats, top_k, seed)
