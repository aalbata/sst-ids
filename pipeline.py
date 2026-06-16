"""
pipeline.py
===========
End-to-end training/evaluation utilities for SST-IDS and baselines:

  * self-supervised pretraining (masked-feature reconstruction, MSE)
  * supervised fine-tuning (BCE) and torch-baseline training
  * metric computation with single-class handling (precision/recall/F1/AUC are
    returned as NaN when only one class is present, e.g. benign-only CIC-IDS-2017
    days such as Monday and Friday-morning)
  * repeated runs over seeds -> mean +/- std
  * cross-dataset transfer with an EXPLICIT feature-alignment map
  * SHAP global importance + local force/waterfall for TP/FP/FN
  * per-flow inference latency + memory footprint
  * paired t-tests across seeds, reporting significant AND non-significant
"""
from __future__ import annotations
import time, warnings, resource
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from scipy import stats as sps

import config as C
from models import (build_torch_model, build_sklearn_baseline, count_parameters)


# --------------------------------------------------------------------------- #
# infra
# --------------------------------------------------------------------------- #
def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_device(cfg=None):
    want = getattr(cfg, "device", None) or C.RUN.device
    return torch.device("cuda" if (want == "cuda" and torch.cuda.is_available())
                        else "cpu")


def _loader(X, y, batch_size, shuffle, device):
    X = torch.as_tensor(X, dtype=torch.float32)
    y = None if y is None else torch.as_tensor(y, dtype=torch.float32)
    ds = torch.utils.data.TensorDataset(X) if y is None else \
        torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# --------------------------------------------------------------------------- #
# SST-IDS training
# --------------------------------------------------------------------------- #
def pretrain_sstids(model, X_train, cfg, device, verbose=False):
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate,
                           weight_decay=cfg.weight_decay)
    loader = _loader(X_train, None, cfg.batch_size, True, device)
    mse = nn.MSELoss()
    for ep in range(cfg.pretrain_epochs):
        tot = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            mask = torch.rand(xb.shape, device=device) < cfg.mask_rate
            empty = ~mask.any(dim=1)                       # guarantee >=1 masked/row
            if empty.any():
                j = torch.randint(0, xb.shape[1], (int(empty.sum()),), device=device)
                mask[empty, j] = True
            recon = model.forward_pretrain(xb, mask)
            loss = mse(recon[mask], xb[mask])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip); opt.step()
            tot += loss.item() * xb.size(0)
        if verbose:
            print(f"  [pretrain] epoch {ep+1}/{cfg.pretrain_epochs}  mse={tot/len(X_train):.4f}")
    return model


def _train_classifier(model, X_tr, y_tr, X_val, y_val, cfg, device, verbose=False):
    model.to(device)
    if cfg.freeze_encoder_in_finetune and hasattr(model, "encoder"):
        for p in model.encoder.parameters():
            p.requires_grad = False
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    pos = float(max(1, (y_tr == 1).sum())); neg = float(max(1, (y_tr == 0).sum()))
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / pos, device=device))
    loader = _loader(X_tr, y_tr, cfg.batch_size, True, device)

    best_auc, best_state = -1.0, None
    for ep in range(cfg.finetune_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = crit(logits, yb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip); opt.step()
        if X_val is not None and len(np.unique(y_val)) > 1:
            p = predict_proba_torch(model, X_val, device)
            auc = roc_auc_score(y_val, p)
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"  [finetune] epoch {ep+1}/{cfg.finetune_epochs}  val_auc={best_auc:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_proba_torch(model, X, device, batch_size=2048):
    model.to(device).eval()
    out = []
    for (xb,) in _loader(X, None, batch_size, False, device):
        out.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
    return np.concatenate(out)


# --------------------------------------------------------------------------- #
# metrics  (single-class safe)
# --------------------------------------------------------------------------- #
def compute_metrics(y_true, y_prob, threshold=None):
    threshold = C.RUN.threshold if threshold is None else threshold
    y_true = np.asarray(y_true); y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    single = len(np.unique(y_true)) < 2
    m = {"accuracy": accuracy_score(y_true, y_pred)}
    if single:
        # precision/recall/F1/AUC are undefined with one class present.
        m.update({k: np.nan for k in ("precision", "recall", "f1", "auc")})
        m["note"] = "single-class test set: P/R/F1/AUC undefined"
    else:
        m["precision"] = precision_score(y_true, y_pred, zero_division=0)
        m["recall"]    = recall_score(y_true, y_pred, zero_division=0)
        m["f1"]        = f1_score(y_true, y_pred, zero_division=0)
        m["auc"]       = roc_auc_score(y_true, y_prob)
        m["note"]      = ""
    return m


# --------------------------------------------------------------------------- #
# one full run (SST-IDS or a baseline) on a prepared split
# --------------------------------------------------------------------------- #
def run_sstids_once(prep, cfg, seed, device):
    set_seed(seed)
    model = build_torch_model("sst-ids", prep["X_train"].shape[1], cfg)
    pretrain_sstids(model, prep["X_train"], cfg, device)         # SSL stage
    _train_classifier(model, prep["X_train"], prep["y_train"],
                      prep["X_val"], prep["y_val"], cfg, device) # fine-tune stage
    prob = predict_proba_torch(model, prep["X_test"], device)
    return compute_metrics(prep["y_test"], prob), model


def run_baseline_once(name, prep, cfg, seed, device):
    set_seed(seed)
    if name.lower() in ("random forest", "rf", "xgboost", "xgb"):
        clf = build_sklearn_baseline(name, random_state=seed)
        clf.fit(prep["X_train"], prep["y_train"])
        prob = clf.predict_proba(prep["X_test"])[:, 1]
        return compute_metrics(prep["y_test"], prob), clf
    model = build_torch_model(name, prep["X_train"].shape[1], cfg)
    _train_classifier(model, prep["X_train"], prep["y_train"],
                      prep["X_val"], prep["y_val"], cfg, device)
    prob = predict_proba_torch(model, prep["X_test"], device)
    return compute_metrics(prep["y_test"], prob), model


# --------------------------------------------------------------------------- #
# repeated runs -> mean +/- std
# --------------------------------------------------------------------------- #
def repeated_runs(prepare_fn, models, cfg, seeds=None, device=None):
    """prepare_fn(seed)->prep. Returns {model: {metric: (mean,std)}} and the raw
    per-seed arrays {model: {metric: [v0,v1,...]}} for paired significance tests."""
    seeds = seeds or list(C.RUN.seeds)
    device = device or get_device(cfg)
    raw = {m: {k: [] for k in ("accuracy", "precision", "recall", "f1", "auc")}
           for m in models}
    for s in seeds:
        prep = prepare_fn(s)
        for m in models:
            try:
                met, _ = (run_sstids_once(prep, cfg, s, device) if m.lower() in ("sst-ids", "sstids")
                          else run_baseline_once(m, prep, cfg, s, device))
            except Exception as e:                       # e.g. xgboost not installed
                warnings.warn(f"model '{m}' failed on seed {s}: {e}; recording NaN.")
                met = {k: float("nan") for k in raw[m]}
            for k in raw[m]:
                raw[m][k].append(met[k])
    summary = {m: {k: (float(np.nanmean(v)), float(np.nanstd(v))) for k, v in d.items()}
               for m, d in raw.items()}
    return summary, raw


# --------------------------------------------------------------------------- #
# cross-dataset transfer with an EXPLICIT feature-alignment map
# --------------------------------------------------------------------------- #
# Behavioral features that are semantically comparable across schemas.
CIC_TO_UNSW = {
    "Flow Duration": "dur",
    "Total Fwd Packets": "spkts",
    "Total Backward Packets": "dpkts",
    "Total Length of Fwd Packets": "sbytes",
    "Total Length of Bwd Packets": "dbytes",
    "Flow Packets/s": "rate",
}


def transfer_cic_to_unsw(cfg, seed=0, device=None, sample_per_file=20000):
    """Train SST-IDS on CIC aligned features, zero-shot test on UNSW (same
    scaler). Returns metrics + the explicit feature map used."""
    import data
    device = device or get_device(cfg)
    cic_cols = list(CIC_TO_UNSW.keys()); unsw_cols = list(CIC_TO_UNSW.values())

    Xc, yc = data.load_cic_pooled(sample_per_file, random_state=seed)
    Xc = Xc[[c for c in cic_cols if c in Xc.columns]]
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(Xc.values)
    Xc_s = scaler.transform(Xc.values).astype(np.float32)

    Xu, yu = data.load_unsw("test", random_state=seed)
    Xu = Xu[[c for c in unsw_cols if c in Xu.columns]]
    Xu_s = scaler.transform(Xu.values).astype(np.float32)   # SOURCE scaler (true transfer)

    set_seed(seed)
    model = build_torch_model("sst-ids", Xc_s.shape[1], cfg)
    pretrain_sstids(model, Xc_s, cfg, device)
    _train_classifier(model, Xc_s, yc.to_numpy(), None, None, cfg, device)
    prob = predict_proba_torch(model, Xu_s, device)
    return compute_metrics(yu.to_numpy(), prob), CIC_TO_UNSW


# --------------------------------------------------------------------------- #
# explainability: global SHAP + local force/waterfall (TP/FP/FN)
# --------------------------------------------------------------------------- #
def shap_analysis(model, X_background, X_explain, feature_names, device,
                  out_prefix, y_true=None):
    """Real SHAP on the trained SST-IDS model over the ACTUAL selected
    (behavioral) features. Saves a global bar plot and local plots for a
    TP/FP/FN when labels are supplied. Falls back gracefully if shap is absent."""
    import os
    try:
        import shap, matplotlib
        matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        warnings.warn(f"shap/matplotlib unavailable ({e}); skipping SHAP plots.")
        return None

    def f(x):                                  # numpy wrapper for Kernel/Gradient SHAP
        return predict_proba_torch(model, x.astype(np.float32), device)

    bg = X_background[np.random.choice(len(X_background),
                                       min(100, len(X_background)), replace=False)]
    explainer = shap.KernelExplainer(f, bg)
    sv = explainer.shap_values(X_explain[:min(200, len(X_explain))], nsamples=100)
    sv = np.asarray(sv)

    plt.figure()
    shap.summary_plot(sv, X_explain[:sv.shape[0]], feature_names=feature_names,
                      plot_type="bar", show=False)
    plt.tight_layout(); plt.savefig(f"{out_prefix}_shap_global_bar.png", dpi=150)
    plt.close()

    if y_true is not None:
        prob = f(X_explain[:sv.shape[0]]); pred = (prob >= C.RUN.threshold).astype(int)
        yt = np.asarray(y_true)[:sv.shape[0]]
        cases = {"TP": np.where((pred == 1) & (yt == 1))[0],
                 "FP": np.where((pred == 1) & (yt == 0))[0],
                 "FN": np.where((pred == 0) & (yt == 1))[0]}
        base = explainer.expected_value
        base = base[0] if np.ndim(base) else base
        for tag, idx in cases.items():
            if len(idx) == 0:
                continue
            i = int(idx[0])
            plt.figure()
            shap.plots._waterfall.waterfall_legacy(
                base, sv[i], feature_names=feature_names, show=False)
            plt.tight_layout(); plt.savefig(f"{out_prefix}_shap_local_{tag}.png", dpi=150)
            plt.close()
    return np.abs(sv).mean(axis=0)


# --------------------------------------------------------------------------- #
# deployment cost: per-flow latency + memory footprint
# --------------------------------------------------------------------------- #
def benchmark_latency_memory(model, X_sample, device, is_torch=True, repeats=5):
    """Return per-flow latency (ms, batch=1), throughput batch latency, and a
    memory proxy (param MB for torch; peak process RSS delta)."""
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # MB (Linux: KB)
    if is_torch:
        model.to(device).eval()
        x1 = torch.as_tensor(X_sample[:1], dtype=torch.float32, device=device)
        with torch.no_grad():                          # warmup
            for _ in range(3):
                model(x1)
        t = []
        with torch.no_grad():
            for _ in range(repeats):
                s = time.perf_counter()
                for i in range(min(1000, len(X_sample))):
                    model(torch.as_tensor(X_sample[i:i+1], dtype=torch.float32, device=device))
                t.append((time.perf_counter() - s) / min(1000, len(X_sample)))
        per_flow_ms = float(np.median(t)) * 1e3
        param_mb = count_parameters(model) * 4 / (1024 ** 2)
    else:
        s = time.perf_counter(); model.predict(X_sample[:1000]); 
        per_flow_ms = (time.perf_counter() - s) / min(1000, len(X_sample)) * 1e3
        param_mb = float("nan")
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return {"per_flow_ms": per_flow_ms, "param_size_mb": param_mb,
            "peak_rss_mb": max(rss0, rss1)}


# --------------------------------------------------------------------------- #
# paired significance testing across seeds (sig + non-sig)
# --------------------------------------------------------------------------- #
def paired_ttests(raw, reference="SST-IDS", alpha=0.05):
    """raw: {model:{metric:[per-seed values]}}. Paired t-test reference vs each
    other model, per metric. Returns a list of rows (sig and non-sig)."""
    rows = []
    ref = raw[reference]
    for model, metrics in raw.items():
        if model == reference:
            continue
        for k in ("accuracy", "precision", "recall", "f1", "auc"):
            a = np.asarray(ref[k], float); b = np.asarray(metrics[k], float)
            ok = ~(np.isnan(a) | np.isnan(b))
            if ok.sum() < 2:
                rows.append((f"{reference} vs {model}", k, np.nan, np.nan, "n/a"))
                continue
            t, p = sps.ttest_rel(a[ok], b[ok])
            rows.append((f"{reference} vs {model}", k, float(t), float(p),
                         "significant" if p <= alpha else "not significant"))
    return rows
