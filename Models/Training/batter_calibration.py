"""
Multiclass calibration metrics for the batter pitch-result model.

Brier (multiclass), top-1 ECE with fixed bin count, macro marginal per-class ECE,
log-loss vs train-frequency baseline, and a top-1 reliability diagram.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def multiclass_brier_score(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    labels: np.ndarray | None = None,
) -> float:
    """Mean squared error between one-hot labels and predicted probabilities (sum over classes)."""
    y = np.asarray(y_true).astype(int).ravel()
    p = np.asarray(proba, dtype=float)
    if p.ndim != 2:
        raise ValueError("proba must be 2D (n_samples, n_classes)")
    n, k = p.shape
    if labels is None:
        labels = np.arange(k)
    k = len(labels)
    if p.shape[1] != k:
        raise ValueError("proba columns must match labels count")
    oh = np.zeros((n, k), dtype=float)
    for i, lab in enumerate(labels):
        oh[:, i] = (y == lab).astype(float)
    return float(np.mean(np.sum((p - oh) ** 2, axis=1)))


def ece_top1(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Top-1 expected calibration error: bin by max predicted probability, compare
    in-bin accuracy to mean confidence. Guo et al. style (multiclass).
    """
    y = np.asarray(y_true).astype(int).ravel()
    p = np.asarray(proba, dtype=float)
    conf = np.max(p, axis=1)
    pred = np.argmax(p, axis=1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y)
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        if b == n_bins - 1:
            m = (conf >= lo) & (conf <= hi)
        else:
            m = (conf >= lo) & (conf < hi)
        nb = int(np.sum(m))
        if nb == 0:
            continue
        ece += (nb / n) * abs(float(np.mean(correct[m])) - float(np.mean(conf[m])))
    return float(ece)


def ece_per_class_macro(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_bins: int = 15,
) -> tuple[float, np.ndarray]:
    """
    For each class k, bin predicted p_k in [0,1]; in each bin compare mean p_k to
    empirical frequency of class k. Return macro-averaged ECE and per-class ECE vector.
    """
    y = np.asarray(y_true).astype(int).ravel()
    p = np.asarray(proba, dtype=float)
    n_classes = p.shape[1]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y)
    per_class = np.zeros(n_classes, dtype=float)
    for k in range(n_classes):
        pk = p[:, k]
        yk = (y == k).astype(float)
        ece_k = 0.0
        for b in range(n_bins):
            lo, hi = bins[b], bins[b + 1]
            if b == n_bins - 1:
                m = (pk >= lo) & (pk <= hi)
            else:
                m = (pk >= lo) & (pk < hi)
            nb = int(np.sum(m))
            if nb == 0:
                continue
            ece_k += (nb / n) * abs(float(np.mean(yk[m])) - float(np.mean(pk[m])))
        per_class[k] = ece_k
    return float(np.mean(per_class)), per_class


def train_frequency_proba(y_train_idx: np.ndarray, n_classes: int) -> np.ndarray:
    """Smoothed empirical class frequencies from training labels (same prob every row)."""
    y = np.asarray(y_train_idx).astype(int).ravel()
    counts = np.bincount(y, minlength=n_classes).astype(float)
    alpha = 1e-6
    p = (counts + alpha) / (counts.sum() + alpha * n_classes)
    return p


def log_loss_vs_constant_baseline(
    y_true: np.ndarray,
    proba_model: np.ndarray,
    train_freq: np.ndarray,
    *,
    labels: np.ndarray | None = None,
) -> dict[str, float]:
    """Log-loss of model vs frequency baseline; relative improvement (fraction)."""
    from sklearn.metrics import log_loss

    y = np.asarray(y_true).astype(int).ravel()
    if labels is None:
        labels = np.arange(proba_model.shape[1])
    ll_model = log_loss(y, proba_model, labels=labels)
    p0 = np.asarray(train_freq, dtype=float).ravel()
    p0 = p0 / p0.sum()
    proba_base = np.tile(p0, (len(y), 1))
    ll_base = log_loss(y, proba_base, labels=labels)
    rel = (ll_base - ll_model) / ll_base if ll_base > 0 else 0.0
    return {"log_loss_model": float(ll_model), "log_loss_freq_baseline": float(ll_base), "log_loss_rel_improvement": float(rel)}


def summarize_calibration(
    y_true: np.ndarray,
    proba: np.ndarray,
    y_train_idx: np.ndarray,
    *,
    class_names: list[str] | None = None,
    n_bins: int = 15,
    labels: np.ndarray | None = None,
) -> dict[str, Any]:
    """One dict for printing: Brier, ECE (top-1 + macro marginal), log-loss vs baseline."""
    from sklearn.metrics import log_loss

    y = np.asarray(y_true).astype(int).ravel()
    p = np.asarray(proba, dtype=float)
    if labels is None:
        labels = np.arange(p.shape[1])
    n_classes = len(labels)
    freq = train_frequency_proba(y_train_idx, n_classes)
    ll_d = log_loss_vs_constant_baseline(y, p, freq, labels=labels)
    brier = multiclass_brier_score(y, p, labels=labels)
    brier_base = multiclass_brier_score(y, np.tile(freq, (len(y), 1)), labels=labels)
    ece1 = ece_top1(y, p, n_bins=n_bins)
    ece_macro, ece_per = ece_per_class_macro(y, p, n_bins=n_bins)
    out: dict[str, Any] = {
        "n_samples": int(len(y)),
        "n_bins": int(n_bins),
        "multiclass_brier": brier,
        "multiclass_brier_freq_baseline": brier_base,
        "ece_top1": ece1,
        "ece_macro_marginal": ece_macro,
        "ece_per_class_marginal": ece_per,
        **ll_d,
    }
    if class_names is not None and len(class_names) == len(ece_per):
        out["ece_per_class_by_name"] = {class_names[i]: float(ece_per[i]) for i in range(len(class_names))}
    return out


def plot_top1_reliability(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    n_bins: int = 15,
    ax: Any = None,
    title: str = "Top-1 reliability (test)",
):
    """Draw reliability diagram: mean accuracy vs mean confidence per bin."""
    import matplotlib.pyplot as plt

    y = np.asarray(y_true).astype(int).ravel()
    p = np.asarray(proba, dtype=float)
    conf = np.max(p, axis=1)
    pred = np.argmax(p, axis=1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = []
    accs = []
    confs = []
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        if b == n_bins - 1:
            m = (conf >= lo) & (conf <= hi)
        else:
            m = (conf >= lo) & (conf < hi)
        if not np.any(m):
            continue
        centers.append(0.5 * (lo + hi))
        accs.append(float(np.mean(correct[m])))
        confs.append(float(np.mean(conf[m])))
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect")
    ax.plot(confs, accs, "o-", label="model")
    ax.set_xlabel("Mean confidence (max p) in bin")
    ax.set_ylabel("Accuracy in bin")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax.figure if created else None
