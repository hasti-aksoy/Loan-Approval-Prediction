# src/evaluate.py 
from typing import Dict, Optional, Sequence, Union
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

ArrayLike = Union[Sequence, np.ndarray]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    return np.asarray(x)


def _extract_positive_score(y_score: ArrayLike) -> np.ndarray:
    """Return 1D positive-class scores. If 2D (n,2), use last column."""
    arr = _to_numpy(y_score)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, -1]
    return arr.ravel()


def _maybe_map_labels(y_true: ArrayLike, pos_label: Union[int, str] = 1) -> np.ndarray:
    y = _to_numpy(y_true)
    if y.dtype.kind in {"i", "u", "b"}:  # ints/uints/bool
        return y.astype(int)
    # Map non-numeric labels to 0/1
    return np.where(y == pos_label, 1, 0).astype(int)


def _scores_are_prob(y_score: np.ndarray) -> bool:
    if y_score.size == 0:
        return False
    smin, smax = float(np.nanmin(y_score)), float(np.nanmax(y_score))
    return smin >= 0.0 and smax <= 1.0


def _default_threshold(y_score: np.ndarray) -> float:
    """0.5 for probabilities, else 0.0 for decision scores."""
    return 0.5 if _scores_are_prob(y_score) else 0.0


def _auto_threshold_f1(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Pick threshold that maximizes F1 over a simple 0..1 quantile grid."""
    grid = np.unique(np.quantile(y_score, np.linspace(0, 1, 101)))
    best_t, best_f1 = _default_threshold(y_score), -1.0
    for t in grid:
        pred = (y_score >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def compute_metrics(
    y_true: ArrayLike,
    y_score: ArrayLike,
    y_pred: Optional[ArrayLike] = None,
    *,
    threshold: Optional[float] = None,
    auto_threshold: bool = False,
    pos_label: Union[int, str] = 1,
    sample_weight: Optional[ArrayLike] = None,
) -> Dict[str, float]:
    """
    Simple binary classification metrics.

    - y_score: 1D probabilities/decision scores or 2D predict_proba (n,2).
    - If y_pred is None, derive it using:
      provided threshold, else F1-optimized threshold if auto_threshold=True,
      else default (0.5 for probabilities, 0.0 for decision scores).
    """
    y_true_bin = _maybe_map_labels(y_true, pos_label=pos_label)
    y_score_pos = _extract_positive_score(y_score)

    if y_pred is None:
        thr = threshold
        if thr is None:
            thr = _auto_threshold_f1(y_true_bin, y_score_pos) if auto_threshold else _default_threshold(y_score_pos)
        y_pred_arr = (y_score_pos >= float(thr)).astype(int)
        thr_used = float(thr)
    else:
        y_pred_arr = _to_numpy(y_pred).astype(int)
        thr_used = float("nan")

    metrics: Dict[str, float] = {}

    # Ranking metrics
    try:
        metrics["roc_auc"] = roc_auc_score(y_true_bin, y_score_pos, sample_weight=sample_weight)
    except ValueError:
        metrics["roc_auc"] = np.nan
    try:
        metrics["pr_auc"] = average_precision_score(y_true_bin, y_score_pos, sample_weight=sample_weight)
    except ValueError:
        metrics["pr_auc"] = np.nan

    # Threshold metrics
    metrics["accuracy"] = accuracy_score(y_true_bin, y_pred_arr, sample_weight=sample_weight)
    metrics["f1"] = f1_score(y_true_bin, y_pred_arr, zero_division=0, sample_weight=sample_weight)
    metrics["precision"] = precision_score(y_true_bin, y_pred_arr, zero_division=0, sample_weight=sample_weight)
    metrics["recall"] = recall_score(y_true_bin, y_pred_arr, zero_division=0, sample_weight=sample_weight)
    metrics["threshold_used"] = thr_used

    return metrics


def print_report(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    labels: Sequence = (0, 1),
    target_names: Optional[Sequence[str]] = None,
    digits: int = 4,
) -> None:
    y_true_arr = _to_numpy(y_true)
    y_pred_arr = _to_numpy(y_pred)
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
    print("Confusion Matrix (labels 0/1):\n", cm)
    print(
        "\nClassification Report:\n",
        classification_report(
            y_true_arr, y_pred_arr, labels=labels, target_names=target_names, digits=digits
        ),
    )

