"""Whole-dataset metrics, with explicit undefined values and grouped uncertainty."""
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)


def classification(y, scores, threshold=0.5):
    y, scores = np.asarray(y, dtype=int), np.asarray(scores, dtype=float)
    if len(y) == 0 or y.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("Need nonempty aligned finite labels and scores")
    if not set(y.tolist()) <= {0, 1}:
        raise ValueError("Labels must be binary")
    pred = (scores >= threshold).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {"n": len(y), "positives": int(y.sum()), "threshold": float(threshold),
            "accuracy": accuracy_score(y, pred), "precision": p, "recall": r, "f1": f,
            "average_precision": average_precision_score(y, scores) if len(set(y)) == 2 else None,
            "roc_auc": roc_auc_score(y, scores) if len(set(y)) == 2 else None,
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist()}


def choose_threshold(y, scores):
    if set(np.asarray(y, dtype=int)) != {0, 1}:
        raise ValueError("Validation needs both classes to select a threshold")
    thresholds = np.r_[np.unique(scores), np.nextafter(max(scores), np.inf)]
    # Tie-break toward the higher threshold, favouring fewer alerts.
    return max(thresholds, key=lambda t: (classification(y, scores, t)["f1"], t))


def group_bootstrap(y, scores, groups, threshold, repeats=1000, seed=42):
    if repeats < 1:
        raise ValueError("bootstrap repeats must be positive")
    y, scores, groups = np.asarray(y), np.asarray(scores), np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) < 2:
        return {"groups": len(unique), "f1_ci95": None}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        selected = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == g) for g in selected])
        values.append(classification(y[indices], scores[indices], threshold)["f1"])
    return {"groups": len(unique), "f1_ci95": np.quantile(values, [.025, .975]).tolist()}


def confusion_seg(target, pred):
    target, pred = np.asarray(target), np.asarray(pred)
    if target.shape != pred.shape:
        raise ValueError("Segmentation shapes differ")
    valid = target != 255
    if not set(np.unique(target[valid])) <= {0, 1} or not set(np.unique(pred)) <= {0, 1}:
        raise ValueError("Segmentation masks must contain 0, 1 (target may contain 255)")
    return np.bincount(2 * target[valid].astype(int) + pred[valid].astype(int),
                       minlength=4).reshape(2, 2)


def segmentation(hist):
    hist = np.asarray(hist, dtype=float)
    tp = np.diag(hist)
    union = hist.sum(0) + hist.sum(1) - tp
    iou = np.divide(tp, union, out=np.full(2, np.nan), where=union > 0)
    denom = hist[1].sum() + hist[:, 1].sum()
    return {"confusion_matrix": hist.astype(int).tolist(),
            "river_iou": float(iou[1]), "mean_iou": float(np.nanmean(iou)),
            "dice": float(2 * tp[1] / denom) if denom else None,
            "pixel_accuracy": float(tp.sum() / hist.sum()) if hist.sum() else None}


def boundary_counts(target, pred, tolerance=2):
    """Exclude void-adjacent pixels from both predicted and target boundaries."""
    if tolerance < 0:
        raise ValueError("boundary tolerance must be nonnegative")
    valid = np.asarray(target) != 255
    safe = binary_erosion(valid, iterations=max(1, tolerance + 1), border_value=0)
    a, b = np.asarray(target) == 1, np.asarray(pred) == 1
    ea = (a ^ binary_erosion(a)) & safe
    eb = (b ^ binary_erosion(b)) & safe
    da = binary_dilation(ea, iterations=tolerance) if tolerance else ea
    db = binary_dilation(eb, iterations=tolerance) if tolerance else eb
    return np.array([(ea & db).sum(), ea.sum(), (eb & da).sum(), eb.sum()], dtype=np.int64)


def boundary_f1(counts):
    hit_true, total_true, hit_pred, total_pred = counts
    if total_true == 0 and total_pred == 0:
        return None
    recall = hit_true / total_true if total_true else 0
    precision = hit_pred / total_pred if total_pred else 0
    return float(2 * recall * precision / (recall + precision)) if recall + precision else 0.0
