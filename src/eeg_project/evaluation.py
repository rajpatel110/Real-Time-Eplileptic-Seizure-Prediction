"""Validated binary window metrics and explicit recording/subject splits."""
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score)


def validate_scores(y, scores):
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if y.ndim != 1 or scores.shape != y.shape or not y.size:
        raise ValueError("Labels and scores must be nonempty matching vectors")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Labels must be binary and scores finite")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("Probability scores must lie in [0, 1]")
    return y, scores


def binary_metrics(y, scores, threshold):
    """Window metrics; average precision is not trapezoidal PR-AUC.

    Average precision is undefined here for a single-class evaluation set.
    Confusion-matrix order is [[TN, FP], [FN, TP]].
    """
    y, scores = validate_scores(y, scores)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must lie in [0, 1]")
    prediction = scores >= threshold
    return {
        "windows": len(y), "positive_fraction": float(y.mean()),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
        "average_precision": float(average_precision_score(y, scores)) if len(np.unique(y)) == 2 else None,
    }


def select_threshold(y, scores, min_precision=0.5, min_recall=0.5):
    """Use development labels only; return None when the target is infeasible."""
    y, scores = validate_scores(y, scores)
    if len(np.unique(y)) != 2:
        raise ValueError("Threshold selection requires both classes")
    if not 0 <= min_precision <= 1 or not 0 <= min_recall <= 1:
        raise ValueError("Precision and recall targets must lie in [0, 1]")
    precision, recall, thresholds = precision_recall_curve(y, scores)
    eligible = np.flatnonzero((precision[:-1] >= min_precision) & (recall[:-1] >= min_recall))
    return float(thresholds[eligible[np.argmax(recall[eligible])]]) if eligible.size else None


def split_masks(recordings, subjects, splits, patient_independent=False):
    """Require exhaustive disjoint recording groups, optionally disjoint patients."""
    recordings, subjects = np.asarray(recordings), np.asarray(subjects)
    if recordings.ndim != 1 or subjects.shape != recordings.shape or not recordings.size:
        raise ValueError("Recording and subject identifiers must be matching nonempty vectors")
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("Define train, validation and test recording groups")
    assigned, masks = set(), {}
    for name, group in splits.items():
        if not group or len(set(group)) != len(group) or assigned.intersection(group):
            raise ValueError("Recording groups must be nonempty, unique and disjoint")
        if not set(group).issubset(set(recordings)):
            raise ValueError(f"Unknown recording in {name} split")
        assigned.update(group)
        masks[name] = np.isin(recordings, group)
    if not np.all(sum(masks.values()) == 1):
        raise ValueError("Every window must belong to exactly one split")
    for recording in set(recordings):
        if len(set(subjects[recordings == recording])) != 1:
            raise ValueError("Recording identifiers must be unique across subjects")
    if patient_independent:
        seen = set()
        for mask in masks.values():
            group = set(subjects[mask])
            if seen.intersection(group):
                raise ValueError("Patient leakage between splits")
            seen.update(group)
    return masks
