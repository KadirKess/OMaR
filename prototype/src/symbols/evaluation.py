from dataclasses import dataclass

import numpy as np

from src.models.primitives import PrimitiveKind, PrimitiveSplit


@dataclass(frozen=True)
class MaskScores:
    precision: float
    recall: float
    f1: float
    iou: float


def score_mask(predicted: np.ndarray, truth: np.ndarray) -> MaskScores:
    predicted = np.asarray(predicted, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    if predicted.shape != truth.shape:
        raise ValueError("predicted and truth masks must have the same shape")
    tp = int(np.count_nonzero(predicted & truth))
    fp = int(np.count_nonzero(predicted & ~truth))
    fn = int(np.count_nonzero(~predicted & truth))
    precision = tp / (tp + fp) if tp + fp else float(not truth.any())
    recall = tp / (tp + fn) if tp + fn else float(not predicted.any())
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = tp + fp + fn
    iou = tp / union if union else 1.0
    return MaskScores(precision, recall, f1, iou)


def evaluate_split(
    result: PrimitiveSplit,
    truth_masks: dict[str, np.ndarray],
    foreground_mask: np.ndarray | None = None,
) -> dict[str, MaskScores]:
    """Score supported masks and, when possible, unexplained foreground."""
    mapping = {
        "notehead": PrimitiveKind.NOTEHEAD,
        "stem": PrimitiveKind.STEM,
        "beam": PrimitiveKind.BEAM,
    }
    scores = {
        name: score_mask(result.class_mask(kind), truth_masks[name])
        for name, kind in mapping.items()
    }
    if foreground_mask is not None:
        known_truth = np.logical_or.reduce(tuple(truth_masks[name] for name in mapping))
        unknown_truth = np.asarray(foreground_mask, dtype=bool) & ~known_truth
        scores["unknown"] = score_mask(
            result.class_mask(PrimitiveKind.UNKNOWN), unknown_truth
        )
    return scores
