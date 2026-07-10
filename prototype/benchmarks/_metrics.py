from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def iou_box(a, b) -> float:
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass(frozen=True)
class PairMatch:
    pred_id: str
    gt_id: str
    iou: float
    pred_label: str
    gt_label: str


def match_pairs(pred_anns, gt_anns, iou_threshold: float = 0.5) -> list[PairMatch]:
    # Greedy by IoU descending; cross-class matching allowed so a wrong label
    # on an otherwise-correct detection is visible in the confusion matrix.
    scored = []
    for p in pred_anns:
        for g in gt_anns:
            iou = iou_box(p.bbox, g.bbox)
            if iou >= iou_threshold:
                scored.append((iou, p, g))
    scored.sort(key=lambda x: -x[0])
    used_pred: set[str] = set()
    used_gt: set[str] = set()
    matches = []
    for iou, p, g in scored:
        if p.ann_id in used_pred or g.ann_id in used_gt:
            continue
        used_pred.add(p.ann_id)
        used_gt.add(g.ann_id)
        matches.append(PairMatch(p.ann_id, g.ann_id, iou, p.cat_name, g.cat_name))
    return matches


def average_precision(
    pred_anns, gt_anns, class_name: str, iou_threshold: float = 0.5
) -> float | None:
    preds = [
        (a.bbox, getattr(a, "confidence", 1.0))
        for a in pred_anns
        if a.cat_name == class_name
    ]
    gts = [a.bbox for a in gt_anns if a.cat_name == class_name]
    if not gts:
        return None
    if not preds:
        return 0.0
    preds.sort(key=lambda x: -x[1])
    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))
    used_gt: set[int] = set()
    for i, (p_bbox, _) in enumerate(preds):
        best_j, best_iou = -1, iou_threshold
        for j, g_bbox in enumerate(gts):
            if j in used_gt:
                continue
            iou = iou_box(p_bbox, g_bbox)
            if iou >= best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0:
            tp[i] = 1
            used_gt.add(best_j)
        else:
            fp[i] = 1
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / len(gts)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    rec = np.concatenate(([0.0], recall, [1.0]))
    pre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(pre) - 2, -1, -1):
        pre[i] = max(pre[i], pre[i + 1])
    indices = np.where(rec[1:] != rec[:-1])[0]
    return float(np.sum((rec[indices + 1] - rec[indices]) * pre[indices + 1]))


def mean_average_precision(
    pred_anns, gt_anns, classes, iou_threshold: float = 0.5
) -> tuple[float, dict]:
    per_class: dict[str, float] = {}
    for c in classes:
        ap = average_precision(pred_anns, gt_anns, c, iou_threshold)
        if ap is not None:
            per_class[c] = ap
    mean = float(np.mean(list(per_class.values()))) if per_class else 0.0
    return mean, per_class


def confusion_matrix(matches, classes) -> tuple[np.ndarray, float]:
    idx = {c: i for i, c in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    correct = 0
    for m in matches:
        if m.gt_label in idx and m.pred_label in idx:
            matrix[idx[m.gt_label], idx[m.pred_label]] += 1
            if m.gt_label == m.pred_label:
                correct += 1
    accuracy = correct / len(matches) if matches else 0.0
    return matrix, accuracy


def edge_prf1(pred_edges, gt_edges, matches) -> tuple[float, float, float]:
    pred_to_gt = {m.pred_id: m.gt_id for m in matches}
    translated: set[tuple[str, str, str]] = set()
    fp_unmatched = 0
    for e in pred_edges:
        if e.src_id in pred_to_gt and e.dst_id in pred_to_gt:
            translated.add((e.relation, pred_to_gt[e.src_id], pred_to_gt[e.dst_id]))
        else:
            # predicted edge touching an unmatched primitive: hallucinated
            fp_unmatched += 1
    gt_set = {(e.relation, e.src_id, e.dst_id) for e in gt_edges}
    tp = len(translated & gt_set)
    fp = len(translated - gt_set) + fp_unmatched
    fn = len(gt_set - translated)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def symbol_error_rate(hyp, ref) -> float:
    # Levenshtein distance normalized by the (non-empty) reference length,
    # PRIMUS convention (Calvo-Zaragoza et al.).
    if not ref:
        return 0.0 if not hyp else 1.0
    n = len(ref)
    prev = list(range(n + 1))
    for i in range(1, len(hyp) + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if hyp[i - 1] == ref[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n] / max(1, n)
