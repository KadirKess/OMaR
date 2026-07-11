import argparse
import csv
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import color, filters

from benchmarks._metrics import (
    confusion_matrix,
    edge_prf1,
    match_pairs,
    mean_average_precision,
    symbol_error_rate,
)
from src.deepscores.dataset import DeepScoresDataset
from src.deepscores.io import (
    SCOPED_CATEGORY_NAMES,
    build_predicted_page,
    derive_ground_truth,
)
from src.preprocessing.preprocessor import Preprocessor
from src.reconstruction.assembly import (
    compute_staff_positions,
    graph_from_split,
    run_rewrite,
)
from src.reconstruction.semantics import assign_durations, assign_pitches
from src.symbols.evaluation import evaluate_split
from src.symbols.splitter import PrimitiveSplitter


DATA_DIR = Path(__file__).parent.parent / "data" / "ds2_dense"
DEFAULT_OUTPUT = Path(__file__).parent.parent / "output" / "pipeline_eval"
CLASSES = tuple(SCOPED_CATEGORY_NAMES)


def _read_page_ids_file(path):
    """Read a TSV manifest (band, image_id, python_n_crops) preserving order."""
    entries = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            entries.append(
                {
                    "band": row["band"],
                    "image_id": str(row["image_id"]),
                    "python_n_crops": int(row["python_n_crops"]),
                }
            )
    return entries


def _recover_staff_rows(image_path, frac=0.5):
    array = np.asarray(Image.open(image_path))
    if array.ndim == 3:
        array = color.rgb2gray(array)
    threshold = filters.threshold_otsu(array)
    ink = np.where(array < threshold, 0, 255).astype(np.uint8)
    pre = Preprocessor(ink)
    rows = list(pre.staffline_detection_projection(frac=frac))
    return tuple(float(r) for r in rows)


def _system_for_crop(rows, crop_y_center):
    rows = sorted(rows)
    if len(rows) <= 5:
        return tuple(rows)
    # pick the window of 5 consecutive rows whose midpoint is nearest to the crop
    best_window, best_dist = None, float("inf")
    for i in range(len(rows) - 4):
        window = rows[i : i + 5]
        window_center = (window[0] + window[-1]) / 2.0
        d = abs(window_center - crop_y_center)
        if d < best_dist:
            best_dist = d
            best_window = window
    return tuple(best_window)


def _process_crop(splitter, sample, staff_rows, staff_line):
    result = splitter.split(
        sample.staffless_mask,
        sample.staff_space,
        sample.staff_line,
        sample.original_mask,
    )
    crop_y_center = sample.origin[1] + sample.staffless_mask.shape[0] / 2.0
    system_lines = _system_for_crop(staff_rows, crop_y_center)
    graph = graph_from_split(result)
    graph.staff_lines = tuple(y - sample.origin[1] for y in system_lines)
    graph.staff_space = sample.staff_space
    compute_staff_positions(graph)
    run_rewrite(graph)
    assign_durations(graph)
    assign_pitches(graph)
    return result, graph, system_lines


def _make_crop_bbox_filter(samples):
    # Restrict GT to primitives whose bbox center falls inside a processed crop.
    # Without this the GT count dwarfs the predictions (iter_supported_crops
    # only yields a few crops per page), capping recall near zero.
    if not samples:
        return lambda bbox: False
    bounds = []
    for s in samples:
        ox, oy = s.origin
        h, w = s.staffless_mask.shape
        bounds.append((ox, oy, ox + w, oy + h))

    def inside(bbox):
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        return any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in bounds)

    return inside


def evaluate_page(
    dataset,
    splitter,
    image_id,
    samples,
    iou_threshold=0.5,
    band=None,
    crop_diagnostics=None,
):
    image = dataset.images[str(image_id)]
    image_path = DATA_DIR / "images" / image["filename"]
    staff_rows = _recover_staff_rows(image_path) if image_path.exists() else ()
    staff_line = (
        max(1.0, round(0.15 * float(np.median([s.staff_line for s in samples]))))
        if samples
        else 2.0
    )

    crop_results = []
    staff_lines_per_crop = []
    mask_scores_per_kind = defaultdict(list)
    for sample in samples:
        result, graph, system_lines = _process_crop(
            splitter, sample, staff_rows, staff_line
        )
        crop_results.append((result, graph, sample.origin))
        staff_lines_per_crop.append(system_lines)
        if sample.truth_masks:
            for kind, score in evaluate_split(
                result, sample.truth_masks, sample.staffless_mask
            ).items():
                mask_scores_per_kind[kind].append(score)

    predicted = build_predicted_page(
        image, crop_results, staff_lines_per_crop, staff_line=staff_line
    )
    # Ground truth is loaded only after inference and covers the full page.
    truth = derive_ground_truth(dataset, image_id, staff_lines=staff_rows)

    useful_crops = set()
    covered_gt = 0
    duplicate_covered_gt = 0
    for annotation in truth.annotations:
        bbox = annotation.bbox
        cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
        hits = 0
        for index, sample in enumerate(samples):
            ox, oy = sample.origin
            h, w = sample.staffless_mask.shape
            if ox <= cx < ox + w and oy <= cy < oy + h:
                hits += 1
                useful_crops.add(index)
        covered_gt += hits > 0
        duplicate_covered_gt += hits > 1

    matches = match_pairs(
        predicted.annotations, truth.annotations, iou_threshold=iou_threshold
    )
    mAP, per_class_ap = mean_average_precision(
        predicted.annotations, truth.annotations, CLASSES, iou_threshold=iou_threshold
    )
    confusion, accuracy = confusion_matrix(matches, CLASSES)
    edge_p, edge_r, edge_f1 = edge_prf1(
        predicted.graph_edges, truth.graph_edges, matches
    )
    ser = symbol_error_rate(predicted.duration_sequence, truth.duration_sequence)
    pitch_ser = symbol_error_rate(predicted.pitch_sequence, truth.pitch_sequence)

    mask_means = {
        k: {
            "precision": float(np.mean([s.precision for s in v])) if v else 0.0,
            "recall": float(np.mean([s.recall for s in v])) if v else 0.0,
            "f1": float(np.mean([s.f1 for s in v])) if v else 0.0,
            "iou": float(np.mean([s.iou for s in v])) if v else 0.0,
        }
        for k, v in mask_scores_per_kind.items()
    }
    return {
        "image_id": str(image_id),
        "filename": image["filename"],
        "band": band,
        "n_crops": len(samples),
        "crop_diagnostics": crop_diagnostics.to_dict() if crop_diagnostics else None,
        "mask_prf1": mask_means,
        "mAP": mAP,
        "per_class_ap": per_class_ap,
        "accuracy": accuracy,
        "confusion_matrix": confusion.tolist(),
        "edge_precision": edge_p,
        "edge_recall": edge_r,
        "edge_f1": edge_f1,
        "ser": ser,
        "pitch_ser": pitch_ser,
        "n_pred": len(predicted.annotations),
        "n_gt": len(truth.annotations),
        "coverage": covered_gt / len(truth.annotations) if truth.annotations else 0.0,
        "covered_gt": covered_gt,
        "duplicate_covered_gt": duplicate_covered_gt,
        "useful_crops": len(useful_crops),
        "irrelevant_crops": len(samples) - len(useful_crops),
        "pred_duration_sequence": list(predicted.duration_sequence),
        "gt_duration_sequence": list(truth.duration_sequence),
        "pred_pitch_sequence": list(predicted.pitch_sequence),
        "gt_pitch_sequence": list(truth.pitch_sequence),
    }


def _add_timings(page, t0, t1, t2):
    """Attach wall-clock timings (ms) covering the whole per-page pipeline.

    discover_ms: crop discovery incl. image decode / I/O.
    eval_ms: inference + reconstruction + metrics.
    elapsed_ms: total per-page wall time (discover + eval).
    """
    page["discover_ms"] = (t1 - t0) * 1000.0
    page["eval_ms"] = (t2 - t1) * 1000.0
    page["elapsed_ms"] = (t2 - t0) * 1000.0


def _format_row(page, width):
    mask_f1 = page["mask_prf1"].get("notehead", {}).get("f1", 0.0)
    return (
        f"{page['image_id']:>{width // 2}} "
        f"{page['n_crops']:>4} {page['n_pred']:>5} {page['n_gt']:>5} "
        f"{mask_f1:>8.4f} {page['mAP']:>8.4f} {page['accuracy']:>8.4f} "
        f"{page['edge_f1']:>8.4f} {page['ser']:>8.4f}"
    )


def _aggregate(pages):
    if not pages:
        return {}
    keys_float = (
        "mAP",
        "accuracy",
        "edge_precision",
        "edge_recall",
        "edge_f1",
        "ser",
        "pitch_ser",
    )
    means = {k: float(np.mean([p[k] for p in pages])) for k in keys_float}
    mask_kinds = set()
    for p in pages:
        mask_kinds.update(p["mask_prf1"].keys())
    mask_means = {}
    for k in mask_kinds:
        per_field = defaultdict(list)
        for p in pages:
            entry = p["mask_prf1"].get(k)
            if entry:
                for f in ("precision", "recall", "f1", "iou"):
                    per_field[f].append(entry[f])
        mask_means[k] = {
            f: float(np.mean(per_field[f])) if per_field[f] else 0.0
            for f in ("precision", "recall", "f1", "iou")
        }
    class_aps = defaultdict(list)
    for p in pages:
        for c, ap in p["per_class_ap"].items():
            class_aps[c].append(ap)
    per_class_mean = {c: float(np.mean(v)) for c, v in class_aps.items() if v}
    confusion_total = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for p in pages:
        confusion_total += np.array(p["confusion_matrix"], dtype=int)

    n_crops_list = [p["n_crops"] for p in pages]
    crop_stats = {
        "total": int(sum(n_crops_list)),
        "mean": float(np.mean(n_crops_list)) if n_crops_list else 0.0,
        "median": float(statistics.median(n_crops_list)) if n_crops_list else 0.0,
        "min": int(min(n_crops_list)) if n_crops_list else 0,
        "max": int(max(n_crops_list)) if n_crops_list else 0,
        "zero_crop_pages": [p["image_id"] for p in pages if p["n_crops"] == 0],
        "per_page": {p["image_id"]: int(p["n_crops"]) for p in pages},
    }
    by_band = defaultdict(list)
    for p in pages:
        by_band[p.get("band") or "unknown"].append(p["n_crops"])
    crop_stats["by_band"] = {
        b: {"total": int(sum(v)), "mean": float(np.mean(v)) if v else 0.0}
        for b, v in sorted(by_band.items())
    }

    return {
        "n_pages": len(pages),
        "crop_stats": crop_stats,
        "coverage": float(np.mean([p["coverage"] for p in pages])),
        "useful_crops": int(sum(p["useful_crops"] for p in pages)),
        "irrelevant_crops": int(sum(p["irrelevant_crops"] for p in pages)),
        "duplicate_covered_gt": int(sum(p["duplicate_covered_gt"] for p in pages)),
        "mask_prf1": mask_means,
        "mAP": means["mAP"],
        "per_class_ap": per_class_mean,
        "accuracy": means["accuracy"],
        "confusion_matrix": confusion_total.tolist(),
        "edge_precision": means["edge_precision"],
        "edge_recall": means["edge_recall"],
        "edge_f1": means["edge_f1"],
        "ser": means["ser"],
        "pitch_ser": means["pitch_ser"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end staged OMR benchmark on DeepScores"
    )
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write-json-pages", action="store_true")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="shuffle image order for reproducible random page selection",
    )
    parser.add_argument(
        "--page-ids-file",
        type=Path,
        default=None,
        help="TSV manifest (band, image_id, python_n_crops) to evaluate a "
        "fixed page set in manifest order, including zero-crop pages",
    )
    args = parser.parse_args()

    dataset = DeepScoresDataset(DATA_DIR, split=args.split)
    splitter = PrimitiveSplitter()
    args.output.mkdir(parents=True, exist_ok=True)

    pages = []
    requested_page_ids = None

    if args.page_ids_file is not None:
        manifest = _read_page_ids_file(args.page_ids_file)
        requested_page_ids = [e["image_id"] for e in manifest]
        band_by_id = {e["image_id"]: e["band"] for e in manifest}
        print(f"evaluating {len(requested_page_ids)} pages from manifest")
        for image_id in requested_page_ids:
            t0 = time.perf_counter()
            samples = dataset.inference_page_compounds(image_id)
            t1 = time.perf_counter()
            try:
                page = evaluate_page(
                    dataset,
                    splitter,
                    image_id,
                    samples,
                    args.iou_threshold,
                    band=band_by_id.get(image_id),
                    crop_diagnostics=None,
                )
                t2 = time.perf_counter()
                _add_timings(page, t0, t1, t2)
            except Exception as exc:  # pragma: no cover - benchmark safety net
                print(f"[skip] image {image_id}: {exc}")
                t2 = time.perf_counter()
                page = {
                    "image_id": str(image_id),
                    "filename": dataset.images.get(str(image_id), {}).get(
                        "filename", ""
                    ),
                    "band": band_by_id.get(image_id),
                    "n_crops": 0,
                    "crop_diagnostics": None,
                    "mask_prf1": {},
                    "mAP": 0.0,
                    "per_class_ap": {},
                    "accuracy": 0.0,
                    "confusion_matrix": [],
                    "edge_precision": 0.0,
                    "edge_recall": 0.0,
                    "edge_f1": 0.0,
                    "ser": 0.0,
                    "pitch_ser": 0.0,
                    "n_pred": 0,
                    "n_gt": 0,
                    "pred_duration_sequence": [],
                    "gt_duration_sequence": [],
                    "pred_pitch_sequence": [],
                    "gt_pitch_sequence": [],
                    "error": str(exc),
                }
                _add_timings(page, t0, t1, t2)
            pages.append(page)
            print(_format_row(page, width=12))
            if args.write_json_pages:
                out_path = args.output / f"page_{image_id}.json"
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(page, f, indent=2)
    else:
        image_ids = sorted(dataset.images, key=int)
        if args.seed is not None:
            rng = np.random.default_rng(args.seed)
            rng.shuffle(image_ids)
        for image_id in image_ids[: max(0, args.limit)]:
            t0 = time.perf_counter()
            samples = dataset.inference_page_compounds(image_id)
            t1 = time.perf_counter()
            try:
                page = evaluate_page(
                    dataset, splitter, image_id, samples, args.iou_threshold
                )
            except Exception as exc:  # pragma: no cover - benchmark safety net
                print(f"[skip] image {image_id}: {exc}")
                continue
            t2 = time.perf_counter()
            _add_timings(page, t0, t1, t2)
            pages.append(page)
            print(_format_row(page, width=12))
            if args.write_json_pages:
                out_path = args.output / f"page_{image_id}.json"
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(page, f, indent=2)

    summary = _aggregate(pages)
    summary["classes"] = list(CLASSES)
    summary["iou_threshold"] = float(args.iou_threshold)
    summary["inference_ground_truth_free"] = True
    summary["crop_discovery"] = "image_connected_components"
    if requested_page_ids is not None:
        summary["requested_page_ids"] = requested_page_ids
        summary["evaluated_page_ids"] = [p["image_id"] for p in pages]
    summary_path = args.output / "report.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"=== summary over {summary.get('n_pages', 0)} pages ===")
    print(
        f"  mask notehead F1: {summary['mask_prf1'].get('notehead', {}).get('f1', 0.0):.4f}   "
        f"(stage 5, splitting)"
    )
    print(
        f"  mAP @IoU={args.iou_threshold:g}: {summary['mAP']:.4f}   (stage 5, detection)"
    )
    print(f"  accuracy:         {summary['accuracy']:.4f}   (stage 5, classification)")
    print(
        f"  edge P/R/F1:      {summary['edge_precision']:.4f} / "
        f"{summary['edge_recall']:.4f} / {summary['edge_f1']:.4f}   (stage 6, reconstruction)"
    )
    print(f"  SER:              {summary['ser']:.4f}   (end-to-end, duration)")
    print(f"  pitch SER:        {summary['pitch_ser']:.4f}   (end-to-end, pitch)")
    print()
    print(f"report written to {summary_path}")


if __name__ == "__main__":
    main()
