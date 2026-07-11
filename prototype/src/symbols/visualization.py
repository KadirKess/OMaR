from pathlib import Path
import os
import tempfile
from collections import Counter

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "musical-score-matplotlib")
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from src.deepscores.dataset import CropSample
from src.models.primitives import PrimitiveKind, PrimitiveSplit


COLORS = {
    PrimitiveKind.NOTEHEAD: np.array([0.85, 0.10, 0.55]),
    PrimitiveKind.STEM: np.array([0.10, 0.45, 0.90]),
    PrimitiveKind.BEAM: np.array([1.00, 0.55, 0.05]),
    PrimitiveKind.UNKNOWN: np.array([0.90, 0.12, 0.12]),
}


def _show_binary(axis, mask, title):
    axis.imshow(mask, cmap="gray_r", vmin=0, vmax=1)
    axis.set_title(title)
    axis.axis("off")


def _overlay(base_mask, masks):
    image = np.ones((*base_mask.shape, 3), dtype=float)
    image[base_mask] = 0.72
    for kind, mask in masks.items():
        image[np.asarray(mask, dtype=bool)] = COLORS[kind]
    return image


def _center(primitive):
    x0, y0, x1, y1 = primitive.bbox
    return (x0 + x1) / 2, (y0 + y1) / 2


def render_split_diagnostics(
    sample: CropSample,
    result: PrimitiveSplit,
    output_path: str | Path,
    title: str | None = None,
):
    """Render the input, intermediate class masks, links, and ground truth."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predicted = {
        kind: np.zeros_like(sample.staffless_mask, dtype=bool)
        for kind in (
            PrimitiveKind.NOTEHEAD,
            PrimitiveKind.STEM,
            PrimitiveKind.BEAM,
            PrimitiveKind.UNKNOWN,
        )
    }
    for primitive in result.primitives:
        predicted[primitive.kind] |= primitive.mask
    truth = {
        PrimitiveKind.NOTEHEAD: sample.truth_masks["notehead"],
        PrimitiveKind.STEM: sample.truth_masks["stem"],
        PrimitiveKind.BEAM: sample.truth_masks["beam"],
    }

    figure, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    _show_binary(axes[0, 0], sample.original_mask, "1. Original binary crop")
    _show_binary(axes[0, 1], sample.staffless_mask, "2. Staffless compound")
    _show_binary(axes[0, 2], predicted[PrimitiveKind.BEAM], "3. Beam mask")
    _show_binary(axes[0, 3], predicted[PrimitiveKind.STEM], "4. Stem masks")
    _show_binary(axes[1, 0], predicted[PrimitiveKind.NOTEHEAD], "5. Notehead masks")
    _show_binary(axes[1, 1], predicted[PrimitiveKind.UNKNOWN], "6. Unknown residual")

    axes[1, 2].imshow(_overlay(sample.staffless_mask, predicted))
    axes[1, 2].set_title(f"7. Prediction: {result.status}")
    axes[1, 2].axis("off")
    for source, target in (*result.head_to_stem, *result.stem_to_beam):
        x0, y0 = _center(result.primitives[source])
        x1, y1 = _center(result.primitives[target])
        axes[1, 2].plot((x0, x1), (y0, y1), color="lime", linewidth=1.5)

    axes[1, 3].imshow(_overlay(sample.staffless_mask, truth))
    axes[1, 3].set_title("8. DeepScores ground truth")
    axes[1, 3].axis("off")

    legend = [
        Line2D([0], [0], color=COLORS[kind], lw=6, label=str(kind)) for kind in COLORS
    ]
    legend.append(Line2D([0], [0], color="lime", lw=2, label="association"))
    figure.legend(handles=legend, loc="lower center", ncol=5, frameon=False)
    heading = title or f"DeepScores image {sample.image_id}"
    reasons = ", ".join(result.reasons) if result.reasons else "none"
    figure.suptitle(f"{heading} — reasons: {reasons}", fontsize=14)
    figure.savefig(output_path, dpi=150, facecolor="white")
    plt.close(figure)


def _page_overlay(staffless_mask, samples_and_results):
    shape = staffless_mask.shape
    page_masks = {
        kind: np.zeros(shape, dtype=bool)
        for kind in (
            PrimitiveKind.NOTEHEAD,
            PrimitiveKind.STEM,
            PrimitiveKind.BEAM,
            PrimitiveKind.UNKNOWN,
        )
    }
    links = []
    crop_boxes = []

    for sample, result in samples_and_results:
        for primitive in result.primitives:
            _paste(page_masks[primitive.kind], primitive.mask, sample.origin)
        ox, oy = sample.origin
        for source, target in (*result.head_to_stem, *result.stem_to_beam):
            sx, sy = _center(result.primitives[source])
            tx, ty = _center(result.primitives[target])
            links.append(((sx + ox, sy + oy), (tx + ox, ty + oy)))
        if str(result.status) != "rejected":
            crop_boxes.append(
                (
                    ox,
                    oy,
                    sample.staffless_mask.shape[1],
                    sample.staffless_mask.shape[0],
                    result.status,
                )
            )
    return page_masks, links, crop_boxes


def render_page_overlay(
    staffless_mask: np.ndarray,
    samples_and_results: list[tuple[CropSample, PrimitiveSplit]],
    output_path: str | Path,
    title: str,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    page_masks, links, crop_boxes = _page_overlay(staffless_mask, samples_and_results)

    dpi = 220
    height, width = staffless_mask.shape
    figsize = (width / dpi, height / dpi)
    figure = plt.figure(figsize=figsize, dpi=dpi)
    axis = figure.add_axes([0, 0, 1, 1])
    axis.imshow(_overlay(staffless_mask, page_masks))
    axis.axis("off")
    for (x0, y0), (x1, y1) in links:
        axis.plot((x0, x1), (y0, y1), color="lime", linewidth=0.6)
    status_colors = {"accepted": "lime", "uncertain": "gold"}
    for x, y, w, height, status in crop_boxes:
        rectangle = plt.Rectangle(
            (x, y),
            w,
            height,
            fill=False,
            edgecolor=status_colors[str(status)],
            linewidth=0.5,
        )
        axis.add_patch(rectangle)

    figure.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(figure)


def _paste(target, source, origin):
    x0, y0 = origin
    y1 = min(target.shape[0], y0 + source.shape[0])
    x1 = min(target.shape[1], x0 + source.shape[1])
    target[y0:y1, x0:x1] |= source[: y1 - y0, : x1 - x0]


def render_page_diagnostics(
    original_mask: np.ndarray,
    staffless_mask: np.ndarray,
    samples_and_results: list[tuple[CropSample, PrimitiveSplit]],
    output_path: str | Path,
    title: str,
):
    """Render all supported compound predictions on one complete score page."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shape = original_mask.shape
    compounds = np.zeros(shape, dtype=bool)
    page_masks = {
        kind: np.zeros(shape, dtype=bool)
        for kind in (
            PrimitiveKind.NOTEHEAD,
            PrimitiveKind.STEM,
            PrimitiveKind.BEAM,
            PrimitiveKind.UNKNOWN,
        )
    }
    links = []
    crop_boxes = []

    for sample, result in samples_and_results:
        _paste(compounds, sample.staffless_mask, sample.origin)
        for primitive in result.primitives:
            _paste(page_masks[primitive.kind], primitive.mask, sample.origin)
        ox, oy = sample.origin
        for source, target in (*result.head_to_stem, *result.stem_to_beam):
            sx, sy = _center(result.primitives[source])
            tx, ty = _center(result.primitives[target])
            links.append(((sx + ox, sy + oy), (tx + ox, ty + oy)))
        if str(result.status) != "rejected":
            crop_boxes.append(
                (
                    ox,
                    oy,
                    sample.staffless_mask.shape[1],
                    sample.staffless_mask.shape[0],
                    result.status,
                )
            )

    figure, axes = plt.subplots(2, 4, figsize=(18, 22), constrained_layout=True)
    _show_binary(axes[0, 0], original_mask, "1. Original full page")
    _show_binary(axes[0, 1], staffless_mask, "2. Staffless full page")
    _show_binary(axes[0, 2], compounds, "3. Connected-component candidates")
    _show_binary(axes[0, 3], page_masks[PrimitiveKind.BEAM], "4. Beam masks")
    _show_binary(axes[1, 0], page_masks[PrimitiveKind.STEM], "5. Stem masks")
    _show_binary(axes[1, 1], page_masks[PrimitiveKind.NOTEHEAD], "6. Notehead masks")
    _show_binary(axes[1, 2], page_masks[PrimitiveKind.UNKNOWN], "7. Unknown residuals")

    axes[1, 3].imshow(_overlay(staffless_mask, page_masks))
    axes[1, 3].set_title("8. Full-page prediction and associations")
    axes[1, 3].axis("off")
    for (x0, y0), (x1, y1) in links:
        axes[1, 3].plot((x0, x1), (y0, y1), color="lime", linewidth=0.6)
    status_colors = {"accepted": "lime", "uncertain": "gold"}
    for x, y, width, height, status in crop_boxes:
        rectangle = plt.Rectangle(
            (x, y),
            width,
            height,
            fill=False,
            edgecolor=status_colors[str(status)],
            linewidth=0.5,
        )
        axes[1, 3].add_patch(rectangle)

    legend = [
        Line2D([0], [0], color=COLORS[kind], lw=6, label=str(kind)) for kind in COLORS
    ]
    legend.append(Line2D([0], [0], color="lime", lw=2, label="association/accepted"))
    figure.legend(handles=legend, loc="lower center", ncol=5, frameon=False)
    statuses = Counter(str(result.status) for _, result in samples_and_results)
    status_summary = ", ".join(f"{name}: {count}" for name, count in statuses.items())
    figure.suptitle(
        f"{title} — {len(samples_and_results)} candidates — {status_summary}",
        fontsize=16,
    )
    figure.savefig(output_path, dpi=220, facecolor="white")
    plt.close(figure)
