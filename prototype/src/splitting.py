from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from skimage import draw, measure, morphology

from .primitives import Primitive, PrimitiveKind, PrimitiveSplit, SplitStatus


@dataclass(frozen=True)
class SplitterConfig:
    beam_min_length_spaces: float = 2.2
    beam_max_angle_degrees: float = 25.0
    stem_min_length_spaces: float = 1.6
    head_width_spaces: float = 1.4
    head_height_spaces: float = 1.05
    head_threshold: float = 0.56
    filled_head_min_occupancy: float = 0.68
    stem_threshold: float = 0.55
    beam_threshold: float = 0.55
    unknown_min_area_spaces: float = 0.12


@dataclass
class _Candidate:
    mask: np.ndarray
    confidence: float


def _line_footprint(length: int, thickness: int, angle_degrees: float) -> np.ndarray:
    half = max(1, length // 2)
    dy = int(round(math.tan(math.radians(angle_degrees)) * half))
    height = 2 * abs(dy) + 2 * thickness + 3
    width = 2 * half + 1
    footprint = np.zeros((height, width), dtype=bool)
    cy = height // 2
    rr, cc = draw.line(cy - dy, 0, cy + dy, width - 1)
    footprint[rr, cc] = True
    if thickness > 1:
        footprint = morphology.dilation(
            footprint, morphology.disk(max(1, thickness // 2))
        )
    return footprint


def _ellipse(shape, cx, cy, rx, ry):
    mask = np.zeros(shape, dtype=bool)
    rr, cc = draw.ellipse(cy, cx, ry, rx, shape=shape)
    mask[rr, cc] = True
    return mask


def _mask_distance(a: np.ndarray, b: np.ndarray, radius: int) -> bool:
    return bool(np.any(morphology.dilation(a, morphology.disk(radius)) & b))


class PrimitiveSplitter:
    """Split a pre-isolated compound into supported graphical primitives."""

    def __init__(self, config: SplitterConfig | None = None):
        self.config = config or SplitterConfig()

    def _detect_beam(self, ink, staff_space, staff_line) -> _Candidate | None:
        cfg = self.config
        length = max(5, int(round(cfg.beam_min_length_spaces * staff_space)))
        thickness = max(1, int(round(staff_line)))
        best = None

        angles = np.linspace(
            -cfg.beam_max_angle_degrees, cfg.beam_max_angle_degrees, 11
        )
        for angle in angles:
            footprint = _line_footprint(length, thickness, float(angle))
            opened = morphology.opening(ink, footprint)
            for region in measure.regionprops(measure.label(opened)):
                minr, minc, maxr, maxc = region.bbox
                if maxc - minc < length:
                    continue
                coords = region.coords
                xs = coords[:, 1].astype(float)
                ys = coords[:, 0].astype(float)
                if np.ptp(xs) == 0:
                    continue
                slope, intercept = np.polyfit(xs, ys, 1)
                detected_angle = abs(math.degrees(math.atan(float(slope))))
                if detected_angle > cfg.beam_max_angle_degrees + 2:
                    continue

                residuals = np.abs(ys - (slope * xs + intercept))
                band_half = max(1.0, float(np.percentile(residuals, 95)) + 0.75)
                band_half = min(0.4 * staff_space, band_half)
                yy, xx = np.indices(ink.shape)
                within_x = (xx >= minc) & (xx < maxc)
                band = within_x & (np.abs(yy - (slope * xx + intercept)) <= band_half)
                candidate_mask = ink & band
                area = int(candidate_mask.sum())
                if area == 0:
                    continue
                span_score = min(1.0, (maxc - minc) / (3.5 * staff_space))
                fill_score = min(1.0, area / max(1.0, (maxc - minc) * staff_line * 1.5))
                confidence = 0.65 * span_score + 0.35 * fill_score
                candidate = _Candidate(candidate_mask, confidence)
                if best is None or candidate.confidence > best.confidence:
                    best = candidate

        if best is None or best.confidence < cfg.beam_threshold:
            return None
        return best

    def _detect_stems(self, ink, beam_mask, staff_space, staff_line):
        cfg = self.config
        work = ink & ~morphology.dilation(
            beam_mask, morphology.disk(max(1, int(round(staff_line))))
        )
        length = max(5, int(round(cfg.stem_min_length_spaces * staff_space)))
        # A one-pixel probe preserves thin stems in DeepScores fonts.
        footprint = np.ones((length, 1), dtype=bool)
        opened = morphology.opening(work, footprint)
        active_columns = opened.any(axis=0)
        column_regions = measure.regionprops(measure.label(active_columns[None, :]))
        candidates = []
        for column_region in column_regions:
            _, x0, _, x1 = column_region.bbox
            expanded_x0 = max(0, x0 - int(round(staff_line)))
            expanded_x1 = min(ink.shape[1], x1 + int(round(staff_line)))
            ys, _ = np.nonzero(opened[:, x0:x1])
            if len(ys) == 0:
                continue
            y0, y1 = int(ys.min()), int(ys.max() + 1)
            mask = np.zeros_like(ink)
            mask[y0:y1, expanded_x0:expanded_x1] = work[y0:y1, expanded_x0:expanded_x1]
            height = y1 - y0
            width = max(1, expanded_x1 - expanded_x0)
            length_score = min(1.0, height / (2.8 * staff_space))
            narrow_score = min(1.0, (2.5 * staff_line) / width)
            confidence = 0.7 * length_score + 0.3 * narrow_score
            if confidence >= cfg.stem_threshold and mask.any():
                candidates.append(_Candidate(mask, confidence))
        return candidates

    def _best_head_for_stem(self, ink, stem, beam_mask, staff_space, staff_line):
        cfg = self.config
        ys, xs = np.nonzero(stem.mask)
        if len(xs) == 0:
            return None
        stem_x = float(np.median(xs))
        endpoints = (int(ys.min()), int(ys.max()))
        rx = max(2, int(round(cfg.head_width_spaces * staff_space / 2)))
        ry = max(2, int(round(cfg.head_height_spaces * staff_space / 2)))
        # Larger closing elements make open noteheads look filled.
        repaired = morphology.closing(ink & ~beam_mask, morphology.disk(1))
        best = None
        best_center = None
        x_offsets = range(-rx, rx + 1, max(1, rx // 3))
        y_offsets = range(-ry, ry + 1, max(1, ry // 3))
        for endpoint in endpoints:
            for dx in x_offsets:
                for dy in y_offsets:
                    cx = int(round(stem_x + dx))
                    cy = endpoint + dy
                    if not (0 <= cx < ink.shape[1] and 0 <= cy < ink.shape[0]):
                        continue
                    ellipse = _ellipse(ink.shape, cx, cy, rx, ry)
                    inside = int(ellipse.sum())
                    if inside == 0:
                        continue
                    occupancy = float(np.count_nonzero(repaired & ellipse) / inside)
                    if occupancy < cfg.filled_head_min_occupancy:
                        continue
                    # Ink must spread in both axes to avoid matching line fragments.
                    local = repaired & ellipse
                    row_spread = np.count_nonzero(local.any(axis=1)) / max(1, 2 * ry)
                    col_spread = np.count_nonzero(local.any(axis=0)) / max(1, 2 * rx)
                    score = (
                        0.65 * occupancy
                        + 0.175 * min(1.0, row_spread)
                        + 0.175 * min(1.0, col_spread)
                    )
                    if best is None or score > best:
                        best, best_center = score, (cx, cy, ellipse)

        if best is None or best < cfg.head_threshold:
            return None
        cx, cy, ellipse = best_center
        growth = max(1, int(round(staff_line / 3)))
        support = _ellipse(
            ink.shape,
            cx,
            cy,
            rx + growth,
            ry + growth,
        )
        mask = ink & support
        if not mask.any():
            return None
        return _Candidate(mask, min(1.0, best))

    def split(
        self,
        staffless_mask: np.ndarray,
        staff_space: float,
        staff_line: float,
        original_mask: np.ndarray | None = None,
        staff_lines: tuple[float, ...] = (),
    ) -> PrimitiveSplit:
        """Split one isolated compound."""
        ink = np.asarray(staffless_mask, dtype=bool)
        if ink.ndim != 2:
            raise ValueError("staffless_mask must be a two-dimensional image")
        if original_mask is not None and np.asarray(original_mask).shape != ink.shape:
            raise ValueError(
                "original_mask and staffless_mask must have the same shape"
            )
        if staff_space <= 0 or staff_line <= 0:
            raise ValueError("staff dimensions must be positive")

        empty = np.zeros_like(ink)
        beam = self._detect_beam(ink, staff_space, staff_line)
        beam_mask = beam.mask.copy() if beam is not None else empty.copy()
        stems = self._detect_stems(ink, beam_mask, staff_space, staff_line)

        head_candidates = []
        for stem_index, stem in enumerate(stems):
            head = self._best_head_for_stem(
                ink, stem, beam_mask, staff_space, staff_line
            )
            if head is not None:
                head_candidates.append((stem_index, head))

        # One head cannot belong to two stems.
        accepted_heads = []
        claimed = np.zeros_like(ink)
        for stem_index, head in sorted(
            head_candidates, key=lambda item: item[1].confidence, reverse=True
        ):
            overlap = np.count_nonzero(claimed & head.mask) / max(1, head.mask.sum())
            if overlap < 0.5:
                accepted_heads.append((stem_index, head))
                claimed |= head.mask

        head_union = np.zeros_like(ink)
        for _, head in accepted_heads:
            head_union |= head.mask
        beam_mask &= ~head_union
        stem_masks = [stem.mask & ~head_union & ~beam_mask for stem in stems]

        primitives = []
        stem_primitive_indices = []
        associated_stems = {stem_index for stem_index, _ in accepted_heads}
        for candidate_index, (stem, mask) in enumerate(zip(stems, stem_masks)):
            # Headless vertical shapes may be barlines or clef fragments.
            if candidate_index in associated_stems and mask.any():
                stem_primitive_indices.append(len(primitives))
                primitives.append(Primitive(PrimitiveKind.STEM, mask, stem.confidence))
            else:
                stem_primitive_indices.append(None)

        head_to_stem = []
        for stem_index, head in accepted_heads:
            head_index = len(primitives)
            primitives.append(
                Primitive(PrimitiveKind.NOTEHEAD, head.mask, head.confidence)
            )
            target_stem = stem_primitive_indices[stem_index]
            if target_stem is not None:
                head_to_stem.append((head_index, target_stem))

        beam_index = None
        if beam is not None and beam_mask.any():
            beam_index = len(primitives)
            primitives.append(Primitive(PrimitiveKind.BEAM, beam_mask, beam.confidence))

        stem_to_beam = []
        if beam_index is not None:
            radius = max(1, int(round(max(staff_line + 1, 0.35 * staff_space))))
            for stem_index in stem_primitive_indices:
                if stem_index is not None and _mask_distance(
                    primitives[stem_index].mask, beam_mask, radius
                ):
                    stem_to_beam.append((stem_index, beam_index))

        assigned = np.zeros_like(ink)
        for primitive in primitives:
            assigned |= primitive.mask
        residual = ink & ~assigned
        min_unknown_area = max(
            2, int(round(self.config.unknown_min_area_spaces * staff_space**2))
        )
        unknown_count = 0
        for region in measure.regionprops(measure.label(residual)):
            if region.area < min_unknown_area:
                continue
            mask = np.zeros_like(ink)
            coords = region.coords
            mask[coords[:, 0], coords[:, 1]] = True
            primitives.append(Primitive(PrimitiveKind.UNKNOWN, mask, 1.0))
            unknown_count += 1

        head_count = sum(p.kind == PrimitiveKind.NOTEHEAD for p in primitives)
        stem_count = sum(p.kind == PrimitiveKind.STEM for p in primitives)
        beam_count = sum(p.kind == PrimitiveKind.BEAM for p in primitives)
        reasons = []
        topology_valid = (
            head_count == stem_count == len(head_to_stem) and head_count > 0
        )
        if beam_count:
            topology_valid &= (
                beam_count == 1 and len(stem_to_beam) == stem_count and stem_count >= 2
            )
        else:
            topology_valid &= head_count == 1
        if not topology_valid:
            reasons.append("unsupported_or_incomplete_topology")
        if unknown_count:
            reasons.append("unexplained_ink")

        if not topology_valid:
            status = SplitStatus.REJECTED
        elif unknown_count:
            status = SplitStatus.UNCERTAIN
        else:
            status = SplitStatus.ACCEPTED

        return PrimitiveSplit(
            primitives=tuple(primitives),
            head_to_stem=tuple(head_to_stem),
            stem_to_beam=tuple(stem_to_beam),
            status=status,
            reasons=tuple(reasons),
            staff_space=float(staff_space),
            staff_line=float(staff_line),
            staff_lines=tuple(float(y) for y in staff_lines),
        )
