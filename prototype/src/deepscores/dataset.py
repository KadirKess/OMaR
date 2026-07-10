from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image
from skimage import color, filters, measure

from src.preprocessing.preprocessor import Preprocessor


TARGET_CATEGORY_NAMES = {
    "noteheadBlackOnLine": "notehead",
    "noteheadBlackInSpace": "notehead",
    "stem": "stem",
    "beam": "beam",
}

UNKNOWN_CATEGORY_PREFIXES = (
    "noteheadHalf",
    "noteheadWhole",
    "noteheadDoubleWhole",
    "accidental",
    "clef",
    "rest",
    "flag",
)


@dataclass(frozen=True)
class CropSample:
    original_mask: np.ndarray
    staffless_mask: np.ndarray
    truth_masks: dict[str, np.ndarray]
    staff_space: float
    staff_line: float
    image_id: str
    annotation_ids: tuple[str, ...]
    origin: tuple[int, int]


def _bbox_distance(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    dx = max(ax0 - bx1, bx0 - ax1, 0)
    dy = max(ay0 - by1, by0 - ay1, 0)
    return float(np.hypot(dx, dy))


class DeepScoresDataset:
    """Read the dense DeepScores annotations already shipped with the project."""

    def __init__(self, root: str | Path, split: str = "train"):
        self.root = Path(root)
        annotation_path = self.root / f"deepscores_{split}.json"
        with annotation_path.open(encoding="utf-8") as handle:
            data = json.load(handle)

        self.categories = data["categories"]
        self.annotations = data["annotations"]
        self.images = {str(image["id"]): image for image in data["images"]}
        self.category_ids = {
            category["name"]: category_id
            for category_id, category in self.categories.items()
            if category.get("annotation_set") == "deepscores"
        }
        self._by_image = defaultdict(list)
        for annotation_id, annotation in self.annotations.items():
            self._by_image[str(annotation["img_id"])].append(
                (str(annotation_id), annotation)
            )

    def _semantic_mask(self, image):
        path = (
            self.root / "segmentation" / image["filename"].replace(".png", "_seg.png")
        )
        return np.asarray(Image.open(path))

    def _ink_mask(self, image):
        array = np.asarray(Image.open(self.root / "images" / image["filename"]))
        if array.ndim == 3:
            array = color.rgb2gray(array)
        threshold = filters.threshold_otsu(array)
        return array < threshold

    def _staffless_ink(self, image):
        """Remove staff lines while preserving symbol strokes crossing them."""
        ink = self._ink_mask(image)
        preprocessor = Preprocessor(np.where(ink, 0, 255).astype(np.uint8))
        staff_line, staff_gap = preprocessor.compute_staff_space_height()
        rows = preprocessor.staffline_detection_projection()
        preprocessor.remove_stafflines(rows=rows)
        staffless = preprocessor.image == 0
        # Adjacent-line distance keeps geometry scale-invariant.
        staff_space = staff_gap + staff_line
        return ink, staffless, float(staff_space), float(staff_line)

    def _category_mask(self, semantic, annotation):
        category_id = annotation["cat_id"][0]
        color_index = self.categories[category_id]["color"]
        return semantic == color_index

    def load_page(self, image_id: str):
        """Return ``(original_ink, staffless_ink, image_metadata)``."""
        image = self.images[str(image_id)]
        ink, staffless, _, _ = self._staffless_ink(image)
        return ink, staffless, image

    def page_compounds(self, image_id: str) -> list[CropSample]:
        image_id = str(image_id)
        if not any(
            self.categories[ann["cat_id"][0]]["name"]
            in ("noteheadBlackOnLine", "noteheadBlackInSpace")
            for _, ann in self._by_image[image_id]
        ):
            return []
        original, staffless, staff_space, staff_line = self._staffless_ink(
            self.images[image_id]
        )
        semantic = self._semantic_mask(self.images[image_id])
        labels = measure.label(staffless)
        minimum_area = max(2, int(round(0.08 * staff_space**2)))
        margin = int(round(0.6 * staff_space))
        category_masks = {kind: [] for kind in ("notehead", "stem", "beam")}
        for name, output_name in TARGET_CATEGORY_NAMES.items():
            category_masks[output_name].append(
                semantic == self.categories[self.category_ids[name]]["color"]
            )
        samples = []
        for region in measure.regionprops(labels):
            if region.area < minimum_area:
                continue
            y0, x0, y1, x1 = region.bbox
            crop_x0 = max(0, x0 - margin)
            crop_x1 = min(original.shape[1], x1 + margin)
            crop_y0 = max(0, y0 - margin)
            crop_y1 = min(original.shape[0], y1 + margin)
            component = labels[crop_y0:crop_y1, crop_x0:crop_x1] == region.label
            truth_masks = {}
            for kind in ("notehead", "stem", "beam"):
                masks = [
                    mask[crop_y0:crop_y1, crop_x0:crop_x1]
                    for mask in category_masks[kind]
                ]
                truth_masks[kind] = (
                    np.logical_or.reduce(masks)
                    if masks
                    else np.zeros_like(component, dtype=bool)
                ) & component
            samples.append(
                CropSample(
                    original_mask=original[crop_y0:crop_y1, crop_x0:crop_x1],
                    staffless_mask=component,
                    truth_masks=truth_masks,
                    staff_space=staff_space,
                    staff_line=staff_line,
                    image_id=image_id,
                    annotation_ids=(),
                    origin=(crop_x0, crop_y0),
                )
            )
        return samples

    def iter_supported_crops(
        self, limit: int | None = None, shuffle_seed: int | None = None
    ) -> Iterator[CropSample]:
        import random as _random

        yielded = 0
        target_ids = {self.category_ids[name] for name in TARGET_CATEGORY_NAMES}
        staff_id = self.category_ids.get("staff")

        image_ids = list(self.images.keys())
        if shuffle_seed is not None:
            _random.Random(shuffle_seed).shuffle(image_ids)

        for image_id in image_ids:
            image = self.images[image_id]
            entries = self._by_image[image_id]
            targets = [
                (aid, ann) for aid, ann in entries if ann["cat_id"][0] in target_ids
            ]
            if not targets:
                continue

            heads = [
                item
                for item in targets
                if TARGET_CATEGORY_NAMES[self.categories[item[1]["cat_id"][0]]["name"]]
                == "notehead"
            ]
            if not heads:
                continue
            head_height = float(
                np.median([h[1]["a_bbox"][3] - h[1]["a_bbox"][1] for h in heads])
            )
            staff_space = max(4.0, head_height)
            link_distance = max(2.0, 0.35 * staff_space)

            graph = defaultdict(set)
            for index, (aid, ann) in enumerate(targets):
                graph[aid]
                for bid, other in targets[index + 1 :]:
                    if _bbox_distance(ann["a_bbox"], other["a_bbox"]) <= link_distance:
                        graph[aid].add(bid)
                        graph[bid].add(aid)

            seen = set()
            semantic = ink = labels = None
            for seed, _ in targets:
                if seed in seen:
                    continue
                stack = [seed]
                component_ids = set()
                while stack:
                    current = stack.pop()
                    if current in component_ids:
                        continue
                    component_ids.add(current)
                    stack.extend(graph[current])
                seen |= component_ids
                component = [(aid, self.annotations[aid]) for aid in component_ids]

                kinds = [
                    TARGET_CATEGORY_NAMES[self.categories[ann["cat_id"][0]]["name"]]
                    for _, ann in component
                ]
                head_count, stem_count, beam_count = (
                    kinds.count("notehead"),
                    kinds.count("stem"),
                    kinds.count("beam"),
                )

                component_heads = [
                    ann
                    for (_, ann), kind in zip(component, kinds)
                    if kind == "notehead"
                ]
                centers = [
                    (
                        (ann["a_bbox"][0] + ann["a_bbox"][2]) / 2,
                        ann["a_bbox"][2] - ann["a_bbox"][0],
                    )
                    for ann in component_heads
                ]
                if any(
                    abs(x1 - x2) < 0.6 * min(w1, w2)
                    for i, (x1, w1) in enumerate(centers)
                    for x2, w2 in centers[i + 1 :]
                ):
                    continue

                if semantic is None:
                    semantic = self._semantic_mask(image)
                    ink = self._ink_mask(image)
                    staff_mask = (
                        semantic == self.categories[staff_id]["color"]
                        if staff_id is not None
                        else np.zeros_like(ink)
                    )
                    labels = measure.label(ink & ~staff_mask)
                truth_full = {
                    kind: np.zeros_like(ink) for kind in ("notehead", "stem", "beam")
                }
                for (_, ann), kind in zip(component, kinds):
                    x0, y0, x1, y1 = map(int, ann["a_bbox"])
                    category = self._category_mask(semantic, ann)
                    truth_full[kind][y0 : y1 + 1, x0 : x1 + 1] |= category[
                        y0 : y1 + 1, x0 : x1 + 1
                    ]

                union = np.logical_or.reduce(tuple(truth_full.values()))
                touched = np.unique(labels[union])
                touched = touched[touched != 0]
                if len(touched) == 0:
                    continue
                compound = np.isin(labels, touched)

                ys, xs = np.nonzero(union)
                margin = int(round(1.5 * staff_space))
                x0 = max(0, int(xs.min()) - margin)
                x1 = min(ink.shape[1], int(xs.max()) + margin + 1)
                y0 = max(0, int(ys.min()) - margin)
                y1 = min(ink.shape[0], int(ys.max()) + margin + 1)

                crop_truth = {
                    kind: mask[y0:y1, x0:x1].copy() for kind, mask in truth_full.items()
                }
                yield CropSample(
                    original_mask=ink[y0:y1, x0:x1].copy(),
                    staffless_mask=compound[y0:y1, x0:x1].copy(),
                    truth_masks=crop_truth,
                    staff_space=staff_space,
                    staff_line=max(1.0, round(0.15 * staff_space)),
                    image_id=image_id,
                    annotation_ids=tuple(sorted(component_ids)),
                    origin=(x0, y0),
                )
                yielded += 1
                if limit is not None and yielded >= limit:
                    return

    def iter_unknown_crops(self, limit: int | None = None) -> Iterator[CropSample]:
        yielded = 0
        staff_id = self.category_ids.get("staff")
        target_names = set(TARGET_CATEGORY_NAMES)

        for image_id, image in self.images.items():
            entries = self._by_image[image_id]
            regular_heads = [
                ann
                for _, ann in entries
                if self.categories[ann["cat_id"][0]]["name"]
                in ("noteheadBlackOnLine", "noteheadBlackInSpace")
            ]
            if not regular_heads:
                continue
            staff_space = max(
                4.0,
                float(
                    np.median(
                        [ann["a_bbox"][3] - ann["a_bbox"][1] for ann in regular_heads]
                    )
                ),
            )
            semantic = self._semantic_mask(image)
            ink = self._ink_mask(image)
            staff_mask = (
                semantic == self.categories[staff_id]["color"]
                if staff_id is not None
                else np.zeros_like(ink)
            )
            for annotation_id, annotation in entries:
                name = self.categories[annotation["cat_id"][0]]["name"]
                if name in target_names or not name.startswith(
                    UNKNOWN_CATEGORY_PREFIXES
                ):
                    continue
                x0, y0, x1, y1 = map(int, annotation["a_bbox"])
                margin = int(round(1.25 * staff_space))
                crop_x0 = max(0, x0 - margin)
                crop_x1 = min(ink.shape[1], x1 + margin + 1)
                crop_y0 = max(0, y0 - margin)
                crop_y1 = min(ink.shape[0], y1 + margin + 1)
                local_ink = ink[crop_y0:crop_y1, crop_x0:crop_x1]
                local_staff = staff_mask[crop_y0:crop_y1, crop_x0:crop_x1]
                local_labels = measure.label(local_ink & ~local_staff)
                category = self._category_mask(semantic, annotation)
                local_instance = category[crop_y0:crop_y1, crop_x0:crop_x1]
                touched = np.unique(local_labels[local_instance])
                touched = touched[touched != 0]
                if len(touched) == 0:
                    continue
                compound = np.isin(local_labels, touched)
                shape = compound.shape
                yield CropSample(
                    original_mask=local_ink,
                    staffless_mask=compound,
                    truth_masks={
                        "notehead": np.zeros(shape, dtype=bool),
                        "stem": np.zeros(shape, dtype=bool),
                        "beam": np.zeros(shape, dtype=bool),
                    },
                    staff_space=staff_space,
                    staff_line=max(1.0, round(0.15 * staff_space)),
                    image_id=image_id,
                    annotation_ids=(annotation_id,),
                    origin=(crop_x0, crop_y0),
                )
                yielded += 1
                if limit is not None and yielded >= limit:
                    return
