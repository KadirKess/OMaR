from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from src.deepscores.dataset import DeepScoresDataset
from src.models.graph import NodeKind, NotationGraph
from src.models.primitives import PrimitiveKind
from src.reconstruction.semantics import EIGHTH, QUARTER, _step_to_pitch


SCOPED_CATEGORY_NAMES = (
    "noteheadBlackOnLine",
    "noteheadBlackInSpace",
    "stem",
    "beam",
)

_DURATION_TOKENS = {
    QUARTER: "quarter",
    EIGHTH: "eighth",
}

_PRIMITIVE_TO_CAT = {
    PrimitiveKind.STEM: "stem",
    PrimitiveKind.BEAM: "beam",
}

_LINK_DISTANCE_FACTOR = 0.35


@dataclass(frozen=True)
class Annotation:
    ann_id: str
    bbox: tuple[float, float, float, float]
    cat_name: str
    area: int
    confidence: float = 1.0


@dataclass(frozen=True)
class GraphEdgeSpec:
    relation: str
    src_id: str
    dst_id: str


@dataclass(frozen=True)
class PageIO:
    image_id: str
    filename: str
    width: int
    height: int
    annotations: tuple[Annotation, ...]
    graph_edges: tuple[GraphEdgeSpec, ...]
    duration_sequence: tuple[str, ...]
    pitch_sequence: tuple[str, ...] = ()


def _bbox_distance(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    dx = max(ax0 - bx1, bx0 - ax1, 0)
    dy = max(ay0 - by1, by0 - ay1, 0)
    return float(np.hypot(dx, dy))


def _link_distance(staff_space: float) -> float:
    return max(2.0, _LINK_DISTANCE_FACTOR * staff_space)


def _notehead_cat_name(head_bbox, staff_lines, staff_line) -> str:
    # head_bbox and staff_lines must be in the same (page) frame; OnLine if the
    # head center sits within two staff-line thicknesses of a recovered line.
    # Tolerance absorbs projection/deskew residuals in staff-line recovery.
    if not staff_lines:
        return "noteheadBlackOnLine"
    x0, y0, x1, y1 = head_bbox
    cy = (y0 + y1) / 2.0
    tol = max(1.0, 2.0 * float(staff_line))
    on_line = any(abs(cy - y) <= tol for y in staff_lines)
    return "noteheadBlackOnLine" if on_line else "noteheadBlackInSpace"


def _annotations_from_split(split, crop_index, origin, staff_lines, staff_line):
    ox, oy = origin
    out = []
    id_map = {}
    for prim_idx, prim in enumerate(split.primitives):
        if prim.kind == PrimitiveKind.UNKNOWN:
            continue
        x0, y0, x1, y1 = prim.bbox
        page_bbox = (float(x0 + ox), float(y0 + oy), float(x1 + ox), float(y1 + oy))
        if prim.kind == PrimitiveKind.NOTEHEAD:
            cat_name = _notehead_cat_name(page_bbox, staff_lines, staff_line)
        else:
            cat_name = _PRIMITIVE_TO_CAT[prim.kind]
        ann_id = f"pred-{crop_index}-{prim_idx}"
        out.append(
            Annotation(
                ann_id=ann_id,
                bbox=page_bbox,
                cat_name=cat_name,
                area=int(prim.mask.sum()),
                confidence=float(prim.confidence),
            )
        )
        id_map[prim_idx] = ann_id
    return out, id_map


def _edges_from_split(split, id_map) -> list[GraphEdgeSpec]:
    out = []
    for head_idx, stem_idx in split.head_to_stem:
        if head_idx in id_map and stem_idx in id_map:
            out.append(
                GraphEdgeSpec("head_to_stem", id_map[head_idx], id_map[stem_idx])
            )
    for stem_idx, beam_idx in split.stem_to_beam:
        if stem_idx in id_map and beam_idx in id_map:
            out.append(
                GraphEdgeSpec("stem_to_beam", id_map[stem_idx], id_map[beam_idx])
            )
    return out


def _node_x(node, origin_x):
    head = node.attrs.get("notehead")
    if head is None:
        heads = node.attrs.get("noteheads") or ()
        head = heads[0] if heads else None
    if head is None:
        return float(origin_x)
    return float(head.bbox[0] + origin_x)


def _duration_sequence_from_graph(graph, origin=(0, 0)) -> list[tuple[float, str]]:
    ox, _ = origin
    nodes = [
        n
        for n in graph.nodes.values()
        if n.kind in (NodeKind.NOTE, NodeKind.CHORD, NodeKind.BEAMED_GROUP)
    ]
    nodes.sort(key=lambda n: n.attrs.get("x_order", 0))
    pairs = []
    for n in nodes:
        if n.kind == NodeKind.BEAMED_GROUP:
            for m in n.attrs.get("members", ()):
                pairs.append((_node_x(m, ox), _DURATION_TOKENS.get(m.attrs.get("duration"), "?")))
        else:
            pairs.append((_node_x(n, ox), _DURATION_TOKENS.get(n.attrs.get("duration"), "?")))
    return pairs


def _pitch_sequence_from_graph(graph, origin=(0, 0)) -> list[tuple[float, str]]:
    ox, _ = origin
    nodes = [
        n
        for n in graph.nodes.values()
        if n.kind in (NodeKind.NOTE, NodeKind.CHORD, NodeKind.BEAMED_GROUP)
    ]
    nodes.sort(key=lambda n: n.attrs.get("x_order", 0))
    pairs = []
    for n in nodes:
        if n.kind == NodeKind.BEAMED_GROUP:
            for m in n.attrs.get("members", ()):
                pairs.append((_node_x(m, ox), m.attrs.get("pitch", "?")))
        else:
            pairs.append((_node_x(n, ox), n.attrs.get("pitch", "?")))
    return pairs


def build_predicted_page(
    image, crop_results, staff_lines_per_crop, staff_line=2.0
) -> PageIO:
    # staff_lines_per_crop aligns with crop_results by index; () falls back to OnLine.
    if staff_lines_per_crop is None:
        staff_lines_per_crop = [()] * len(crop_results)
    all_anns: list[Annotation] = []
    all_edges: list[GraphEdgeSpec] = []
    all_duration_pairs: list[tuple[float, str]] = []
    all_pitch_pairs: list[tuple[float, str]] = []
    for crop_index, ((split, graph, origin), staff_lines) in enumerate(
        zip(crop_results, staff_lines_per_crop)
    ):
        anns, id_map = _annotations_from_split(
            split, crop_index, origin, staff_lines, staff_line
        )
        all_anns.extend(anns)
        all_edges.extend(_edges_from_split(split, id_map))
        all_duration_pairs.extend(_duration_sequence_from_graph(graph, origin))
        all_pitch_pairs.extend(_pitch_sequence_from_graph(graph, origin))
    all_duration_pairs.sort(key=lambda p: p[0])
    all_pitch_pairs.sort(key=lambda p: p[0])
    return PageIO(
        image_id=str(image["id"]),
        filename=image["filename"],
        width=int(image["width"]),
        height=int(image["height"]),
        annotations=tuple(all_anns),
        graph_edges=tuple(all_edges),
        duration_sequence=tuple(t for _, t in all_duration_pairs),
        pitch_sequence=tuple(t for _, t in all_pitch_pairs),
    )


def _nearest_system_bottom(staff_rows, y_center):
    rows = sorted(float(r) for r in staff_rows)
    if len(rows) <= 5:
        return max(rows) if rows else None
    best_bottom, best_dist = None, float("inf")
    for i in range(len(rows) - 4):
        window = rows[i : i + 5]
        window_center = (window[0] + window[-1]) / 2.0
        d = abs(window_center - y_center)
        if d < best_dist:
            best_dist = d
            best_bottom = window[-1]
    return best_bottom


def derive_ground_truth(
    dataset: DeepScoresDataset,
    image_id: str,
    staff_space: float | None = None,
    bbox_filter=None,
    staff_lines: tuple[float, ...] = (),
) -> PageIO:
    # bbox_filter: optional predicate (a_bbox -> bool) that scopes the GT to a
    # region of interest (e.g. the crops we actually processed). When provided,
    # the edges and duration sequence are derived over the filtered set only,
    # so mAP / accuracy / SER measure only the symbols we attempted.
    image = dataset.images[str(image_id)]
    entries = dataset._by_image[str(image_id)]
    scoped_cat_ids = {
        dataset.category_ids[name]
        for name in SCOPED_CATEGORY_NAMES
        if name in dataset.category_ids
    }
    gt: list[tuple[str, dict, str]] = []
    for aid, ann in entries:
        cid = ann["cat_id"][0]
        if cid not in scoped_cat_ids:
            continue
        if bbox_filter is not None and not bbox_filter(ann["a_bbox"]):
            continue
        gt.append((aid, ann, dataset.categories[cid]["name"]))

    annotations = tuple(
        Annotation(
            ann_id=str(aid),
            bbox=tuple(float(v) for v in ann["a_bbox"]),
            cat_name=cat_name,
            area=int(ann.get("area", 0)),
        )
        for aid, ann, cat_name in gt
    )
    bboxes = [ann["a_bbox"] for _, ann, _ in gt]
    ann_ids = [a.ann_id for a in annotations]
    cat_names = [c for _, _, c in gt]

    if staff_space is None:
        head_heights = [
            bboxes[i][3] - bboxes[i][1]
            for i in range(len(gt))
            if cat_names[i] in ("noteheadBlackOnLine", "noteheadBlackInSpace")
        ]
        staff_space = (
            max(4.0, float(np.median(head_heights))) if head_heights else 4.0
        )
    link_d = _link_distance(staff_space)

    stem_idx = [i for i, n in enumerate(cat_names) if n == "stem"]
    beam_idx = [i for i, n in enumerate(cat_names) if n == "beam"]

    edges: list[GraphEdgeSpec] = []
    head_tokens: list[tuple[float, str]] = []
    pitch_tokens: list[tuple[float, str]] = []
    for i, name in enumerate(cat_names):
        if name not in ("noteheadBlackOnLine", "noteheadBlackInSpace"):
            continue
        if not stem_idx:
            continue
        s = min(stem_idx, key=lambda j: _bbox_distance(bboxes[i], bboxes[j]))
        if _bbox_distance(bboxes[i], bboxes[s]) > link_d:
            continue
        edges.append(GraphEdgeSpec("head_to_stem", ann_ids[i], ann_ids[s]))
        token = "quarter"
        if beam_idx:
            b = min(beam_idx, key=lambda j: _bbox_distance(bboxes[s], bboxes[j]))
            if _bbox_distance(bboxes[s], bboxes[b]) <= link_d:
                edges.append(GraphEdgeSpec("stem_to_beam", ann_ids[s], ann_ids[b]))
                token = "eighth"
        head_tokens.append((bboxes[i][0], token))

        if staff_lines:
            cy = (bboxes[i][1] + bboxes[i][3]) / 2.0
            ref_y = _nearest_system_bottom(staff_lines, cy)
            if ref_y is not None:
                step = round((ref_y - cy) / (staff_space / 2.0))
                pitch_tokens.append((bboxes[i][0], _step_to_pitch(step, "E", 4)))

    head_tokens.sort(key=lambda x: x[0])
    duration_sequence = tuple(t for _, t in head_tokens)
    pitch_tokens.sort(key=lambda x: x[0])
    pitch_sequence = tuple(t for _, t in pitch_tokens)

    return PageIO(
        image_id=str(image["id"]),
        filename=image["filename"],
        width=int(image["width"]),
        height=int(image["height"]),
        annotations=annotations,
        graph_edges=tuple(edges),
        duration_sequence=duration_sequence,
        pitch_sequence=pitch_sequence,
    )


def _a_bbox_to_o_bbox(a_bbox):
    x0, y0, x1, y1 = a_bbox
    return (x0, y0, x1, y0, x1, y1, x0, y1)


def write_predictions_json(
    page: PageIO, source_categories: dict, out_path: Path
) -> None:
    name_to_id = {
        cat["name"]: cid
        for cid, cat in source_categories.items()
        if cat.get("annotation_set") == "deepscores"
    }
    scoped_categories = {
        cid: source_categories[cid]
        for cid, cat in source_categories.items()
        if cat.get("annotation_set") == "deepscores"
        and cat["name"] in SCOPED_CATEGORY_NAMES
    }
    annotations_json = {}
    for ann in page.annotations:
        cat_id = name_to_id.get(ann.cat_name)
        if cat_id is None:
            continue
        annotations_json[ann.ann_id] = {
            "a_bbox": list(ann.bbox),
            "o_bbox": list(_a_bbox_to_o_bbox(ann.bbox)),
            "cat_id": [cat_id],
            "area": ann.area,
            "img_id": page.image_id,
            "comments": "",
        }
    image_id_value = (
        int(page.image_id) if str(page.image_id).isdigit() else page.image_id
    )
    payload = {
        "info": {
            "description": "OMR pipeline predictions",
            "version": "1.0",
        },
        "annotation_sets": ["deepscores"],
        "categories": scoped_categories,
        "images": [
            {
                "id": image_id_value,
                "filename": page.filename,
                "width": page.width,
                "height": page.height,
            }
        ],
        "annotations": annotations_json,
        "graph": {
            "edges": [
                {"relation": e.relation, "src": e.src_id, "dst": e.dst_id}
                for e in page.graph_edges
            ]
        },
        "duration_sequence": list(page.duration_sequence),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
