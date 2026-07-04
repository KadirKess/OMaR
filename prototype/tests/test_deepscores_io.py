import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.deepscores.io import (
    Annotation,
    GraphEdgeSpec,
    PageIO,
    SCOPED_CATEGORY_NAMES,
    _notehead_cat_name,
    build_predicted_page,
    derive_ground_truth,
    write_predictions_json,
)
from src.reconstruction.assembly import compute_staff_positions, graph_from_split, run_rewrite
from src.models.primitives import Primitive, PrimitiveKind, PrimitiveSplit, SplitStatus
from src.reconstruction.semantics import assign_durations


def make_primitive(kind, x0, y0, x1, y1, confidence=1.0):
    mask = np.zeros((64, 64), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return Primitive(kind=kind, mask=mask, confidence=confidence)


def quarter_note_split(origin_offset=0):
    head = make_primitive(PrimitiveKind.NOTEHEAD, 10 + origin_offset, 46, 24 + origin_offset, 54)
    stem = make_primitive(PrimitiveKind.STEM, 22 + origin_offset, 20, 24 + origin_offset, 46)
    return PrimitiveSplit(
        primitives=(head, stem),
        head_to_stem=((0, 1),),
        status=SplitStatus.ACCEPTED,
        staff_space=10.0,
        staff_lines=(10.0, 20.0, 30.0, 40.0, 50.0),
    )


def beamed_pair_split():
    head_a = make_primitive(PrimitiveKind.NOTEHEAD, 10, 44, 24, 52)
    stem_a = make_primitive(PrimitiveKind.STEM, 22, 18, 24, 46)
    head_b = make_primitive(PrimitiveKind.NOTEHEAD, 40, 44, 54, 52)
    stem_b = make_primitive(PrimitiveKind.STEM, 52, 18, 54, 46)
    beam = make_primitive(PrimitiveKind.BEAM, 22, 14, 54, 18)
    return PrimitiveSplit(
        primitives=(head_a, stem_a, head_b, stem_b, beam),
        head_to_stem=((0, 1), (2, 3)),
        stem_to_beam=((1, 4), (3, 4)),
        status=SplitStatus.ACCEPTED,
        staff_space=10.0,
        staff_lines=(10.0, 20.0, 30.0, 40.0, 50.0),
    )


class NoteheadCatNameTests(unittest.TestCase):
    def test_on_line_when_center_matches_a_staff_line(self):
        bbox = (0, 46, 14, 54)  # center y=50 == bottom staff line
        self.assertEqual(
            _notehead_cat_name(bbox, (10.0, 20.0, 30.0, 40.0, 50.0), staff_line=2.0),
            "noteheadBlackOnLine",
        )

    def test_in_space_when_center_between_lines(self):
        bbox = (0, 41, 14, 49)  # center y=45, halfway between lines 40 and 50
        self.assertEqual(
            _notehead_cat_name(bbox, (10.0, 20.0, 30.0, 40.0, 50.0), staff_line=2.0),
            "noteheadBlackInSpace",
        )

    def test_defaults_to_on_line_without_staff_lines(self):
        bbox = (0, 41, 14, 49)
        self.assertEqual(_notehead_cat_name(bbox, (), staff_line=2.0), "noteheadBlackOnLine")


class BuildPredictedPageTests(unittest.TestCase):
    def _finish_graph(self, split):
        graph = graph_from_split(split)
        compute_staff_positions(graph)
        run_rewrite(graph)
        assign_durations(graph)
        return graph

    def _image(self):
        return {"id": 42, "filename": "page.png", "width": 200, "height": 200}

    def test_annotations_carry_origin_translation_and_correct_categories(self):
        split = quarter_note_split(origin_offset=0)
        graph = self._finish_graph(split)
        page = build_predicted_page(
            self._image(),
            [(split, graph, (100, 200))],
            staff_lines_per_crop=[(10.0, 20.0, 30.0, 40.0, 50.0)],
            staff_line=2.0,
        )
        # head at x0=10 y0=46..y1=54 -> with origin (100,200): 110..114..124
        head_ann = next(a for a in page.annotations if a.cat_name.startswith("notehead"))
        self.assertAlmostEqual(head_ann.bbox[0], 110.0)
        self.assertAlmostEqual(head_ann.bbox[1], 246.0)
        self.assertAlmostEqual(head_ann.bbox[2], 124.0)
        self.assertAlmostEqual(head_ann.bbox[3], 254.0)
        stem_ann = next(a for a in page.annotations if a.cat_name == "stem")
        self.assertAlmostEqual(stem_ann.bbox[0], 122.0)
        self.assertAlmostEqual(stem_ann.bbox[1], 220.0)
        self.assertEqual(head_ann.ann_id, "pred-0-0")
        self.assertEqual(stem_ann.ann_id, "pred-0-1")

    def test_quarter_note_yields_quarter_token(self):
        split = quarter_note_split()
        graph = self._finish_graph(split)
        page = build_predicted_page(
            self._image(),
            [(split, graph, (0, 0))],
            staff_lines_per_crop=[(10.0, 20.0, 30.0, 40.0, 50.0)],
            staff_line=2.0,
        )
        self.assertEqual(page.duration_sequence, ("quarter",))

    def test_beamed_pair_yields_two_eighth_tokens_in_x_order(self):
        split = beamed_pair_split()
        graph = self._finish_graph(split)
        page = build_predicted_page(
            self._image(),
            [(split, graph, (0, 0))],
            staff_lines_per_crop=[(10.0, 20.0, 30.0, 40.0, 50.0)],
            staff_line=2.0,
        )
        self.assertEqual(page.duration_sequence, ("eighth", "eighth"))

    def test_predicted_edge_specs_match_split_pairings(self):
        split = beamed_pair_split()
        graph = self._finish_graph(split)
        page = build_predicted_page(
            self._image(),
            [(split, graph, (0, 0))],
            staff_lines_per_crop=[(10.0, 20.0, 30.0, 40.0, 50.0)],
            staff_line=2.0,
        )
        relations = sorted(e.relation for e in page.graph_edges)
        self.assertEqual(
            relations,
            sorted(["head_to_stem", "head_to_stem", "stem_to_beam", "stem_to_beam"]),
        )
        for e in page.graph_edges:
            self.assertTrue(e.src_id.startswith("pred-0-"))
            self.assertTrue(e.dst_id.startswith("pred-0-"))

    def test_multiple_crops_use_distinct_ids(self):
        split = quarter_note_split()
        graph = self._finish_graph(split)
        page = build_predicted_page(
            self._image(),
            [(split, graph, (0, 0)), (split, graph, (50, 60))],
            staff_lines_per_crop=[
                (10.0, 20.0, 30.0, 40.0, 50.0),
                (10.0, 20.0, 30.0, 40.0, 50.0),
            ],
            staff_line=2.0,
        )
        ids = {a.ann_id for a in page.annotations}
        self.assertEqual(len(ids), 4)
        self.assertIn("pred-0-0", ids)
        self.assertIn("pred-1-1", ids)


class WritePredictionsJsonTests(unittest.TestCase):
    def _fake_page(self):
        return PageIO(
            image_id="42",
            filename="page.png",
            width=200,
            height=200,
            annotations=(
                Annotation(
                    ann_id="pred-0-0",
                    bbox=(10.0, 20.0, 30.0, 40.0),
                    cat_name="noteheadBlackOnLine",
                    area=120,
                ),
                Annotation(
                    ann_id="pred-0-1",
                    bbox=(22.0, 5.0, 24.0, 40.0),
                    cat_name="stem",
                    area=70,
                ),
            ),
            graph_edges=(
                GraphEdgeSpec("head_to_stem", "pred-0-0", "pred-0-1"),
            ),
            duration_sequence=("quarter",),
        )

    def _fake_categories(self):
        return {
            "135": {"name": "stem", "annotation_set": "deepscores", "color": 7},
            "208": {
                "name": "noteheadBlackOnLine",
                "annotation_set": "deepscores",
                "color": 8,
            },
            "209": {
                "name": "noteheadBlackInSpace",
                "annotation_set": "deepscores",
                "color": 9,
            },
            "1": {"name": "clefG", "annotation_set": "deepscores", "color": 2},
            "300": {"name": "beam", "annotation_set": "deepscores", "color": 11},
            "999": {"name": "restQuarter", "annotation_set": "muscima++", "color": 5},
        }

    def test_round_trip_emissions_match_deep_scores_schema(self):
        page = self._fake_page()
        cats = self._fake_categories()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out" / "page_42.json"
            write_predictions_json(page, cats, out)
            self.assertTrue(out.exists())
            data = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(data["info"]["version"], "1.0")
        self.assertEqual(data["annotation_sets"], ["deepscores"])

        # categories scoped to the 4 supported symbol classes; clefs/rests excluded
        kept_names = {c["name"] for c in data["categories"].values()}
        self.assertEqual(kept_names, set(SCOPED_CATEGORY_NAMES))

        # dict-keyed annotations (DeepScores wire format), with cat_id resolved via name
        self.assertIn("pred-0-0", data["annotations"])
        ann = data["annotations"]["pred-0-0"]
        self.assertEqual(ann["cat_id"], ["208"])
        self.assertEqual(ann["a_bbox"], [10.0, 20.0, 30.0, 40.0])
        self.assertEqual(len(ann["o_bbox"]), 8)  # OBB = 4 corners x 2
        self.assertEqual(ann["img_id"], "42")
        self.assertEqual(ann["area"], 120)

        stem_ann = data["annotations"]["pred-0-1"]
        self.assertEqual(stem_ann["cat_id"], ["135"])

        # graph + duration extensions
        self.assertEqual(len(data["graph"]["edges"]), 1)
        self.assertEqual(data["graph"]["edges"][0]["relation"], "head_to_stem")
        self.assertEqual(data["duration_sequence"], ["quarter"])

    def test_unknown_cat_name_is_silently_dropped(self):
        page = PageIO(
            image_id="1",
            filename="x.png",
            width=10,
            height=10,
            annotations=(
                Annotation(ann_id="x", bbox=(0.0, 0.0, 1.0, 1.0), cat_name="notInTaxonomy", area=1),
            ),
            graph_edges=(),
            duration_sequence=(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.json"
            write_predictions_json(page, self._fake_categories(), out)
            data = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(data["annotations"], {})


@unittest.skipUnless(
    (Path(__file__).parent.parent / "data" / "ds2_dense" / "deepscores_test.json").exists(),
    "local DeepScores subset is not installed",
)
class DeriveGroundTruthTests(unittest.TestCase):
    DATA_DIR = Path(__file__).parent.parent / "data" / "ds2_dense"

    def _first_test_image(self, dataset):
        for image_id, image in dataset.images.items():
            if any(
                dataset.categories[ann["cat_id"][0]]["name"]
                in ("noteheadBlackOnLine", "noteheadBlackInSpace")
                for _, ann in dataset._by_image[image_id]
            ):
                return image_id
        self.skipTest("no test image with scoped noteheads")

    def test_ground_truth_only_keeps_scoped_categories(self):
        from src.deepscores.dataset import DeepScoresDataset

        ds = DeepScoresDataset(self.DATA_DIR, split="test")
        image_id = self._first_test_image(ds)
        gt = derive_ground_truth(ds, image_id)
        for ann in gt.annotations:
            self.assertIn(ann.cat_name, SCOPED_CATEGORY_NAMES)

    def test_gt_duration_tokens_are_in_vocabulary(self):
        from src.deepscores.dataset import DeepScoresDataset

        ds = DeepScoresDataset(self.DATA_DIR, split="test")
        image_id = self._first_test_image(ds)
        gt = derive_ground_truth(ds, image_id)
        for token in gt.duration_sequence:
            self.assertIn(token, {"quarter", "eighth"})

    def test_gt_edge_endpoints_are_present_in_annotations(self):
        from src.deepscores.dataset import DeepScoresDataset

        ds = DeepScoresDataset(self.DATA_DIR, split="test")
        image_id = self._first_test_image(ds)
        gt = derive_ground_truth(ds, image_id)
        ids = {a.ann_id for a in gt.annotations}
        for e in gt.graph_edges:
            self.assertIn(e.src_id, ids)
            self.assertIn(e.dst_id, ids)


if __name__ == "__main__":
    unittest.main()
