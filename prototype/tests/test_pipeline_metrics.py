import unittest

from benchmarks._metrics import (
    PairMatch,
    average_precision,
    confusion_matrix,
    edge_prf1,
    iou_box,
    match_pairs,
    mean_average_precision,
    symbol_error_rate,
)
from src.deepscores.io import Annotation, GraphEdgeSpec


def ann(ann_id, bbox, cat_name, area=10, confidence=1.0):
    return Annotation(
        ann_id=ann_id,
        bbox=bbox,
        cat_name=cat_name,
        area=area,
        confidence=confidence,
    )


class IouBoxTests(unittest.TestCase):
    def test_identical_boxes_score_one(self):
        self.assertAlmostEqual(iou_box((0, 0, 4, 4), (0, 0, 4, 4)), 1.0)

    def test_disjoint_boxes_score_zero(self):
        self.assertEqual(iou_box((0, 0, 2, 2), (5, 5, 7, 7)), 0.0)

    def test_partial_overlap(self):
        a = (0, 0, 4, 4)
        b = (2, 2, 6, 6)
        # intersection 2x2=4, areas 16 each, union = 16+16-4 = 28 -> 4/28
        self.assertAlmostEqual(iou_box(a, b), 4 / 28)

    def test_contained_box(self):
        outer = (0, 0, 10, 10)
        inner = (2, 2, 4, 4)
        # intersection = 4, area_outer = 100, area_inner = 4, union = 100
        self.assertAlmostEqual(iou_box(outer, inner), 4 / 100)

    def test_zero_area_box_is_zero(self):
        self.assertEqual(iou_box((0, 0, 0, 0), (0, 0, 4, 4)), 0.0)


class MatchPairsTests(unittest.TestCase):
    def test_perfect_one_to_one_match(self):
        preds = [ann("p1", (0, 0, 10, 10), "stem")]
        gts = [ann("g1", (0, 0, 10, 10), "stem")]
        matches = match_pairs(preds, gts)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].pred_id, "p1")
        self.assertEqual(matches[0].gt_id, "g1")
        self.assertTrue(matches[0].pred_label == matches[0].gt_label)

    def test_below_threshold_is_unmatched(self):
        preds = [ann("p1", (0, 0, 2, 2), "stem")]
        gts = [ann("g1", (10, 10, 12, 12), "stem")]
        self.assertEqual(match_pairs(preds, gts, iou_threshold=0.5), [])

    def test_greedy_takes_highest_iou_first(self):
        preds = [
            ann("p_best", (0, 0, 10, 10), "stem"),
            ann("p_overlap", (4, 4, 14, 14), "stem"),
        ]
        gts = [ann("g1", (0, 0, 10, 10), "stem")]
        matches = match_pairs(preds, gts)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].pred_id, "p_best")

    def test_cross_class_match_still_pairs(self):
        # a bbox matched across class labels: detection correct, label wrong
        preds = [ann("p1", (0, 0, 10, 10), "noteheadBlackOnLine")]
        gts = [ann("g1", (0, 0, 10, 10), "noteheadBlackInSpace")]
        matches = match_pairs(preds, gts)
        self.assertEqual(len(matches), 1)
        self.assertNotEqual(matches[0].pred_label, matches[0].gt_label)


class AveragePrecisionTests(unittest.TestCase):
    def test_perfect_predictions_give_ap_one(self):
        preds = [
            ann("p1", (0, 0, 10, 10), "stem", confidence=0.9),
            ann("p2", (20, 0, 30, 10), "stem", confidence=0.8),
        ]
        gts = [
            ann("g1", (0, 0, 10, 10), "stem"),
            ann("g2", (20, 0, 30, 10), "stem"),
        ]
        ap = average_precision(preds, gts, "stem")
        self.assertAlmostEqual(ap, 1.0)

    def test_all_false_positives_give_ap_zero(self):
        preds = [ann("p1", (0, 0, 10, 10), "stem", confidence=0.9)]
        gts = [ann("g1", (100, 100, 110, 110), "stem")]
        ap = average_precision(preds, gts, "stem")
        self.assertEqual(ap, 0.0)

    def test_no_predictions_when_gt_exists_give_zero(self):
        gts = [ann("g1", (0, 0, 10, 10), "stem")]
        self.assertEqual(average_precision([], gts, "stem"), 0.0)

    def test_absent_class_in_gt_returns_none(self):
        preds = [ann("p1", (0, 0, 10, 10), "stem")]
        self.assertIsNone(average_precision(preds, [], "stem"))

    def test_mean_average_precision_skips_absent_classes(self):
        preds = [
            ann("p1", (0, 0, 10, 10), "stem"),
            ann("p2", (0, 0, 10, 10), "beam"),
        ]
        gts = [ann("g1", (0, 0, 10, 10), "stem")]
        mAP, per_class = mean_average_precision(
            preds, gts, ["stem", "beam", "noteheadBlackOnLine"]
        )
        self.assertIn("stem", per_class)
        self.assertNotIn("beam", per_class)
        self.assertAlmostEqual(mAP, per_class["stem"])


class ConfusionMatrixTests(unittest.TestCase):
    CLASSES = ["noteheadBlackOnLine", "stem", "beam"]

    def test_perfect_diagonal(self):
        matches = [
            PairMatch("p1", "g1", 0.9, "stem", "stem"),
            PairMatch("p2", "g2", 0.9, "stem", "stem"),
            PairMatch("p3", "g3", 0.9, "beam", "beam"),
        ]
        matrix, accuracy = confusion_matrix(matches, self.CLASSES)
        self.assertAlmostEqual(accuracy, 1.0)
        self.assertEqual(matrix[1, 1], 2)
        self.assertEqual(matrix[2, 2], 1)
        self.assertEqual(matrix.sum(), 3)

    def test_misclassification_off_diagonal(self):
        matches = [
            PairMatch("p1", "g1", 0.9, "stem", "beam"),
            PairMatch("p2", "g2", 0.9, "stem", "stem"),
        ]
        matrix, accuracy = confusion_matrix(matches, self.CLASSES)
        self.assertAlmostEqual(accuracy, 0.5)
        self.assertEqual(matrix[2, 1], 1)
        self.assertEqual(matrix[1, 1], 1)

    def test_empty_matches_yield_zero_matrix_and_zero_accuracy(self):
        matrix, accuracy = confusion_matrix([], self.CLASSES)
        self.assertEqual(accuracy, 0.0)
        self.assertEqual(matrix.sum(), 0)


class EdgePrf1Tests(unittest.TestCase):
    def test_perfect_translation_gives_one(self):
        matches = [
            PairMatch("p1", "g1", 0.9, "stem", "stem"),
            PairMatch("p2", "g2", 0.9, "stem", "stem"),
        ]
        pred = [GraphEdgeSpec("head_to_stem", "p1", "p2")]
        gt = [GraphEdgeSpec("head_to_stem", "g1", "g2")]
        p, r, f1 = edge_prf1(pred, gt, matches)
        self.assertAlmostEqual(p, 1.0)
        self.assertAlmostEqual(r, 1.0)
        self.assertAlmostEqual(f1, 1.0)

    def test_missing_predicted_edge_is_recall_loss(self):
        matches = [
            PairMatch("p1", "g1", 0.9, "stem", "stem"),
            PairMatch("p2", "g2", 0.9, "stem", "stem"),
        ]
        pred = []
        gt = [GraphEdgeSpec("head_to_stem", "g1", "g2")]
        _, r, _ = edge_prf1(pred, gt, matches)
        self.assertEqual(r, 0.0)

    def test_extra_predicted_edge_is_precision_loss(self):
        matches = [
            PairMatch("p1", "g1", 0.9, "stem", "stem"),
            PairMatch("p2", "g2", 0.9, "stem", "stem"),
        ]
        pred = [
            GraphEdgeSpec("head_to_stem", "p1", "p2"),  # correct after translation
            GraphEdgeSpec("stem_to_beam", "p1", "p2"),  # wrong relation
        ]
        gt = [GraphEdgeSpec("head_to_stem", "g1", "g2")]
        p, _, _ = edge_prf1(pred, gt, matches)
        self.assertAlmostEqual(p, 0.5)

    def test_edge_with_unmatched_endpoint_counts_as_fp(self):
        # stem primitive never matched a GT bbox; its edge is hallucinated
        matches = [PairMatch("p1", "g1", 0.9, "notehead", "notehead")]
        pred = [GraphEdgeSpec("head_to_stem", "p1", "p_unmatched")]
        gt = []
        p, r, f1 = edge_prf1(pred, gt, matches)
        self.assertEqual(p, 0.0)
        self.assertEqual(r, 0.0)


class SymbolErrorRateTests(unittest.TestCase):
    def test_identical_sequences_are_zero(self):
        self.assertEqual(
            symbol_error_rate(
                ("quarter", "eighth", "eighth"), ("quarter", "eighth", "eighth")
            ),
            0.0,
        )

    def test_single_substitution(self):
        # ref length 4, one substitution -> 1/4
        self.assertAlmostEqual(
            symbol_error_rate(
                ("quarter", "eighth", "eighth", "eighth"),
                ("quarter", "quarter", "eighth", "eighth"),
            ),
            0.25,
        )

    def test_one_insertion_in_hypothesis(self):
        # ref length 2, one extra symbol -> 1 edit / 2
        self.assertAlmostEqual(
            symbol_error_rate(("quarter", "eighth", "eighth"), ("quarter", "eighth")),
            0.5,
        )

    def test_one_deletion_from_hypothesis(self):
        self.assertAlmostEqual(
            symbol_error_rate(("quarter",), ("quarter", "eighth")), 0.5
        )

    def test_completely_different_sequences(self):
        # both length 2, both differ -> 2 edits / 2 = 1.0
        self.assertAlmostEqual(
            symbol_error_rate(("quarter", "quarter"), ("eighth", "eighth")), 1.0
        )

    def test_empty_reference_with_hypothesis_is_one(self):
        self.assertEqual(symbol_error_rate(("quarter",), ()), 1.0)

    def test_both_empty_is_zero(self):
        self.assertEqual(symbol_error_rate((), ()), 0.0)


if __name__ == "__main__":
    unittest.main()
