import unittest

import numpy as np
from skimage import draw

from src.primitives import PrimitiveKind, SplitStatus
from src.split_evaluation import score_mask
from src.splitting import PrimitiveSplitter


def add_head(image, x, y, rx=7, ry=5):
    rr, cc = draw.ellipse(y, x, ry, rx, shape=image.shape)
    image[rr, cc] = True


class PrimitiveSplitterTests(unittest.TestCase):
    def setUp(self):
        self.splitter = PrimitiveSplitter()

    def test_upstem_quarter_note(self):
        image = np.zeros((60, 50), dtype=bool)
        add_head(image, 20, 40)
        image[12:41, 26:28] = True

        result = self.splitter.split(image, staff_space=10, staff_line=2)

        self.assertEqual(result.status, SplitStatus.ACCEPTED)
        self.assertEqual(len(result.of_kind(PrimitiveKind.NOTEHEAD)), 1)
        self.assertEqual(len(result.of_kind(PrimitiveKind.STEM)), 1)
        self.assertEqual(len(result.of_kind(PrimitiveKind.BEAM)), 0)
        self.assertEqual(result.head_to_stem, ((1, 0),))

    def test_single_beam_group(self):
        image = np.zeros((70, 90), dtype=bool)
        for x, y in ((20, 45), (45, 48), (70, 42)):
            add_head(image, x, y)
            image[12 : y + 1, x + 6 : x + 8] = True
        image[10:15, 26:78] = True

        result = self.splitter.split(image, staff_space=10, staff_line=2)

        self.assertEqual(result.status, SplitStatus.ACCEPTED)
        self.assertEqual(len(result.of_kind(PrimitiveKind.NOTEHEAD)), 3)
        self.assertEqual(len(result.of_kind(PrimitiveKind.STEM)), 3)
        self.assertEqual(len(result.of_kind(PrimitiveKind.BEAM)), 1)
        self.assertEqual(len(result.head_to_stem), 3)
        self.assertEqual(len(result.stem_to_beam), 3)

    def test_unexplained_region_makes_result_uncertain(self):
        image = np.zeros((70, 70), dtype=bool)
        add_head(image, 20, 45)
        image[15:46, 26:28] = True
        image[8:20, 50:62] = True

        result = self.splitter.split(image, staff_space=10, staff_line=2)

        self.assertEqual(result.status, SplitStatus.UNCERTAIN)
        self.assertEqual(len(result.of_kind(PrimitiveKind.UNKNOWN)), 1)
        self.assertIn("unexplained_ink", result.reasons)

    def test_open_head_is_not_accepted_as_supported(self):
        image = np.zeros((60, 50), dtype=bool)
        rr, cc = draw.ellipse_perimeter(40, 20, 5, 7, shape=image.shape)
        image[rr, cc] = True
        image[12:41, 26:28] = True

        result = self.splitter.split(image, staff_space=10, staff_line=2)

        self.assertNotEqual(result.status, SplitStatus.ACCEPTED)

    def test_mask_metric(self):
        truth = np.array([[1, 1], [0, 0]], dtype=bool)
        predicted = np.array([[1, 0], [1, 0]], dtype=bool)
        score = score_mask(predicted, truth)
        self.assertAlmostEqual(score.precision, 0.5)
        self.assertAlmostEqual(score.recall, 0.5)
        self.assertAlmostEqual(score.f1, 0.5)
        self.assertAlmostEqual(score.iou, 1 / 3)

    def test_invalid_dimensions(self):
        with self.assertRaises(ValueError):
            self.splitter.split(np.zeros((3, 3), dtype=bool), 0, 1)


if __name__ == "__main__":
    unittest.main()
