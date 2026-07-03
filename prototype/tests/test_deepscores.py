import unittest
from pathlib import Path

from src.deepscores import DeepScoresDataset
from src.primitives import PrimitiveKind
from src.splitting import PrimitiveSplitter


DATA_DIR = Path(__file__).parent.parent / "data" / "ds2_dense"


@unittest.skipUnless(DATA_DIR.exists(), "local DeepScores subset is not installed")
class DeepScoresSmokeTests(unittest.TestCase):
    def test_first_supported_crop_can_be_split(self):
        sample = next(DeepScoresDataset(DATA_DIR).iter_supported_crops(limit=1))
        result = PrimitiveSplitter().split(
            sample.staffless_mask,
            sample.staff_space,
            sample.staff_line,
            sample.original_mask,
        )

        self.assertGreater(len(result.of_kind(PrimitiveKind.NOTEHEAD)), 0)
        self.assertEqual(
            len(result.of_kind(PrimitiveKind.NOTEHEAD)),
            len(result.of_kind(PrimitiveKind.STEM)),
        )
        self.assertEqual(
            len(result.head_to_stem), len(result.of_kind(PrimitiveKind.STEM))
        )

    def test_unsupported_symbol_is_not_accepted(self):
        sample = next(DeepScoresDataset(DATA_DIR).iter_unknown_crops(limit=1))
        result = PrimitiveSplitter().split(
            sample.staffless_mask,
            sample.staff_space,
            sample.staff_line,
            sample.original_mask,
        )
        self.assertNotEqual(str(result.status), "accepted")
        self.assertGreater(len(result.of_kind(PrimitiveKind.UNKNOWN)), 0)


if __name__ == "__main__":
    unittest.main()
