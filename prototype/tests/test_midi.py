import tempfile
import unittest
from pathlib import Path

from src.midi import duration_token_to_ticks, pitch_token_to_midi, write_midi


class PitchTokenTests(unittest.TestCase):
    def test_a4_is_69(self):
        self.assertEqual(pitch_token_to_midi("A4"), 69)

    def test_c4_is_60(self):
        self.assertEqual(pitch_token_to_midi("C4"), 60)

    def test_c5_is_72(self):
        self.assertEqual(pitch_token_to_midi("C5"), 72)

    def test_g4_is_67(self):
        self.assertEqual(pitch_token_to_midi("G4"), 67)

    def test_unknown_returns_none(self):
        self.assertIsNone(pitch_token_to_midi("?"))
        self.assertIsNone(pitch_token_to_midi(""))
        self.assertIsNone(pitch_token_to_midi("X9"))


class DurationTokenTests(unittest.TestCase):
    def test_quarter_at_default_ppq(self):
        self.assertEqual(duration_token_to_ticks("quarter"), 480)

    def test_eighth_at_default_ppq(self):
        self.assertEqual(duration_token_to_ticks("eighth"), 240)

    def test_quarter_scales_with_ppq(self):
        self.assertEqual(duration_token_to_ticks("quarter", ppq=960), 960)

    def test_unknown_returns_none(self):
        self.assertIsNone(duration_token_to_ticks("whole"))
        self.assertIsNone(duration_token_to_ticks("?"))


class WriteMidiTests(unittest.TestCase):
    def test_writes_valid_smf_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "midi.mid"
            written = write_midi(
                ("quarter", "eighth", "quarter", "eighth"),
                ("C4", "D4", "E4", "G4"),
                out,
            )
            self.assertEqual(written, 4)
            data = out.read_bytes()
            self.assertTrue(data.startswith(b"MThd"))
            self.assertEqual(data[9], 0)
            track_chunk = data[14:]
            self.assertTrue(track_chunk.startswith(b"MTrk"))
            self.assertEqual(
                int.from_bytes(track_chunk[4:8], "big"),
                len(track_chunk[8:]),
            )

    def test_skips_unknown_pitch(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "midi.mid"
            written = write_midi(
                ("quarter", "eighth", "quarter"),
                ("C4", "?", "E4"),
                out,
            )
            self.assertEqual(written, 2)
            self.assertTrue(out.read_bytes().startswith(b"MThd"))


if __name__ == "__main__":
    unittest.main()
