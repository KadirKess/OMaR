import unittest

import numpy as np

from src.assembly import compute_staff_positions, graph_from_split, run_rewrite
from src.graph import NodeKind
from src.primitives import Primitive, PrimitiveKind, PrimitiveSplit, SplitStatus
from src.semantics import (
    EIGHTH,
    HALF,
    QUARTER,
    assign_durations,
    assign_pitches,
    validate_measure_durations,
)

STAFF_LINES = (10.0, 20.0, 30.0, 40.0, 50.0)
STAFF_SPACE = 10.0


def filled_head(x0, y0, x1, y1):
    """A solid notehead mask (high fill ratio)."""
    mask = np.zeros((64, 80), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return Primitive(kind=PrimitiveKind.NOTEHEAD, mask=mask, confidence=1.0)


def open_head(x0, y0, x1, y1):
    """A hollow notehead mask: filled box with an empty centre (low fill ratio)."""
    mask = np.zeros((64, 80), dtype=bool)
    mask[y0:y1, x0:x1] = True
    mask[y0 + 2 : y1 - 2, x0 + 2 : x1 - 2] = False
    return Primitive(kind=PrimitiveKind.NOTEHEAD, mask=mask, confidence=1.0)


def stem(x0, y0, x1, y1):
    mask = np.zeros((64, 80), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return Primitive(kind=PrimitiveKind.STEM, mask=mask, confidence=1.0)


def beam(x0, y0, x1, y1):
    mask = np.zeros((64, 80), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return Primitive(kind=PrimitiveKind.BEAM, mask=mask, confidence=1.0)


def build(primitives, head_to_stem=(), stem_to_beam=()):
    split = PrimitiveSplit(
        primitives=primitives,
        head_to_stem=head_to_stem,
        stem_to_beam=stem_to_beam,
        status=SplitStatus.ACCEPTED,
        staff_space=STAFF_SPACE,
        staff_lines=STAFF_LINES,
    )
    graph = graph_from_split(split)
    compute_staff_positions(graph)
    run_rewrite(graph)
    return graph


class DurationTests(unittest.TestCase):
    def test_filled_head_with_stem_is_a_quarter(self):
        graph = build(
            (filled_head(20, 44, 34, 52), stem(32, 20, 34, 46)),
            head_to_stem=((0, 1),),
        )
        assign_durations(graph)
        (note,) = graph.nodes.values()
        self.assertEqual(note.attrs["duration"], QUARTER)

    def test_open_head_with_stem_is_a_half(self):
        graph = build(
            (open_head(20, 40, 34, 54), stem(32, 16, 34, 42)),
            head_to_stem=((0, 1),),
        )
        assign_durations(graph)
        (note,) = graph.nodes.values()
        self.assertEqual(note.attrs["duration"], HALF)

    def test_beamed_notes_are_eighths(self):
        graph = build(
            (
                filled_head(10, 44, 24, 52),
                stem(22, 18, 24, 46),
                filled_head(40, 44, 54, 52),
                stem(52, 18, 54, 46),
                beam(22, 14, 54, 18),
            ),
            head_to_stem=((0, 1), (2, 3)),
            stem_to_beam=((1, 4), (3, 4)),
        )
        assign_durations(graph)
        (group,) = graph.nodes.values()
        self.assertEqual(group.kind, NodeKind.BEAMED_GROUP)
        for member in group.attrs["members"]:
            self.assertEqual(member.attrs["duration"], EIGHTH)


class PitchTests(unittest.TestCase):
    def test_bottom_line_is_e4_in_treble(self):
        # notehead centred on y=50 (bottom line) -> staff_pos 0 -> E4
        graph = build(
            (filled_head(20, 46, 34, 54), stem(32, 22, 34, 48)),
            head_to_stem=((0, 1),),
        )
        assign_pitches(graph, clef="treble")
        (note,) = graph.nodes.values()
        self.assertEqual(note.attrs["pitch"], "E4")

    def test_next_line_up_is_g4(self):
        # centred on y=40 (second line) -> staff_pos 2 -> G4
        graph = build(
            (filled_head(20, 36, 34, 44), stem(32, 12, 34, 40)),
            head_to_stem=((0, 1),),
        )
        assign_pitches(graph, clef="treble")
        (note,) = graph.nodes.values()
        self.assertEqual(note.attrs["pitch"], "G4")

    def test_bass_clef_bottom_line_is_g2(self):
        graph = build(
            (filled_head(20, 46, 34, 54), stem(32, 22, 34, 48)),
            head_to_stem=((0, 1),),
        )
        assign_pitches(graph, clef="bass")
        (note,) = graph.nodes.values()
        self.assertEqual(note.attrs["pitch"], "G2")

    def test_unsupported_clef_raises(self):
        graph = build(
            (filled_head(20, 46, 34, 54), stem(32, 22, 34, 48)),
            head_to_stem=((0, 1),),
        )
        with self.assertRaises(ValueError):
            assign_pitches(graph, clef="tenor")


class ValidationTests(unittest.TestCase):
    def four_quarters(self):
        primitives = []
        head_to_stem = []
        for i in range(4):
            x = 5 + i * 15
            primitives.append(filled_head(x, 44, x + 12, 52))
            primitives.append(stem(x + 10, 20, x + 12, 46))
            head_to_stem.append((2 * i, 2 * i + 1))
        return build(tuple(primitives), head_to_stem=tuple(head_to_stem))

    def test_four_quarters_fill_a_4_4_measure(self):
        graph = self.four_quarters()
        assign_durations(graph)
        check = validate_measure_durations(graph, time_signature=(4, 4))
        self.assertTrue(check.is_valid)
        self.assertAlmostEqual(check.actual, 1.0)
        self.assertAlmostEqual(check.expected, 1.0)

    def test_missing_beat_is_invalid(self):
        graph = build(
            (filled_head(20, 44, 34, 52), stem(32, 20, 34, 46)),
            head_to_stem=((0, 1),),
        )
        assign_durations(graph)
        check = validate_measure_durations(graph, time_signature=(4, 4))
        self.assertFalse(check.is_valid)
        self.assertAlmostEqual(check.actual, 0.25)


if __name__ == "__main__":
    unittest.main()
