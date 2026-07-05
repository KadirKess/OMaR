"""Duration, pitch and measure-consistency passes over the notation graph.

Runs after ``assembly``. Scope is noteheads/stems/beams only: no clefs, key
signatures or accidentals, treble clef assumed.
"""

from dataclasses import dataclass

from src.models.graph import NodeKind, NotationGraph

WHOLE = 1.0
HALF = 0.5
QUARTER = 0.25
EIGHTH = 0.125

_LETTERS = "CDEFGAB"

_CLEF_BOTTOM_LINE = {
    "treble": ("E", 4),
    "bass": ("G", 2),
    "alto": ("F", 3),
}


def _iter_note_like(graph: NotationGraph):
    """Yield ``(node, is_beamed)`` for every note/chord, including beamed members."""
    for node in graph.nodes.values():
        if node.kind in (NodeKind.NOTE, NodeKind.CHORD):
            yield node, False
        elif node.kind == NodeKind.BEAMED_GROUP:
            for member in node.attrs.get("members", ()):
                yield member, True


def _first_notehead(node):
    """The representative notehead primitive of a note or chord node."""
    if "notehead" in node.attrs:
        return node.attrs["notehead"]
    noteheads = node.attrs.get("noteheads")
    return noteheads[0] if noteheads else None


def _is_filled(primitive, threshold: float = 0.5) -> bool:
    """Filled (quarter/eighth) vs open (half/whole) from the mask fill ratio."""
    x0, y0, x1, y1 = primitive.bbox
    area = (x1 - x0) * (y1 - y0)
    if area <= 0:
        return False
    return primitive.mask.sum() / area >= threshold


def assign_durations(graph: NotationGraph) -> None:
    """Write ``attrs['duration']`` (whole-note fraction) on every note/chord."""
    for node, is_beamed in _iter_note_like(graph):
        head = _first_notehead(node)
        stem = node.attrs.get("stem")

        if stem is None:
            duration = WHOLE
        elif head is not None and not _is_filled(head):
            duration = HALF
        elif is_beamed:
            duration = EIGHTH
        else:
            duration = QUARTER

        node.attrs["duration"] = duration


def _step_to_pitch(step: int, base_letter: str, base_octave: int) -> str:
    """Map a diatonic step above the clef's bottom line to a pitch like 'G4'."""
    total = _LETTERS.index(base_letter) + step
    letter = _LETTERS[total % 7]
    octave = base_octave + total // 7
    return f"{letter}{octave}"


def assign_pitches(graph: NotationGraph, clef: str = "treble") -> None:
    """Write ``attrs['pitch']`` from each node's ``staff_pos`` under ``clef``.

    Assumes no key signature and no accidentals. Chords get ``attrs['pitches']``
    (a list, low to high) instead.
    """
    base_letter, base_octave = _CLEF_BOTTOM_LINE[clef]

    for node, _ in _iter_note_like(graph):
        if node.kind == NodeKind.CHORD:
            positions = node.attrs.get("staff_positions", ())
            node.attrs["pitches"] = [
                _step_to_pitch(round(p), base_letter, base_octave)
                for p in positions
                if p is not None
            ]
        else:
            staff_pos = node.attrs.get("staff_pos")
            if staff_pos is not None:
                node.attrs["pitch"] = _step_to_pitch(
                    round(staff_pos), base_letter, base_octave
                )


@dataclass(frozen=True)
class MeasureCheck:
    expected: float
    actual: float
    is_valid: bool


def _event_duration(node) -> float:
    """Total duration a top-level node contributes to the measure."""
    if node.kind == NodeKind.BEAMED_GROUP:
        return sum(m.attrs.get("duration", 0.0) for m in node.attrs.get("members", ()))
    return node.attrs.get("duration", 0.0)


def validate_measure_durations(
    graph: NotationGraph, time_signature: tuple[int, int] = (4, 4), tol: float = 1e-6
) -> MeasureCheck:
    """Check that the summed note durations equal one measure.

    ``assign_durations`` must have run first. With no barlines available the
    whole graph is treated as a single measure.
    """
    numerator, denominator = time_signature
    expected = numerator / denominator

    actual = sum(
        _event_duration(node)
        for node in graph.nodes.values()
        if node.kind in (NodeKind.NOTE, NodeKind.CHORD, NodeKind.BEAMED_GROUP)
    )

    return MeasureCheck(
        expected=expected, actual=actual, is_valid=abs(actual - expected) <= tol
    )
