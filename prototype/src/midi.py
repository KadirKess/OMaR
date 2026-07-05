"""Encode pipeline output as a monophonic MIDI file.

A '?' pitch in the sequence skips that note.
"""

from __future__ import annotations

from pathlib import Path

import mido

_PITCH_LETTER_OFFSET = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def pitch_token_to_midi(token: str) -> int | None:
    """Convert a token like 'A4' to a MIDI note number, or None if unknown."""
    if not token or token == "?" or len(token) < 2:
        return None
    letter = token[0].upper()
    if letter not in _PITCH_LETTER_OFFSET:
        return None
    try:
        octave = int(token[1:])
    except ValueError:
        return None
    return 12 * (octave + 1) + _PITCH_LETTER_OFFSET[letter]


def duration_token_to_ticks(token: str, ppq: int = 480) -> int | None:
    """Ticks for a duration token at the given pulses-per-quarter, or None."""
    if token == "quarter":
        return ppq
    if token == "eighth":
        return ppq // 2
    return None


def write_midi(
    duration_sequence,
    pitch_sequence,
    out_path: Path,
    ppq: int = 480,
    tempo_bpm: int = 120,
) -> int:
    """Write a monophonic format-0 SMF. Returns the number of notes written.

    Pairs each duration token with the pitch at the same index; skips any pair
    whose pitch is ``'?'`` or unparseable. Notes are sequential (no overlap).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mid = mido.MidiFile(ticks_per_beat=ppq, type=0)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm)))

    written = 0
    for duration_token, pitch_token in zip(duration_sequence, pitch_sequence):
        duration = duration_token_to_ticks(duration_token, ppq)
        if duration is None:
            continue
        note = pitch_token_to_midi(pitch_token)
        if note is None:
            continue
        track.append(
            mido.Message("note_on", note=note, velocity=64, time=0)
        )
        track.append(
            mido.Message("note_off", note=note, velocity=0, time=duration)
        )
        written += 1

    track.append(mido.MetaMessage("end_of_track"))
    mid.save(str(out_path))
    return written
