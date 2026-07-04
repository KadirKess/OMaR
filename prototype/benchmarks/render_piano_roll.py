"""Render a piano-roll PNG from a MIDI file for visual verification."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mido


def render_piano_roll(mid_path: Path, out_path: Path) -> None:
    mid = mido.MidiFile(str(mid_path))
    track = mid.tracks[0]

    abs_tick = 0
    active: dict[int, int] = {}
    segments: list[tuple[int, int, int]] = []
    for msg in track:
        abs_tick += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            active[msg.note] = abs_tick
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            start = active.pop(msg.note, None)
            if start is not None:
                segments.append((start, abs_tick, msg.note))

    if not segments:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    for start, end, note in segments:
        ax.barh(note, end - start, left=start, height=0.8, color="black")
    ax.set_xlabel("tick (ppq=480, quarter=480, eighth=240)")
    ax.set_ylabel("MIDI note")
    ax.set_title(f"{mid_path.stem} piano roll")
    notes = [s[2] for s in segments]
    ax.set_ylim(min(notes) - 2, max(notes) + 2)
    ax.set_yticks(range(min(notes), max(notes) + 1))
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    midi_dir = Path(__file__).parent.parent / "output" / "midi"
    for mid in sorted(midi_dir.glob("*.mid")):
        out = midi_dir / f"{mid.stem}_pianoroll.png"
        render_piano_roll(mid, out)
        print(f"{mid.name} -> {out.name}")
