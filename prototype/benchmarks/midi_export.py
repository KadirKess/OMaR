"""Export pipeline predictions for hand-picked clean DeepScores pages to MIDI.

The page IDs below were selected by scanning the dense DeepScores test split
for pages whose only notational symbols are blacks and croches -- no
accidentals, no open/half/whole noteheads, no chords (verified via the
dataset's own proximity graph in `iter_supported_crops`), and a single clef
family per page so the hardcoded treble clef in `assign_pitches` is correct.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmarks.pipeline_eval import DATA_DIR, _process_crop, _recover_staff_rows
from src.deepscores.dataset import DeepScoresDataset
from src.deepscores.io import build_predicted_page
from src.midi import write_midi
from src.symbols.splitter import PrimitiveSplitter

CLEAN_PAGE_IDS = ("1339", "1316", "966", "1257", "84", "1317")

DEFAULT_OUTPUT = Path(__file__).parent.parent / "output" / "midi"
DEFAULT_TEMPO_BPM = 120


def _predict_page(dataset, splitter, image_id, samples):
    image = dataset.images[str(image_id)]
    image_path = DATA_DIR / "images" / image["filename"]
    staff_rows = _recover_staff_rows(image_path) if image_path.exists() else ()
    staff_line = (
        max(1.0, round(0.15 * float(np.median([s.staff_line for s in samples]))))
        if samples
        else 2.0
    )

    crop_results = []
    staff_lines_per_crop = []
    for sample in samples:
        result, graph, system_lines = _process_crop(
            splitter, sample, staff_rows, staff_line
        )
        crop_results.append((result, graph, sample.origin))
        staff_lines_per_crop.append(system_lines)

    return build_predicted_page(
        image, crop_results, staff_lines_per_crop, staff_line=staff_line
    )


def main():
    parser = argparse.ArgumentParser(
        description="Export clean DeepScores pages to MIDI"
    )
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--limit", type=int, default=len(CLEAN_PAGE_IDS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tempo", type=int, default=DEFAULT_TEMPO_BPM)
    args = parser.parse_args()

    dataset = DeepScoresDataset(DATA_DIR, split=args.split)
    splitter = PrimitiveSplitter()

    grouped: dict[str, list] = defaultdict(list)
    wanted = set(CLEAN_PAGE_IDS[: args.limit])
    for sample in dataset.iter_supported_crops():
        if sample.image_id in wanted:
            grouped[sample.image_id].append(sample)

    args.output.mkdir(parents=True, exist_ok=True)
    for image_id in CLEAN_PAGE_IDS[: args.limit]:
        samples = grouped.get(image_id, [])
        if not samples:
            print(f"[skip] image {image_id}: no supported crops")
            continue
        try:
            page = _predict_page(dataset, splitter, image_id, samples)
        except Exception as exc:
            print(f"[skip] image {image_id}: {exc}")
            continue

        mid_path = args.output / f"{image_id}.mid"
        written = write_midi(
            page.duration_sequence,
            page.pitch_sequence,
            mid_path,
            tempo_bpm=args.tempo,
        )
        sidecar = {
            "image_id": image_id,
            "filename": page.filename,
            "n_crops": len(samples),
            "duration_sequence": list(page.duration_sequence),
            "pitch_sequence": list(page.pitch_sequence),
            "midi_file": str(mid_path),
            "notes_written": written,
            "tempo_bpm": args.tempo,
        }
        with (args.output / f"{image_id}.json").open("w", encoding="utf-8") as f:
            json.dump(sidecar, f, indent=2)
        print(
            f"image {image_id}: {written} notes, {len(page.duration_sequence)} tokens "
            f"-> {mid_path.name}"
        )


if __name__ == "__main__":
    main()
