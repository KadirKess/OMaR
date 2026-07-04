# Prototype

End-to-end OMR prototype for the DeepScores dataset: binarize and deskew score
images, detect staff geometry by projection, remove staff lines, split raster
ink into primitives (noteheads, stems, beams), rewrite them into a notation
graph, and assign each note a duration and pitch. Each stage ships its own
metric so a regression can be localized.

## Scope

The benchmark is restricted to **blacks and croches only** (filled quarter
noteheads and beamed eighth notes, plus their stems and beams). Clefs, key
signatures, accidentals, rests, flags, open/half/whole noteheads, and small
(grace) heads are out of scope and excluded from both predictions and the
ground truth.

## Layout

```
src/
  models/            shared data types (primitives, notation graph)
  preprocessing/     stage 1: binarize, deskew, staff handling, DIBCO metrics
  symbols/           stage 2: primitive splitting, mask evaluation, visualization
  deepscores/        dataset loader + DeepScores-format I/O
  reconstruction/    stage 3: assembly (graph rewrite) + semantics (duration/pitch)
benchmarks/
  _metrics.py        pure detection/classification/SER metrics
  pipeline_eval.py  end-to-end staged benchmark
tests/               unit tests for splitting, assembly, semantics, metrics, I/O
```

Run the test suite with:

```sh
uv run python -m unittest discover -s tests -v
```

## Primitive splitting

The splitter currently supports:

- filled noteheads (small/grace heads excluded);
- one stem per notehead;
- either one isolated quarter note or a group joined by one beam;
- no chords, open noteheads, multiple beams, flags, rests, clefs, or accidentals.

Other foreground remains `unknown`. Results are `accepted`, `uncertain` when
unknown ink remains, or `rejected` when the topology is unsupported.

### Splitting evaluation

DeepScores semantic annotations select benchmark crops, remove annotated staff
pixels, and provide ground-truth masks. The splitter itself receives raster ink.
Negative crops cover open noteheads, accidentals, clefs, rests, and flags.

```sh
uv run python -m unittest discover -s tests -v
uv run python -m benchmarks.splitting_eval --limit 100 --unknown-limit 100
uv run python -m benchmarks.render_splitting_pages --pages 2
```

The page renderer writes PNG diagnostics to `output/splitting_pages/`.

## End-to-end staged benchmark

`benchmarks/pipeline_eval.py` runs the full pipeline (preprocess -> split ->
assemble -> assign durations) on each DeepScores test page and reports one
metric per stage so a regression can be localized:

| Metric | Stage | What it localizes |
|---|---|---|
| Mask P/R/F1 @IoU | 5 splitting | pixel-level primitive segmentation quality |
| mAP @IoU | 5 detection | bbox-level detection (see note on thin classes) |
| Accuracy + confusion matrix | 5 classification | notehead OnLine/InSpace + stem + beam |
| Edge P/R/F1 | 6 reconstruction | head-to-stem and stem-to-beam pairing |
| SER (duration) | end-to-end | quarter/eighth sequence vs. ground truth |

The ground truth is filtered to the crops actually processed, so mAP/accuracy
measure only the symbols the pipeline attempted (otherwise recall would be
capped near zero on long pages).

**Thin-class caveat.** DeepScores annotates stems as 1-pixel-wide boxes. At
IoU=0.5 a 4-pixel-wide predicted stem can't score above ~0.25 IoU even when
correct, so stem mAP is structurally deflated. Notehead mAP (wide shapes) is
the reliable signal. Pass `--iou-threshold 0.3` for a more lenient view.

```sh
uv run python -m benchmarks.pipeline_eval --split test --limit 50
uv run python -m benchmarks.pipeline_eval --split test --limit 50 \
    --iou-threshold 0.3 --write-json-pages --output output/pipeline_eval
```

The stdout table and a JSON report land in `output/pipeline_eval/report.json`
by default (one `page_<id>.json` per page with `--write-json-pages`).

## MIDI export

`benchmarks/midi_export.py` runs the pipeline on a fixed set of clean
DeepScores test pages and writes a monophonic MIDI file per page. The page
IDs (`CLEAN_PAGE_IDS` in the script) were selected by scanning the dense
annotations for pages whose only notational symbols are blacks and croches
-- no accidentals, no open/half/whole noteheads, no chords, and a single
clef family per page (treble) so the hardcoded clef in `assign_pitches` is
correct.

```sh
uv run python -m benchmarks.midi_export --limit 6
```

Each page produces `output/midi/<image_id>.mid` (format-0 SMF, 480 PPQ,
120 BPM) and a sidecar `<image_id>.json` with the duration and pitch token
sequences the pipeline emitted.

