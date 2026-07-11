# Automatic Musical Score Analysis

An end-to-end optical music recognition (OMR) project: a printed score image
is preprocessed, split into musical primitives, reconstructed as a notation
graph, assigned pitch and duration, and exported as MIDI.

The project has two implementations of the same pipeline:

- `prototype/`: the Python reference implementation.
- `cpp/`: the C++20 port, using OpenCV only for low-level image primitives,
  nlohmann/json for data interchange, and GoogleTest for tests.

The supported musical scope is deliberately narrow: filled quarter notes and
beamed eighth notes (their noteheads, stems, and beams) on printed treble-clef
pages. Clefs, rests, accidentals, flags, open noteheads, chords, and other
symbols are outside the current scope.

## Pipeline

`binarization -> staff-line removal -> primitive splitting -> notation graph -> pitch and duration -> MIDI`

Each stage has a metric, including mask F1, detection mAP, classification
accuracy, graph-edge F1, duration SER, and pitch SER. The C++ pipeline also
includes inference-free crop discovery, visualization overlays, and a
Python-versus-C++ comparison runner.

## Quick start

### Python reference

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```sh
cd prototype
uv sync
uv run python -m unittest discover -s tests -v
```

### C++ port

Requires CMake 3.20+, a C++20 compiler, OpenCV (`core`, `imgproc`, and
`imgcodecs`), and nlohmann/json. On macOS:

```sh
brew install opencv nlohmann-json
```

Build and test from `cpp/`:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Running the end-to-end pipeline

The DeepScoresV2 dense subset is not included in the repository. Place it at
`prototype/data/ds2_dense/` (or pass another directory with `--data`). The
directory must contain the DeepScores JSON annotations and the corresponding
`images/`, `instance/`, and `segmentation/` directories. Dataset-dependent
tests skip cleanly when it is absent.

With the dataset installed, run the C++ evaluator:

```sh
cd cpp
./build/omr_pipeline_eval --split test --limit 10 --iou-threshold 0.3 \
  --data ../prototype/data/ds2_dense --output output/pipeline_eval
```

To export MIDI for the selected clean pages:

```sh
./build/omr_midi_export --split test --limit 6 \
  --data ../prototype/data/ds2_dense --output output/midi
```

For the corresponding Python benchmark and a controlled implementation
comparison, see [prototype/README.md](prototype/README.md).

## Project layout

```text
cpp/        C++20 pipeline, command-line apps, and GoogleTest suite
prototype/  Python reference, benchmarks, and unit tests
public/     GitLab Pages presentation site
reports/    submitted LaTeX deliverables
```

## Presentation site

The static project presentation lives in `public/`. GitLab CI deploys it with
GitLab Pages from the default branch.
