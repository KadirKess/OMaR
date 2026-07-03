# Prototype

The preprocessing pipeline binarizes and deskews score images, detects staff
geometry by projection, and removes staff lines.

## Primitive splitting

The splitter currently supports:

- filled noteheads (small/grace heads excluded);
- one stem per notehead;
- either one isolated quarter note or a group joined by one beam;
- no chords, open noteheads, multiple beams, flags, rests, clefs, or accidentals.

Other foreground remains `unknown`. Results are `accepted`, `uncertain` when
unknown ink remains, or `rejected` when the topology is unsupported.

### Evaluation

DeepScores semantic annotations select benchmark crops, remove annotated staff
pixels, and provide ground-truth masks. The splitter itself receives raster ink.
Negative crops cover open noteheads, accidentals, clefs, rests, and flags.

```sh
uv run python -m unittest discover -s tests -v
uv run python -m benchmarks.splitting_eval --limit 100 --unknown-limit 100
uv run python -m benchmarks.render_splitting_pages --pages 2
```

The page renderer writes PNG diagnostics to `output/splitting_pages/`.
