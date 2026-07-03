# Full-page primitive splitting diagnostics

Generate complete score-page panels from `prototype/` with:

```sh
uv run python -m benchmarks.render_splitting_pages --pages 2
```

Each PNG shows the original and staffless page, connected components, primitive
masks, unknown residuals, and predicted associations.
