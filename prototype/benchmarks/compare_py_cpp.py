import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from src.deepscores.dataset import DeepScoresDataset


DATA_DIR = Path(__file__).parent.parent / "data" / "ds2_dense"
DEFAULT_OUTPUT = Path(__file__).parent.parent / "output" / "compare_py_cpp"
DEFAULT_CPP_BIN = (
    Path(__file__).parent.parent.parent / "cpp" / "build" / "omr_pipeline_eval"
)

# Scalar per-page metrics emitted for both implementations as py_/cpp_ columns.
SCALAR_METRICS = (
    "n_crops",
    "n_pred",
    "n_gt",
    "mAP",
    "accuracy",
    "edge_precision",
    "edge_recall",
    "edge_f1",
    "ser",
    "pitch_ser",
    "coverage",
    "covered_gt",
    "duplicate_covered_gt",
    "useful_crops",
    "irrelevant_crops",
)
# Nested mask_prf1[kind][field] metrics flattened to <kind>_<field>.
MASK_METRICS = (
    ("notehead", "f1"),
    ("notehead", "iou"),
    ("stem", "f1"),
    ("beam", "f1"),
)
# Timing fields (milliseconds) added by pipeline_eval, whole pipeline incl. I/O.
TIMING_METRICS = ("discover_ms", "eval_ms", "elapsed_ms")


def _read_manifest(path):
    """Read a TSV manifest (band, image_id, python_n_crops), preserving order."""
    entries = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            entries.append(
                {
                    "band": row.get("band", ""),
                    "image_id": str(row["image_id"]),
                    "python_n_crops": int(row.get("python_n_crops", 0) or 0),
                }
            )
    return entries


def _band_from_filename(filename):
    """Best-effort engraving/font tag, e.g. '...-aug-emmentaler--page-4.png'."""
    stem = Path(filename).stem
    if "-aug-" in stem:
        tail = stem.split("-aug-", 1)[1]
        return tail.split("--", 1)[0].rstrip("-")
    return ""


def _build_manifest(dataset, limit, seed):
    """Select a deterministic page set and return manifest rows."""
    image_ids = sorted(dataset.images, key=int)
    if seed is not None:
        rng = np.random.default_rng(seed)
        rng.shuffle(image_ids)
    image_ids = image_ids[: max(0, limit)]
    return [
        {
            "band": _band_from_filename(dataset.images[i]["filename"]),
            "image_id": str(i),
            "python_n_crops": 0,
        }
        for i in image_ids
    ]


def _write_manifest(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["band", "image_id", "python_n_crops"])
        for r in rows:
            writer.writerow([r["band"], r["image_id"], r["python_n_crops"]])


def _run_python(manifest_path, out_dir, iou_threshold, split):
    cmd = [
        sys.executable,
        "-m",
        "benchmarks.pipeline_eval",
        "--split",
        split,
        "--iou-threshold",
        str(iou_threshold),
        "--page-ids-file",
        str(manifest_path),
        "--write-json-pages",
        "--output",
        str(out_dir),
    ]
    print("[python]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=Path(__file__).parent.parent)


def _run_cpp(cpp_bin, manifest_path, out_dir, iou_threshold, split, data_dir):
    cmd = [
        str(cpp_bin),
        "--split",
        split,
        "--iou-threshold",
        str(iou_threshold),
        "--data",
        str(data_dir),
        "--page-ids-file",
        str(manifest_path),
        "--write-json-pages",
        "--output",
        str(out_dir),
    ]
    print("[cpp]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _load_page(out_dir, image_id):
    path = out_dir / f"page_{image_id}.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _get_mask(page, kind, field):
    entry = (page.get("mask_prf1") or {}).get(kind)
    if not entry:
        return ""
    return entry.get(field, "")


def _columns():
    cols = ["image_id", "band"]
    for m in SCALAR_METRICS:
        cols += [f"py_{m}", f"cpp_{m}"]
    for kind, field in MASK_METRICS:
        name = f"{kind}_{field}"
        cols += [f"py_{name}", f"cpp_{name}"]
    for m in TIMING_METRICS:
        cols += [f"py_{m}", f"cpp_{m}"]
    cols.append("speedup")
    return cols


def _row(image_id, band, py, cpp):
    row = {"image_id": image_id, "band": band}
    py = py or {}
    cpp = cpp or {}
    for m in SCALAR_METRICS:
        row[f"py_{m}"] = py.get(m, "")
        row[f"cpp_{m}"] = cpp.get(m, "")
    for kind, field in MASK_METRICS:
        name = f"{kind}_{field}"
        row[f"py_{name}"] = _get_mask(py, kind, field)
        row[f"cpp_{name}"] = _get_mask(cpp, kind, field)
    for m in TIMING_METRICS:
        row[f"py_{m}"] = py.get(m, "")
        row[f"cpp_{m}"] = cpp.get(m, "")
    py_e = py.get("elapsed_ms")
    cpp_e = cpp.get("elapsed_ms")
    row["speedup"] = (py_e / cpp_e) if (py_e and cpp_e) else ""
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--page-ids-file",
        type=Path,
        default=None,
        help="TSV manifest (band, image_id, python_n_crops); overrides "
        "--limit/--seed selection so both runs share exactly these pages",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--cpp-bin", type=Path, default=DEFAULT_CPP_BIN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip-python", action="store_true", help="reuse existing PY page JSON"
    )
    parser.add_argument(
        "--skip-cpp", action="store_true", help="reuse existing CPP page JSON"
    )
    args = parser.parse_args()

    if not args.cpp_bin.exists():
        parser.error(
            f"cpp binary not found: {args.cpp_bin}\n"
            "build it first, e.g. `cmake --build cpp/build --target "
            "omr_pipeline_eval`, or pass --cpp-bin"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    dataset = DeepScoresDataset(args.data, split=args.split)

    if args.page_ids_file is not None:
        rows = _read_manifest(args.page_ids_file)
        manifest_path = args.page_ids_file
    else:
        rows = _build_manifest(dataset, args.limit, args.seed)
        manifest_path = args.output / "manifest.tsv"
        _write_manifest(rows, manifest_path)
    print(f"comparing {len(rows)} pages (manifest: {manifest_path})")

    py_dir = args.output / "python"
    cpp_dir = args.output / "cpp"
    if not args.skip_python:
        _run_python(manifest_path, py_dir, args.iou_threshold, args.split)
    if not args.skip_cpp:
        _run_cpp(
            args.cpp_bin,
            manifest_path,
            cpp_dir,
            args.iou_threshold,
            args.split,
            args.data,
        )

    csv_rows = []
    py_total = cpp_total = 0.0
    for entry in rows:
        image_id = entry["image_id"]
        py = _load_page(py_dir, image_id)
        cpp = _load_page(cpp_dir, image_id)
        csv_rows.append(_row(image_id, entry.get("band", ""), py, cpp))
        if py and py.get("elapsed_ms"):
            py_total += py["elapsed_ms"]
        if cpp and cpp.get("elapsed_ms"):
            cpp_total += cpp["elapsed_ms"]

    csv_path = args.output / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_columns())
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nper-page comparison written to {csv_path}")
    print(f"=== timing over {len(rows)} pages (whole pipeline, incl. I/O) ===")
    print(f"  python total: {py_total / 1000.0:8.3f} s")
    print(f"  cpp total:    {cpp_total / 1000.0:8.3f} s")
    if cpp_total > 0:
        print(
            f"  speedup:      {py_total / cpp_total:8.3f}x "
            f"({'cpp faster' if py_total > cpp_total else 'python faster'})"
        )


if __name__ == "__main__":
    main()
