import argparse
import csv
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import numpy as np

from src.deepscores.dataset import DeepScoresDataset
from src.symbols.splitter import PrimitiveSplitter
from src.symbols.visualization import render_page_overlay


DATA_DIR = Path(__file__).parent.parent / "data" / "ds2_dense"
DEFAULT_OUTPUT = Path(__file__).parent.parent / "output" / "side_by_side"
DEFAULT_CPP_BIN = (
    Path(__file__).parent.parent.parent / "cpp" / "build" / "omr_pipeline_eval"
)


def _render_python(dataset, splitter, image_id, out_png):
    """Render the Python full-page overlay for one page."""
    samples = dataset.inference_page_compounds(image_id)
    original, staffless, metadata = dataset.load_page(image_id)
    results = [
        (
            sample,
            splitter.split(
                sample.staffless_mask,
                sample.staff_space,
                sample.staff_line,
                sample.original_mask,
            ),
        )
        for sample in samples
    ]
    render_page_overlay(staffless, results, out_png, metadata["filename"])


def _render_cpp(cpp_bin, image_id, band, split, iou_threshold, data_dir, viz_dir):
    """Invoke the C++ eval on a one-page manifest to dump its overlay PNG."""
    viz_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        manifest = tmp / "manifest.tsv"
        with manifest.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["band", "image_id", "python_n_crops"])
            writer.writerow([band, str(image_id), 0])
        cmd = [
            str(cpp_bin),
            "--split",
            split,
            "--iou-threshold",
            str(iou_threshold),
            "--data",
            str(data_dir),
            "--page-ids-file",
            str(manifest),
            "--viz-dir",
            str(viz_dir),
            "--output",
            str(tmp / "out"),
        ]
        print("[cpp]", " ".join(cmd))
        subprocess.run(cmd, check=True)
    return viz_dir / f"{image_id}.png"


def _label(image, text):
    """Return a copy of image with a titled header band on top."""
    width = image.width
    font_size = max(24, width // 40)
    band_h = font_size * 2
    font = ImageFont.load_default(size=font_size)
    canvas = Image.new("RGB", (width, image.height + band_h), "white")
    canvas.paste(image, (0, band_h))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, width, band_h], fill=(30, 30, 30))
    draw.text((band_h // 3, band_h // 2), text, fill="white", anchor="lm", font=font)
    return canvas


def _stitch(py_png, cpp_png, out_png, image_id):
    left = Image.open(py_png).convert("RGB")
    right = Image.open(cpp_png).convert("RGB")
    # Scale both to a common height (the taller of the two).
    height = max(left.height, right.height)
    if left.height != height:
        left = left.resize((round(left.width * height / left.height), height))
    if right.height != height:
        right = right.resize((round(right.width * height / right.height), height))
    left = _label(left, f"PYTHON  -  page {image_id}")
    right = _label(right, f"C++  -  page {image_id}")
    gap = 16
    canvas = Image.new(
        "RGB",
        (left.width + gap + right.width, max(left.height, right.height)),
        "white",
    )
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + gap, 0))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image_id",
        nargs="?",
        default=None,
        help="DeepScores image id to render; if omitted, --seed picks one",
    )
    parser.add_argument("--seed", type=int, default=None, help="random page seed")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--cpp-bin", type=Path, default=DEFAULT_CPP_BIN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.cpp_bin.exists():
        parser.error(f"cpp binary not found: {args.cpp_bin} (pass --cpp-bin)")

    dataset = DeepScoresDataset(args.data, split=args.split)

    if args.image_id is None:
        image_ids = sorted(dataset.images, key=int)
        if not image_ids:
            parser.error(f"no images in split '{args.split}'")
        rng = np.random.default_rng(args.seed)
        image_id = rng.choice(image_ids)
        print(f"[pick] seed={args.seed} -> image_id {image_id}")
    else:
        image_id = str(args.image_id)

    if image_id not in dataset.images:
        parser.error(f"image id {image_id} not found in split '{args.split}'")
    splitter = PrimitiveSplitter()

    args.output.mkdir(parents=True, exist_ok=True)
    py_png = args.output / f"py_{image_id}.png"
    band = Path(dataset.images[image_id]["filename"]).stem

    print(f"[python] rendering page {image_id}")
    _render_python(dataset, splitter, image_id, py_png)
    cpp_png = _render_cpp(
        args.cpp_bin,
        image_id,
        band,
        args.split,
        args.iou_threshold,
        args.data,
        args.output / "cpp_viz",
    )
    if not cpp_png.exists():
        parser.error(
            f"cpp did not produce {cpp_png}; the page may have yielded no crops"
        )

    out_png = args.output / f"side_by_side_{image_id}.png"
    _stitch(py_png, cpp_png, out_png, image_id)
    print(f"\nwrote {out_png}")


if __name__ == "__main__":
    main()
