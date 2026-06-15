"""
Quality benchmark for the binarization preprocessing step.

Runs :class:`Preprocessor` over every ``N.bmp`` / ``N_gt.bmp`` pair in
``data/binarisation`` and reports precision, recall and F-measure per image
plus the average.

Usage::

    uv run python -m benchmarks.binarization_eval
"""

from pathlib import Path

import skimage

from src.metrics import scores
from src.preprocessing import Preprocessor

DATA_DIR = Path(__file__).parent.parent / "data" / "binarisation"


def find_pairs(data_dir):
    """Yields ``(name, input_path, gt_path)`` for each input/GT image pair."""
    for gt_path in sorted(data_dir.glob("*_gt.bmp")):
        name = gt_path.name[: -len("_gt.bmp")]
        input_path = data_dir / f"{name}.bmp"
        if input_path.exists():
            yield name, input_path, gt_path


def evaluate_pair(input_path, gt_path):
    img = skimage.io.imread(input_path)
    gt = skimage.io.imread(gt_path)

    pre = Preprocessor(img)
    pre.apply()

    return scores(pre.image, gt)


def main():
    results = []
    header = (
        f"{'image':>8} {'precision':>10} {'recall':>10} "
        f"{'f_measure':>10} {'psnr':>8} {'drd':>8}"
    )
    print(header)
    print("-" * len(header))

    for name, input_path, gt_path in find_pairs(DATA_DIR):
        s = evaluate_pair(input_path, gt_path)
        results.append(s)
        print(
            f"{name:>8} {s['precision']:>10.4f} {s['recall']:>10.4f} "
            f"{s['f_measure']:>10.4f} {s['psnr']:>8.2f} {s['drd']:>8.2f}"
        )

    if not results:
        print("no image pairs found in", DATA_DIR)
        return

    n = len(results)

    def mean(key):
        return sum(r[key] for r in results) / n

    print("-" * len(header))
    print(
        f"{'mean':>8} {mean('precision'):>10.4f} {mean('recall'):>10.4f} "
        f"{mean('f_measure'):>10.4f} {mean('psnr'):>8.2f} {mean('drd'):>8.2f}"
    )


if __name__ == "__main__":
    main()
