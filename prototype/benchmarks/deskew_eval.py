"""
Quality benchmark for the deskew (skew-angle estimation) preprocessing step.

Runs :meth:`Preprocessor.get_skew_angle` over every ``{id}_{angle}.png`` image
in ``data/hough_transform`` and reports the residual angular error per image
against the known applied rotation, plus the mean and max.

The ground-truth rotation is encoded in the filename: ``5_13.45.png`` was
rotated by ``+13.45`` degrees.

Usage::

    uv run python -m benchmarks.deskew_eval
"""

from pathlib import Path

import numpy as np
import skimage

from src.metrics import angular_error
from src.preprocessing import Preprocessor

DATA_DIR = Path(__file__).parent.parent / "data" / "hough_transform"

# Below this residual the deskew is considered excellent (see evaluation.pdf).
GOOD_THRESHOLD_DEG = 0.5


def to_gray(image):
    """Converts an RGB or RGBA image to single-channel grayscale."""
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = skimage.color.rgba2rgb(image)
        image = skimage.color.rgb2gray(image)
    return image


def find_images(data_dir):
    """Yields ``(name, path, true_angle)`` for each ``{id}_{angle}.png``."""
    for path in sorted(data_dir.glob("*.png")):
        name = path.stem
        try:
            true_angle = float(name.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            continue
        yield name, path, true_angle


def evaluate_one(path, true_angle):
    img = skimage.io.imread(path)

    pre = Preprocessor(img)
    estimated = float(np.ravel(pre.get_skew_angle(to_gray(img)))[0])

    return estimated, angular_error(estimated, true_angle)


def main():
    results = []
    header = f"{'image':>10} {'true_deg':>10} {'est_deg':>10} {'abs_err':>10}"
    print(header)
    print("-" * len(header))

    for name, path, true_angle in find_images(DATA_DIR):
        estimated, error = evaluate_one(path, true_angle)
        results.append(error)
        print(
            f"{name:>10} {true_angle:>10.2f} {estimated:>10.2f} {error:>10.4f}"
        )

    if not results:
        print("no images found in", DATA_DIR)
        return

    n = len(results)
    mean_err = sum(results) / n
    max_err = max(results)

    print("-" * len(header))
    print(f"{'mean':>10} {'':>10} {'':>10} {mean_err:>10.4f}")
    print(f"{'max':>10} {'':>10} {'':>10} {max_err:>10.4f}")

    verdict = "OK" if mean_err < GOOD_THRESHOLD_DEG else "ABOVE"
    print(
        f"\nmean angular error {mean_err:.4f} deg "
        f"({verdict} threshold {GOOD_THRESHOLD_DEG} deg)"
    )


if __name__ == "__main__":
    main()
