from pathlib import Path
from matplotlib import pyplot as plt
from src.preprocessing.preprocessor import Preprocessor

import skimage
from skimage import morphology

DATA_DIR = Path(__file__).parent / "data" / "binarisation"
SCORE_DIR = Path(__file__).parent / "data" / "ds2_dense" / "images"


def test_preprocessing():
    img = skimage.io.imread(DATA_DIR / "1.bmp")
    gt = skimage.io.imread(DATA_DIR / "1_gt.bmp")
    pre = Preprocessor(img)
    pre.apply()

    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original")

    plt.subplot(1, 2, 2)
    plt.imshow(pre.image, cmap="gray")
    plt.title("Binarized")

    plt.show()


def test_staff_detection_projection():
    img = skimage.io.imread(SCORE_DIR / "lg-169480697-aug-gonville--page-8.png")

    pre = Preprocessor(img)
    pre.apply()
    rows = pre.staffline_detection_projection(frac=0.5)
    cols = pre.barline_detection_projection(frac=0.2)
    print(f"{len(rows)} staff lines detected at rows: {list(rows)}")
    print(f"{len(cols)} barlines detected at columns: {list(cols)}")
    overlay = pre.draw_staff(rows)
    overlay = pre.draw_barlines(cols, canvas=overlay)

    pre.remove_barlines(cols, rows)
    cleaned = pre.remove_stafflines(rows).copy()

    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(overlay)
    plt.title(f"Projection: {len(rows)} staff lines, {len(cols)} barlines detected")

    plt.subplot(1, 2, 2)
    plt.imshow(cleaned, cmap="gray")
    plt.title("Staff lines and barlines removed")

    plt.show()


def main():
    test_staff_detection_projection()


if __name__ == "__main__":
    main()
