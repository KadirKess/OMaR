from pathlib import Path
from matplotlib import pyplot as plt
from src.preprocessing import Preprocessor

import skimage


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
    print(f"{len(rows)} staff lines detected at rows: {list(rows)}")
    overlay = pre.draw_staff(rows)

    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(pre.image, cmap="gray")
    plt.title("Binarized")

    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title(f"Projection: {len(rows)} staff lines")

    plt.show()


def main():
    test_staff_detection_projection()


if __name__ == "__main__":
    main()
