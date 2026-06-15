from pathlib import Path
from matplotlib import pyplot as plt
from src.preprocessing import Preprocessor

import skimage


DATA_DIR = Path(__file__).parent / "data" / "binarisation"


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


def main():
    test_preprocessing()


if __name__ == "__main__":
    main()
