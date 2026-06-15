import numpy as np
import skimage


class Preprocessor:
    def __init__(self, image):
        self.image = image

    def gaussian_filter(self, sigma):
        """
        Applies a Gaussian filter to the image.
        """
        self.image = skimage.filters.gaussian(self.image, sigma=sigma)
        return self.image

    def binarization(self, window_size=11, k=0.08):
        """
        Binarizes the image using Sauvola's local adaptive thresholding.

        Sauvola computes a per-pixel threshold from the local mean and
        standard deviation, which keeps flat background regions clean
        instead of turning paper texture into noise (as tiled Otsu does).

        ``window_size`` (odd) should roughly match the stroke/staff
        spacing; higher ``k`` yields less foreground.
        """
        if self.image.ndim == 3:
            self.image = skimage.color.rgb2gray(self.image)

        threshold = skimage.filters.threshold_sauvola(
            self.image, window_size=window_size, k=k
        )
        self.image = self.image > threshold

    def apply(self):
        """
        Applies the preprocessing steps to the image.
        """
        self.gaussian_filter(sigma=1)
        self.binarization()

        footprint = skimage.morphology.disk(2)
        self.image = skimage.morphology.erosion(self.image, footprint=footprint)
        self.image = skimage.morphology.dilation(self.image, footprint=footprint)
