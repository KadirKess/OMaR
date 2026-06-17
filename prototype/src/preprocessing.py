import numpy as np
import skimage
import scipy


class Preprocessor:
    def __init__(self, image):
        self.image = image

    def gaussian_filter(self, sigma):
        """
        Applies a Gaussian filter to the image.
        """
        self.image = skimage.filters.gaussian(self.image, sigma=sigma)
        return self.image

    def binarization(self, window_size=31, k=0.12):
        """
        Binarizes the image with Sauvola local adaptive thresholding and
        returns it as uint8 (ink = 0, background = 255).

        ``window_size`` (odd) roughly matches the stroke/staff spacing;
        higher ``k`` yields less foreground.
        """
        if self.image.ndim == 3:
            self.image = skimage.color.rgb2gray(self.image)

        threshold = skimage.filters.threshold_sauvola(
            self.image, window_size=window_size, k=k
        )
        self.image = np.where(self.image > threshold, 255, 0).astype(np.uint8)
        return self.image

    def get_skew_angle(self, image):
        # convert to edges
        edges = skimage.feature.canny(image)
        # Classic straight-line Hough transform between 0.1 - 180 degrees.
        tested_angles = np.deg2rad(np.arange(0.1, 180.0))
        h, theta, d = skimage.transform.hough_line(edges, theta=tested_angles)

        # find line peaks and angles
        accum, angles, dists = skimage.transform.hough_line_peaks(h, theta, d)

        # round the angles to 2 decimal places and find the most common angle.
        most_common_angle = scipy.stats.mode(np.around(angles, decimals=2))[0]

        # convert the angle to degree for rotation.
        skew_angle = np.rad2deg(most_common_angle - np.pi / 2)
        return skew_angle

    def apply(self):
        """
        Applies the preprocessing steps to the image.
        """
        self.gaussian_filter(sigma=0.5)
        self.binarization()

        # Erosion then dilation on an ink=0 image is a closing of the ink:
        # it fills small gaps in strokes without removing thin features.
        footprint = skimage.morphology.disk(2)
        self.image = skimage.morphology.erosion(self.image, footprint=footprint)
        self.image = skimage.morphology.dilation(self.image, footprint=footprint)
