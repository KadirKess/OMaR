import numpy as np
from skimage import color, filters, morphology, feature, transform
from scipy import signal, stats


class Preprocessor:
    def __init__(self, image):
        self.image = image
        self._image_width = image.shape[1]
        self._image_height = image.shape[0]

    def gaussian_filter(self, sigma):
        """
        Applies a Gaussian filter to the image.
        """
        self.image = filters.gaussian(self.image, sigma=sigma)
        return self.image

    def binarization(self, window_size=31, k=0.12):
        """
        Binarizes the image with Sauvola local adaptive thresholding and
        returns it as uint8 (ink = 0, background = 255).

        ``window_size`` (odd) roughly matches the stroke/staff spacing;
        higher ``k`` yields less foreground.
        """
        if self.image.ndim == 3:
            self.image = color.rgb2gray(self.image)

        threshold = filters.threshold_sauvola(self.image, window_size=window_size, k=k)
        self.image = np.where(self.image > threshold, 255, 0).astype(np.uint8)
        return self.image

    def get_skew_angle(self):
        edges = feature.canny(self.image)
        tested_angles = np.deg2rad(np.arange(0.1, 180.0))
        h, theta, d = transform.hough_line(edges, theta=tested_angles)

        accum, angles, dists = transform.hough_line_peaks(h, theta, d)

        most_common_angle = stats.mode(np.around(angles, decimals=2))[0]

        skew_angle = np.rad2deg(most_common_angle - np.pi / 2)
        return skew_angle

    def compute_staff_space_height(self):
        """
        Computes the staff space height of the image and the staff line height of the image.

        Returns a tuple (staff_line_height, staff_space_height).
        """
        cur_black_run = 0
        cur_white_run = 0
        black_runs = dict()
        white_runs = dict()

        for col in range(self.image.shape[1]):
            for row in range(self.image.shape[0]):
                if self.image[row, col] == 0:
                    cur_black_run += 1
                    if cur_white_run > 0:
                        if cur_white_run not in white_runs:
                            white_runs[cur_white_run] = 0
                        white_runs[cur_white_run] += 1
                        cur_white_run = 0
                else:
                    cur_white_run += 1
                    if cur_black_run > 0:
                        if cur_black_run not in black_runs:
                            black_runs[cur_black_run] = 0
                        black_runs[cur_black_run] += 1
                        cur_black_run = 0

        staff_height = max(black_runs, key=black_runs.get)
        staff_space_height = max(white_runs, key=white_runs.get)

        return (staff_height, staff_space_height)

    def staffline_detection_projection(self, frac=0.5):
        """
        Fast staff line detection via the horizontal projection profile
        (Fujinaga, "Optical Music Recognition Using Projections", 1988).

        After deskewing, a staff line spans (nearly) the full width, so it shows
        up as a maximum of the per-row ink count. Returns the sorted row indices
        of the detected lines.
        """
        _, staff_space_height = self.compute_staff_space_height()

        profile = (self.image == 0).sum(axis=1)
        rows, _ = signal.find_peaks(
            profile,
            height=frac * self._image_width,
            distance=max(1, staff_space_height // 2),
        )
        return rows

    def draw_staff(self, rows=None, color=(255, 0, 0)):
        """
        Returns an RGB copy of the current image with the staff lines drawn on
        top. ``rows`` is the sequence of row indices returned by
        :meth:`staffline_detection_projection`.
        """
        if rows is None:
            rows = self.staffline_detection_projection()

        if self.image.ndim == 2:
            canvas = np.stack([self.image] * 3, axis=-1).astype(np.uint8)
        else:
            canvas = self.image[..., :3].astype(np.uint8).copy()

        for row in rows:
            canvas[int(row), :] = color

        return canvas

    def apply(self):
        """
        Applies the preprocessing steps to the image.
        """
        self.gaussian_filter(sigma=0.5)
        self.binarization()

        # Erosion then dilation on an ink=0 image is a closing of the ink:
        # it fills small gaps in strokes without removing thin features.
        footprint = morphology.disk(2)
        self.image = morphology.erosion(self.image, footprint=footprint)
        self.image = morphology.dilation(self.image, footprint=footprint)
