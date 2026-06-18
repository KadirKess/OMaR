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
        edges = skimage.feature.canny(image)
        tested_angles = np.deg2rad(np.arange(0.1, 180.0))
        h, theta, d = skimage.transform.hough_line(edges, theta=tested_angles)

        accum, angles, dists = skimage.transform.hough_line_peaks(h, theta, d)

        most_common_angle = scipy.stats.mode(np.around(angles, decimals=2))[0]

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

    """
    StaffLine_Detection(IMAGE, CORRCOEF)
    {
        STAFFSPACE_HEIGHT = computeStaffSpaceHeight();
        BLACK_PERC = 0.75;
        BLACK_RUN = STAFFSPACE_HEIGHT;
        STRIP_HEIGHT = STAFFSPACE_HEIGHT;
        for (int row = 0; row < ImageHeight; row++)
        {
        Point2D start(0, row);
        Point2D end(ImageWidth-1, row);
        Path path = findShortestPath (start, end);
        if(blackPercentage(path) < BLACK_PERC)
        continue;
        path = trimPath(path);
        if(corrcoef(path) < CORRCOEF)
        continue;
        addPathToSetOfStaffLines(path);
        }
    }
    """

    def staffline_detection(self, corrcoef):
        pass

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
