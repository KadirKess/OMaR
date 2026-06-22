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

    def barline_detection_projection(self, frac=0.5):
        """
        Measure-divider (barline) detection via the vertical projection profile,
        the column-wise dual of :meth:`staffline_detection_projection`.
        """
        _, staff_space_height = self.compute_staff_space_height()

        profile = (self.image == 0).sum(axis=0)
        cols, _ = signal.find_peaks(
            profile,
            height=frac * self._image_height,
            distance=max(1, staff_space_height // 2),
        )
        return cols

    def _group_staves(self, rows):
        """
        Groups staff line rows into ``(top_row, bottom_row)`` system extents.
        """
        _, staff_space_height = self.compute_staff_space_height()
        rows = sorted(int(r) for r in rows)
        if not rows:
            return []

        systems = []
        group = [rows[0]]
        for r in rows[1:]:
            if r - group[-1] > 2 * staff_space_height:
                systems.append((group[0], group[-1]))
                group = [r]
            else:
                group.append(r)
        systems.append((group[0], group[-1]))
        return systems

    def _blank_staff_spanning_runs(self, col, systems, tol):
        """
        Blanks the vertical ink runs in column ``col`` that span a full staff
        (within ``tol`` pixels); stems never reach both extremes so they survive.
        Returns True if any run was blanked.
        """
        column = self.image[:, col]
        height = self.image.shape[0]
        blanked = False

        def spans_staff(run_top, run_bottom):
            return any(
                run_top <= top + tol and run_bottom >= bottom - tol
                for top, bottom in systems
            )

        run_start = None
        for row in range(height):
            if column[row] == 0 and run_start is None:
                run_start = row
            elif column[row] != 0 and run_start is not None:
                if spans_staff(run_start, row - 1):
                    column[run_start:row] = 255
                    blanked = True
                run_start = None
        if run_start is not None and spans_staff(run_start, height - 1):
            column[run_start:] = 255
            blanked = True
        return blanked

    def remove_barlines(self, cols=None, rows=None, tol=None):
        """
        Erases the detected barlines while keeping note stems. Only blanks the
        vertical ink runs that span a full staff (within ``tol``), growing left
        and right from each peak column to cover the barline thickness. Mutates
        and returns ``self.image``.
        """
        if cols is None:
            cols = self.barline_detection_projection()
        if rows is None:
            rows = self.staffline_detection_projection()

        staff_line_height, staff_space_height = self.compute_staff_space_height()
        if tol is None:
            tol = staff_space_height // 2

        max_growth = max(2, 3 * staff_line_height)

        systems = self._group_staves(rows)
        width = self.image.shape[1]
        for col in cols:
            col = int(col)
            self._blank_staff_spanning_runs(col, systems, tol)

            right = col + 1
            while (
                right < width
                and right - col <= max_growth
                and self._blank_staff_spanning_runs(right, systems, tol)
            ):
                right += 1

            left = col - 1
            while (
                left >= 0
                and col - left <= max_growth
                and self._blank_staff_spanning_runs(left, systems, tol)
            ):
                left -= 1

        return self.image

    def remove_stafflines(self, rows=None, max_thickness=None):
        """
        Erases the detected staff lines while keeping notes that cross them. At
        each row, blanks the vertical ink run through every column only when it
        is no taller than ``max_thickness`` (default twice the staff line
        height). Mutates and returns ``self.image``.
        """
        if rows is None:
            rows = self.staffline_detection_projection()

        staff_line_height, _ = self.compute_staff_space_height()
        if max_thickness is None:
            max_thickness = 2 * staff_line_height

        height = self.image.shape[0]
        width = self.image.shape[1]
        for row in rows:
            row = int(row)
            for col in range(width):
                if self.image[row, col] != 0:
                    continue
                top = row
                while top - 1 >= 0 and self.image[top - 1, col] == 0:
                    top -= 1
                bottom = row
                while bottom + 1 < height and self.image[bottom + 1, col] == 0:
                    bottom += 1
                if bottom - top + 1 <= max_thickness:
                    self.image[top : bottom + 1, col] = 255

        return self.image

    def _rgb_canvas(self):
        """
        Returns an RGB uint8 copy of the current image for overlays.
        """
        if self.image.ndim == 2:
            return np.stack([self.image] * 3, axis=-1).astype(np.uint8)
        return self.image[..., :3].astype(np.uint8).copy()

    def draw_staff(self, rows=None, color=(255, 0, 0), canvas=None):
        """
        Returns an RGB copy with the staff lines drawn on top. Pass ``canvas``
        to draw onto an existing overlay instead of a fresh copy.
        """
        if rows is None:
            rows = self.staffline_detection_projection()

        if canvas is None:
            canvas = self._rgb_canvas()

        for row in rows:
            canvas[int(row), :] = color

        return canvas

    def apply_morphological_closing(self, disk_radius=2):
        """
        Closes the image with a disk of ``disk_radius`` to fill small gaps.
        """
        footprint = morphology.disk(disk_radius)
        self.image = morphology.erosion(self.image, footprint=footprint)
        self.image = morphology.dilation(self.image, footprint=footprint)
        return self.image

    def apply_morphological_opening(self, disk_radius=2):
        """
        Opens the image with a disk of ``disk_radius`` to remove small noise.
        """
        footprint = morphology.disk(disk_radius)
        self.image = morphology.dilation(self.image, footprint=footprint)
        self.image = morphology.erosion(self.image, footprint=footprint)
        return self.image

    def draw_barlines(self, cols=None, color=(0, 0, 255), canvas=None):
        """
        Returns an RGB copy with the barlines drawn on top. Pass ``canvas`` to
        draw onto an existing overlay instead of a fresh copy.
        """
        if cols is None:
            cols = self.barline_detection_projection()

        if canvas is None:
            canvas = self._rgb_canvas()

        for col in cols:
            canvas[:, int(col)] = color

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
