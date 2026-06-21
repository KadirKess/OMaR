import numpy as np
from scipy.sparse.csgraph import dijkstra
from skimage.graph import pixel_graph
from skimage import color, filters, morphology, feature, transform
from scipy import stats


class Preprocessor:
    def __init__(self, image):
        self.image = image
        self._image_width = image.shape[1]
        self._image_height = image.shape[0]

        self._current_strip = None

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

    def get_skew_angle(self, image):
        edges = feature.canny(image)
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

    def build_pixel_graph(self, black):

        black = np.ascontiguousarray(black, dtype=bool)
        H, W = black.shape

        image = black.astype(float)

        def edge_function(v0, v1, distances):
            touches_black = (v0 == 1.0) | (v1 == 1.0)
            # cost 2 if either pixel is black, 6 otherwise
            f = np.where(touches_black, 2.0, 6.0)
            return f * distances

        graph, node_to_pixel = pixel_graph(
            image, connectivity=2, edge_function=edge_function
        )

        pixel_to_node = np.full(H * W, -1, dtype=np.int64)
        pixel_to_node[node_to_pixel] = np.arange(len(node_to_pixel))

        return graph, node_to_pixel, pixel_to_node

    def shortest_path_row(self, graph, node_to_pixel, pixel_to_node, shape, row):
        H, W = shape
        start_node = pixel_to_node[row * W + 0]
        end_node = pixel_to_node[row * W + (W - 1)]

        dist, predecessors = dijkstra(
            graph, directed=True, indices=start_node, return_predecessors=True
        )

        # reconstruct the path
        path_nodes = []
        node = end_node
        while node != start_node and node >= 0:
            path_nodes.append(node)
            node = predecessors[node]
        path_nodes.append(start_node)
        path_nodes.reverse()

        pix = node_to_pixel[path_nodes]
        coords = np.stack([pix // W, pix % W], axis=1)
        return coords

    def _black_percentage(self, coords):
        if coords.shape[0] == 0:
            return 0.0
        pixels = self.image[coords[:, 0], coords[:, 1]]
        return np.mean(pixels == 0)

    def _trim_path(self, coords):
        is_black = self.image[coords[:, 0], coords[:, 1]] == 0
        if not is_black.any():
            return coords[:0]
        first = np.argmax(is_black)
        last = len(is_black) - np.argmax(is_black[::-1])
        return coords[first:last]

    def _path_corrcoef(self, coords):
        rows = coords[:, 0].astype(float)
        cols = coords[:, 1].astype(float)
        if rows.std() == 0 or cols.std() == 0:
            return 1.0
        return np.corrcoef(cols, rows)[0, 1]

    def _staffline_detection_shortest_path(self, corrcoef):
        _, staff_space_height = self.compute_staff_space_height()
        black_perc = 0.75
        strip_height = max(staff_space_height, 3)
        half = strip_height // 2

        H, _ = self.image.shape

        staff_lines = []
        for row in range(H):
            # Restrict the search to a strip around the current row; this is the
            # STRIP_HEIGHT of the paper and keeps each shortest path cheap.
            top = max(0, row - half)
            bottom = min(H, row + half + 1)
            strip = self.image[top:bottom]

            graph, node_to_pixel, pixel_to_node = self.build_pixel_graph(strip == 0)
            path = self.shortest_path_row(
                graph, node_to_pixel, pixel_to_node, strip.shape, row - top
            )
            path[:, 0] += top  # strip-local rows back to full-image coordinates

            if self._black_percentage(path) < black_perc:
                continue

            path = self._trim_path(path)
            if path.shape[0] < 2:
                continue

            if abs(self._path_corrcoef(path)) < corrcoef:
                continue

            staff_lines.append(path)

        return staff_lines

    def staffline_detection(self):
        return self._staffline_detection_shortest_path(corrcoef=0.5)

    def draw_staff(self, staff_lines=None, corrcoef=0.5, color=(255, 0, 0)):
        if staff_lines is None:
            staff_lines = self.staffline_detection(corrcoef=corrcoef)

        canvas = np.zeros((*self.image.shape, 3), dtype=np.uint8)

        for path in staff_lines:
            canvas[path[:, 0], path[:, 1]] = color

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
