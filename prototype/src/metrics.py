import numpy as np
from scipy.ndimage import correlate


def to_foreground_mask(image):
    """
    Returns a boolean mask where ``True`` marks foreground (ink) pixels.

    Both the binarization output and the ground-truth ``.bmp`` files share the
    same black-on-white convention (ink = 0, background = 255); the ground
    truth is just stored as 3-channel RGB, so we drop the extra channels.
    """
    image = np.asarray(image)

    if image.ndim == 3:
        image = image[..., 0]

    return image == 0


def confusion(pred_fg, gt_fg):
    """Pixel-wise true/false positives and negatives between two masks."""
    pred_fg = np.asarray(pred_fg, dtype=bool)
    gt_fg = np.asarray(gt_fg, dtype=bool)

    tp = int(np.count_nonzero(pred_fg & gt_fg))
    fp = int(np.count_nonzero(pred_fg & ~gt_fg))
    fn = int(np.count_nonzero(~pred_fg & gt_fg))
    tn = int(np.count_nonzero(~pred_fg & ~gt_fg))
    return tp, fp, fn, tn


def psnr(pred_fg, gt_fg):
    """
    Peak signal-to-noise ratio between two binary foreground masks, in dB.

    For binary images the peak amplitude is ``C = 1`` and the MSE equals the
    fraction of misclassified pixels. Higher is better (DIBCO: >20 dB is
    excellent). Returns ``inf`` when the masks are identical.
    """
    pred_fg = np.asarray(pred_fg, dtype=float)
    gt_fg = np.asarray(gt_fg, dtype=float)

    mse = np.mean((pred_fg - gt_fg) ** 2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(1.0 / mse))


def _drd_weight_matrix():
    """5x5 normalized reciprocal-distance weights (center weight = 0)."""
    coords = np.arange(5) - 2
    dy, dx = np.meshgrid(coords, coords, indexing="ij")
    dist = np.sqrt(dy**2 + dx**2)
    weights = np.zeros((5, 5))
    nonzero = dist != 0
    weights[nonzero] = 1.0 / dist[nonzero]
    return weights / weights.sum()


def drd(pred_fg, gt_fg, block_size=8):
    """
    Distance Reciprocal Distortion (Lu et al., 2004), as used in DIBCO.

    Each misclassified pixel is penalised by how much its 5x5 ground-truth
    neighbourhood disagrees with it, weighted by reciprocal distance, then
    normalised by the number of non-uniform 8x8 blocks. Lower is better
    (0 = perfect, <3 very good).
    """
    pred_fg = np.asarray(pred_fg, dtype=float)
    gt_fg = np.asarray(gt_fg, dtype=float)

    wnm = _drd_weight_matrix()
    # Weighted agreement of each pixel's neighbourhood with foreground (=1).
    neighbour_fg = correlate(gt_fg, wnm, mode="constant", cval=0.0)

    flipped = pred_fg != gt_fg
    # If the prediction said background (0), distortion = weight on fg pixels;
    # if it said foreground (1), distortion = weight on background pixels.
    per_pixel = np.where(pred_fg == 0, neighbour_fg, 1.0 - neighbour_fg)
    drd_sum = float(per_pixel[flipped].sum())

    # Number of non-uniform block_size x block_size blocks in the ground truth.
    h, w = gt_fg.shape
    bh, bw = h // block_size, w // block_size
    cropped = gt_fg[: bh * block_size, : bw * block_size]
    blocks = cropped.reshape(bh, block_size, bw, block_size)
    block_sums = blocks.sum(axis=(1, 3))
    uniform = (block_sums == 0) | (block_sums == block_size**2)
    nubn = int(np.count_nonzero(~uniform))

    if nubn == 0:
        return 0.0
    return drd_sum / nubn


def angular_error(estimated_deg, true_deg):
    """
    Residual angular error ``|theta_hat - theta|`` in degrees between an
    estimated skew angle and the known applied rotation.

    The difference is folded into ``[-90, 90]`` modulo 180 degrees, so an
    estimate that is correct up to the 180-degree ambiguity of a line
    orientation counts as exact. Lower is better (<0.5 deg is excellent).
    """
    err = (float(estimated_deg) - float(true_deg) + 90.0) % 180.0 - 90.0
    return abs(err)


def scores(pred, gt):
    """
    Computes the DIBCO binarization metrics (F-measure, PSNR, DRD) of a
    binarization result against a ground-truth image (both in any supported
    convention).
    """
    pred_fg = to_foreground_mask(pred)
    gt_fg = to_foreground_mask(gt)

    if pred_fg.shape != gt_fg.shape:
        raise ValueError(
            f"shape mismatch: prediction {pred_fg.shape} vs gt {gt_fg.shape}"
        )

    tp, fp, fn, _ = confusion(pred_fg, gt_fg)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f_measure = (
        2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f_measure": f_measure,
        "psnr": psnr(pred_fg, gt_fg),
        "drd": drd(pred_fg, gt_fg),
    }


def angular_error(pred_angle, gt_angle):
    """
    Computes the angular error between two angles in degrees.

    The angular error is defined as the absolute difference between the two
    angles, wrapped to the range [0, 180] degrees. Lower is better (0 = perfect).
    """
    pred_angle = np.asarray(pred_angle)
    gt_angle = np.asarray(gt_angle)

    # Compute the absolute difference and wrap to [0, 180]
    diff = np.abs(pred_angle - gt_angle) % 360
    angular_error = np.minimum(diff, 360 - diff)

    return float(angular_error)
