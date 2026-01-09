import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim

from reconstruction.eval.metrics import Metric


class StructuralSimilarityIndex(Metric):
    def __init__(self, window_size: int = 11):
        super().__init__("SSIM")
        self.window_size = window_size

    def compute(self, ref, other) -> float:
        """
        Compute the Peak Signal-to-Noise Ratio (PSNR) between two RGBD images.

        Parameters
        ----------
        ref : RGBDImage
            The reference RGBD image.
        other : RGBDImage
            The other RGBD image to compare against the reference.

        Returns
        -------
        float
            The computed masked Structural Similarity Index value for the given image pair and reference mask.
        """
        ref_mask = gaussian_filter(ref.mask.astype(np.float32), sigma=5.0) # (H, W, 1)
        ssim_map = ssim(ref.rgb, other.rgb, win_size=self.window_size, data_range=255, channel_axis=2, full=True)[-1].mean(axis=2)
        return np.sum(ssim_map * ref_mask) / np.sum(ref_mask)
