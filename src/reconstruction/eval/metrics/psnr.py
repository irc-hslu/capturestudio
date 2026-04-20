import numpy as np

from reconstruction.eval.metrics import Metric
from reconstruction.primitive.pcd import RGBDImage


class PeakSignalToNoiseRatio(Metric):
    def __init__(self):
        super().__init__("PSNR")

    def compute(self, ref: RGBDImage, other: RGBDImage) -> float:
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
            The computed PSNR value (in dB).
        """
        ref_rgb = ref.rgb.astype(np.float32)
        ref_mask = ref.mask.astype(np.float32)[..., None]  # (H, W, 1)
        other_rgb = other.color_match_to(ref).rgb.astype(np.float32)

        masked_mse = np.sum(((ref_rgb - other_rgb) ** 2) * ref_mask) / ref_mask.sum()
        return 10 * np.log10((255.0 ** 2) / masked_mse)
