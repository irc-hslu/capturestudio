from typing import Literal

import lpips
import torch

from reconstruction.eval.metrics import Metric
from reconstruction.primitive.pcd import RGBDImage


class PerceptualSimilarity(Metric):
    def __init__(self, lpips_net: Literal['alex', 'vgg', 'squeeze'] = 'squeeze'):
        super().__init__("LPIPS")
        self.lpips_fn = lpips.LPIPS(net=lpips_net)

    def compute(self, ref: RGBDImage, other: RGBDImage) -> float:
        """
        Compute the Learned Perceptual Image Patch Similarity (LPIPS) between two RGBD images.

        Parameters
        ----------
        ref : RGBDImage
            The reference RGBD image.
        other : RGBDImage
            The other RGBD image to compare against the reference.

        Returns
        -------
        float
            The computed LPIPS value for the given image pair.
        """
        ref_rgb = torch.from_numpy(ref.rgb).permute(2, 0, 1).unsqueeze(0) / 255.0  # (1, 3, H, W)
        ref_mask = torch.from_numpy(ref.mask).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        other_rgb = torch.from_numpy(other.rgb).permute(2, 0, 1).unsqueeze(0) / 255.0  # (1, 3, H, W)
        with torch.no_grad():
            lpips_map = self.lpips_fn(ref_rgb, other_rgb, normalize=True)  # (1, 1, H, W)
        return ((lpips_map * ref_mask).sum() / ref_mask.sum()).item() if ref_mask.sum() > 0 else 0.0
