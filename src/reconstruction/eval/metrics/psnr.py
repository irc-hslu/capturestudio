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

        # # Plot stuff
        # import cv2
        # # Prepare images
        # ref_rgb = cv2.blur(ref.rgb, (11, 11)).astype(np.float32)
        # ref_mask = ref.mask.astype(np.float32)[..., None]  # (H, W, 1)
        # other_rgb = cv2.blur(other.rgb, (11, 11)).astype(np.float32)
        #
        # # Apply mask
        # ref_masked = ref_rgb * ref_mask
        # other_masked = other_rgb * ref_mask
        # diff_masked = np.abs(ref_masked - other_masked)
        #
        # # Convert to uint8 for drawing
        # ref_uint8 = np.ascontiguousarray(np.clip(ref_masked, 0, 255).astype(np.uint8))
        # other_uint8 = np.ascontiguousarray(np.clip(other_masked, 0, 255).astype(np.uint8))
        # diff_uint8 = np.ascontiguousarray(np.clip(diff_masked, 0, 255).astype(np.uint8))
        #
        # # Find contours from the binary mask (squeeze channel, convert to uint8)
        # mask_u8 = (ref_mask[..., 0] > 0).astype(np.uint8) * 255
        # contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #
        # # Draw contours in red on each image
        # cv2.drawContours(ref_uint8, contours, -1, (255, 0, 1), thickness=2)
        # cv2.drawContours(other_uint8, contours, -1, (255, 0, 1), thickness=2)
        # cv2.drawContours(diff_uint8, contours, -1, (255, 0, 1), thickness=2)
        #
        # # Concatenate and save
        # combined = np.concatenate([ref_uint8, other_uint8, diff_uint8], axis=1)
        # cv2.imwrite('psnr.png', cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))

        masked_mse = np.sum(((ref_rgb - other_rgb) ** 2) * ref_mask) / ref_mask.sum()
        return 10 * np.log10((255.0 ** 2) / masked_mse)
