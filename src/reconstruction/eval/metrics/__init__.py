import abc
from collections import defaultdict
from typing import Dict, Union, Optional

import numpy as np

from reconstruction.primitive.pcd import RGBDImage, PCDUtils


class Metric(metaclass=abc.ABCMeta):
    def __init__(self, name: str):
        self.name = name

    @abc.abstractmethod
    def compute(self, ref: RGBDImage, other: RGBDImage) -> float:
        """ Compute the evaluation metric between two RGBD images.

        Parameters
        ----------
        ref : RGBDImage
            The reference RGBD image.
        other : RGBDImage
            The other RGBD image to compare against the reference.

        Returns
        -------
        float
            The computed metric value.
        """
        raise NotImplementedError("This method should be overridden by subclasses")


class MetricsAggregator:
    def __init__(self, *metrics: Metric):
        self.metrics = metrics
        self.comparisons = {
            metric.name: []
            for metric in metrics
        }
        # For group-wise running average
        self.group_sums = defaultdict(lambda: {metric.name: 0.0 for metric in metrics})
        self.group_counts = defaultdict(int)

    def __len__(self):
        """
        Returns the number of comparison that have been stored.

        Returns
        -------
        int
            The number of comparisons.
        """
        return len(self.comparisons[self.metrics[0].name]) if self.metrics else 0

    def __call__(self, ref: RGBDImage, other: RGBDImage, align: bool = True, group: Optional[Union[int, str]] = None) -> Dict[str, float]:
        """
        Appends the metric for the given image pair to the self.metrics.

        Parameters
        ----------
        ref : RGBDImage
            The reference RGBD image.
        other : RGBDImage
            The other RGBD image to compare against the reference.
        group : Optional[Union[int, str]], optional
            An optional group identifier for running average calculations. If provided, the metric will be computed
            and stored in a running average for that group. If None, the metric will be computed and returned
            as instantaneous values.

        Returns
        -------
        dict
            A dictionary containing the metric names and their corresponding computed values for the given image pair.
        """
        if align:
            aligned_images, aligned_depths, aligned_valid = PCDUtils.align_projected_to_original(
                projected_images=other.rgb,
                projected_depths=other.depth,
                valid_pixels=other.mask,
                original_images=ref.rgb,
            )
            aligned_images = aligned_images.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
            aligned_depths = aligned_depths.detach().cpu().numpy().squeeze()
            aligned_valid = aligned_valid.detach().cpu().numpy().squeeze()
            if other.rgb.dtype == np.uint8:
                aligned_images = (aligned_images * 255.0 + 0.5).astype(np.uint8)
            if other.depth.max() > 1000.0:
                aligned_depths = aligned_depths * 1000.0
            other.rgb = aligned_images
            other.depth = aligned_depths
            other.mask = aligned_valid

        result_dict = {}
        for metric in self.metrics:
            value = metric.compute(ref, other)
            self.comparisons[metric.name].append(value)
            if group is not None:
                # Update running sum and count for the group
                self.group_sums[group][metric.name] += value
            result_dict[metric.name] = value

        # If group is set, update running average of that group and return the running average values
        if group is not None:
            self.group_counts[group] += 1
            # Return running average for the group
            averaged_result = {
                metric.name: self.group_sums[group][metric.name] / self.group_counts[group]
                for metric in self.metrics
            }
            return averaged_result

        # Otherwise return instantaneous values
        return result_dict

    def reset(self) -> None:
        """
        Resets the stored comparisons for all metrics.
        """
        self.comparisons = {name: [] for name in self.comparisons.keys()}
        self.group_sums.clear()
        self.group_counts.clear()
        import gc
        gc.collect()

    def gather(self) -> dict:
        """
        Returns the aggregated results of all metrics.

        Returns
        -------
        dict
            A dictionary containing the metric names and their corresponding values.
        """
        return {name: np.nanmean(values) if values else 0.0
                for name, values in self.comparisons.items()}

    def gather_grouped(self) -> Dict[Union[int, str], Dict[str, float]]:
        """
        Returns the aggregated results of all metrics grouped by the group identifier.

        Returns
        -------
        Dict[Union[int, str], Dict[str, float]]
            A dictionary where keys are group identifiers and values are dictionaries containing metric names and their corresponding values.
        """
        return {
            group: {name: self.group_sums[group][name] / self.group_counts[group] if self.group_counts[group] > 0 else 0.0
                    for name in self.group_sums[group]}
            for group in self.group_sums
        }

    @classmethod
    def full(cls) -> 'MetricsAggregator':
        """
        Creates a MetricsAggregator with all available metrics.

        Returns
        -------
        MetricsAggregator
            An instance of MetricsAggregator with all available metrics.
        """
        from reconstruction.eval.metrics.psnr import PeakSignalToNoiseRatio
        from reconstruction.eval.metrics.ssim import StructuralSimilarityIndex
        from reconstruction.eval.metrics.perceptual_similarity import PerceptualSimilarity

        return cls(
            PeakSignalToNoiseRatio(),
            StructuralSimilarityIndex(),
            PerceptualSimilarity()
        )

    @classmethod
    def for_psnr(cls) -> 'MetricsAggregator':
        """
        Creates a MetricsAggregator with only the PSNR metric.

        Returns
        -------
        MetricsAggregator
            An instance of MetricsAggregator with only the PSNR metric.
        """
        from reconstruction.eval.metrics.psnr import PeakSignalToNoiseRatio

        return cls(PeakSignalToNoiseRatio())

    @classmethod
    def for_psnr_and_ssim(cls) -> 'MetricsAggregator':
        """
        Creates a MetricsAggregator with PSNR and SSIM metrics.

        Returns
        -------
        MetricsAggregator
            An instance of MetricsAggregator with PSNR and SSIM metrics.
        """
        from reconstruction.eval.metrics.psnr import PeakSignalToNoiseRatio
        from reconstruction.eval.metrics.ssim import StructuralSimilarityIndex

        return cls(
            PeakSignalToNoiseRatio(),
            StructuralSimilarityIndex()
        )
