"""R-FCN head"""

import torch
from torch import nn, Tensor
import numpy as np

from . import PSROIPool


class _RFCNHead(nn.Module):
    """R-FCN head. see https://arxiv.org/abs/1605.06409.

    Args:
        in_channels: input feature map channels.
        n_targets: if classifier - number of classes + 1 (for background).
                   if regressor - 4 (ijhw).
        k: height and width of spatial grid. see paper.
    """
    def __init__(self, in_channels: int, n_targets: int, k: int) -> None:
        super().__init__()
        self.sm_conv = nn.Conv2d(in_channels, n_targets*k**2, kernel_size=1)
        self.roi_pool = PSROIPool(n_targets, k)

        self.n_targets = n_targets

    def forward(self, x: Tensor, regions: np.ndarray) -> Tensor:
        """
        Args:
            x: (C, H, W) input feature map.
            regions: (|R|, 4) predicted regions of interest. (ijhw, fractional).

        Returns:
            scores: (|R|, n_targets) scores for each target for each region.
        """
        score_map = self.sm_conv(x)  # (n_targets*k^2, H, W)
        pooled = self.roi_pool(score_map, torch.as_tensor(regions))  # (|R|, n_targets, k, k)
        scores = pooled.mean(-1).mean(-1)  # (|R|, n_targets)

        return scores


class RFCNClsHead(_RFCNHead):
    """RFCN head for classification."""
    def __init__(self, in_channels: int, n_classes: int, k: int) -> None:
        super().__init__(in_channels, n_classes + 1, k)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: Tensor, regions: np.ndarray) -> Tensor:
        scores = super().forward(x, regions)  # (|R|, n_classes + 1)
        preds = self.softmax(scores)  # (|R|, n_classes+1)

        return preds


class RFCNRegHead(_RFCNHead):
    """RFCN head for bounding box regression."""
    def __init__(self, in_channels: int, k: int) -> None:
        super().__init__(in_channels, 4, k)
