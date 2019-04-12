"""R-FCN head"""

from typing import Tuple

from torch import nn, Tensor

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

    def forward(self, x: Tensor, regions: Tensor) -> Tensor:
        """
        Args:
            x: (C, H, W) input feature map.
            regions: (|R|, 4) predicted regions of interest. (ijhw, fractional).

        Returns:
            scores: (|R|, n_targets) scores for each target for each region.
        """
        x = x[None, :, :, :]  # (1, C, H, W)
        score_map = self.sm_conv(x)  # (1, n_targets*k^2, H, W)
        score_map = score_map.squeeze(0)  # (n_targets*k^2, H, W)

        pooled = self.roi_pool(score_map, regions)  # (|R|, n_targets, k, k)
        scores = pooled.mean(-1).mean(-1)  # (|R|, n_targets)

        return scores


class RFCN(nn.Module):
    """R-FCN. see https://arxiv.org/abs/1605.06409.

    Args:
        in_channels: input feature map channels.
        n_classes: number of non-background clases.
        k: height and width of spatial grid. see paper.
    """
    def __init__(self, in_channels: int, n_classes: int, k: int) -> None:
        super().__init__()
        self.cls_head = _RFCNHead(in_channels, n_classes + 1, k)
        self.reg_head = _RFCNHead(in_channels, 4, k)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: Tensor, regions: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: (C, H, W) input feature map.
            regions: (|R|, 4) region proposals.

        Returns:
            c_hat: (|R|, n_classes) region classification scores.
            b_hat: (|R|, 4) object bounding box offsets from regions.
        """
        c_hat = self.cls_head(x, regions)
        c_hat = self.softmax(c_hat)

        b_hat = self.reg_head(x, regions)

        return c_hat, b_hat
