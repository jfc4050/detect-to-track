"""correlation based tracking module"""

import torch
from torch import Tensor
from torch import nn

from . import PointwiseCorrelation
from . import ROIPool
from . import ResNetFeatures


class CorrelationTracker(nn.Module):
    """given features from time steps t and t+tau, predict bounding box
    transformations between time steps.
    see https://arxiv.org/abs/1710.03958.

    Args:
        d_max: maximum displacement for pointwise correlations.
        r_hw: height and width of pooled feature maps.
        reg_channels: RPN feature map channels.
        stride: correlation stride.
    """
    def __init__(
            self,
            d_max: int,
            r_hw: int,
            reg_channels: int,
            stride: int = 1
    ) -> None:
        super().__init__()
        self.point_corr = PointwiseCorrelation(d_max, stride)
        self.pool = ROIPool(r_hw)

        self.reg_fc = nn.Linear(
            (3*(2*d_max+1)**2 + 2*reg_channels) * r_hw**2,
            4
        )

    def forward(
            self,
            fm_pyr_0: ResNetFeatures,
            fm_pyr_1: ResNetFeatures,
            reg_fm_0: Tensor,
            reg_fm_1: Tensor,
            rois: Tensor
    ) -> Tensor:
        """
        Args:
            fm_pyr_0: backbone feature map pyramid from time t.
            fm_pyr_1: backbone feature map pyramid from time t+tau.
            reg_fm_0: (Cr, H, W); RPN features from time t.
            reg_fm_1: (Cr, H, W); RPN features from time t+tau.
            rois: (|R|, 4); ROIs from time t.

        Returns:
            t_hat: (|R|, 4); predicted box transformations from time t to t+tau.
        """
        ### preprocess inputs
        # insert batch dimensions: (C, H, W) -> (1, C, H, W)
        c3_0, c4_0, c5_0 = [
            fm[None, :, :, :] for fm in [fm_pyr_0.c3, fm_pyr_0.c4, fm_pyr_0.c5]
        ]
        c3_1, c4_1, c5_1 = [
            fm[None, :, :, :] for fm in [fm_pyr_1.c3, fm_pyr_1.c4, fm_pyr_1.c5]
        ]
        # resize c3 (c3 has half the stride of c4 and c5)
        c3_0 = nn.functional.interpolate(c3_0, scale_factor=1/2)
        c3_1 = nn.functional.interpolate(c3_1, scale_factor=1/2)

        ### compute correlation features
        corr_feats = [
            # (1, H, W, (2d+1), (2d+1)) -> ((2d+1)^2, H, W)
            cf.squeeze(0).view(cf.size(1), cf.size(2), -1).permute(2, 0, 1)
            for cf in [
                self.point_corr(c3_0, c3_1),
                self.point_corr(c4_0, c4_1),
                self.point_corr(c5_0, c5_1)
            ]
        ]

        track_feats = torch.cat([
            reg_fm_0,  # (Cr, H, W)
            reg_fm_1,  # (Cr, H, W)
            *corr_feats,  # 3 * ((2d+1)^2, H, W)
        ])  # (3*(2d+1)^2 + 2Cr, H, W)

        pooled_feats = self.pool(track_feats, rois)  # (|R|, C, rHW, rHW)
        pooled_feats = pooled_feats.view(pooled_feats.size(0), -1)  # (|R|, roi_features)

        t_hat = self.reg_fc(pooled_feats)  # (|R|, 4)

        return t_hat
