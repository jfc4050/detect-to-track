"""pooling operations for two stage detectors"""

import numpy as np
import torch
from torch import Tensor, nn
from ml_utils.bbox_utils import ijhw_to_ijij


class PSROIPool(nn.Module):
    """position-sensitive roi pooling operation
    see https://arxiv.org/abs/1605.06409.

    Args:
        k: height and width of spatial grid. see paper.
    """
    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k

    def _partition_region(self, region: np.ndarray) -> np.ndarray:
        """given a region, get coordinates of partitioned (k x k) grid.

        Args:
            region: (4,) ijhw for region.

        Returns:
            partitions: (k, k, 4) partitions[i][j] = ijhw for partition i,j.
        """
        _, _, h, w = region
        i0, j0, i1, j1 = ijhw_to_ijij(region)

        ivals = np.linspace(i0, i1, self.k + 2)[1:-1]  # center vals
        jvals = np.linspace(j0, j1, self.k + 2)[1:-1]  # center vals
        i_grid, j_grid = np.meshgrid(ivals, jvals, indexing='ij')  # 2* (k, k)
        ij_grid = np.stack([i_grid, j_grid], axis=-1)  # (k, k, 2)

        hw = np.array([h/self.k, w/self.k])  # (2,)
        hw_grid = np.broadcast_to(hw[None, None, :], (self.k, self.k, 2))

        partitions = np.concatenate([ij_grid, hw_grid], axis=-1)  # (k, k, 4)

        return partitions

    def forward(self, x: Tensor, region: np.ndarray) -> Tensor:
        """
        Args:
            x: (k^2, H, W) score map to extract pooled features from.
            region: (4,) single region of interest. (ijhw, fractional).

        Returns:
            region_score: (scalar) pooled and averaged scores.
        """
        _, x_h, x_w = x.shape
        partitions = self._partition_region(region)  # (k, k, 4)
        partitions = partitions.view(-1, 4)  # (k^2, 4)
        partitions = ijhw_to_ijij * [x_h, x_w, x_h, x_w]  # (k^2, 4)
        partitions = np.concatenate([
            np.floor(partitions[:2]), np.ceil([partitions[2:]])
        ]).astype(int)  # (k^2, 4)

        bin_scores = torch.stack([
            x[bin_num, i0:i1, j0:j1].mean()  # averaged bin slice
            for bin_num, (i0, j0, i1, j1) in enumerate(partitions)
        ])
        region_score = bin_scores.mean()

        return region_score
