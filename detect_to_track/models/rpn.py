"""region proposal network."""

from typing import Tuple

from torch import Tensor, nn
from torch.nn.functional import relu, softmax


class RPN(nn.Module):
    """region proposal network. see https://arxiv.org/abs/1506.01497.

    Args:
        in_channels: input feature map channels.
        n_anchors: number of anchors per feature map cell.
    """
    def __init__(self, in_channels: int, n_anchors: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        self.cls_fc = nn.Conv2d(in_channels, 2*n_anchors, kernel_size=1)
        self.reg_fc = nn.Conv2d(in_channels, 4*n_anchors, kernel_size=1)

    @staticmethod
    def _flatten_outputs(x: Tensor, targets_p_anchor: int) -> Tensor:
        """flatten outputs, keeping anchors together."""
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(x.size(0), -1, targets_p_anchor)

        return x

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """get objectness scores and bounding box regression offsets from
        feature map.

        Args:
            x: (|B|, C, H, W) feature map.

        Returns:
            x: (|B|, C', H', W') regression features.
            c_hat: (|B|, |A|, 2) anchorwise class score (object and not object).
            b_hat: (|B|, |A|, 4) anchorwise bounding box offsets.
        """
        x = relu(self.conv(x))  # (|B|, C', H', W')
        o_hat = self.cls_fc(x)  # (|B|, 2*a, H', W'), a = anchors per cell.
        b_hat = self.reg_fc(x)  # (|B|, 4*a, H', W')

        o_hat = self._flatten_outputs(o_hat, 2)  # (|B|, |A|, 2)
        b_hat = self._flatten_outputs(b_hat, 4)  # (|B|, |A|, 4)

        o_hat = softmax(o_hat, dim=2)  # object vs not object

        return x, o_hat, b_hat
