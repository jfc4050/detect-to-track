"""general loss functions for object detection."""

import torch
from torch import nn, Tensor


def make_param(x: Tensor) -> nn.Parameter:
    return nn.Parameter(torch.as_tensor(x), requires_grad=False)


class FocalLoss(nn.Module):
    """focal loss for dense object detection.
    see https://arxiv.org/abs/1708.02002

    Args:
        gamma: focusing factor.
        alpha: class balancing factor.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.) -> None:
        super().__init__()
        self.alpha = make_param(alpha)
        self.gamma = make_param(gamma)
        self.bce_module = nn.BCELoss(reduction='none')

    def forward(self, c_hat: Tensor, c_star: Tensor) -> Tensor:
        """compute focal loss.

        Args:
            c_hat: (|B|, |A|, C) predicted class scores.
            c_star: (|B|, |A|) ground truth classes.

        Returns:
            focal_loss: (|B|, |A|) anchorwise focal loss.
        """
        c_star_oh = torch.zeros_like(c_hat)  # (|B|, |A|, C)
        c_star_oh.scatter_(-1, c_star.unsqueeze(-1), 1)  # one-hot representation

        # pt = 1-c_hat if c_star==1, c_hat otherwise
        pt = torch.where(c_star_oh, 1 - c_hat, c_hat)  # (|B|, |A|, C)
        # at = 1-alpha if c_star==1, alpha otherwise
        at = torch.where(c_star_oh, 1 - self.alpha, self.alpha)  # (|B|, |A|, C)
        bce = self.bce_module(c_hat, c_star)  # (|B|, |A|, C)
        fl = pt.pow(self.gamma) * at * bce  # (|B|, |A|, C)

        fl = fl.mean(-1)  # (|B|, |A|) mean loss across all classes

        return fl
