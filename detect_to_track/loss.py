"""general loss functions for object detection."""

from typing import Tuple

import torch
from torch import nn, Tensor


def make_param(x: Tensor) -> nn.Parameter:
    return nn.Parameter(torch.as_tensor(x), requires_grad=False)


class FocalLoss(nn.BCELoss):
    """focal loss for dense object detection.
    see https://arxiv.org/abs/1708.02002

    Args:
        gamma: focusing factor.
        alpha: class balancing factor.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.) -> None:
        super().__init__(reduction='none')
        self.alpha = make_param(alpha)
        self.gamma = make_param(gamma)

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
        bce = super().forward(c_hat, c_star)  # (|B|, |A|, C)
        fl = pt.pow(self.gamma) * at * bce  # (|B|, |A|, C)

        fl = fl.mean(-1)  # (|B|, |A|) mean loss across all classes

        return fl


class BBoxLoss(nn.SmoothL1Loss):
    """smooth L1 loss applied only at positive anchors"""
    def __init__(self) -> None:
        super().__init__(reduction='none')

    def forward(self, b_hat: Tensor, b_star: Tensor, c_star: Tensor) -> Tensor:
        """compute bounding box loss.

        Args:
            b_hat: (|B|, |A|, 4) predicted anchor offsets.
            b_star: (|B|, |A|, 4) ground-truth anchor offsets.
            c_star: (|B|, |A|) true class values.

        Returns:
            bbox_loss: (|B|, |A|) anchorwise bounding box loss.
        """
        l1 = super().forward(b_hat, b_star).mean(-1)  # (|B|, |A|) anchorwise avg
        l1[c_star == 0] = 0  # no penalties for negative anchors.

        return l1


class RPNLoss(nn.Module):
    """loss function for region proposal network.

    Args:
        alpha: see FocalLoss.
        gamma: see FocalLoss.
    """
    def __init__(self, alpha: float, gamma: float) -> None:
        super().__init__()
        self._o_loss_func = FocalLoss(alpha, gamma)
        self._b_loss_func = BBoxLoss()

    def forward(
            self,
            lw: Tensor,
            o_hat: Tensor,
            o_star: Tensor,
            b_hat: Tensor,
            b_star: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """compute objectness and regression loss.

        Args:
            lw: (|B|, |A|); anchorwise loss weights.
            o_hat: (|B|, |A|, 2) class predictions (object or not object).
            o_star: (|B|, |A|) ground-truth objectness or classes.
            b_hat: (|B|, |A|, 4) predicted bounding box offsets from anchors.
            b_star: (|B|, |A|, 4) ground truth bounding box offsets from anchors.

        Returns:
            o_loss: (scalar) classification loss.
            b_loss: (scalar) regression loss.
        """
        o_loss = self._o_loss_func(o_hat, o_star)
        b_loss = self._b_loss_func(b_hat, b_star, o_star)

        o_loss = (lw * o_loss).mean()
        b_loss = b_loss.mean()

        return o_loss, b_loss


class RCNNLoss(nn.Module):
    """loss function for region based convolutional neural network.

    Args:
        alpha: see FocalLoss.
        gamma: see FocalLoss.
    """
    def __init__(self, alpha: float, gamma: float) -> None:
        super().__init__()
        self._c_loss_func = FocalLoss(alpha, gamma)
        self._b_loss_func = BBoxLoss()

    def forward(self, c_hat, c_star, b_hat, b_star):
        """compute classification and regression loss.

        Args:
            c_hat: (|R|, n_classes) class predictions.
            c_star: (|R|) ground-truth classes.
            b_hat: (|R|, 4) predicted bounding box offsets from anchors.
            b_star: (|R|, 4) ground truth bounding box offsets from anchors.

        Returns:
            c_loss: (scalar) classification loss.
            b_loss: (scalar) regression loss.
        """
        # can't have meaningful batch dimension, because different images
        # from a batch may have different numbers of regions.
        # losses for region-based predictions from different images in batch
        # can still be made to benefit from batch computation by concatenating
        # along region dimension.
        c_hat = c_hat[None, :, :]  # (1, |R|, n_classes)
        c_star = c_star[None, :]  # (1, |R|)
        b_hat = b_hat[None, :, :]  # (1, |R|, 4)
        b_star = b_star[None, :, :]  # (1, |R|, 4)

        c_loss = self._c_loss_func(c_hat, c_star)
        b_loss = self._b_loss_func(b_hat, b_star, c_star)

        c_loss = c_loss.mean()
        b_loss = b_loss.mean()

        return c_loss, b_loss


class TrackLoss(nn.Module):
    """smooth L1 loss for track regression."""
    def __init__(self) -> None:
        super().__init__()
        self.l1_module = nn.SmoothL1Loss()

    def forward(self, t_hat: Tensor, t_star: Tensor) -> Tensor:
        """compute track regression loss.

        Args:
            t_hat: (|R|, 4) predicted track regression offsets.
            t_star: (|R|, 4) ground-truth track regression offsets.

        Returns:
            l1: smooth l1 loss for track regression.
        """
        l1 = self.l1_module(t_hat, t_star)  # (|R|, 4)

        l1 = l1.mean()

        return l1
