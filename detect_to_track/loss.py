"""general loss functions for object detection."""

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
    def __init__(self):
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


class DetectionLoss(nn.Module):
    """generic loss function for object detection.

    Args:
        alpha: see FocalLoss.
        gamma: see FocalLoss.
    """
    def __init__(
            self,
            cls_loss_module: nn.Module,
            reg_loss_module: nn.Module
    ) -> None:
        super().__init__()
        self.cls_loss_module = cls_loss_module
        self.reg_loss_module = reg_loss_module

    def forward(
            self,
            weights: Tensor,
            o_hat: Tensor,
            o_star: Tensor,
            b_hat: Tensor,
            b_star: Tensor
    ) -> Tensor:
        """compute classification and regression loss.

        Args:
            weights (FloatTensor): (|B|, |A|) anchorwise loss weights.
            c_hat (FloatTensor): (|B|, |A|, 2) class predictions
                (object or not object).
            c_star (FloatTensor): (|B|, |A|, 2) ground-truth classes.
            b_hat (FloatTensor): (|B|, |A|, 4) predicted bounding box offsets
                from anchors.
            b_star (FloatTensor): (|B|, |A|, 4) ground truth bounding box
                offsets from anchors.

        Returns:
            cls_loss: (scalar) classification loss.
            FloatTensor: (scalar) regression loss.
        """
        cls_loss = self.cls_loss_module(o_hat, o_star)
        reg_loss = self.reg_loss_module(b_hat, b_star, o_star)

        cls_loss = (weights * cls_loss).sum()
        reg_loss = (weights * reg_loss).sum()

        return cls_loss, reg_loss
