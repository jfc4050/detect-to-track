"""misc. utilities"""

from typing import Union, Tuple, Optional, Sequence

import numpy as np
import torch
from torch import Tensor
from torchvision import transforms


def tensor_to_ndarray(tensor: Tensor) -> np.ndarray:
    """convert torch Tensor to numpy ndarray."""
    return tensor.detach().cpu().numpy()


def make_input_transform(
        net_input_shape: Union[int, Tuple[int, int]]
) -> object:
    """return transform layer for PIL Image -> network input"""
    if isinstance(net_input_shape, int):
        net_input_shape = (net_input_shape, net_input_shape)

    return transforms.Compose([
        transforms.Resize(net_input_shape),
        transforms.ToTensor()
    ])


def build_anchors(
        fm_shape: Union[Tuple[int, int], int],
        base_area: float,
        scale_factors: Sequence[float],
        aspect_ratios: Sequence[float],
        flatten: bool = True
) -> np.ndarray:
    """build flattened anchor grid.

    Args:
        fm_shape: prediction map height and width.
        base_area: base anchor area, pre-scaling.
        scale_factors: anchor area scaling factors.
        aspect_ratios: anchor aspect ratios (h/w).

    Returns:
        anchors: (H * W * |SF x AR|, 4) bounding box priors.
    """
    if isinstance(fm_shape, int):
        fm_shape = (fm_shape, fm_shape)

    ### get anchor heights and widths.
    anchor_dims = np.zeros((len(scale_factors), len(aspect_ratios), 2))
    for i, scale_factor in enumerate(scale_factors):
        scaled_area = scale_factor * base_area
        for j, aspect_ratio in enumerate(aspect_ratios):
            h = np.sqrt(scaled_area * aspect_ratio)
            w = scaled_area / h
            anchor_dims[i, j, :] = [h, w]
    anchor_dims = anchor_dims.reshape(-1, 2)  # (|SF x AR|, 2)

    ### get centered anchor coordinates.
    fm_h, fm_w = fm_shape
    iv, jv = np.meshgrid(
        np.linspace(0, 1, fm_h + 2)[1:-1],
        np.linspace(0, 1, fm_w + 2)[1:-1],
        indexing='ij'
    )  # 2*(H, W) grid of center i, j values for each feature map cell
    ij_grid = np.stack([iv, jv], axis=-1)  # (H, W, 2)

    ### expand and concatenate coordinates and shapes.
    # expand ij_grid and hw_grid: (H, W, |SF x AR|, 2)
    target_shape = (fm_h, fm_w, len(anchor_dims), 2)
    ij_grid = np.broadcast_to(ij_grid[:, :, None, :], target_shape)
    hw_grid = np.broadcast_to(anchor_dims[None, None, :, :], target_shape)
    anchors = np.concatenate([ij_grid, hw_grid], 3)  # (H, W, |SF x AR|, 4)

    if flatten:
        anchors = anchors.reshape(-1, 4)  # (H * W * |SF x AR|, 4)

    anchors.flags.writeable = False

    return anchors


class DTLoss(object):
    """encapsulate loss tensors and provide some convenience methods."""
    def __init__(
            self,
            o_loss: Optional[Tensor] = None,
            b_loss_rpn: Optional[Tensor] = None,
            c_loss: Optional[Tensor] = None,
            b_loss_rcnn: Optional[Tensor] = None,
            t_loss: Optional[Tensor] = None
    ) -> None:
        # cloning in case inputs are leaves. otherwise no effect.
        self.o_loss = o_loss.clone() or 0
        self.b_loss_rpn = b_loss_rpn.clone() or 0
        self.c_loss = c_loss.clone() or 0
        self.b_loss_rcnn = b_loss_rcnn.clone() or 0
        self.t_loss = t_loss.clone() or 0

        if any(
                x is not None
                for x in [o_loss, b_loss_rpn, c_loss, b_loss_rcnn, t_loss]
        ):
            self.count = 1
        else:
            self.count = 0

    def __iadd__(self, lhs: 'DTLoss') -> None:
        self.o_loss += lhs.o_loss
        self.b_loss_rpn += lhs.b_loss_rpn
        self.c_loss += lhs.c_loss
        self.b_loss_rcnn += lhs.b_loss_rcnn
        self.t_loss += lhs.t_loss

        self.count += lhs.count

        return self

    def as_tensor(self) -> Tensor:
        """convert to tensor without breaking computation graph."""
        return torch.stack([
            self.o_loss,
            self.b_loss_rpn,
            self.c_loss,
            self.b_loss_rcnn,
            self.t_loss
        ])

    def to_scalar(self, coefs: Optional[Tensor] = None) -> Tensor:
        """linear combination of individual losses as specified by coefs"""
        if coefs is None:
            coefs = torch.ones(5)

        loss_tensor = self.as_tensor()
        coefs = coefs.to(loss_tensor)

        scalar_loss = torch.dot(coefs, loss_tensor)
        scalar_loss /= self.count  # normalize

        return scalar_loss

    def backward(
            self,
            grad_tensors: Optional[Tensor] = None,
            retain_graph: Optional[bool] = None,
            create_graph: bool = False
    ) -> None:
        """loss backprop. mimics tensor.backward() interface."""
        self.to_scalar(grad_tensors).backward(
            retain_graph=retain_graph,
            create_graph=create_graph
        )

    def __repr__(self) -> str:
        as_dict = {
            'o': self.o_loss,
            'b_rpn': self.b_loss_rpn,
            'c': self.c_loss,
            'b_rcnn': self.b_loss_rcnn,
            't': self.t_loss
        }

        # str(as_dict) isn't compact enough
        as_str = ' '.join([
            f'{k}:{v / self.count:.2f}'  # normalize and round
            for k, v in as_dict.items()
        ])

        return as_str
