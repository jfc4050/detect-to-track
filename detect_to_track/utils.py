"""misc. utilities"""

import re
from typing import Union, Tuple, Optional, Sequence

import numpy as np
import torch
from torch import Tensor
from torchvision import transforms


class DTLoss(object):
    """encapsulate loss tensors and provide some convenience methods."""

    def __init__(
        self,
        o_loss: Tensor = None,
        b_loss_rpn: Tensor = None,
        c_loss: Tensor = None,
        b_loss_rcnn: Tensor = None,
        t_loss: Tensor = None,
        requires_grad: bool = False,
    ) -> None:
        # initial loss values.
        self.o = self._process_init_tensor(o_loss)
        self.b_rpn = self._process_init_tensor(b_loss_rpn)
        self.c = self._process_init_tensor(c_loss)
        self.b_rcnn = self._process_init_tensor(b_loss_rcnn)
        self.t = self._process_init_tensor(t_loss)

        self.requires_grad = requires_grad
        self.count = int(
            any(
                x is not None for x in [o_loss, b_loss_rpn, c_loss, b_loss_rcnn, t_loss]
            )
        )

    @staticmethod
    def _process_init_tensor(init_tensor: Tensor) -> Tensor:
        """return zero tensor if init_tensor is None, otherwise clone
        (in case inputs are leaves)."""
        return torch.as_tensor(0.0) if init_tensor is None else init_tensor.clone()

    def _process_addend(self, addend: Tensor) -> Union[float, Tensor]:
        """convert to float if this `DTLoss` instance doesn't `require_grad`"""
        if not self.requires_grad:
            addend = addend.detach()
        return addend

    def __iadd__(self, lhs: "DTLoss") -> None:
        self.o += self._process_addend(lhs.o).to(self.o.device)
        self.b_rpn += self._process_addend(lhs.b_rpn).to(self.b_rpn.device)
        self.c += self._process_addend(lhs.c).to(self.c.device)
        self.b_rcnn += self._process_addend(lhs.b_rcnn).to(self.b_rcnn.device)
        self.t += self._process_addend(lhs.t).to(self.t.device)

        self.count += lhs.count

        return self

    def as_tensor(self) -> Tensor:
        """convert to tensor without breaking computation graph."""
        return torch.stack([self.o, self.b_rpn, self.c, self.b_rcnn, self.t])

    def to_scalar(self, coefs: Optional[Tensor] = None) -> Tensor:
        """linear combination of individual losses as specified by coefs"""
        loss_tensor = self.as_tensor()
        if coefs is None:
            coefs = torch.ones(len(loss_tensor))
        coefs = coefs.to(loss_tensor)

        scalar_loss = torch.dot(coefs, loss_tensor)
        scalar_loss /= self.count  # normalize

        return scalar_loss

    def backward(
        self,
        grad_tensors: Optional[Tensor] = None,
        retain_graph: Optional[bool] = None,
        create_graph: bool = False,
    ) -> None:
        """loss backprop. mimics tensor.backward() interface."""
        if not self.requires_grad:
            raise RuntimeError(f"this DTLoss instance does not require_grad.")
        self.to_scalar(grad_tensors).backward(
            retain_graph=retain_graph, create_graph=create_graph
        )

    def asdict(self) -> dict:
        """convert self to dict."""
        return {
            "o": self.o,
            "a": self.b_rpn,
            "c": self.c,
            "r": self.b_rcnn,
            "t": self.t,
        }

    def __repr__(self) -> str:
        """str(self.asdict()) isn't compact enough."""
        return " ".join(
            [
                re.sub(
                    r"e([+-])0(\d)",
                    lambda m: f"e{m.group(1)}{m.group(2)}",
                    f"{k}:{v / self.count:.2e}",
                )
                for k, v in self.asdict().items()
            ]
        )


def build_anchors(
    fm_shape: Union[Tuple[int, int], int],
    anchor_areas: Sequence[float],
    aspect_ratios: Sequence[float],
    flatten: bool = True,
) -> np.ndarray:
    """build (optionally) flattened anchor grid.

    Args:
        fm_shape: prediction map height and width.
        anchor_areas: anchor areas.
        aspect_ratios: anchor aspect ratios (h/w).

    Returns:
        anchors: (H * W * |A x R|, 4) bounding box priors.
    """
    if isinstance(fm_shape, int):
        fm_shape = (fm_shape, fm_shape)

    ### get anchor heights and widths.
    anchor_dims = np.zeros((len(anchor_areas), len(aspect_ratios), 2))
    for i, anchor_area in enumerate(anchor_areas):
        for j, aspect_ratio in enumerate(aspect_ratios):
            h = np.sqrt(anchor_area * aspect_ratio)
            w = anchor_area / h
            anchor_dims[i, j, :] = [h, w]
    anchor_dims = anchor_dims.reshape(-1, 2)  # (|A x R|, 2)

    ### get centered anchor coordinates.
    fm_h, fm_w = fm_shape
    iv, jv = np.meshgrid(
        np.linspace(0, 1, fm_h, endpoint=False) + 1 / fm_h / 2,
        np.linspace(0, 1, fm_w, endpoint=False) + 1 / fm_w / 2,
        indexing="ij",
    )  # 2*(H, W) grid of center i, j values for each feature map cell
    ij_grid = np.stack([iv, jv], axis=-1)  # (H, W, 2)

    ### expand and concatenate coordinates and shapes.
    # expand ij_grid and hw_grid: (H, W, |A x R|, 2)
    target_shape = (fm_h, fm_w, len(anchor_dims), 2)
    ij_grid = np.broadcast_to(ij_grid[:, :, None, :], target_shape)
    hw_grid = np.broadcast_to(anchor_dims[None, None, :, :], target_shape)
    anchors = np.concatenate([ij_grid, hw_grid], 3)  # (H, W, |A x R|, 4)

    if flatten:
        anchors = anchors.reshape(-1, 4)  # (H * W * |A x R|, 4)

    anchors.flags.writeable = False

    return anchors


def tensor_to_ndarray(tensor: Tensor) -> np.ndarray:
    """convert torch Tensor to numpy ndarray."""
    return tensor.detach().cpu().numpy()


def make_input_transform(net_input_shape: Union[int, Tuple[int, int]]) -> object:
    """return transform layer for PIL Image -> network input"""
    if isinstance(net_input_shape, int):
        net_input_shape = (net_input_shape, net_input_shape)

    return transforms.Compose(
        [transforms.Resize(net_input_shape), transforms.ToTensor()]
    )


def get_subset_lengths(dataset_length, subset_length):
    """get lengths of dataset splits"""
    n_full_subsets = int(dataset_length / subset_length)
    remainder = dataset_length % subset_length
    lengths = n_full_subsets * [subset_length] + [remainder]

    return lengths
