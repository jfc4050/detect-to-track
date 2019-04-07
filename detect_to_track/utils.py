"""misc. utilities"""

from typing import Union, Tuple, Optional

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
        transforms.ToTensor
    ])


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

    def backward(
            self,
            grad_tensors: Optional[Tensor] = None,
            retain_graph: Optional[bool] = None,
            create_graph: bool = False
    ) -> None:
        """loss backprop. mimics tensor.backward() interface."""
        loss_tensor = self.as_tensor()
        if grad_tensors is None:
            grad_tensors = torch.ones(5).to(loss_tensor)

        # linear combination of individual losses as specified by grad_tensors
        scalar_loss = torch.dot(grad_tensors, loss_tensor)
        scalar_loss /= self.count  # normalize

        scalar_loss.backward(
            retain_graph=retain_graph, create_graph=create_graph
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
