"""misc. utilities"""

from typing import Union, Tuple

import numpy as np
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
