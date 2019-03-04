"""tests for stride reduced resnet with dilated convolutions"""
import pytest

import torch
from torch.utils.model_zoo import load_url
from torchvision.models.resnet import model_urls

from ..models.backbones import resnet


@pytest.mark.parametrize('arch', [50, 101, 152])
def test_load_state_dict(arch):
    """make sure weights are loaded in properly"""
    rn = resnet(arch)
    state_dict = load_url(model_urls[f'resnet{arch}'])

    rn.load_state_dict(state_dict)
    for param_name, param in rn.named_parameters():
        if param_name in state_dict:
            assert torch.allclose(state_dict[param_name], param)
