"""tests for stride reduced resnet with dilated convolutions"""

import re

import pytest
import torch
from torch.utils.model_zoo import load_url
from torchvision.models.resnet import model_urls

from detect_to_track.models.resnet import resnet


depths = [50, 101, 152]
n_stages = 5


@pytest.mark.parametrize('depth', depths)
def test_load_state_dict(depth):
    """make sure weights are loaded in properly, does not yet test
    renamed params"""
    rn = resnet(depth)
    state_dict = load_url(model_urls[f'resnet{depth}'])

    rn.load_state_dict(state_dict, strict=False)
    for param_name, param in rn.named_parameters():
        if param_name in state_dict:
            assert torch.allclose(state_dict[param_name], param)


@pytest.mark.parametrize('depth', depths)
@pytest.mark.parametrize('first_trainable_stage', range(1, n_stages+1))
def test_train_eval(depth, first_trainable_stage):
    rn = resnet(
        depth, first_trainable_stage=first_trainable_stage, pretrained=False
    )
    rn.train()
    for child_name, child in rn.named_children():
        stage_num = int(re.search(r'^stage(\d)', child_name).group(1))
        should_be_frozen = stage_num < first_trainable_stage

        if should_be_frozen:
            assert not child.training
        else:
            assert child.training

    rn.eval()
    for child_name, child in rn.named_children():
        assert not child.training


@pytest.mark.parametrize('depth', depths)
@pytest.mark.parametrize('first_trainable_stage', range(1, n_stages+1))
def test_freeze(depth, first_trainable_stage):
    rn = resnet(
        depth, first_trainable_stage=first_trainable_stage, pretrained=False
    )
    rn.freeze()

    for param_name, param in rn.named_parameters():
        stage_num = int(re.search(r'^stage(\d)', param_name).group(1))
        if stage_num < first_trainable_stage:
            assert param.requires_grad is False
        else:
            assert param.requires_grad is True


@pytest.mark.parametrize('depth', depths)
def test_resnet_fm_shapes(depth):
    rn = resnet(depth, pretrained=False).cuda()
    rn.eval()

    input_h, input_w = (608, 1200)
    with torch.no_grad():
        x = torch.rand(1, 3, input_h, input_w).cuda()

        features = rn(x)
        c3 = features.c3.cpu()
        c4 = features.c4.cpu()
        c5 = features.c5.cpu()

    fmaps = [c3, c4, c5]
    expected_channels = [
        rn.stage3.out_channels,
        rn.stage4.out_channels,
        rn.stage5.out_channels
    ]
    expected_strides = [8, 16, 16]

    for fmap, expected_c, expected_stride in zip(
            fmaps, expected_channels, expected_strides
    ):
        stride_h = x.size(2) / fmap.size(2)
        stride_w = x.size(3) / fmap.size(3)

        assert fmap.size(0) == x.size(0)
        assert fmap.size(1) == expected_c
        assert stride_h == expected_stride
        assert stride_w == expected_stride
