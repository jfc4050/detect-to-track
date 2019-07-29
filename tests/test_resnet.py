"""tests for stride reduced resnet with dilated convolutions"""

import pytest
import torch

from detect_to_track.models.resnet import resnet_backbone


@pytest.mark.parametrize("depth", [50, 101])
def test_resnet_fm_shapes(depth):
    rn = resnet_backbone(f"resnet{depth}", 5).cuda()
    rn.eval()

    input_h, input_w = (608, 1200)
    with torch.no_grad():
        x = torch.rand(1, 3, input_h, input_w).cuda()

        features = rn(x)
        c3 = features["c3"].cpu()
        c4 = features["c4"].cpu()
        c5 = features["c5"].cpu()

    fmaps = [c3, c4, c5]
    expected_strides = [8, 16, 16]

    for fmap, expected_stride in zip(fmaps, expected_strides):
        assert fmap.size(0) == x.size(0)
        assert x.size(2) / fmap.size(2) == expected_stride
        assert x.size(3) / fmap.size(3) == expected_stride
