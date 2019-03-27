"""unit tests for roipool CUDA op."""

import pytest
import torch
from torch.autograd import gradcheck

from detect_to_track.models import ROIPool


@pytest.mark.parametrize('r_hw', [5, 6])
@pytest.mark.parametrize('fm_c', [2])
@pytest.mark.parametrize('fm_hw', [10, 11])
def test_roipool_gradients(r_hw, fm_c, fm_hw):
    rp = ROIPool(r_hw)
    fm = torch.rand(fm_c, fm_hw, fm_hw).double().cuda().requires_grad_(True)

    rois = torch.Tensor([
        [0.5, 0.5, 0.5, 0.5],
        [0.1, 0.1, 0.2, 0.3]
    ]).double().cuda().requires_grad_(False)

    grad_passed = gradcheck(rp, (fm, rois))

    assert grad_passed
