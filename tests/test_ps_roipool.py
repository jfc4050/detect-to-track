import pytest
import torch
from torch.autograd import gradcheck

from detect_to_track.models import PSROIPool


@pytest.mark.parametrize('n_targets', [1, 2])
@pytest.mark.parametrize('r_hw', [6, 7])
@pytest.mark.parametrize('fm_hw', [10, 11])
def test_ps_roipool_gradients(n_targets, r_hw, fm_hw):
    pr = PSROIPool(n_targets, r_hw)
    fm = torch.rand(
        n_targets * r_hw**2, fm_hw, fm_hw
    ).double().cuda().requires_grad_(True)

    rois = torch.Tensor([
        [0.5, 0.5, 0.1, 0.1],
        [0.1, 0.1, 0.2, 0.3]
    ]).double().cuda().requires_grad_(False)

    grad_pass = gradcheck(pr, (fm, rois))

    assert grad_pass
