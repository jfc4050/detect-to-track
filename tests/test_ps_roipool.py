import pytest
import torch
from torch.autograd import gradcheck

from detect_to_track.models import PSROIPool


@pytest.mark.parametrize("n_targets", [1, 2])
@pytest.mark.parametrize("r_hw", [6, 7])
@pytest.mark.parametrize("fm_h", [10, 11])
@pytest.mark.parametrize("fm_w", [10, 11])
def test_ps_roipool_gradients(n_targets, r_hw, fm_h, fm_w):
    pr = PSROIPool(n_targets, r_hw)
    fm = (
        torch.rand(n_targets * r_hw ** 2, fm_h, fm_w)
        .double()
        .cuda()
        .requires_grad_(True)
    )

    rois = (
        torch.Tensor([[0.5, 0.5, 0.1, 0.1], [0.1, 0.1, 0.2, 0.3], [1.5, 1.5, 0.2, 0.2]])
        .double()
        .cuda()
        .requires_grad_(False)
    )

    grad_pass = gradcheck(pr, (fm, rois))

    assert grad_pass


@pytest.mark.parametrize("n_targets", [1, 2])
@pytest.mark.parametrize("r_hw", [6, 7])
@pytest.mark.parametrize("fm_h", [10, 11])
@pytest.mark.parametrize("fm_w", [10, 11])
def test_ps_roipool_can_handle_oob(n_targets, r_hw, fm_h, fm_w):
    pr = PSROIPool(n_targets, r_hw).cuda()
    fm = torch.full((n_targets * r_hw ** 2, fm_h, fm_w), 10).cuda()
    rois = torch.as_tensor([[3.0, 3.0, 0.5, 0.5]]).cuda()

    ans = pr(fm, rois).cpu()

    assert torch.allclose(ans, torch.zeros(len(rois), n_targets, r_hw, r_hw))
