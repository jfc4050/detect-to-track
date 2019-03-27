import pytest
import torch
from torch.autograd import gradcheck

from detect_to_track.models import PointwiseCorrelation


@pytest.mark.parametrize('d_max', [3])
@pytest.mark.parametrize('stride', [1, 2])
@pytest.mark.parametrize('input_b', [1, 2])
@pytest.mark.parametrize('input_c', [2])
@pytest.mark.parametrize('input_hw', [10, 11])
def test_pointwise_correlation_gradients(
        d_max,
        stride,
        input_b,
        input_c,
        input_hw,
):
    pc = PointwiseCorrelation(d_max, stride).cuda()

    fm_shape = (input_b, input_c, input_hw, input_hw)
    fm0 = torch.rand(*fm_shape).double().cuda().requires_grad_(True)
    fm1 = torch.rand(*fm_shape).double().cuda().requires_grad_(True)

    grad_pass = gradcheck(pc, (fm0, fm1))

    assert grad_pass
