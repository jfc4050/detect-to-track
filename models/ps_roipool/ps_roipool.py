"""position-sensitive ROI pooling function and module."""

from pathlib import Path

from torch import Tensor
from torch.nn import Module
from torch.autograd import Function
from torch.utils import cpp_extension


# JIT compilation
this_dir = Path(__file__).resolve().parent
_ext = cpp_extension.load(
    name='ext',
    sources=[
        Path(this_dir, srcfile) for srcfile in
        ['ps_roipool.cpp', 'ps_roipool_cuda.cu']
    ],
    extra_cuda_cflags=['-arch=sm_60']
)


class PSROIPoolFunction(Function):
    """position-sensitive ROI-pooling function.
    see https://arxiv.org/abs/1605.06409"""
    @staticmethod
    def forward(
            ctx,
            FM: Tensor,
            rois: Tensor,
            n_targets: int,
            r_hw: int
    ) -> Tensor:
        """
        Args:
            FM: (n_targets * r_hw^2, H, W); feature map for position-sensitive
                pooling.
            rois: (|R|, 4); region of interest bounding boxes.
            n_targets: number of targets per ROI.
            r_hw: height and width of pooled features.

        Returns:
            pooled: (|R|, n_targets, r_hw, r_hw) pooled feature map.
        """
        ctx.save_for_backward(rois)
        ctx.fm_h, ctx.fm_w = FM.shape[-2:]

        expected_channels = n_targets * r_hw ** 2
        if FM.size(0) != expected_channels:
            raise ValueError(
                f'expected {expected_channels} feature map channels, '
                f'recieved feature map of shape {tuple(FM.shape)}'
            )

        pooled = _ext.ps_roipool_forward(FM, rois, n_targets, r_hw)

        return pooled

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> Tensor:
        """
        Args:
            grad_out: (|R|, n_targets, r_hw, r_hw); loss derivatives wrt
                pooling output.

        Returns:
            grad_FM: (n_targets * r_hw^2, H, W); loss derivatives wrt
                pooling input.
        """
        rois, = ctx.saved_tensors
        grad_FM = _ext.ps_roipool_backward(grad_out, rois, ctx.fm_h, ctx.fm_w)

        return grad_FM, None, None, None


class PSROIPool(Module):
    """position-sensitive ROI-Pooling layer.
    see https://arxiv.org/abs/1605.06409

    Args:
        n_targets: number of targets per ROI.
        r_hw: height and with of pooled features.
    """
    def __init__(self, n_targets: int, r_hw: int):
        super().__init__()
        self.n_targets = n_targets
        self.r_hw = r_hw

    def forward(self, FM: Tensor, rois: Tensor) -> Tensor:
        """
        Args:
            FM: (n_targets * r_hw^2, H, W); feature map for position-sensitive
                pooling.
            rois: (|R|, 4) region of interest bounding boxes.

        Returns:
            pooled: (|R|, n_targets, r_hw, r_hw): pooled output.
        """
        return PSROIPoolFunction.apply(FM, rois, self.n_targets, self.r_hw)
