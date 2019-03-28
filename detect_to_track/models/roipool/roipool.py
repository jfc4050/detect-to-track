"""RoI Pooling function and module."""

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
        ['roipool.cpp', 'roipool_cuda.cu']
    ],
    extra_include_paths=[str(this_dir.parent)],
    extra_cuda_cflags=['-arch=sm_60']
)

class ROIPoolFunction(Function):
    """RoI Pooling function."""
    @staticmethod
    def forward(ctx, FM: Tensor, rois: Tensor, r_hw: int) -> Tensor:
        """RoI Pooling from FM, directed by rois.

        Args:
            FM: (C, H, W) feature map to pool from.
            rois: (|R|, 4) rois to pool from. (ijhw, fractional)
            r_hw: height and width of pooled feature maps.

        Returns:
            out: (|R|, C, r_hw, r_hw) pooled features.
        """
        ctx.i_h, ctx.i_w = FM.shape[-2:]
        ctx.save_for_backward(rois)
        out = _ext.roipool_forward(FM, rois, r_hw)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        """given loss derivatives wrt output, compute loss derivatives wrt input.

        Args:
            grad_out: (|R|, C, r_hw, r_hw) loss derivatives wrt output.

        Returns:
            grad_fm: (C, H, W) loss derivatives wrt input.
        """
        rois, = ctx.saved_tensors
        grad_fm = _ext.roipool_backward(grad_out, rois, ctx.i_h, ctx.i_w)

        return grad_fm, None, None


class ROIPool(Module):
    """RoI Pooling from FM, directed by rois.
    see https://arxiv.org/abs/1504.08083.

    Args:
        r_hw: height and width of pooled feature maps.
    """
    def __init__(self, r_hw: int) -> None:
        super().__init__()
        self.r_hw = r_hw

    def forward(self, FM: Tensor, rois: Tensor) -> Tensor:
        """
        Args:
            FM0: (C, H, W) input feature map.
            rois: (|R|, 4) regions of interest (ijhw, fractional).

        Returns:
            out: (|R|, C, r_hw, r_hw) pooled features.
        """
        return ROIPoolFunction.apply(FM, rois, self.r_hw)
