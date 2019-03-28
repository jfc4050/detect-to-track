#include <torch/torch.h>
#include "common/cpp_common.hpp"


/* CUDA forward declaration */
at::Tensor psROIPoolCudaForward(
    const at::Tensor& FM,
    const at::Tensor& rois,
    const int nTargets,
    const int rHW
);

/* CUDA forward declaration */
at::Tensor psROIPoolCudaBackward(
    const at::Tensor& gradOut,
    const at::Tensor& rois,
    const int iH,
    const int iW
);


/* check input and forward to CUDA function */
at::Tensor psROIPoolForward(
    const at::Tensor& FM,
    const at::Tensor& rois,
    const int nTargets,
    const int rHW
)
{
    CHECK_INPUT(FM);
    CHECK_INPUT(rois);
    return psROIPoolCudaForward(FM, rois, nTargets, rHW);
}


/* check input and forward to CUDA function */
at::Tensor psROIPoolBackward(
    const at::Tensor& gradOut,
    const at::Tensor& rois,
    const int iH,
    const int iW
)
{
    CHECK_INPUT(gradOut);
    CHECK_INPUT(rois);
    return psROIPoolCudaBackward(gradOut, rois, iH, iW);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "ps_roipool_forward",
        &psROIPoolForward,
        "position-sensitive roi pooling forward pass"
    );
    m.def(
        "ps_roipool_backward",
        &psROIPoolBackward,
        "position-sensitive roi pooling backward pass"
    );
}
