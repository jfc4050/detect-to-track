#include <torch/torch.h>
#include "common/cpp_common.hpp"


/* CUDA forward declaration */
at::Tensor ROIPoolCudaForward(
    const at::Tensor& FM,
    const at::Tensor& rois,
    const int rHW
);

/* CUDA forward declaration */
at::Tensor ROIPoolCudaBackward(
    const at::Tensor& gradOut,
    const at::Tensor& rois,
    const int iH,
    const int iW
);


/* check input and forward to CUDA function */
at::Tensor ROIPoolForward(
    const at::Tensor& FM,
    const at::Tensor& rois,
    const int rHW
)
{
    CHECK_INPUT(FM);
    CHECK_INPUT(rois);
    return ROIPoolCudaForward(FM, rois, rHW);
}


/* check input and forward to CUDA function */
at::Tensor ROIPoolBackward(
    const at::Tensor& gradOut,
    const at::Tensor& rois,
    const int iH,
    const int iW
)
{
    CHECK_INPUT(gradOut);
    CHECK_INPUT(rois);
    return ROIPoolCudaBackward(gradOut, rois, iH, iW);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "roipool_forward",
        &ROIPoolForward,
        "RoI Pool forward pass"
    );
    m.def(
        "roipool_backward",
        &ROIPoolBackward,
        "RoI Pool backward pass"
    );
}
