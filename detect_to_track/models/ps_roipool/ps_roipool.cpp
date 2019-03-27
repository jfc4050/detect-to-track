#include <torch/torch.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), "CPU op not implemented")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

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
