#include <torch/torch.h>


#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), "CPU op not implemented")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


/*
    CUDA forward declarations
*/
at::Tensor pointwiseCorrelationCudaForward(
    const at::Tensor& FM0,
    const at::Tensor& FM1,
    const int dMax,
    const int stride
);


std::tuple<at::Tensor, at::Tensor> pointwiseCorrelationCudaBackward(
    const at::Tensor& gradOutput,
    const at::Tensor& FM0,
    const at::Tensor& FM1,
    const int dMax,
    const int stride
);


/* forward pass */
at::Tensor pointwiseCorrelationForward(
    const at::Tensor& FM0,
    const at::Tensor& FM1,
    const int dMax,
    const int stride
)
{
    CHECK_INPUT(FM0);
    CHECK_INPUT(FM1);
    return pointwiseCorrelationCudaForward(FM0, FM1, dMax, stride);
}

/* backward pass */
std::tuple<at::Tensor, at::Tensor> pointwiseCorrelationBackward(
    const at::Tensor& gradOut,
    const at::Tensor& FM0,
    const at::Tensor& FM1,
    const int dMax,
    const int stride
)
{
    CHECK_INPUT(gradOut);
    CHECK_INPUT(FM0);
    CHECK_INPUT(FM1);
    return pointwiseCorrelationCudaBackward(gradOut, FM0, FM1, dMax, stride);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "pointwise_correlation_forward",
        &pointwiseCorrelationForward,
        "pointwise correlation forward pass"
    );
    m.def(
        "pointwise_correlation_backward",
        &pointwiseCorrelationBackward,
        "pointwise correlation backward pass"
    );
}