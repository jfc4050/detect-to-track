#include <ATen/ATen.h>
#include "common/cuda_common.cuh"


template <typename scalar_t>
__global__ void ROIPoolKernelForward(
    const scalar_t* const __restrict__ FM,  // (C, H, W)
    const scalar_t* const __restrict__ rois,  // (|R|, 4)
    scalar_t* const __restrict__ out,  // (|R|, C, rHW, rHW)
    const int iR,  // number of input rois
    const int iC,  // number of input channels
    const int iH,  // input height
    const int iW,  // input width
    const int rHW  // height and width of pooled output. rHW^2 bins per ROI.
)
{
    for (
        int ind = blockIdx.x * blockDim.x + threadIdx.x;
        ind < (iR * iC * rHW * rHW);
        ind += blockDim.x * gridDim.x
    )
    {
        // this thread is responsible for computing the value of out[r, c, j, k]
        // determine which (r, c, j, k) this thread is responsible for by mapping
        // from offset in contiguous memory to a 4-dimensional index.
        const int r(ind / rHW / rHW / iC);
        const int c(ind / rHW / rHW % iC);
        const int i(ind / rHW % rHW);
        const int j(ind % rHW);

        // get coordinates of roi r (ijhw, fractional)
        const scalar_t* const roi_start(rois + r*4);
        const scalar_t rI(roi_start[0]);
        const scalar_t rJ(roi_start[1]);
        const scalar_t rH(roi_start[2]);
        const scalar_t rW(roi_start[3]);
        
        // 1. get coordinates of bin b (ijhw, fractional)
        const scalar_t bH(rH / rHW);
        const scalar_t bW(rW / rHW);
        const scalar_t bI(clamp(rI - rH/2) + (static_cast<scalar_t>(i) + 0.5) * bH);
        const scalar_t bJ(clamp(rJ - rW/2) + (static_cast<scalar_t>(j) + 0.5) * bW);
        // 2. convert coordinates of b from ijhw to ijij
        // 3. clamp to [0, 1]
        // 4. convert from fractional coordinates to pixel indices
        // 5. i0, j0 get rounded down, i1, j1 get rounded up
        const int BI0(floor(clamp(bI - bH/2) * iH));
        const int BJ0(floor(clamp(bJ - bW/2) * iW));
        const int BI1(ceil (clamp(bI + bH/2) * iH));
        const int BJ1(ceil (clamp(bJ + bW/2) * iW));
        const int binNumel((BI1 - BI0) * (BJ1 - BJ0));

        // iterate over all pixels in bin b, only looking at a single
        // feature map channel
        const int channelOffset(c * iW * iH);
        for (int pI = BI0; pI < BI1; ++pI) {
            for (int pJ = BJ0; pJ < BJ1; ++pJ) {
                out[ind] += FM[channelOffset + pI * iW + pJ];
            }
        }
        out[ind] /= binNumel;  // average pooling
    }
}


// TODO - rewrite without atomic operations
template <typename scalar_t>
__global__ void ROIPoolKernelBackward(
    const scalar_t* const __restrict__ gradOut,  // (|R|, C, rHW, rHW)
    const scalar_t* const __restrict__ rois,  // (|R|, 4)
    scalar_t* const __restrict__ gradIn,  // (C, H, W)
    const int iR,  // number of input rois
    const int iC,  // number of input channels
    const int iH,  // input height
    const int iW,  // input width
    const int rHW  // height and width of pooled output. rHW^2 bins per ROI.
)
{
    for (
        int ind = blockIdx.x * blockDim.x + threadIdx.x;
        ind < (iR * iC * rHW * rHW);
        ind += blockDim.x * gridDim.x
    )
    {
        // this thread is responsible for computing the value of out[r, c, j, k]
        // determine which (r, c, j, k) this thread is responsible for by mapping
        // from offset in contiguous memory to a 4-dimensional index.
        const int r(ind / rHW / rHW / iC);
        const int c(ind / rHW / rHW % iC);
        const int i(ind / rHW % rHW);
        const int j(ind % rHW);

        // get coordinates of roi r (ijhw, fractional)
        const scalar_t* const roi_start(rois + r*4);
        const scalar_t rI(roi_start[0]);
        const scalar_t rJ(roi_start[1]);
        const scalar_t rH(roi_start[2]);
        const scalar_t rW(roi_start[3]);
        
        // 1. get coordinates of bin b (ijhw, fractional)
        const scalar_t bH(rH / rHW);
        const scalar_t bW(rW / rHW);
        const scalar_t bI(clamp(rI - rH/2) + (static_cast<scalar_t>(i) + 0.5) * bH);
        const scalar_t bJ(clamp(rJ - rW/2) + (static_cast<scalar_t>(j) + 0.5) * bW);
        // 2. convert coordinates of b from ijhw to ijij
        // 3. clamp to [0, 1]
        // 4. convert from fractional coordinates to pixel indices
        // 5. i0, j0 get rounded down, i1, j1 get rounded up
        const int BI0(floor(clamp(bI - bH/2) * iH));
        const int BJ0(floor(clamp(bJ - bW/2) * iW));
        const int BI1(ceil (clamp(bI + bH/2) * iH));
        const int BJ1(ceil (clamp(bJ + bW/2) * iW));
        const int binNumel((BI1 - BI0) * (BJ1 - BJ0));

        const scalar_t gradVal(gradOut[ind]);
        // iterate over all pixels in bin b, only looking at a single
        // feature map channel
        const int channelOffset(c * iW * iH);
        for (int pI = BI0; pI < BI1; ++pI) {
            for (int pJ = BJ0; pJ < BJ1; ++pJ) {
                const int inputOffset(channelOffset + pI * iW + pJ);
                // dL/dIn = dL/dOut * (dOut/dIn = 1/n)
                atomicAdd(gradIn + inputOffset, gradVal / binNumel);
            }
        }
    }
}


at::Tensor ROIPoolCudaForward(
    const at::Tensor& FM,  // (C, H, W)
    const at::Tensor& rois,  // (|R|, 4)
    const int rHW  // height and width of pooled feature map
)
{
    const int iR(rois.size(0));
    const int iC(FM.size(0));
    const int iH(FM.size(1));
    const int iW(FM.size(2));

    at::Tensor out = at::zeros({iR, iC, rHW, rHW}, FM.options());

    const dim3 numBlocks(ceilDivide(out.numel(), THREADS_PER_BLOCK));

    AT_DISPATCH_FLOATING_TYPES(
        out.type(), "ROIPoolKernelForward", ([&] {
            ROIPoolKernelForward<scalar_t>
            <<<THREADS_PER_BLOCK, numBlocks>>>(
                FM.data<scalar_t>(),
                rois.data<scalar_t>(),
                out.data<scalar_t>(),
                iR, iC, iW, iW,
                rHW
            );
        })
    );
    return out;
}


at::Tensor ROIPoolCudaBackward(
    const at::Tensor& gradOut,  // (|R|, C, rHW, rHW)
    const at::Tensor& rois,  // (|R|, 4)
    const int iH,  // input height
    const int iW  // input width
)
{
    const int iR(gradOut.size(0));
    const int iC(gradOut.size(1));
    const int rHW(gradOut.size(2));

    at::Tensor gradIn = at::zeros({iC, iH, iW}, gradOut.options());

    const dim3 numBlocks(ceilDivide(gradOut.numel(), THREADS_PER_BLOCK));

    AT_DISPATCH_FLOATING_TYPES(
        gradOut.type(),
        "ROIPoolKernelBackward",
        ([&] {
            ROIPoolKernelBackward<scalar_t>
            <<<THREADS_PER_BLOCK, numBlocks>>>(
                gradOut.data<scalar_t>(),
                rois.data<scalar_t>(),
                gradIn.data<scalar_t>(),
                iR, iC, iH, iW, rHW
            );
        })
    );
    return gradIn;
}