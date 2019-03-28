#include <ATen/ATen.h>
#include "common/cuda_common.cuh"


// TODO - maybe in the future can use ROI-warping concepts so that
// we can get gradients wrt the ROIs as well.


template<typename scalar_t>
__global__ void psROIPoolKernelForward(
    const scalar_t* const __restrict__ FM,  // (nTargets * rHW^2, H, W). input feature map.
    const scalar_t* const __restrict__ rois,  // (|R|, 4). fractional, ijhw rois.
    scalar_t* const __restrict__ out,  // (|R|, nTargets, rHW, rHW). pooled features to output.
    const int nTargets,  // number of prediction targets per roi.
    const int rHW,  // height and width of pooled features.
    const int iR,  // input rois.
    const int iH,  // input feature map height.
    const int iW  // input feature map width.
)
{
    for (  // grid-stride loop
        int ind = blockIdx.x * blockDim.x + threadIdx.x;
        ind < (iR * nTargets * rHW * rHW);  // output numel
        ind += blockDim.x * gridDim.x
    )
    {
        // this thread is responsible for computing the value of out[r, t, i, j]
        // determine which (r, t, i, j) this thread is responsible for by
        // mapping from offset in contiguous memory to a 4-dimensional index.
        const int r(ind / rHW / rHW / nTargets);
        const int t(ind / rHW / rHW % nTargets);
        const int i(ind / rHW % rHW);
        const int j(ind % rHW);

        // get coordinates of roi r (ijhw, fractional)
        const scalar_t* const roi_start(rois + r*4);
        const scalar_t rI(roi_start[0]);
        const scalar_t rJ(roi_start[1]);
        const scalar_t rH(roi_start[2]);
        const scalar_t rW(roi_start[3]);

        // 1. get coordinates of this threads cell c within roi r (ijhw, fractional)
        const scalar_t cH(rH / rHW);  // height of any cell in this roi
        const scalar_t cW(rW / rHW);  // width of any cell in this roi
        const scalar_t cI(clamp(rI - rH/2) + (static_cast<scalar_t>(i) + 0.5) * cH);  // of cell c
        const scalar_t cJ(clamp(rJ - rW/2) + (static_cast<scalar_t>(j) + 0.5) * cW);  // of cell c
        // 2. convert from ijhw to ijij and clamp to [0, 1].
        // 3. convert from fractional to pixel coordinates.
        // 4. take floor of i0, j0 and ceil of i1, j1.
        // result is ijij feature map pixel coordinates for cell c.
        const int cI0(floor(clamp(cI - cH/2) * iH));
        const int cJ0(floor(clamp(cJ - cW/2) * iW));
        const int cI1(ceil(clamp(cI + cH/2) * iH));
        const int cJ1(ceil(clamp(cJ + cW/2) * iW));

        // this thread only considers a single feature map channel,
        // which is determined by (t, i, j).
        const int targetChannel((t + 1) * (i * rHW + j));
        const scalar_t* const FMChannel(FM + targetChannel*iW*iH);
        // iterate over each feature map pixel in target channel of cell c
        for (int pI = cI0; pI < cI1; ++pI) {
            for (int pJ = cJ0; pJ < cJ1; ++pJ) {
                out[ind] += FMChannel[pI * iW + pJ];  // accumulate values for averaging
            }
        }
        // average pool - divide by pooledNumel
        out[ind] /= ((cI1 - cI0) * (cJ1 - cJ0));
    }
}


/* TODO - rewrite without atomic ops */
template<typename scalar_t>
__global__ void psROIPoolKernelBackward(
    const scalar_t* const __restrict__ gradOut, // (|R|, nTargets, rHW, rHW).
    const scalar_t* const __restrict__ rois,  // (|R|, 4).
    scalar_t* const __restrict__ gradIn,  // (nTargets * rHW^2, H, W), assumed to be zero-initialized.
    const int nTargets,  // number of prediction targets per roi.
    const int rHW,  // pool height and width.
    const int iR,  // input rois.
    const int iH,  // input feature map height.
    const int iW  // input feature map width.
)
{
    for (  // grid-stride loop
        int ind = blockIdx.x * blockDim.x + threadIdx.x;
        ind < (iR * nTargets * rHW * rHW);  // output numel
        ind += blockDim.x * gridDim.x
    )
    {
        // this thread is responsible for computing derivatives related to
        // out[r, t, i, j].
        // determine which (r, t, i, j) this thread is responsible for by
        // mapping from offset in contiguous memory to a 4-dimensional index.
        const int r(ind / rHW / rHW / nTargets);
        const int t(ind / rHW / rHW % nTargets);
        const int i(ind / rHW % rHW);
        const int j(ind % rHW);

        // get coordinates of roi r (ijhw, fractional)
        const scalar_t* const roi_start(rois + r*4);
        const scalar_t rI(roi_start[0]);
        const scalar_t rJ(roi_start[1]);
        const scalar_t rH(roi_start[2]);
        const scalar_t rW(roi_start[3]);

        // 1. get coordinates of this threads cell c within roi r (ijhw, fractional)
        const scalar_t cH(rH / rHW);  // height of any cell in this roi
        const scalar_t cW(rW / rHW);  // width of any cell in this roi
        const scalar_t cI(clamp(rI - rH/2) + (static_cast<scalar_t>(i) + 0.5) * cH);  // of cell c
        const scalar_t cJ(clamp(rJ - rW/2) + (static_cast<scalar_t>(j) + 0.5) * cW);  // of cell c
        // 2. convert from ijhw to ijij and clamp to [0, 1].
        // 3. convert from fractional to pixel coordinates.
        // 4. take floor of i0, j0 and ceil of i1, j1.
        // result is ijij feature map pixel coordinates for cell c.
        const int cI0(floor(clamp(cI - cH/2) * iH));
        const int cJ0(floor(clamp(cJ - cW/2) * iW));
        const int cI1(ceil(clamp(cI + cH/2) * iH));
        const int cJ1(ceil(clamp(cJ + cW/2) * iW));

        // number of pixels in this roi cell.
        const int roiNumel((cI1 - cI0) * (cJ1 - cJ0));

        // this thread only considers a single channel of gradients,
        // which is determined by (t, i, j).
        const int targetChannel((t + 1) * (i * rHW + j));
        scalar_t* const gradInChannel(gradIn + targetChannel*iW*iH);
        // iterate over each gradient map pixel in target channel of cell c
        for (int pI = cI0; pI < cI1; ++pI) {
            for (int pJ = cJ0; pJ < cJ1; ++pJ) {
                // dL/dIn = dL/dOut * (dOut/dIn = 1/n)
                atomicAdd(gradInChannel + (pI * iW + pJ), gradOut[ind] / roiNumel);
            }
        }
    }
}


at::Tensor psROIPoolCudaForward(
    const at::Tensor& FM,  // (nTargets * rHW^2, H, W)
    const at::Tensor& rois,  // (|R|, 4)
    const int nTargets,  // number of prediction targets per roi.
    const int rHW  // roi height and width
)
{
    const int iR(rois.size(0));  // number of input ROIS
    const int iH(FM.size(1));  // input feature map height
    const int iW(FM.size(2));  // input feature map width

    at::Tensor out = at::zeros({iR, nTargets, rHW, rHW}, FM.options());

    const dim3 numBlocks(ceilDivide(out.numel(), THREADS_PER_BLOCK));

    AT_DISPATCH_FLOATING_TYPES(
        out.type(), "psROIPoolKernelForward", ([&] {
            psROIPoolKernelForward<scalar_t>
            <<<THREADS_PER_BLOCK, numBlocks>>>(
                FM.data<scalar_t>(),
                rois.data<scalar_t>(),
                out.data<scalar_t>(),
                nTargets,
                rHW,
                iR, iH, iW
            );
        })
    );
    return out;
}


at::Tensor psROIPoolCudaBackward(
    const at::Tensor& gradOut,  // (|R|, nTargets, rHW, rHW)
    const at::Tensor& rois,  // (|R|, 4)
    const int iH,  // input feature map height
    const int iW  // input feature map width
)
{
    const int iR(gradOut.size(0));  // input rois
    const int nTargets(gradOut.size(1));  // number of targets per ROI
    const int rHW(gradOut.size(2));  // height and width of pooled output

    at::Tensor gradIn = at::zeros({nTargets * rHW * rHW, iH, iW}, gradOut.options());

    const dim3 numBlocks(ceilDivide(gradOut.numel(), THREADS_PER_BLOCK));

    AT_DISPATCH_FLOATING_TYPES(
        gradIn.type(), "psROIPoolKernelBackward", ([&] {
            psROIPoolKernelBackward<scalar_t>
            <<<THREADS_PER_BLOCK, numBlocks>>>(
                gradOut.data<scalar_t>(),
                rois.data<scalar_t>(),
                gradIn.data<scalar_t>(),
                nTargets, rHW, iR, iH, iW
            );
        })
    );
    return gradIn;
}
