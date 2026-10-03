// Elementwise / small reduction kernels used by the training graph.
//
// Each call is either encoded as a Metal compute shader (no host block) or
// executed as a plain CPU loop, depending on the calling thread's GEMM
// backend. All functions operate on padded Mats (see mat.h).
#pragma once

#include <vector>

#include "nn/mat.h"

namespace nn {

// y[i,j] += b[j]
void kAddBias(Mat& y, const std::vector<float>& b);

// in-place ReLU / its backward
void kRelu(Mat& x);
void kReluBwd(const Mat& pre, const Mat& gradOut, Mat& gradIn);

// gp[b,q] = gi[b,q] + gh[b,q] + bi[q] + bh[q]
void kGateAdd(Mat& gp, const Mat& gi, const Mat& gh,
              const std::vector<float>& bi, const std::vector<float>& bh);

// LSTM cell: given gate preactivations gp (B x 4h) and previous c,
// write h/c. cp == nullptr denotes the t == 0 zero state.
void kLstmCellFwd(const Mat& gp, const float* cp, Mat& hOut, Mat& cOut,
                  int hidden);

// Backward through the cell. dh/dc are the running next-state gradients
// (updated in place); dg is this step's gate gradient (B x 4h).
// gh is the external gradient added to dh. cp may be null at t == 0.
void kLstmCellBwd(const Mat& gh, const Mat& gp, const Mat& cNow,
                  const float* cp, Mat& dh, Mat& dc, Mat& dg, int hidden);

// z[i,0:n1) = a[i,:]; z[i,n1:n1+n2) = b[i,:]
void kConcat(Mat& z, const Mat& a, int n1, const Mat& b, int n2);
// inverse: split z rows into a (first n1) and b (next n2)
void kSplit(const Mat& z, int n1, Mat& a, Mat& b, int n2);

// Fill ghAll (B*T x h) with zero and copy gh into the last time block.
void kZeroAndLast(Mat& ghAll, const Mat& gh, int B, int T, int h);

// logits[i,a] = mask>0.5 ? logits[i,a] + ds[(i*N+a),0] : -1e9
void kMaskDyn(Mat& logits, const Mat& ds, const Mat& mask, int actions);

// dst[0..n) += packed/padded src rows (src is rows x cols padded).
// gSlot is the parameter's persistent device-gradient cache slot.
void kAddTo(float* dst, void** gSlot, int dstCols, const Mat& src);

// db[j] += sum_i g[i,j]   (g is B x o padded)
void kBiasGradAdd(const Mat& g, float* db, void** dbSlot);

// dst[i,:] = src[i, off : off+h]
void kSlice(Mat& dst, const Mat& src, int off, int h);

// flatten B x N (padded) into (B*N) x 1 packed
void kFlatten(Mat& dst, const Mat& src, int N);

// gDst[:,0] = src[i,col0+i offset] generic dense copy between padded mats
void kCopyMat(Mat& dst, const Mat& src);

// Adam step over one packed parameter vector (w/dw/m/v each n floats)
void kAdam(std::vector<float>& w, std::vector<float>& dw,
           std::vector<float>& m, std::vector<float>& v, int n, float lr,
           float beta1, float beta2, float eps, float bc1, float bc2);

}  // namespace nn
