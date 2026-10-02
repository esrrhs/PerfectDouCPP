// Dense GEMM dispatch layer.
//
//   C = op(A) * op(B), row-major, FP32
//     transA/transB == 'N': no transpose, 'T': transpose
//   physical A dims: transA=='N' ? M x K : K x M
//   physical B dims: transB=='N' ? K x N : N x K
//   C dims: M x N
//
// On Apple platforms a Metal/MPS implementation is compiled in and selected
// for large products (small products stay on CPU where dispatch overhead
// would dominate). Elsewhere everything runs on CPU.
#pragma once

namespace nn {

void gemmInit();                       // initialize GPU backend (if present)
bool gemmHasGpu();
void gemmSetGpu(bool enabled);         // process-wide default
bool gemmGpuEnabled();
void gemmSetThreadGpu(int enabled);    // per-thread override (-1 = follow default)

void sgemm(char transA, char transB, int M, int N, int K,
           const float* A, int lda, const float* B, int ldb,
           float* C, int ldc);

}  // namespace nn
