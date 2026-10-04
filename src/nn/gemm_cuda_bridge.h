#pragma once

#include <cstddef>

struct ID3D12Device;
struct ID3D12CommandQueue;
struct ID3D12Fence;
struct ID3D12Resource;

namespace nn {

// Runs one row-major FP32 GEMM in cuBLAS over shared D3D12 resources.
// The caller must have completed all D3D work that touches these resources.
bool cudaBridgeGemm(ID3D12Device* device,
                    ID3D12CommandQueue* queue,
                    ID3D12Fence* fence, unsigned long long waitValue,
                    unsigned long long signalValue,
                    ID3D12Resource* a, size_t aOff,
                    ID3D12Resource* b, size_t bOff,
                    ID3D12Resource* c, size_t cOff,
                    char transA, char transB,
                    int M, int N, int K, int lda, int ldb, int ldc,
                    float* hostOut, size_t hostOutBytes);

const char* cudaBridgeError();

}  // namespace nn
