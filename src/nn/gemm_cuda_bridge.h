#pragma once

#include <cstddef>

struct ID3D12Device;
struct ID3D12Resource;

namespace nn {

// Queues cuBLAS / CUDA work on one stream. Mappings of D3D12 resources are
// cached. Nothing is synchronized until cudaBridgeSync.
bool cudaBridgeGemm(ID3D12Device* device,
                    ID3D12Resource* a, size_t aOff,
                    ID3D12Resource* b, size_t bOff,
                    ID3D12Resource* c, size_t cOff,
                    char transA, char transB,
                    int M, int N, int K, int lda, int ldb, int ldc);

bool cudaBridgeKernel(ID3D12Device* device, const char* name,
                      ID3D12Resource** resources, const size_t* offsets, int nbuf,
                      const void* cst, size_t cbytes,
                      unsigned gridX, unsigned gridY);

// Seat index selects which CUDA stream later launches and syncs use.
// Streams do not wait for each other.
void cudaBridgeSetStream(int seat);
bool cudaBridgeHasWork();
bool cudaBridgeSync();
bool cudaBridgeSyncAll();
bool cudaBridgeDtoH(ID3D12Device* device, ID3D12Resource* resource, size_t offset,
                    void* host, size_t bytes);
void cudaBridgeDrop(ID3D12Resource* resource);
void cudaBridgeReset();

const char* cudaBridgeError();

}  // namespace nn
