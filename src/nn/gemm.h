// Dense GEMM dispatch + GPU stream layer.
//
//   C = op(A) * op(B), row-major, FP32
//     transA/transB == 'N': no transpose, 'T': transpose
//   physical A dims: transA=='N' ? M x K : K x M
//   physical B dims: transB=='N' ? K x N : N x K
//   C dims: M x N
//
// lda/ldb/ldc are row strides in *floats* (may exceed the logical column
// count when matrices are padded).
//
// On Apple platforms a Metal implementation (in-house GEMM + compute kernels)
// is compiled in. On Windows the same stream is implemented with Direct3D 12
// compute shaders (gemm_d3d.cpp); discrete GPUs copy host memory explicitly.
// The learning graph uses a streaming model: gpuCommitGemms()
// / the elementwise kernel
// wrappers encode work onto one per-thread command queue WITHOUT blocking,
// and gpuWait() fences only at the few points where the host has to read
// results. Command buffers on a queue execute FIFO, so data dependencies are
// respected without per-op round trips.
#pragma once
#include <cstddef>
#include <functional>
namespace nn {

void gemmInit();                       // initialize GPU backend (if present)
bool gemmHasGpu();
void gemmSetGpu(bool enabled);         // process-wide default
bool gemmGpuEnabled();
const char* gemmGpuLabel();            // "GPU (Metal)" / "GPU (D3D12: ...)"
void gemmSetThreadGpu(int enabled);    // per-thread override (-1 = follow default)

void sgemm(char transA, char transB, int M, int N, int K,
           const float* A, int lda, const float* B, int ldb,
           float* C, int ldc);

// One independent GEMM in a batch.
struct GemmOp {
    char transA = 'N', transB = 'N';
    int M = 0, N = 0, K = 0;
    const float* A = nullptr;
    int lda = 0;
    const float* B = nullptr;
    int ldb = 0;
    float* C = nullptr;
    int ldc = 0;
    // When non-null, B is a model weight and *wSlot caches its persistent
    // device copy (owned by the host-side Mat/Param) for the whole update.
    void** wSlot = nullptr;
};

// Encode a set of independent products and return immediately (GPU) or run
// them straight away (CPU fallback).
void gpuCommitGemms(int count, const GemmOp* ops);
// One product on the custom tiled kernel. epi 0 = plain; 1 = bias then ReLU.
// biasFloats gives the bias vector length (0 if none).
void gpuGemm(const GemmOp& op, const float* bias, int biasFloats, int epi);
// Block until everything previously encoded has finished. Marked device
// results are flushed to host; when keepWindow is true the residency binds
// stay alive (needed so the backward pass reads forward activations).
void gpuWaitEx(bool keepWindow);
inline void gpuWait() { gpuWaitEx(false); }
// Bind later GPU work and stream syncs on this thread to a seat. Seats do
// not share scratch or a CUDA stream. The D3D queue stays the one owner queue.
void gpuBindSeat(int seat);
// Launch the queued wave and return. The stream keeps running.
void gpuSubmit();
// Wait for the stream selected by gpuBindSeat. Other seats are not waited on.
void gpuSync();
// Copy a host buffer into the current residency window (read-write scratch
// initialized on the host, e.g. zeroed recurrent-gradient states).
void gpuStageInput(float* p, int floats);
// Mark a device buffer as needed by the host at the next fence (flush it).
void gpuMarkHost(float* p);
// Release an opaque per-object device cache slot (no-op when empty/CPU).
void gpuDropCache(void** slot);
// Persistent device copy of a model weight, cached in the host object's
// *slot; lazily (re)seeded from host after an invalidation/optimizer step.
// Returns an opaque buffer reference for immediate encoding.
void* gpuWeightCache(void** slot, const void* host, size_t bytes);
// Persistent device gradient accumulator cached in *slot, seeded once from
// the zeroed host gradient and reused across residency windows.
void* gpuGradCache(void** slot, const void* host, size_t bytes);
// Copy a persistent gradient cache back into its host vector after a fence.
void gpuFlushGrad(void* slot, void* host, size_t bytes);
// Print/reset per-thread backend stats (no-op without Metal).
void gpuPrintStats(const char* tag);
// True when the calling thread dispatches work to the GPU.
bool gpuActive();
// False after the D3D12 device is removed. Host weights are unchanged until
// the next successful fence. Other platforms are always ok.
bool gpuDeviceOk();
// Run fn on the single thread that owns the D3D12 compute queue. Nested
// calls from that thread run inline. Other platforms call fn directly.
void gpuInvoke(const std::function<void()>& fn);
// Drop the compute queue and every device allocation it owns. The queue is
// released on the thread that created it.
void gpuReleaseThread();
// Recreate the process-wide device after every learner has released its
// queue. Returns false when the GPU cannot be brought back.
bool gpuRecreate();

}  // namespace nn
