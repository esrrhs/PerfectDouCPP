// CPU GEMM reference implementation and dispatch logic.
#include "nn/gemm.h"

#include <algorithm>

namespace nn {

namespace {
bool g_enabled =
#ifdef PD_HAVE_MPS
    true;
#else
    false;
#endif
thread_local int t_override = -1;  // -1: follow g_enabled, 0/1: explicit

// Above this MAC count the GPU is used; below it, launch/sync overhead would
// dominate small products (256x256 and smaller).
constexpr long long kGpuMacThreshold = 1000000;
}  // namespace

void sgemmCpu(char tA, char tB, int M, int N, int K,
              const float* A, int lda, const float* B, int ldb,
              float* C, int ldc) {
    if (tA == 'N' && tB == 'N') {
        // i-k-j: contiguous inner loops, auto-vectorizes
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j) C[size_t(i) * ldc + j] = 0.0f;
        for (int i = 0; i < M; ++i) {
            const float* ai = A + size_t(i) * lda;
            float* ci = C + size_t(i) * ldc;
            for (int t = 0; t < K; ++t) {
                float a = ai[t];
                if (a == 0.0f) continue;
                const float* bt = B + size_t(t) * ldb;
                for (int j = 0; j < N; ++j) ci[j] += a * bt[j];
            }
        }
        return;
    }
    // transposed cases (only small products reach CPU): plain dot products
    auto aAt = [&](int i, int t) {
        return tA == 'N' ? A[size_t(i) * lda + t] : A[size_t(t) * lda + i];
    };
    auto bAt = [&](int j, int t) {
        return tB == 'N' ? B[size_t(t) * ldb + j] : B[size_t(j) * ldb + t];
    };
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float s = 0.0f;
            for (int t = 0; t < K; ++t) s += aAt(i, t) * bAt(j, t);
            C[size_t(i) * ldc + j] = s;
        }
}

// MPS implementation (Apple only), defined in gemm_mps.mm
#ifdef PD_HAVE_MPS
void mpsCommitGemms(int count, const GemmOp* ops, long long macs);
void mpsCustomGemm(const GemmOp& op, const void* bias, size_t biasBytes,
                   int epi);
void mpsWait(bool keepWindow);
void mpsStageInput(const void* p, size_t bytes);
void mpsPrintStats(const char* tag);
void mpsMarkHost(const void* p);
void mpsDropCache(void** slot);
void* mpsWeightCache(void** slot, const void* host, size_t bytes);
void* mpsGradCache(void** slot, const void* host, size_t bytes);
void mpsFlushGrad(void* slot, void* host, size_t bytes);
#endif

void gemmInit() {
#ifdef PD_HAVE_MPS
    extern bool mpsInit();
    if (!mpsInit()) g_enabled = false;
#endif
}
bool gemmHasGpu() {
#ifdef PD_HAVE_MPS
    extern bool mpsAvailable();
    return mpsAvailable();
#else
    return false;
#endif
}
void gemmSetGpu(bool enabled) { g_enabled = enabled && gemmHasGpu(); }
void gemmSetThreadGpu(int enabled) { t_override = enabled; }
bool gemmGpuEnabled() {
    return t_override >= 0 ? (t_override != 0) : g_enabled;
}
bool gpuActive() {
#ifdef PD_HAVE_MPS
    return gemmGpuEnabled();
#else
    return false;
#endif
}

void sgemm(char tA, char tB, int M, int N, int K,
           const float* A, int lda, const float* B, int ldb,
           float* C, int ldc) {
    long long macs = (long long)M * N * K;
#ifdef PD_HAVE_MPS
    if (gemmGpuEnabled() && macs >= kGpuMacThreshold) {
        GemmOp g{tA, tB, M, N, K, A, lda, B, ldb, C, ldc};
        mpsCommitGemms(1, &g, macs);
        mpsMarkHost(C);  // standalone call: result is read on host
        mpsWait(false);
        return;
    }
#endif
    sgemmCpu(tA, tB, M, N, K, A, lda, B, ldb, C, ldc);
}

void gpuGemm(const GemmOp& g, const float* bias, int biasFloats, int epi) {
#ifdef PD_HAVE_MPS
    if (gemmGpuEnabled()) {
        size_t bbytes = biasFloats
            ? ((size_t(biasFloats) * 4 + 15) & ~size_t(15)) : 0;
        mpsCustomGemm(g, bias, bbytes, epi);
        return;
    }
#else
    (void)bias; (void)biasFloats; (void)epi;
#endif
    sgemmCpu(g.transA, g.transB, g.M, g.N, g.K, g.A, g.lda, g.B, g.ldb,
             g.C, g.ldc);
    if (epi == 1 && bias) {
        for (int i = 0; i < g.M; ++i)
            for (int j = 0; j < g.N; ++j) {
                float v = g.C[i * g.ldc + j] + bias[j];
                g.C[i * g.ldc + j] = v > 0.0f ? v : 0.0f;
            }
    }
}

void gpuCommitGemms(int count, const GemmOp* ops) {
    if (count <= 0) return;
    long long macs = 0;
    for (int i = 0; i < count; ++i)
        macs += (long long)ops[i].M * ops[i].N * ops[i].K;
#ifdef PD_HAVE_MPS
    // The residency model keeps intermediates on the device, so even tiny
    // products must go through the GPU (a CPU fallback would read stale
    // memory).
    if (gemmGpuEnabled()) {
        mpsCommitGemms(count, ops, macs);
        return;
    }
#endif
    for (int i = 0; i < count; ++i) {
        const GemmOp& g = ops[i];
        sgemmCpu(g.transA, g.transB, g.M, g.N, g.K, g.A, g.lda, g.B, g.ldb,
                 g.C, g.ldc);
    }
}

void gpuWaitEx(bool keepWindow) {
#ifdef PD_HAVE_MPS
    if (gemmGpuEnabled()) mpsWait(keepWindow);
#else
    (void)keepWindow;
#endif
}

void gpuStageInput(float* p, int floats) {
#ifdef PD_HAVE_MPS
    if (gemmGpuEnabled()) mpsStageInput(p, size_t(floats) * 4);
#else
    (void)p; (void)floats;
#endif
}

void gpuPrintStats(const char* tag) {
#ifdef PD_HAVE_MPS
    if (gemmGpuEnabled()) mpsPrintStats(tag);
#else
    (void)tag;
#endif
}

void gpuMarkHost(float* p) {
#ifdef PD_HAVE_MPS
    if (gemmGpuEnabled()) mpsMarkHost(p);
#else
    (void)p;
#endif
}

void gpuDropCache(void** slot) {
#ifdef PD_HAVE_MPS
    if (slot != nullptr && *slot != nullptr) mpsDropCache(slot);
#else
    (void)slot;
#endif
}

void* gpuWeightCache(void** slot, const void* host, size_t bytes) {
#ifdef PD_HAVE_MPS
    if (gemmGpuEnabled()) return mpsWeightCache(slot, host, bytes);
#else
    (void)slot; (void)host; (void)bytes;
#endif
    return nullptr;
}

void* gpuGradCache(void** slot, const void* host, size_t bytes) {
#ifdef PD_HAVE_MPS
    if (gemmGpuEnabled()) return mpsGradCache(slot, host, bytes);
#else
    (void)slot; (void)host; (void)bytes;
#endif
    return nullptr;
}

void gpuFlushGrad(void* slot, void* host, size_t bytes) {
#ifdef PD_HAVE_MPS
    if (gemmGpuEnabled()) mpsFlushGrad(slot, host, bytes);
#else
    (void)slot; (void)host; (void)bytes;
#endif
}

}  // namespace nn
