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
void sgemmMps(char tA, char tB, int M, int N, int K,
              const float* A, int lda, const float* B, int ldb,
              float* C, int ldc);
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

void sgemm(char tA, char tB, int M, int N, int K,
           const float* A, int lda, const float* B, int ldb,
           float* C, int ldc) {
    long long macs = (long long)M * N * K;
#ifdef PD_HAVE_MPS
    if (gemmGpuEnabled() && macs >= kGpuMacThreshold) {
        sgemmMps(tA, tB, M, N, K, A, lda, B, ldb, C, ldc);
        return;
    }
#endif
    sgemmCpu(tA, tB, M, N, K, A, lda, B, ldb, C, ldc);
}

}  // namespace nn
