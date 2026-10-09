// AVX2/FMA micro-kernel for the Windows CPU inference path.
// This translation unit is compiled with AVX2 enabled, but callers check
// pdAvx2Available() before entering it so older x86 machines still work.
#include <algorithm>
#include <cstddef>
#include <immintrin.h>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace nn {

bool pdAvx2Available() {
#if defined(_MSC_VER)
    int info[4] = {};
    __cpuid(info, 1);
    const bool osxsave = (info[2] & (1 << 27)) != 0;
    const bool avx = (info[2] & (1 << 28)) != 0;
    const bool fma = (info[2] & (1 << 12)) != 0;
    if (!osxsave || !avx || !fma || (_xgetbv(0) & 6) != 6) return false;
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 5)) != 0;
#elif defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

namespace {

inline void zeroN(float* c, int n) {
    const __m256 z = _mm256_setzero_ps();
    int j = 0;
    for (; j + 32 <= n; j += 32) {
        _mm256_storeu_ps(c + j, z);
        _mm256_storeu_ps(c + j + 8, z);
        _mm256_storeu_ps(c + j + 16, z);
        _mm256_storeu_ps(c + j + 24, z);
    }
    for (; j + 8 <= n; j += 8) _mm256_storeu_ps(c + j, z);
    for (; j < n; ++j) c[j] = 0.0f;
}

#if defined(_MSC_VER)
#define PD_RESTRICT __restrict
#else
#define PD_RESTRICT __restrict__
#endif

inline void saxpy(float* PD_RESTRICT c, const float* PD_RESTRICT b, float a,
                  int n) {
    const __m256 va = _mm256_set1_ps(a);
    int j = 0;
    for (; j + 32 <= n; j += 32) {
        __m256 c0 = _mm256_loadu_ps(c + j);
        __m256 c1 = _mm256_loadu_ps(c + j + 8);
        __m256 c2 = _mm256_loadu_ps(c + j + 16);
        __m256 c3 = _mm256_loadu_ps(c + j + 24);
        c0 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b + j), c0);
        c1 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b + j + 8), c1);
        c2 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b + j + 16), c2);
        c3 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b + j + 24), c3);
        _mm256_storeu_ps(c + j, c0);
        _mm256_storeu_ps(c + j + 8, c1);
        _mm256_storeu_ps(c + j + 16, c2);
        _mm256_storeu_ps(c + j + 24, c3);
    }
    for (; j + 8 <= n; j += 8) {
        __m256 cv = _mm256_fmadd_ps(
            va, _mm256_loadu_ps(b + j), _mm256_loadu_ps(c + j));
        _mm256_storeu_ps(c + j, cv);
    }
    for (; j < n; ++j) c[j] += a * b[j];
}

}  // namespace

void pdAvx2SgemmPanel(int M, int N, int K, const float* A, int lda,
                      const float* B, int ldb, float* C, int ldc,
                      bool transA) {
    constexpr int MR = 4;
    for (int i0 = 0; i0 < M; i0 += MR) {
        const int mr = std::min(MR, M - i0);
        for (int i = 0; i < mr; ++i) zeroN(C + size_t(i0 + i) * ldc, N);
        for (int k = 0; k < K; ++k) {
            float a[MR];
            bool any = false;
            for (int i = 0; i < mr; ++i) {
                a[i] = transA ? A[size_t(k) * lda + i0 + i]
                              : A[size_t(i0 + i) * lda + k];
                any = any || a[i] != 0.0f;
            }
            if (!any) continue;
            const float* bk = B + size_t(k) * ldb;
            for (int i = 0; i < mr; ++i)
                if (a[i] != 0.0f)
                    saxpy(C + size_t(i0 + i) * ldc, bk, a[i], N);
        }
    }
}

}  // namespace nn
