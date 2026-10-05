// Dispatch wrappers for the elementwise training kernels.
//
// GPU entry points are compiled in the Metal (Apple) or D3D12 (Windows)
// backend; on other platforms identical loops run on the CPU.
#include <cmath>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include "nn/gemm.h"
#include "nn/kernels.h"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#if defined(__APPLE__)
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#define PD_HAVE_ACCEL
#endif

#if defined(PD_HAVE_MPS)
#define PD_HAVE_GPU 1
#define PD_BE(fn) mps##fn
#elif defined(PD_HAVE_D3D)
#define PD_HAVE_GPU 1
#define PD_BE(fn) d3d##fn
#endif

namespace nn {

#ifdef PD_HAVE_GPU
void PD_BE(AddBias)(Mat& y, const std::vector<float>& b);
void PD_BE(Relu)(Mat& x);
void PD_BE(ReluBwd)(const Mat& pre, const Mat& gout, Mat& gin);
void PD_BE(GateAdd)(Mat& gp, const Mat& gi, const Mat& gh,
                const std::vector<float>& bi, const std::vector<float>& bh);
void PD_BE(LstmCellFwd)(const Mat& gp, const float* cp, Mat& hOut, Mat& cOut,
                    int hidden);
void PD_BE(LstmCellBwd)(const Mat& gh, const Mat& gp, const Mat& cNow,
                    const float* cp, Mat& dh, Mat& dc, Mat& dg, int hidden);
#if defined(PD_HAVE_D3D)
bool d3dLstmSeqFwd(Mat& statesH, Mat& wTh, const Mat& gateI,
                   const std::vector<float>& bi, const std::vector<float>& bh,
                   Mat& gpre, Mat& statesC, int B, int T, int hidden);
bool d3dLstmSeqBwd(const Mat& ghAll, const Mat& gpre, const Mat& statesC,
                   const float* Wh, void** whSlot, Mat& dgAll,
                   int B, int T, int hidden);
#endif
void PD_BE(Concat)(Mat& z, const Mat& a, int n1, const Mat& b, int n2);
void PD_BE(Split)(const Mat& z, int n1, Mat& a, Mat& b, int n2);
void PD_BE(ZeroAndLast)(Mat& all, const Mat& gh, int B, int T, int h);
void PD_BE(MaskDyn)(Mat& logits, const Mat& ds, const Mat& mask, int N);
void PD_BE(AddTo)(float* dst, void** gSlot, int dstCols, const Mat& src);
void PD_BE(BiasGradAdd)(const Mat& g, float* db, void** dbSlot);
void PD_BE(CopyMat)(Mat& dst, const Mat& src);
void PD_BE(Adam)(std::vector<float>& w, std::vector<float>& dw,
             std::vector<float>& m, std::vector<float>& v, int n, float lr,
             float b1, float b2, float eps, float bc1, float bc2);
#endif

void kAddBias(Mat& y, const std::vector<float>& b) {
#ifdef PD_HAVE_GPU
    if (gpuActive()) { PD_BE(AddBias)(y, b); return; }
#endif
    for (int i = 0; i < y.r; ++i)
        for (int j = 0; j < y.c; ++j) y.row(i)[j] += b[j];
}

void kRelu(Mat& x) {
#ifdef PD_HAVE_GPU
    if (gpuActive()) { PD_BE(Relu)(x); return; }
#endif
    reluFwd(x);
}

void kReluBwd(const Mat& pre, const Mat& gout, Mat& gin) {
    // The GPU path encodes asynchronously and never touches the host Mat, so
    // make sure the output storage/layout exists up front.
    gin.resize(pre.r, pre.c);
#ifdef PD_HAVE_GPU
    if (gpuActive()) { PD_BE(ReluBwd)(pre, gout, gin); return; }
#endif
    reluBwd(pre, gout, gin);
}

void kGateAdd(Mat& gp, const Mat& gi, const Mat& gh,
              const std::vector<float>& bi, const std::vector<float>& bh) {
#ifdef PD_HAVE_GPU
    if (gpuActive()) { PD_BE(GateAdd)(gp, gi, gh, bi, bh); return; }
#endif
    for (int i = 0; i < gp.r; ++i)
        for (int q = 0; q < gp.c; ++q)
            gp.row(i)[q] = gi.row(i)[q] + gh.row(i)[q] + bi[q] + bh[q];
}

namespace {
inline float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }

#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
#define PD_AVX2 __attribute__((target("avx2,fma")))
#else
#define PD_AVX2
#endif

// Cephes expf, about 1 ulp. The libm double exp was the rollout hotspot.
PD_AVX2 __m256 lstmExp8(__m256 x) {
    x = _mm256_min_ps(x, _mm256_set1_ps(88.3762626647949f));
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.3762626647949f));
    const __m256 half = _mm256_set1_ps(0.5f);
    __m256 fx = _mm256_fmadd_ps(x, _mm256_set1_ps(1.44269504088896341f), half);
    fx = _mm256_floor_ps(fx);
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693359375f), x);
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.12194440e-4f), x);
    __m256 z = _mm256_mul_ps(x, x);
    __m256 y = _mm256_set1_ps(1.9875691500E-4f);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507E-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073E-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894E-2f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459E-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201E-1f));
    y = _mm256_fmadd_ps(y, z, x);
    y = _mm256_add_ps(y, _mm256_set1_ps(1.0f));
    __m256i ei = _mm256_add_epi32(_mm256_cvtps_epi32(fx), _mm256_set1_epi32(127));
    ei = _mm256_slli_epi32(ei, 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(ei));
}

PD_AVX2 __m256 lstmSigm8(__m256 x) {
    __m256 e = lstmExp8(_mm256_sub_ps(_mm256_setzero_ps(), x));
    return _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_add_ps(_mm256_set1_ps(1.0f), e));
}

PD_AVX2 __m256 lstmTanh8(__m256 x) {
    __m256 e = lstmExp8(_mm256_mul_ps(x, _mm256_set1_ps(-2.0f)));
    return _mm256_div_ps(_mm256_sub_ps(_mm256_set1_ps(1.0f), e),
                         _mm256_add_ps(_mm256_set1_ps(1.0f), e));
}

PD_AVX2 void lstmCellRowAvx2(const float* g, const float* cp, float* hOut, float* cOut,
                             float* og, float* th, int h) {
    int u = 0;
    for (; u + 8 <= h; u += 8) {
        __m256 iv = lstmSigm8(_mm256_loadu_ps(g + u));
        __m256 fv = lstmSigm8(_mm256_loadu_ps(g + h + u));
        __m256 gv = lstmTanh8(_mm256_loadu_ps(g + 2 * h + u));
        __m256 ov = lstmSigm8(_mm256_loadu_ps(g + 3 * h + u));
        __m256 pc = cp ? _mm256_loadu_ps(cp + u) : _mm256_setzero_ps();
        __m256 cc = _mm256_fmadd_ps(iv, gv, _mm256_mul_ps(fv, pc));
        __m256 tn = lstmTanh8(cc);
        _mm256_storeu_ps(cOut + u, cc);
        _mm256_storeu_ps(hOut + u, _mm256_mul_ps(ov, tn));
        if (og) {
            _mm256_storeu_ps(og + u, iv);
            _mm256_storeu_ps(og + h + u, fv);
            _mm256_storeu_ps(og + 2 * h + u, gv);
            _mm256_storeu_ps(og + 3 * h + u, ov);
        }
        if (th) _mm256_storeu_ps(th + u, tn);
    }
    for (; u < h; ++u) {
        float iv = sigm(g[u]);
        float fv = sigm(g[h + u]);
        float gv = tanhf(g[2 * h + u]);
        float ov = sigm(g[3 * h + u]);
        float pc = cp ? cp[u] : 0.0f;
        float cc = fv * pc + iv * gv;
        float tn = tanhf(cc);
        cOut[u] = cc;
        hOut[u] = ov * tn;
        if (og) {
            og[u] = iv;
            og[h + u] = fv;
            og[2 * h + u] = gv;
            og[3 * h + u] = ov;
        }
        if (th) th[u] = tn;
    }
}
#endif
}

bool pdAvx2Available();

bool kLstmSeqFwd(Mat& statesH, Mat& wTh, const Mat& gateI,
                 const std::vector<float>& bi, const std::vector<float>& bh,
                 Mat& gpre, Mat& statesC, int B, int T, int hidden) {
#if defined(PD_HAVE_D3D)
    if (gpuActive())
        return d3dLstmSeqFwd(statesH, wTh, gateI, bi, bh, gpre, statesC, B, T, hidden);
#else
    (void)statesH; (void)wTh; (void)gateI; (void)bi; (void)bh;
    (void)gpre; (void)statesC; (void)B; (void)T; (void)hidden;
#endif
    return false;
}

bool kLstmSeqBwd(const Mat& ghAll, const Mat& gpre, const Mat& statesC,
                 const float* Wh, void** whSlot, Mat& dgAll,
                 int B, int T, int hidden) {
#if defined(PD_HAVE_D3D)
    if (gpuActive())
        return d3dLstmSeqBwd(ghAll, gpre, statesC, Wh, whSlot, dgAll, B, T, hidden);
#else
    (void)ghAll; (void)gpre; (void)statesC; (void)Wh; (void)whSlot;
    (void)dgAll; (void)B; (void)T; (void)hidden;
#endif
    return false;
}

void kLstmCellFwd(const Mat& gp, const float* cp, Mat& hOut, Mat& cOut,
                  int h, Mat* gates, Mat* tanhC) {
#ifdef PD_HAVE_GPU
    if (gpuActive()) { PD_BE(LstmCellFwd)(gp, cp, hOut, cOut, h); return; }
#endif
#ifdef PD_HAVE_ACCEL
    // vForce exp/tanh over the whole step. Saved gates let backward skip this.
    const int B = gp.r;
    if (B <= 0 || h <= 0) return;
    const int n3 = B * 3 * h, nh = B * h;
    thread_local std::vector<float> neg, ex, sig, gin, gout, cc, tc;
    neg.resize(n3); ex.resize(n3); sig.resize(n3);
    gin.resize(nh); gout.resize(nh); cc.resize(nh); tc.resize(nh);
    for (int b = 0; b < B; ++b) {
        const float* g = gp.row(b);
        float* n = neg.data() + size_t(b) * 3 * h;
        float* gi = gin.data() + size_t(b) * h;
        for (int u = 0; u < h; ++u) {
            n[u] = -g[u];
            n[h + u] = -g[h + u];
            n[2 * h + u] = -g[3 * h + u];
            gi[u] = g[2 * h + u];
        }
    }
    int ns = n3, nt = nh;
    vvexpf(ex.data(), neg.data(), &ns);
    for (int i = 0; i < n3; ++i) ex[i] += 1.0f;
    vvrecf(sig.data(), ex.data(), &ns);
    vvtanhf(gout.data(), gin.data(), &nt);
    for (int b = 0; b < B; ++b) {
        const float* s = sig.data() + size_t(b) * 3 * h;
        const float* gv = gout.data() + size_t(b) * h;
        float* cbuf = cc.data() + size_t(b) * h;
        float* og = gates ? gates->row(b) : nullptr;
        for (int u = 0; u < h; ++u) {
            float iv = s[u], fv = s[h + u], ov = s[2 * h + u];
            float pc = cp ? cp[size_t(b) * h + u] : 0.0f;
            float c = fv * pc + iv * gv[u];
            cOut.row(b)[u] = c;
            cbuf[u] = c;
            if (og) {
                og[u] = iv;
                og[h + u] = fv;
                og[2 * h + u] = gv[u];
                og[3 * h + u] = ov;
            }
        }
    }
    vvtanhf(tc.data(), cc.data(), &nt);
    for (int b = 0; b < B; ++b) {
        const float* s = sig.data() + size_t(b) * 3 * h;
        const float* th = tc.data() + size_t(b) * h;
        for (int u = 0; u < h; ++u) {
            hOut.row(b)[u] = s[2 * h + u] * th[u];
            if (tanhC) tanhC->row(b)[u] = th[u];
        }
    }
#else
#if defined(__x86_64__) || defined(_M_X64)
    static const bool avx = pdAvx2Available();
#else
    const bool avx = false;
#endif
    for (int b = 0; b < gp.r; ++b) {
        const float* g = gp.row(b);
        const float* cprev = cp ? cp + size_t(b) * h : nullptr;
        float* og = gates ? gates->row(b) : nullptr;
        float* th = tanhC ? tanhC->row(b) : nullptr;
#if defined(__x86_64__) || defined(_M_X64)
        if (avx) {
            lstmCellRowAvx2(g, cprev, hOut.row(b), cOut.row(b), og, th, h);
            continue;
        }
#else
        (void)avx;
#endif
        for (int u = 0; u < h; ++u) {
            float iv = sigm(g[u]);
            float fv = sigm(g[h + u]);
            float gv = tanhf(g[2 * h + u]);
            float ov = sigm(g[3 * h + u]);
            float pc = cprev ? cprev[u] : 0.0f;
            float cc = fv * pc + iv * gv;
            float tn = tanhf(cc);
            cOut.row(b)[u] = cc;
            hOut.row(b)[u] = ov * tn;
            if (og) {
                og[u] = iv;
                og[h + u] = fv;
                og[2 * h + u] = gv;
                og[3 * h + u] = ov;
            }
            if (th) th[u] = tn;
        }
    }
#endif
}

void kLstmCellBwd(const Mat& gh, const Mat& gp, const Mat& cNow,
                  const float* cp, Mat& dh, Mat& dc, Mat& dg, int h,
                  const Mat* gates, const Mat* tanhC) {
#ifdef PD_HAVE_GPU
    if (gpuActive()) { PD_BE(LstmCellBwd)(gh, gp, cNow, cp, dh, dc, dg, h); return; }
#endif
    if (gates && tanhC) {
        for (int b = 0; b < gp.r; ++b) {
            const float* og = gates->row(b);
            const float* th = tanhC->row(b);
            for (int u = 0; u < h; ++u) {
                float iv = og[u], fv = og[h + u], gz = og[2 * h + u], ov = og[3 * h + u];
                float t = th[u];
                float dhn = gh.row(b)[u] + dh.row(b)[u];
                float pc = cp ? cp[size_t(b) * h + u] : 0.0f;
                float dct = dhn * ov * (1.0f - t * t) + dc.row(b)[u];
                dg.row(b)[u] = dct * gz * iv * (1.0f - iv);
                dg.row(b)[h + u] = dct * pc * fv * (1.0f - fv);
                dg.row(b)[2 * h + u] = dct * iv * (1.0f - gz * gz);
                dg.row(b)[3 * h + u] = dhn * t * ov * (1.0f - ov);
                dc.row(b)[u] = dct * fv;
            }
        }
        return;
    }
#ifdef PD_HAVE_ACCEL
    const int B = gp.r;
    if (B <= 0 || h <= 0) return;
    const int n3 = B * 3 * h, nh = B * h;
    thread_local std::vector<float> neg, ex, sig, gin, gout, cin, tc;
    neg.resize(n3); ex.resize(n3); sig.resize(n3);
    gin.resize(nh); gout.resize(nh); cin.resize(nh); tc.resize(nh);
    for (int b = 0; b < B; ++b) {
        const float* g = gp.row(b);
        float* n = neg.data() + size_t(b) * 3 * h;
        float* gi = gin.data() + size_t(b) * h;
        float* ci = cin.data() + size_t(b) * h;
        for (int u = 0; u < h; ++u) {
            n[u] = -g[u];
            n[h + u] = -g[h + u];
            n[2 * h + u] = -g[3 * h + u];
            gi[u] = g[2 * h + u];
            ci[u] = cNow.row(b)[u];
        }
    }
    int ns = n3, nt = nh;
    vvexpf(ex.data(), neg.data(), &ns);
    for (int i = 0; i < n3; ++i) ex[i] += 1.0f;
    vvrecf(sig.data(), ex.data(), &ns);
    vvtanhf(gout.data(), gin.data(), &nt);
    vvtanhf(tc.data(), cin.data(), &nt);
    for (int b = 0; b < B; ++b) {
        const float* s = sig.data() + size_t(b) * 3 * h;
        const float* gv = gout.data() + size_t(b) * h;
        const float* tcv = tc.data() + size_t(b) * h;
        for (int u = 0; u < h; ++u) {
            float iv = s[u], fv = s[h + u], ov = s[2 * h + u];
            float gz = gv[u], t = tcv[u];
            float dhn = gh.row(b)[u] + dh.row(b)[u];
            float pc = cp ? cp[size_t(b) * h + u] : 0.0f;
            float dct = dhn * ov * (1.0f - t * t) + dc.row(b)[u];
            dg.row(b)[u] = dct * gz * iv * (1.0f - iv);
            dg.row(b)[h + u] = dct * pc * fv * (1.0f - fv);
            dg.row(b)[2 * h + u] = dct * iv * (1.0f - gz * gz);
            dg.row(b)[3 * h + u] = dhn * t * ov * (1.0f - ov);
            dc.row(b)[u] = dct * fv;
        }
    }
#else
    for (int b = 0; b < gp.r; ++b) {
        for (int u = 0; u < h; ++u) {
            float tc = std::tanh(cNow.row(b)[u]);
            float iv = sigm(gp.row(b)[u]);
            float fv = sigm(gp.row(b)[h + u]);
            float gv = std::tanh(gp.row(b)[2 * h + u]);
            float ov = sigm(gp.row(b)[3 * h + u]);
            float dhn = gh.row(b)[u] + dh.row(b)[u];
            float pc = cp ? cp[b * h + u] : 0.0f;
            float dct = dhn * ov * (1.0f - tc * tc) + dc.row(b)[u];
            dg.row(b)[u] = dct * gv * iv * (1.0f - iv);
            dg.row(b)[h + u] = dct * pc * fv * (1.0f - fv);
            dg.row(b)[2 * h + u] = dct * iv * (1.0f - gv * gv);
            dg.row(b)[3 * h + u] = dhn * tc * ov * (1.0f - ov);
            dc.row(b)[u] = dct * fv;
        }
    }
#endif
}

void kConcat(Mat& z, const Mat& a, int n1, const Mat& b, int n2) {
#ifdef PD_HAVE_GPU
    if (gpuActive()) { PD_BE(Concat)(z, a, n1, b, n2); return; }
#endif
    for (int i = 0; i < z.r; ++i) {
        std::copy(a.row(i), a.row(i) + n1, z.row(i));
        std::copy(b.row(i), b.row(i) + n2, z.row(i) + n1);
    }
}

void kSplit(const Mat& z, int n1, Mat& a, Mat& b, int n2) {
#ifdef PD_HAVE_GPU
    if (gpuActive()) { PD_BE(Split)(z, n1, a, b, n2); return; }
#endif
    for (int i = 0; i < z.r; ++i) {
        std::copy(z.row(i), z.row(i) + n1, a.row(i));
        std::copy(z.row(i) + n1, z.row(i) + n1 + n2, b.row(i));
    }
}

void kZeroAndLast(Mat& all, const Mat& gh, int B, int T, int h) {
#ifdef PD_HAVE_GPU
    if (gpuActive()) { PD_BE(ZeroAndLast)(all, gh, B, T, h); return; }
#endif
    std::fill(all.d.begin(), all.d.end(), 0.0f);
    for (int i = 0; i < B; ++i)
        std::copy(gh.row(i), gh.row(i) + h, all.row((T - 1) * B + i));
}

void kMaskDyn(Mat& logits, const Mat& ds, const Mat& mask, int N) {
#ifdef PD_HAVE_GPU
    if (gpuActive()) { PD_BE(MaskDyn)(logits, ds, mask, N); return; }
#endif
    for (int i = 0; i < logits.r; ++i)
        for (int a = 0; a < N; ++a) {
            if (mask.row(i)[a] > 0.5f)
                logits.row(i)[a] += ds.row(i * N + a)[0];
            else
                logits.row(i)[a] = -1e9f;
        }
}

void kAddTo(float* dst, void** gSlot, int dstCols, const Mat& src) {
#ifdef PD_HAVE_GPU
    if (gpuActive()) { PD_BE(AddTo)(dst, gSlot, dstCols, src); return; }
#else
    (void)gSlot;
#endif
    for (int i = 0; i < src.r; ++i) {
        const float* s = src.row(i);
        float* d = dst + size_t(i) * dstCols;
        int j = 0;
#if defined(__ARM_NEON)
        for (; j + 4 <= src.c; j += 4)
            vst1q_f32(d + j, vaddq_f32(vld1q_f32(d + j), vld1q_f32(s + j)));
#endif
        for (; j < src.c; ++j) d[j] += s[j];
    }
}

void kBiasGradAdd(const Mat& g, float* db, void** dbSlot) {
#ifdef PD_HAVE_GPU
    if (gpuActive()) { PD_BE(BiasGradAdd)(g, db, dbSlot); return; }
#else
    (void)dbSlot;
#endif
    int j = 0;
#if defined(__ARM_NEON)
    for (; j + 4 <= g.c; j += 4) {
        float32x4_t s = vld1q_f32(db + j);
        for (int i = 0; i < g.r; ++i) s = vaddq_f32(s, vld1q_f32(g.row(i) + j));
        vst1q_f32(db + j, s);
    }
#endif
    for (; j < g.c; ++j) {
        float s = db[j];
        for (int i = 0; i < g.r; ++i) s += g.row(i)[j];
        db[j] = s;
    }
}

void kCopy(float* dst, int dstStride, const float* src, int srcStride,
           int rows, int cols) {
    for (int i = 0; i < rows; ++i)
        std::copy(src + size_t(i) * srcStride,
                  src + size_t(i) * srcStride + cols,
                  dst + size_t(i) * dstStride);
}

void kSlice(Mat& dst, const Mat& src, int off, int h) {
#ifdef PD_HAVE_GPU
    extern void PD_BE(Slice)(Mat& dst, const Mat& src, int off, int h);
    if (gpuActive()) { PD_BE(Slice)(dst, src, off, h); return; }
#endif
    for (int i = 0; i < src.r; ++i)
        std::copy(src.row(i) + off, src.row(i) + off + h, dst.row(i));
}

void kFlatten(Mat& dst, const Mat& src, int N) {
#ifdef PD_HAVE_GPU
    extern void PD_BE(Flatten)(Mat& dst, const Mat& src, int N);
    if (gpuActive()) { PD_BE(Flatten)(dst, src, N); return; }
#endif
    for (int i = 0; i < src.r; ++i)
        for (int a = 0; a < N; ++a) dst.row(i * N + a)[0] = src.row(i)[a];
}

void kCopyMat(Mat& dst, const Mat& src) {
#ifdef PD_HAVE_GPU
    if (gpuActive()) { PD_BE(CopyMat)(dst, src); return; }
#endif
    for (int i = 0; i < src.r; ++i)
        std::copy(src.row(i), src.row(i) + src.c, dst.row(i));
}

void kAdam(std::vector<float>& w, std::vector<float>& dw,
           std::vector<float>& m, std::vector<float>& v, int n, float lr,
           float beta1, float beta2, float eps, float bc1, float bc2) {
#ifdef PD_HAVE_GPU
    if (gpuActive()) {
        PD_BE(Adam)(w, dw, m, v, n, lr, beta1, beta2, eps, bc1, bc2);
        return;
    }
#endif
    for (int i = 0; i < n; ++i) {
        float g = dw[i];
        float mh = (beta1 * m[i] + (1.0f - beta1) * g) / bc1;
        float vh = (beta2 * v[i] + (1.0f - beta2) * g * g) / bc2;
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
        w[i] -= lr * mh / (std::sqrt(vh) + eps);
    }
}

}  // namespace nn
