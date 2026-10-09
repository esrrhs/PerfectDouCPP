// CPU GEMM reference implementation and dispatch logic.
#include "nn/gemm.h"

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#define PD_HAVE_ACCEL
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#if defined(PD_HAVE_MPS)
#define PD_HAVE_GPU 1
#define PD_BE(fn) mps##fn
#elif defined(PD_HAVE_D3D)
#define PD_HAVE_GPU 1
#define PD_BE(fn) d3d##fn
#endif

namespace nn {

namespace {
// Accelerate/AMX beats the Metal GEMM kernels on the training shapes, so the
// default backend is CPU. --backend gpu still opts into Metal.
bool g_enabled = false;
thread_local int t_override = -1;  // -1: follow g_enabled, 0/1: explicit
thread_local int g_boundSeat = 0;

// Above this MAC count the GPU is used; below it, launch/sync overhead would
// dominate small products (256x256 and smaller).
constexpr long long kGpuMacThreshold = 1000000;
}  // namespace

#if defined(_M_X64) || defined(__x86_64__)
bool pdAvx2Available();
void pdAvx2SgemmPanel(int M, int N, int K, const float* A, int lda,
                      const float* B, int ldb, float* C, int ldc,
                      bool transA);
#endif

// K-sequential FMA, matching the GPU kernels. A row of B is applied to a
// panel of output rows while it is still in L1/registers, and zero entries
// of A (binary card features) are skipped.
#if defined(__ARM_NEON)
inline void saxpy(float* __restrict__ c, const float* __restrict__ b, float a,
                  int n) {
    float32x4_t va = vdupq_n_f32(a);
    int j = 0;
    for (; j + 16 <= n; j += 16) {
        float32x4_t c0 = vld1q_f32(c + j);
        float32x4_t c1 = vld1q_f32(c + j + 4);
        float32x4_t c2 = vld1q_f32(c + j + 8);
        float32x4_t c3 = vld1q_f32(c + j + 12);
        c0 = vfmaq_f32(c0, va, vld1q_f32(b + j));
        c1 = vfmaq_f32(c1, va, vld1q_f32(b + j + 4));
        c2 = vfmaq_f32(c2, va, vld1q_f32(b + j + 8));
        c3 = vfmaq_f32(c3, va, vld1q_f32(b + j + 12));
        vst1q_f32(c + j, c0);
        vst1q_f32(c + j + 4, c1);
        vst1q_f32(c + j + 8, c2);
        vst1q_f32(c + j + 12, c3);
    }
    for (; j + 4 <= n; j += 4) {
        float32x4_t cv = vfmaq_f32(vld1q_f32(c + j), va, vld1q_f32(b + j));
        vst1q_f32(c + j, cv);
    }
    for (; j < n; ++j) c[j] += a * b[j];
}
inline void zeroN(float* c, int n) {
    int j = 0;
    float32x4_t z = vdupq_n_f32(0.0f);
    for (; j + 4 <= n; j += 4) vst1q_f32(c + j, z);
    for (; j < n; ++j) c[j] = 0.0f;
}
#else
inline void saxpy(float* c, const float* b, float a, int n) {
    for (int j = 0; j < n; ++j) c[j] += a * b[j];
}
inline void zeroN(float* c, int n) {
    for (int j = 0; j < n; ++j) c[j] = 0.0f;
}
#endif

// C[i,:] += A_panel[i,k] * Brow[k,:], k outer so each B row is reused across
// the panel. aAt(i,k) reads A in either NN (row i) or TN (column i) layout.
void sgemmPanel(int M, int N, int K, const float* A, int lda, const float* B,
                int ldb, float* C, int ldc, bool transA) {
#if defined(_M_X64) || defined(__x86_64__)
    static const bool haveAvx2 = pdAvx2Available();
    if (haveAvx2) {
        pdAvx2SgemmPanel(M, N, K, A, lda, B, ldb, C, ldc, transA);
        return;
    }
#endif
    constexpr int MR = 4;
    for (int i0 = 0; i0 < M; i0 += MR) {
        int mr = std::min(MR, M - i0);
        for (int i = 0; i < mr; ++i) zeroN(C + size_t(i0 + i) * ldc, N);
        for (int k = 0; k < K; ++k) {
            float a[MR];
            bool any = false;
            for (int i = 0; i < mr; ++i) {
                a[i] = transA ? A[size_t(k) * lda + (i0 + i)]
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

void sgemmCpu(char tA, char tB, int M, int N, int K,
              const float* A, int lda, const float* B, int ldb,
              float* C, int ldc) {
#ifdef PD_HAVE_ACCEL
    // Apple AMX. Far above the hand-written NEON kernel on every training
    // shape (about 1–1.6 TFLOP/s vs ~40 GFLOP/s). beta = 0 overwrites C.
    cblas_sgemm(CblasRowMajor,
                tA == 'T' ? CblasTrans : CblasNoTrans,
                tB == 'T' ? CblasTrans : CblasNoTrans,
                M, N, K, 1.0f, A, lda, B, ldb, 0.0f, C, ldc);
#else
    if (tB == 'T') {
        thread_local std::vector<float> pack;
        pack.resize(size_t(K) * N);
        for (int j = 0; j < N; ++j) {
            const float* src = B + size_t(j) * ldb;
            for (int k = 0; k < K; ++k) pack[size_t(k) * N + j] = src[k];
        }
        sgemmPanel(M, N, K, A, lda, pack.data(), N, C, ldc, tA == 'T');
        return;
    }
    sgemmPanel(M, N, K, A, lda, B, ldb, C, ldc, tA == 'T');
#endif
}

// MPS implementation (Apple only), defined in gemm_mps.mm
#ifdef PD_HAVE_GPU
void PD_BE(CommitGemms)(int count, const GemmOp* ops, long long macs);
void PD_BE(CustomGemm)(const GemmOp& op, const void* bias, size_t biasBytes,
                   int epi);
void PD_BE(Wait)(bool keepWindow);
void PD_BE(StageInput)(const void* p, size_t bytes);
void PD_BE(PrintStats)(const char* tag);
void PD_BE(MarkHost)(const void* p);
void PD_BE(DropCache)(void** slot);
void PD_BE(StaleCache)(void** slot);
void* PD_BE(WeightCache)(void** slot, const void* host, size_t bytes);
void* PD_BE(GradCache)(void** slot, const void* host, size_t bytes);
void PD_BE(FlushGrad)(void* slot, void* host, size_t bytes);
#endif

void gemmInit() {
    // The Windows CRT ignores line buffering, and abort() skips the final
    // flush. Unbuffered logs stay visible if a GPU call fails mid-run.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::setvbuf(stderr, nullptr, _IONBF, 0);
#ifdef PD_HAVE_ACCEL
    // One BLAS thread per caller. The trainer already runs one thread per
    // seat; a BLAS thread pool on top of that just contends for AMX.
    setenv("VECLIB_MAXIMUM_THREADS", "1", /*overwrite=*/0);
#endif
#if defined(PD_HAVE_D3D)
    extern bool d3dInit();
    void d3dUseSeat(int);
    void d3dSubmit();
    void d3dSyncSeat();
    g_enabled = d3dInit();
#elif defined(PD_HAVE_GPU)
    extern bool PD_BE(Init)();
    if (!PD_BE(Init)()) g_enabled = false;
#endif
}
bool gemmHasGpu() {
#ifdef PD_HAVE_GPU
    extern bool PD_BE(Available)();
    return PD_BE(Available)();
#else
    return false;
#endif
}
void gemmSetGpu(bool enabled) { g_enabled = enabled && gemmHasGpu(); }
void gemmSetThreadGpu(int enabled) { t_override = enabled; }
bool gemmGpuEnabled() {
    return t_override >= 0 ? (t_override != 0) : g_enabled;
}
const char* gemmGpuLabel() {
#if defined(PD_HAVE_D3D)
    extern const char* d3dLabel();
    return d3dLabel();
#elif defined(PD_HAVE_MPS)
    return "GPU (Metal)";
#else
    return "GPU";
#endif
}
bool gpuActive() {
#ifdef PD_HAVE_GPU
    return gemmGpuEnabled();
#else
    return false;
#endif
}

void sgemm(char tA, char tB, int M, int N, int K,
           const float* A, int lda, const float* B, int ldb,
           float* C, int ldc) {
    long long macs = (long long)M * N * K;
#ifdef PD_HAVE_GPU
    if (gemmGpuEnabled() && macs >= kGpuMacThreshold) {
        GemmOp g{tA, tB, M, N, K, A, lda, B, ldb, C, ldc};
        PD_BE(CommitGemms)(1, &g, macs);
        PD_BE(MarkHost)(C);  // standalone call: result is read on host
        PD_BE(Wait)(false);
        return;
    }
#endif
    sgemmCpu(tA, tB, M, N, K, A, lda, B, ldb, C, ldc);
}

void gpuGemm(const GemmOp& g, const float* bias, int biasFloats, int epi) {
#ifdef PD_HAVE_GPU
    if (gemmGpuEnabled()) {
        size_t bbytes = biasFloats
            ? ((size_t(biasFloats) * 4 + 15) & ~size_t(15)) : 0;
        PD_BE(CustomGemm)(g, bias, bbytes, epi);
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
#ifdef PD_HAVE_GPU
    // The residency model keeps intermediates on the device, so even tiny
    // products must go through the GPU (a CPU fallback would read stale
    // memory).
    if (gemmGpuEnabled()) {
        PD_BE(CommitGemms)(count, ops, macs);
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
#ifdef PD_HAVE_GPU
    if (gemmGpuEnabled()) PD_BE(Wait)(keepWindow);
#else
    (void)keepWindow;
#endif
}

void gpuBindSeat(int seat) {
    if (seat < 0) seat = 0;
    g_boundSeat = seat;
#if defined(PD_HAVE_CUBLAS)
    extern void cudaBridgeSetStream(int);
    cudaBridgeSetStream(seat);
#endif
#if defined(PD_HAVE_D3D)
    if (std::getenv("PD_CUBLAS")) {
        d3dUseSeat(seat);
    }
#endif
}

void gpuSubmit() {
#if defined(PD_HAVE_D3D)
    if (gemmGpuEnabled()) d3dSubmit();
#endif
}

void gpuSync() {
#if defined(PD_HAVE_D3D)
    extern void cudaBridgeSetStream(int);
    cudaBridgeSetStream(g_boundSeat);
    if (gemmGpuEnabled()) d3dSyncSeat();
#endif
}

void gpuStageInput(float* p, int floats) {
#ifdef PD_HAVE_GPU
    if (gemmGpuEnabled()) PD_BE(StageInput)(p, size_t(floats) * 4);
#else
    (void)p; (void)floats;
#endif
}

void gpuPrintStats(const char* tag) {
#ifdef PD_HAVE_GPU
    if (gemmGpuEnabled()) PD_BE(PrintStats)(tag);
#else
    (void)tag;
#endif
}

void gpuMarkHost(float* p) {
#ifdef PD_HAVE_GPU
    if (gemmGpuEnabled()) PD_BE(MarkHost)(p);
#else
    (void)p;
#endif
}

void gpuInvoke(const std::function<void()>& fn) {
#if !defined(PD_HAVE_D3D)
    fn();
#else
    int seat = g_boundSeat;
    auto run = [fn, seat] {
        struct Guard {
            int prev;
            explicit Guard(int s) : prev(g_boundSeat) { g_boundSeat = s; }
            ~Guard() { g_boundSeat = prev; }
        } guard(seat);
        d3dUseSeat(seat);
        fn();
    };
    static int single = -1;
    if (single < 0) single = std::getenv("PD_SINGLE_THREAD") ? 1 : 0;
    if (single) {
        run();
        return;
    }
    // One queue, one thread. Three compute queues on this NVIDIA driver
    // remove the device even when their packets are executed one at a time.
    struct Owner {
        std::mutex mu;
        std::condition_variable cv;
        std::function<void()> job;
        std::exception_ptr error;
        bool pending = false;
        bool done = false;
        bool busy = false;
        bool stop = false;
        std::thread th;
        Owner() {
            th = std::thread([this] {
                for (;;) {
                    std::function<void()> run;
                    {
                        std::unique_lock<std::mutex> lk(mu);
                        cv.wait(lk, [&] { return pending || stop; });
                        if (stop && !pending) return;
                        run = std::move(job);
                        pending = false;
                    }
                    std::exception_ptr ep;
                    try {
                        run();
                    } catch (...) {
                        ep = std::current_exception();
                    }
                    {
                        std::lock_guard<std::mutex> lk(mu);
                        error = ep;
                        done = true;
                    }
                    cv.notify_all();
                }
            });
        }
        ~Owner() {
            {
                std::lock_guard<std::mutex> lk(mu);
                stop = true;
            }
            cv.notify_all();
            if (th.joinable()) th.join();
        }
    };
    static Owner owner;
    if (std::this_thread::get_id() == owner.th.get_id()) {
        run();
        return;
    }
    std::unique_lock<std::mutex> lk(owner.mu);
    owner.cv.wait(lk, [&] { return !owner.busy; });
    owner.busy = true;
    owner.done = false;
    owner.error = nullptr;
    owner.job = std::function<void()>(run);
    owner.pending = true;
    lk.unlock();
    owner.cv.notify_all();
    lk.lock();
    owner.cv.wait(lk, [&] { return owner.done; });
    std::exception_ptr ep = owner.error;
    owner.busy = false;
    lk.unlock();
    owner.cv.notify_all();
    if (ep) std::rethrow_exception(ep);
#endif
}

void gpuDropCache(void** slot) {
#ifdef PD_HAVE_GPU
    if (slot != nullptr && *slot != nullptr) {
#if defined(PD_HAVE_D3D)
        if (std::getenv("PD_CUBLAS")) PD_BE(DropCache)(slot);
        else gpuInvoke([slot] { PD_BE(DropCache)(slot); });
#else
        PD_BE(DropCache)(slot);
#endif
    }
#else
    (void)slot;
#endif
}

void gpuStaleCache(void** slot) {
#ifdef PD_HAVE_GPU
    if (slot != nullptr && *slot != nullptr) {
#if defined(PD_HAVE_D3D)
        if (std::getenv("PD_CUBLAS")) PD_BE(StaleCache)(slot);
        else gpuInvoke([slot] { PD_BE(StaleCache)(slot); });
#else
        PD_BE(StaleCache)(slot);
#endif
    }
#else
    (void)slot;
#endif
}

void* gpuWeightCache(void** slot, const void* host, size_t bytes) {
#ifdef PD_HAVE_GPU
    if (gemmGpuEnabled()) return PD_BE(WeightCache)(slot, host, bytes);
#else
    (void)slot; (void)host; (void)bytes;
#endif
    return nullptr;
}

void* gpuGradCache(void** slot, const void* host, size_t bytes) {
#ifdef PD_HAVE_GPU
    if (gemmGpuEnabled()) return PD_BE(GradCache)(slot, host, bytes);
#else
    (void)slot; (void)host; (void)bytes;
#endif
    return nullptr;
}

void gpuFlushGrad(void* slot, void* host, size_t bytes) {
#ifdef PD_HAVE_GPU
    if (gemmGpuEnabled()) PD_BE(FlushGrad)(slot, host, bytes);
#else
    (void)slot; (void)host; (void)bytes;
#endif
}

bool gpuDeviceOk() {
#if defined(PD_HAVE_D3D)
    extern bool d3dDeviceOk();
    return d3dDeviceOk();
#else
    return true;
#endif
}

void gpuReleaseThread() {
#if defined(PD_HAVE_D3D)
    extern void d3dReleaseThread();
    if (std::getenv("PD_CUBLAS")) d3dReleaseThread();
    else gpuInvoke([] { d3dReleaseThread(); });
#endif
}

bool gpuRecreate() {
#if defined(PD_HAVE_D3D)
    extern bool d3dRecreate();
    return d3dRecreate();
#else
    return true;
#endif
}

}  // namespace nn
