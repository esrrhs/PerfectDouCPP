// Metal backend: resident GEMM + compute-kernel stream.
//
// * Dense products run on in-house tiled Metal GEMM kernels (see gemm_block,
//   gemm_blockn and gemm_tiled), dispatched by shape: register-blocked
//   32x32 for tall-M products, 16x32 for short-M/wide-N products, and the
//   scalar 16x16 kernel for small tiles with long K. MPSMatrixMultiplication
//   was evaluated and rejected because it silently returned wrong results /
//   NaNs when chained inside the residency windows.
// * Model weights and gradients keep their device buffer in a cache slot
//   owned by the host Param/Mat, so its lifetime can never alias freed or
//   reused host storage. All other intermediates live in per-fence residency
//   windows, flushed to host only when the host actually reads them.
// * Elementwise ops are small Metal compute shaders compiled from embedded
//   source once at startup.
// * Every encode goes onto the thread-local command queue and returns without
//   blocking; command buffers run FIFO, which honors all graph dependencies.
//   The host fences explicitly with mpsWait() only when it has to read data.
//   An autorelease pool is pushed per command buffer and popped at its fence,
//   so long-lived threads (e.g. main) never accumulate autoreleased buffers.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <chrono>
#include <cmath>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "nn/gemm.h"
#include "nn/kernels.h"

extern "C" void* objc_autoreleasePoolPush(void);
extern "C" void objc_autoreleasePoolPop(void*);

// Device residency bindings (must live at global Objective-C scope).
@interface PDBind : NSObject {
  @public
    id<MTLBuffer> buf;
    void* host;
    NSUInteger len;
    BOOL dirty;
    BOOL toHost;  // host will read this at the fence (flush it back)
}
@end
@implementation PDBind
@end

@interface PDRef : NSObject {
  @public
    id<MTLBuffer> buf;
    NSUInteger off;
    NSUInteger len;
}
+ (PDRef*)with:(id<MTLBuffer>)b off:(NSUInteger)o;
+ (PDRef*)with:(id<MTLBuffer>)b off:(NSUInteger)o len:(NSUInteger)l;
@end
@implementation PDRef
+ (PDRef*)with:(id<MTLBuffer>)b off:(NSUInteger)o {
    return [self with:b off:o len:0];
}
+ (PDRef*)with:(id<MTLBuffer>)b off:(NSUInteger)o len:(NSUInteger)l {
    PDRef* r = [PDRef new];
    r->buf = b;
    r->off = o;
    r->len = l;
    return r;
}
@end

namespace nn {

namespace {

id<MTLDevice> g_device = nil;
id<MTLLibrary> g_library = nil;
std::mutex g_pipeMutex;
NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* g_pipes = nil;

const char* kKernelSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct P3 { int a, b, c; };
kernel void add_bias(device float* y, constant const float* bi,
                     constant P3& p, uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.b)) return;
    int i = gid / p.a, j = gid - i * p.a;
    y[i * p.c + j] += bi[j];
}

kernel void relu_fwd(device float* x, constant P3& p,
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.b)) return;
    int i = gid / p.a, j = gid - i * p.a;
    float v = x[i * p.c + j];
    x[i * p.c + j] = v > 0.0f ? v : 0.0f;
}

struct P4 { int a, b, c, d; };
kernel void relu_bwd(device const float* pre, device const float* gout,
                     device float* gin, constant P4& p,
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.b)) return;
    int i = gid / p.a, j = gid - i * p.a;
    gin[i * p.d + j] = pre[i * p.c + j] > 0.0f ? gout[i * p.d + j] : 0.0f;
}

struct P2 { int a, b; };
kernel void gate_add(device float* gp, device const float* gi,
                     device const float* gh, constant const float* bi,
                     constant const float* bh, constant P2& p,
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.b * p.a)) return;
    int i = gid / p.a, q = gid - i * p.a;
    gp[gid] = ((gi[gid] + gh[gid]) + bi[q]) + bh[q];
}

struct PCell { int h, B, hasCp; };
kernel void lstm_cell_fwd(device const float* gp, device const float* cp,
                          device float* hout, device float* cout_,
                          constant PCell& p,
                          uint2 gid [[thread_position_in_grid]]) {
    int b = gid.x, u = gid.y, h = p.h;
    if (b >= p.B || u >= h) return;
    float iv = 1.0f / (1.0f + exp(-gp[b * 4 * h + u]));
    float fv = 1.0f / (1.0f + exp(-gp[b * 4 * h + h + u]));
    float gz = tanh(gp[b * 4 * h + 2 * h + u]);
    float ov = 1.0f / (1.0f + exp(-gp[b * 4 * h + 3 * h + u]));
    float pc = p.hasCp != 0 ? cp[b * h + u] : 0.0f;
    float cc = fv * pc + iv * gz;
    cout_[b * h + u] = cc;
    hout[b * h + u] = ov * tanh(cc);
}

kernel void lstm_cell_bwd(device const float* gh, device const float* gp,
                          device const float* cn, device const float* cp,
                          device float* dh, device float* dc,
                          device float* dg, constant PCell& p,
                          uint2 gid [[thread_position_in_grid]]) {
    int b = gid.x, u = gid.y, h = p.h;
    if (b >= p.B || u >= h) return;
    float tc = tanh(cn[b * h + u]);
    float iv = 1.0f / (1.0f + exp(-gp[b * 4 * h + u]));
    float fv = 1.0f / (1.0f + exp(-gp[b * 4 * h + h + u]));
    float gz = tanh(gp[b * 4 * h + 2 * h + u]);
    float ov = 1.0f / (1.0f + exp(-gp[b * 4 * h + 3 * h + u]));
    float dhn = gh[b * h + u] + dh[b * h + u];
    float pc = p.hasCp != 0 ? cp[b * h + u] : 0.0f;
    float dct = dhn * ov * (1.0f - tc * tc) + dc[b * h + u];
    dg[b * 4 * h + u] = dct * gz * iv * (1.0f - iv);
    dg[b * 4 * h + h + u] = dct * pc * fv * (1.0f - fv);
    dg[b * 4 * h + 2 * h + u] = dct * iv * (1.0f - gz * gz);
    dg[b * 4 * h + 3 * h + u] = dhn * tc * ov * (1.0f - ov);
    dc[b * h + u] = dct * fv;
}

struct P6 { int a, b, c, d, e, f; };
kernel void concat2(device float* z, device const float* a,
                    device const float* b, constant P6& p,
                    uint gid [[thread_position_in_grid]]) {
    int cols = p.a + p.b;
    if (gid >= uint(p.c * cols)) return;
    int i = gid / cols, j = gid - i * cols;
    z[i * p.d + j] = (j < p.a) ? a[i * p.e + j] : b[i * p.f + (j - p.a)];
}

kernel void split2(device const float* z, device float* a, device float* b,
                   constant P6& p, uint gid [[thread_position_in_grid]]) {
    int cols = p.a + p.b;
    if (gid >= uint(p.c * cols)) return;
    int i = gid / cols, j = gid - i * cols;
    if (j < p.a) a[i * p.e + j] = z[i * p.d + j];
    else b[i * p.f + (j - p.a)] = z[i * p.d + j];
}

struct P3i { int B, T, h; };
kernel void zero_and_last(device float* all, device const float* gh,
                          constant P3i& p, uint gid [[thread_position_in_grid]]) {
    int n = p.B * p.T * p.h;
    if (gid >= n) return;
    int t = gid / (p.B * p.h);
    int rem = gid - t * p.B * p.h;
    all[gid] = (t == p.T - 1) ? gh[rem] : 0.0f;
}

struct PMask { int N, B, ls, ms; };
kernel void mask_dyn(device float* logits, device const float* ds,
                     device const float* mask, constant PMask& p,
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.B * p.N)) return;
    int i = gid / p.N, a = gid - i * p.N;
    if (mask[i * p.ms + a] > 0.5f)
        logits[i * p.ls + a] += ds[(i * p.N + a) * 4];
    else
        logits[i * p.ls + a] = -1e9f;
}

struct P4i { int rows, cols, dc, sc; };
kernel void add_to(device float* dst, device const float* src,
                   constant P4i& p, uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.rows * p.cols)) return;
    int i = gid / p.cols, j = gid - i * p.cols;
    dst[i * p.dc + j] += src[i * p.sc + j];
}

struct PBG { int B, o, gs; };
kernel void bias_grad_add(device const float* g, device float* db,
                          constant PBG& p, uint gid [[thread_position_in_grid]]) {
    if (gid >= p.o) return;
    float s = 0.0f;
    for (int i = 0; i < p.B; ++i) s += g[i * p.gs + gid];
    db[gid] += s;
}

struct PSl { int B, h, off, ds, ss; };
kernel void slice_cols(device float* dst, device const float* src,
                       constant PSl& p, uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.B * p.h)) return;
    int i = gid / p.h, j = gid - i * p.h;
    dst[i * p.ds + j] = src[i * p.ss + p.off + j];
}

struct PFl { int B, N, ss; };
kernel void flatten_rows(device float* dst, device const float* src,
                         constant PFl& p, uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.B * p.N)) return;
    int i = gid / p.N, a = gid - i * p.N;
    dst[gid * 4] = src[i * p.ss + a];
}

struct PCp { int rows, cols, ds, ss; };
kernel void copy_mat(device float* dst, device const float* src,
                     constant PCp& p, uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.rows * p.cols)) return;
    int i = gid / p.cols, j = gid - i * p.cols;
    dst[i * p.ds + j] = src[i * p.ss + j];
}

// Tiled fp32 GEMM with optional fused epilogue:
//  epi 0: plain    1: C += bias[j], then ReLU
constant constexpr int GEMM_TM = 16;
constant constexpr int GEMM_TN = 16;
constant constexpr int GEMM_TK = 16;
struct GemmP {
    int M, N, K, lda, ldb, ldc, tA, tB, epi;
};
kernel void gemm_tiled(device const float* A, device const float* B,
                       device float* C, constant float* bias,
                       constant GemmP& p,
                       uint2 tid [[thread_position_in_threadgroup]],
                       uint2 gid [[threadgroup_position_in_grid]]) {
    int row = gid.y * GEMM_TM + tid.y;
    int col = gid.x * GEMM_TN + tid.x;
    threadgroup float As[GEMM_TM][GEMM_TK];
    threadgroup float Bs[GEMM_TK][GEMM_TN];
    float acc = 0.0f;
    int tiles = (p.K + GEMM_TK - 1) / GEMM_TK;
    for (int t = 0; t < tiles; ++t) {
        // cooperative load of the A/B tiles (256 threads, tiles are 16x16)
        int tr = tid.y, tc = tid.x;
        int kA = t * GEMM_TK + tc;
        int ar = gid.y * GEMM_TM + tr;
        if (p.tA == 0)
            As[tr][tc] = (ar < p.M && kA < p.K) ? A[ar * p.lda + kA] : 0.0f;
        else
            // A is physically [K x M]: dot index k selects the row, output
            // row i selects the column.
            As[tr][tc] = (kA < p.K && ar < p.M)
                             ? A[kA * p.lda + ar] : 0.0f;
        int kB = t * GEMM_TK + tr;
        int bc = gid.x * GEMM_TN + tc;
        if (p.tB == 0)
            Bs[tr][tc] = (kB < p.K && bc < p.N) ? B[kB * p.ldb + bc] : 0.0f;
        else
            // B is physically [N x K]: output col j selects the row, dot
            // index k selects the column.
            Bs[tr][tc] = (bc < p.N && kB < p.K)
                             ? B[bc * p.ldb + kB] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int z = 0; z < GEMM_TK; ++z) {
            int kk = t * GEMM_TK + z;
            if (kk < p.K) acc += As[tr][z] * Bs[z][tc];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < p.M && col < p.N) {
        float v = acc;
        if (p.epi == 1) {
            v += bias[col];
            v = v > 0.0f ? v : 0.0f;
        }
        C[row * p.ldc + col] = v;
    }
}

// Register-blocked fp32 GEMM. One threadgroup owns a BM x BN output tile;
// each of the 8x8 threads accumulates MR x NR outputs in registers while K is
// streamed through threadgroup memory in BK slices. Compared with the scalar
// 16x16 kernel this halves threadgroup traffic per MAC, vectorizes the
// epilogue and keeps many more threads resident; optional fused bias+ReLU.
constant constexpr int GEMM_BM = 32;
constant constexpr int GEMM_BN = 32;
constant constexpr int GEMM_BK = 16;
constant constexpr int GEMM_BW = 8;              // threads per tile dimension
constant constexpr int GEMM_MR = GEMM_BM / GEMM_BW;  // 4 rows/thread
constant constexpr int GEMM_NR = GEMM_BN / GEMM_BW;  // 4 cols/thread
kernel void gemm_block(device const float* A, device const float* B,
                       device float* C, constant const float* bias,
                       constant GemmP& p,
                       uint2 tpos [[thread_position_in_threadgroup]],
                       uint2 gpos [[threadgroup_position_in_grid]]) {
    threadgroup float As[GEMM_BM][GEMM_BK];
    threadgroup float Bs[GEMM_BK][GEMM_BN];
    float acc[GEMM_MR][GEMM_NR];
    #pragma unroll
    for (int a = 0; a < GEMM_MR; ++a)
        #pragma unroll
        for (int b = 0; b < GEMM_NR; ++b) acc[a][b] = 0.0f;

    const uint i0 = gpos.y * GEMM_BM;
    const uint j0 = gpos.x * GEMM_BN;
    const uint tr = tpos.y, tc = tpos.x;
    const uint tid = tr * GEMM_BW + tc;  // 0..63
    const uint nLoadsA = (GEMM_BM * GEMM_BK) / 64;  // 8
    const uint nLoadsB = (GEMM_BK * GEMM_BN) / 64;  // 8

    for (int k0 = 0; k0 < p.K; k0 += GEMM_BK) {
        #pragma unroll
        for (uint q = 0; q < nLoadsA; ++q) {
            uint idx = tid + 64u * q;
            uint r = idx / GEMM_BK, c = idx % GEMM_BK;
            uint ar = i0 + r, ak = uint(k0) + c;
            float v = 0.0f;
            if (ar < uint(p.M) && ak < uint(p.K))
                v = (p.tA == 0) ? A[ar * p.lda + ak]
                                : A[ak * p.lda + ar];
            As[r][c] = v;
        }
        #pragma unroll
        for (uint q = 0; q < nLoadsB; ++q) {
            uint idx = tid + 64u * q;
            uint r = idx / GEMM_BN, c = idx % GEMM_BN;
            uint bk = uint(k0) + r, bj = j0 + c;
            float v = 0.0f;
            if (bk < uint(p.K) && bj < uint(p.N))
                v = (p.tB == 0) ? B[bk * p.ldb + bj]
                                : B[bj * p.ldb + bk];
            Bs[r][c] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (int z = 0; z < GEMM_BK; ++z) {
            float av[GEMM_MR];
            #pragma unroll
            for (int a = 0; a < GEMM_MR; ++a) av[a] = As[tr * GEMM_MR + a][z];
            #pragma unroll
            for (int a = 0; a < GEMM_MR; ++a)
                #pragma unroll
                for (int b = 0; b < GEMM_NR; ++b)
                    acc[a][b] += av[a] * Bs[z][tc * GEMM_NR + b];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma unroll
    for (int a = 0; a < GEMM_MR; ++a) {
        uint row = i0 + tr * GEMM_MR + a;
        if (row >= uint(p.M)) continue;
        uint col = j0 + tc * GEMM_NR;
        device float* cp = C + row * p.ldc + col;
        if (p.epi == 0) {
            if (col + GEMM_NR <= uint(p.N)) {
                cp[0] = acc[a][0]; cp[1] = acc[a][1];
                cp[2] = acc[a][2]; cp[3] = acc[a][3];
            } else {
                #pragma unroll
                for (int b = 0; b < GEMM_NR; ++b)
                    if (col + b < uint(p.N)) cp[b] = acc[a][b];
            }
        } else {
            constant const float* bp = bias + col;
            if (col + GEMM_NR <= uint(p.N)) {
                #pragma unroll
                for (int b = 0; b < GEMM_NR; ++b) {
                    float v = acc[a][b] + bp[b];
                    cp[b] = v > 0.0f ? v : 0.0f;
                }
            } else {
                #pragma unroll
                for (int b = 0; b < GEMM_NR; ++b)
                    if (col + b < uint(p.N)) {
                        float v = acc[a][b] + bp[b];
                        cp[b] = v > 0.0f ? v : 0.0f;
                    }
            }
        }
    }
}

// Narrow variant of gemm_block for short-M products (small batch LSTM GEMMs):
// 16x32 output tile, 8x8 threads, 2x4 outputs/thread. More threadgroups in M
// keep the GPU busy where the 32x32 tile would launch too few groups.
constant constexpr int GEMM_NM = 16;
constant constexpr int GEMM_NN2 = 32;
constant constexpr int GEMM_NK = 16;
constant constexpr int GEMM_NW = 8;
constant constexpr int GEMM_NMR = GEMM_NM / GEMM_NW;  // 2
constant constexpr int GEMM_NNR = GEMM_NN2 / GEMM_NW; // 4
kernel void gemm_blockn(device const float* A, device const float* B,
                        device float* C, constant const float* bias,
                        constant GemmP& p,
                        uint2 tpos [[thread_position_in_threadgroup]],
                        uint2 gpos [[threadgroup_position_in_grid]]) {
    threadgroup float As[GEMM_NM][GEMM_NK];
    threadgroup float Bs[GEMM_NK][GEMM_NN2];
    float acc[GEMM_NMR][GEMM_NNR];
    #pragma unroll
    for (int a = 0; a < GEMM_NMR; ++a)
        #pragma unroll
        for (int b = 0; b < GEMM_NNR; ++b) acc[a][b] = 0.0f;

    const uint i0 = gpos.y * GEMM_NM;
    const uint j0 = gpos.x * GEMM_NN2;
    const uint tr = tpos.y, tc = tpos.x;
    const uint tid = tr * GEMM_NW + tc;  // 0..63
    const uint nLoadsA = (GEMM_NM * GEMM_NK) / 64;  // 4
    const uint nLoadsB = (GEMM_NK * GEMM_NN2) / 64; // 8

    for (int k0 = 0; k0 < p.K; k0 += GEMM_NK) {
        #pragma unroll
        for (uint q = 0; q < nLoadsA; ++q) {
            uint idx = tid + 64u * q;
            uint r = idx / GEMM_NK, c = idx % GEMM_NK;
            uint ar = i0 + r, ak = uint(k0) + c;
            float v = 0.0f;
            if (ar < uint(p.M) && ak < uint(p.K))
                v = (p.tA == 0) ? A[ar * p.lda + ak]
                                : A[ak * p.lda + ar];
            As[r][c] = v;
        }
        #pragma unroll
        for (uint q = 0; q < nLoadsB; ++q) {
            uint idx = tid + 64u * q;
            uint r = idx / GEMM_NN2, c = idx % GEMM_NN2;
            uint bk = uint(k0) + r, bj = j0 + c;
            float v = 0.0f;
            if (bk < uint(p.K) && bj < uint(p.N))
                v = (p.tB == 0) ? B[bk * p.ldb + bj]
                                : B[bj * p.ldb + bk];
            Bs[r][c] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (int z = 0; z < GEMM_NK; ++z) {
            float av[GEMM_NMR];
            #pragma unroll
            for (int a = 0; a < GEMM_NMR; ++a)
                av[a] = As[tr * GEMM_NMR + a][z];
            #pragma unroll
            for (int a = 0; a < GEMM_NMR; ++a)
                #pragma unroll
                for (int b = 0; b < GEMM_NNR; ++b)
                    acc[a][b] += av[a] * Bs[z][tc * GEMM_NNR + b];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma unroll
    for (int a = 0; a < GEMM_NMR; ++a) {
        uint row = i0 + tr * GEMM_NMR + a;
        if (row >= uint(p.M)) continue;
        uint col = j0 + tc * GEMM_NNR;
        device float* cp = C + row * p.ldc + col;
        if (p.epi == 0) {
            if (col + GEMM_NNR <= uint(p.N)) {
                #pragma unroll
                for (int b = 0; b < GEMM_NNR; ++b) cp[b] = acc[a][b];
            } else {
                #pragma unroll
                for (int b = 0; b < GEMM_NNR; ++b)
                    if (col + b < uint(p.N)) cp[b] = acc[a][b];
            }
        } else {
            constant const float* bp = bias + col;
            #pragma unroll
            for (int b = 0; b < GEMM_NNR; ++b)
                if (col + b < uint(p.N)) {
                    float v = acc[a][b] + bp[b];
                    cp[b] = v > 0.0f ? v : 0.0f;
                }
        }
    }
}

struct AdamP {
    int n;
    float lr, b1, b2, eps, bc1, bc2;
};
kernel void adam_step(device float* w, device float* dw, device float* m,
                      device float* v, constant AdamP& p,
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) return;
    float g = dw[gid];
    float mm = p.b1 * m[gid] + (1.0f - p.b1) * g;
    float vv = p.b2 * v[gid] + (1.0f - p.b2) * g * g;
    m[gid] = mm;
    v[gid] = vv;
    w[gid] -= p.lr * (mm / p.bc1) / (sqrt(vv / p.bc2) + p.eps);
}
)MSL";

// ---------------------------------------------------------------------------
// Device/queue/pipeline state
// ---------------------------------------------------------------------------
struct Ctx {
    id<MTLCommandQueue> queue = nil;
    id<MTLCommandBuffer> cb = nil;     // accumulates all encodes per window
    id<MTLComputeCommandEncoder> enc = nil;
    void* pool = nullptr;             // autorelease pool token
    NSMutableDictionary<NSNumber*, PDBind*>* window = nil;
    NSMutableArray* alive = nil;  // objects retained until the next fence
    long long nWait = 0, nCommit = 0;
    long long stageBytes = 0, flushBytes = 0;
    double waitSec = 0, encSec = 0;
};



Ctx& ctx() {
    static thread_local Ctx c;
    return c;
}

id keepAlive(id x) {
    Ctx& c = ctx();
    if (c.alive == nil) c.alive = [NSMutableArray arrayWithCapacity:64];
    [c.alive addObject:x];
    return x;
}

NSNumber* ptrKey(const void* p) {
    return [NSNumber numberWithUnsignedLongLong:uint64_t(p)];
}

// --- residency window between two host fences ------------------------------
// Binds remember where each host Mat currently lives, so chained
// GEMM/kernel sequences stay device-resident; mpsWait() copies each dirty
// result back to its host Mat once and clears the window. A bind is an
// exact match, else an offset inside an already bound parent bind.
PDRef* bindResolve(const void* p) {
    Ctx& c = ctx();
    PDBind* bd = [c.window objectForKey:ptrKey(p)];
    if (bd != nil) return [PDRef with:bd->buf off:0];
    for (PDBind* e in c.window.allValues) {
        char* lo = (char*)e->host;
        char* cur = (char*)p;
        if (cur >= lo && cur < lo + e->len)
            return [PDRef with:e->buf off:(NSUInteger)(cur - lo)];
    }
    return nil;
}

PDRef* bindOutput(const void* p, size_t bytes) {
    Ctx& c = ctx();
    if (c.window == nil) c.window = [NSMutableDictionary new];
    PDBind* bd = [c.window objectForKey:ptrKey(p)];
    if (bd == nil) {
        bd = [PDBind new];
        bd->host = const_cast<void*>(p);
        bd->len = bytes;
        bd->dirty = YES;
        bd->toHost = NO;
        bd->buf = [g_device newBufferWithLength:bytes
                                        options:MTLResourceStorageModeShared];
        [c.window setObject:bd forKey:ptrKey(p)];
    }
    return [PDRef with:bd->buf off:0];
}

// Kernel input: bound view/parent, else ephemeral copy-in.
PDRef* kerIn(const void* p, size_t bytes) {
    PDRef* r = bindResolve(p);
    if (r != nil) return r;
    id<MTLBuffer> b = [g_device newBufferWithLength:bytes
                                            options:MTLResourceStorageModeShared];
    memcpy(b.contents, p, bytes);
    ctx().stageBytes += (long long)bytes;
    return [PDRef with:b off:0];
}

// Kernel output that fully overwrites its destination (no host seed needed).
PDRef* kerOut(const void* p, size_t bytes) {
    PDRef* r = bindResolve(p);
    if (r != nil) return r;
    return bindOutput(p, bytes);
}

// Kernel output read-modify-write seeded with the current host contents.
PDRef* kerOutSeed(const void* p, size_t bytes) {
    PDRef* r = bindResolve(p);
    if (r != nil) return r;
    r = bindOutput(p, bytes);
    memcpy(r->buf.contents, p, bytes);
    return r;
}

// Read-write host vector touched by a bounded kernel (gradient accumulator,
// Adam state): register it in the residency window, seed it from the host on
// first use this window and flush it back at every fence. Device buffers
// cannot alias unaligned std::vector storage (no-copy requires page-aligned
// pointers), so all host<->device transfers are explicit.
PDRef* scalarRmw(const void* p, size_t bytes) {
    PDRef* r = bindResolve(p);
    if (r != nil) return r;
    Ctx& c = ctx();
    if (c.window == nil) c.window = [NSMutableDictionary new];
    PDBind* bd = [PDBind new];
    bd->host = const_cast<void*>(p);
    bd->len = bytes;
    bd->dirty = YES;
    bd->toHost = YES;  // accumulator result: host reads it at the fence
    bd->buf = [g_device newBufferWithLength:bytes
                                    options:MTLResourceStorageModeShared];
    memcpy(bd->buf.contents, p, bytes);
    [c.window setObject:bd forKey:ptrKey(p)];
    c.stageBytes += (long long)bytes;
    return [PDRef with:bd->buf off:0];
}


id<MTLComputePipelineState> pipeline(NSString* name) {
    std::lock_guard<std::mutex> lk(g_pipeMutex);
    if (g_pipes == nil) g_pipes = [NSMutableDictionary new];
    id p = [g_pipes objectForKey:name];
    if (p != nil) return p;
    NSError* err = nil;
    id<MTLFunction> f = [g_library newFunctionWithName:name];
    p = [g_device newComputePipelineStateWithFunction:f error:&err];
    if (p == nil) {
        fprintf(stderr, "pipeline %s failed: %s\n", name.UTF8String,
                err.localizedDescription.UTF8String);
        std::abort();
    }
    [g_pipes setObject:p forKey:name];
    return p;
}

// ---------------------------------------------------------------------------
// Command buffer accumulation: every encode (custom GEMM or compute kernel)
// goes
// into one per-window command buffer, committed together at the fence.
// ---------------------------------------------------------------------------
void ensureCB() {
    Ctx& c = ctx();
    if (c.queue == nil) c.queue = [g_device newCommandQueue];
    if (c.cb == nil) {
        // One autorelease pool per residency window (command buffer lifetime).
        // It MUST be popped at the fence: a long-lived thread (e.g. the main
        // thread, which never exits) would otherwise accumulate every
        // autoreleased buffer/ref forever, growing memory and eventually
        // corrupting allocator reuse.
        if (c.pool == nullptr) c.pool = objc_autoreleasePoolPush();
        c.cb = [c.queue commandBuffer];
        if (c.alive == nil)
            c.alive = [NSMutableArray arrayWithCapacity:128];
    }
}

void closeEncoder() {
    Ctx& c = ctx();
    if (c.enc != nil) {
        [c.enc endEncoding];
        c.enc = nil;
    }
}

// ---------------------------------------------------------------------------
// Compute kernel dispatch helpers
// ---------------------------------------------------------------------------
struct KArgs {
    int a = 0, b = 0, c = 0, d = 0, e = 0, f = 0, g = 0;
};

void launch1(NSString* name, NSArray<PDRef*>* bufs,
             const void* constants, NSUInteger cbytes, int threads) {
    auto _t0 = std::chrono::steady_clock::now();
    keepAlive(bufs);
    ensureCB();
    closeEncoder();
    Ctx& cx = ctx();
    id<MTLComputeCommandEncoder> en = [cx.cb computeCommandEncoder];
    id<MTLComputePipelineState> pso = pipeline(name);
    [en setComputePipelineState:pso];
    for (NSUInteger i = 0; i < bufs.count; ++i)
        [en setBuffer:bufs[i]->buf offset:bufs[i]->off atIndex:i];
    if (cbytes) [en setBytes:constants length:cbytes atIndex:bufs.count];
    NSUInteger tw = 64;
    NSUInteger groups = (threads + tw - 1) / tw;
    [en dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
    [en endEncoding];
    cx.enc = nil;
    ++cx.nCommit;
    auto _t1 = std::chrono::steady_clock::now();
    cx.encSec += std::chrono::duration<double>(_t1 - _t0).count();
}

void launch2(NSString* name, NSArray<PDRef*>* bufs,
             const void* constants, NSUInteger cbytes, int B, int H) {
    auto _t0 = std::chrono::steady_clock::now();
    keepAlive(bufs);
    ensureCB();
    closeEncoder();
    Ctx& cx = ctx();
    id<MTLComputeCommandEncoder> en = [cx.cb computeCommandEncoder];
    [en setComputePipelineState:pipeline(name)];
    for (NSUInteger i = 0; i < bufs.count; ++i)
        [en setBuffer:bufs[i]->buf offset:bufs[i]->off atIndex:i];
    if (cbytes) [en setBytes:constants length:cbytes atIndex:bufs.count];
    NSUInteger gw = 8, gh = 8;
    [en dispatchThreadgroups:MTLSizeMake((B + gw - 1) / gw, (H + gh - 1) / gh, 1)
              threadsPerThreadgroup:MTLSizeMake(gw, gh, 1)];
    [en endEncoding];
    cx.enc = nil;
    ++cx.nCommit;
    auto _t1 = std::chrono::steady_clock::now();
    cx.encSec += std::chrono::duration<double>(_t1 - _t0).count();
}

}  // namespace

// ---------------------------------------------------------------------------
// Per-object persistent device caches (owned by Param/Mat slots)
// ---------------------------------------------------------------------------
static PDRef* slotRef(void** slot) {
    return *slot ? (__bridge PDRef*)*slot : nil;
}

void* mpsWeightCache(void** slot, const void* p, size_t bytes) {
    PDRef* r = slotRef(slot);
    if (r == nil) {
        id<MTLBuffer> b = [g_device newBufferWithLength:bytes
                                                options:MTLResourceStorageModeShared];
        memcpy(b.contents, p, bytes);
        r = [PDRef with:b off:0 len:bytes];
        *slot = (__bridge_retained void*)r;
        ctx().stageBytes += (long long)bytes;
    }
    return (__bridge void*)r;
}

void* mpsGradCache(void** slot, const void* p, size_t bytes) {
    PDRef* r = slotRef(slot);
    if (r == nil) {
        // Seed once (host gradient is zero after Param::zeroGrad) and keep
        // accumulating device-side until the optimizer flushes the slot.
        id<MTLBuffer> b = [g_device newBufferWithLength:bytes
                                                options:MTLResourceStorageModeShared];
        memcpy(b.contents, p, bytes);
        r = [PDRef with:b off:0 len:bytes];
        *slot = (__bridge_retained void*)r;
        ctx().stageBytes += (long long)bytes;
    }
    return (__bridge void*)r;
}

void mpsFlushGrad(void* slot, void* host, size_t bytes) {
    if (slot == nullptr) return;
    PDRef* r = (__bridge PDRef*)slot;
    memcpy(host, (char*)r->buf.contents + r->off, bytes);
    ctx().flushBytes += (long long)bytes;
}

void mpsDropCache(void** slot) {
    if (slot == nullptr || *slot == nullptr) return;
    id transferred __attribute__((objc_precise_lifetime)) =
        (__bridge_transfer id)*slot;  // releases the retained PDRef
    *slot = nullptr;
    (void)transferred;
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
bool mpsInit() {
    static bool tried = false;
    if (!tried) {
        tried = true;
        @autoreleasepool {
            g_device = MTLCreateSystemDefaultDevice();
            if (g_device != nil) MPSSupportsMTLDevice(g_device);
            if (g_device != nil) {
                NSError* err = nil;
                g_library = [g_device newLibraryWithSource:
                    [NSString stringWithUTF8String:kKernelSrc]
                                                   options:[MTLCompileOptions new]
                                                     error:&err];
                if (g_library == nil)
                    fprintf(stderr, "Metal library compile failed: %s\n",
                            err.localizedDescription.UTF8String);
            }
        }
    }
    return g_device != nil && g_library != nil;
}

bool mpsAvailable() {
    @autoreleasepool {
        if (!mpsInit()) return false;
        Ctx& c = ctx();
        if (c.queue == nil) c.queue = [g_device newCommandQueue];
        return c.queue != nil;
    }
}

void mpsPrintStats(const char* tag) {
    Ctx& c = ctx();
    fprintf(stderr,
            "[mps %s] CB=%lld enc=%lld waitMs=%.1f encMs=%.1f "
            "stageMB=%.1f flushMB=%.1f\n",
            tag, c.nWait, c.nCommit, c.waitSec * 1e3, c.encSec * 1e3,
            c.stageBytes / 1e6, c.flushBytes / 1e6);
    c.nWait = c.nCommit = 0; c.stageBytes = c.flushBytes = 0;
    c.waitSec = c.encSec = 0;
}

void mpsMarkHost(const void* p) {
    Ctx& c = ctx();
    PDBind* bd = [c.window objectForKey:ptrKey(p)];
    if (bd != nil) bd->toHost = YES;
}

void mpsWait(bool keepWindow) {
    Ctx& c = ctx();
    auto t0 = std::chrono::steady_clock::now();
    closeEncoder();
    if (c.cb != nil) {
        [c.cb commit];
        [c.cb waitUntilCompleted];
        if (c.cb.status != MTLCommandBufferStatusCompleted)
            fprintf(stderr, "Metal command buffer failed: %s\n",
                    c.cb.error.localizedDescription.UTF8String);
        ++c.nWait;
        if (c.window != nil) {
            for (PDBind* bd in c.window.allValues) {
                if (bd->toHost) {
                    memcpy(bd->host, bd->buf.contents, bd->len);
                    c.flushBytes += (long long)bd->len;
                    // Stay marked while the window survives: read-write
                    // accumulators can be updated again before the final fence.
                    if (!keepWindow) bd->toHost = NO;
                }
            }
            if (!keepWindow) [c.window removeAllObjects];
        }
        c.cb = nil;
    }
    [c.alive removeAllObjects];
    c.alive = nil;
    // Drain all autoreleased refs/buffers produced in this window. Persistent
    // caches live in Param/Mat slots with their own balance, unaffected.
    if (c.pool != nullptr) {
        objc_autoreleasePoolPop(c.pool);
        c.pool = nullptr;
    }
    auto t1 = std::chrono::steady_clock::now();
    c.waitSec += std::chrono::duration<double>(t1 - t0).count();
}

// Stage host data into the residency window as a read-write dirty bind.
void mpsStageInput(const void* p, size_t bytes) {
    Ctx& c = ctx();
    if (c.window == nil) c.window = [NSMutableDictionary new];
    if ([c.window objectForKey:ptrKey(p)] != nil) return;
    PDBind* bd = [PDBind new];
    bd->host = const_cast<void*>(p);
    bd->len = bytes;
    bd->dirty = YES;
    bd->buf = [g_device newBufferWithLength:bytes
                                    options:MTLResourceStorageModeShared];
    memcpy(bd->buf.contents, p, bytes);
    [c.window setObject:bd forKey:ptrKey(p)];
}

// ---------------------------------------------------------------------------
// GEMM stream
// ---------------------------------------------------------------------------

// Host-side mirror of the MSL GemmP layout.
struct GParam {
    int M, N, K, lda, ldb, ldc, tA, tB, epi;
};

// Must match the gemm_block / gemm_blockn MSL constants.
static constexpr int GB_M = 32, GB_N = 32, GB_W = 8;
static constexpr int GB_NM = 16, GB_NN = 32;

// Custom tiled GEMM (one compute dispatch) with fused bias+relu epilogue.
void mpsCustomGemm(const GemmOp& g, const void* biasV, size_t biasBytes,
                   int epi) {
    const float* bias = static_cast<const float*>(biasV);
    auto _t0 = std::chrono::steady_clock::now();
    ensureCB();
    closeEncoder();
    Ctx& cx = ctx();
    NSUInteger ar = (g.transA == 'N') ? g.M : g.K;
    NSUInteger br = (g.transB == 'N') ? g.K : g.N;
    PDRef* rA = kerIn(g.A, ar * g.lda * 4);
    PDRef* rB = g.wSlot
        ? (__bridge PDRef*)mpsWeightCache(g.wSlot, g.B, br * g.ldb * 4)
        : kerIn(g.B, br * g.ldb * 4);
    PDRef* rC = bindOutput(g.C, size_t(g.M) * g.ldc * 4);
    PDRef* rBias = bias ? kerIn(bias, biasBytes) : nil;
    NSMutableArray<PDRef*>* bufs = [NSMutableArray arrayWithObjects:rA, rB, rC,
                                            rBias ? rBias : rC, nil];
    keepAlive(bufs);
    GParam p{g.M, g.N, g.K, g.lda, g.ldb, g.ldc,
             g.transA == 'T' ? 1 : 0, g.transB == 'T' ? 1 : 0, epi};
    id<MTLComputeCommandEncoder> en = [cx.cb computeCommandEncoder];
    // Kernel selection (override with PD_GEMM_OLD / PD_GEMM_FORCE):
    //  gemm_block  (32x32, 4x4 outputs/thread) wins on tall-M / big tiles;
    //  gemm_blockn (16x32, 2x4) wins on short-M products where 32x32 would
    //  launch too few threadgroups to hide the K-loop latency;
    //  gemm_tiled  (16x16 scalar) retained for degenerate shapes.
    int variant;  // 0=block, 1=blockn, 2=tiled
    if (getenv("PD_GEMM_OLD")) variant = 2;
    else if (getenv("PD_GEMM_FORCE"))
        variant = atoi(getenv("PD_GEMM_FORCE"));
    else if (g.M >= 192)
        variant = 0;                 // tall M: 32x32 register block wins
    else if (g.N >= 512)
        variant = 1;                 // short M, wide N: 16x32 block wins
    else
        variant = 2;                 // small tiles / long K: scalar 16x16
                                     // launches enough groups to hide latency
    NSString* names[3] = {@"gemm_block", @"gemm_blockn", @"gemm_tiled"};
    [en setComputePipelineState:pipeline(names[variant])];
    for (NSUInteger i = 0; i < 4; ++i)
        [en setBuffer:bufs[i]->buf offset:bufs[i]->off atIndex:i];
    [en setBytes:&p length:sizeof(p) atIndex:4];
    if (variant == 0) {
        NSUInteger gx = (g.N + GB_N - 1) / GB_N;
        NSUInteger gy = (g.M + GB_M - 1) / GB_M;
        [en dispatchThreadgroups:MTLSizeMake(gx, gy, 1)
                  threadsPerThreadgroup:MTLSizeMake(GB_W, GB_W, 1)];
    } else if (variant == 1) {
        NSUInteger gx = (g.N + GB_NN - 1) / GB_NN;
        NSUInteger gy = (g.M + GB_NM - 1) / GB_NM;
        [en dispatchThreadgroups:MTLSizeMake(gx, gy, 1)
                  threadsPerThreadgroup:MTLSizeMake(GB_W, GB_W, 1)];
    } else {
        NSUInteger gx = (g.N + 15) / 16, gy = (g.M + 15) / 16;
        [en dispatchThreadgroups:MTLSizeMake(gx, gy, 1)
                  threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    }
    [en endEncoding];
    cx.enc = nil;
    ++cx.nCommit;
    auto _t1 = std::chrono::steady_clock::now();
    cx.encSec += std::chrono::duration<double>(_t1 - _t0).count();
}

void mpsCommitGemms(int count, const GemmOp* ops, long long /*macs*/) {
    // Every GEMM goes through the in-house tiled kernel.
    //
    // MPSMatrixMultiplication was evaluated and rejected: chained inside the
    // residency windows (many small batched products feeding one another on
    // the same command buffer), it returns deterministically wrong results /
    // NaNs on this device, even though isolated one-off GEMMs and Apple's
    // debug layer show nothing. The bounded custom kernel is bit-exact vs the
    // CPU reference across all parity, accumulation and concurrency tests.
    for (int e = 0; e < count; ++e) mpsCustomGemm(ops[e], nullptr, 0, 0);
}

// ---------------------------------------------------------------------------
// Elementwise kernels (Metal entry points)
// ---------------------------------------------------------------------------
PDRef* rIn(const Mat& m) { return kerIn(m.data(), size_t(m.r) * m.s * 4); }
PDRef* rOut(Mat& m) { return kerOut(m.data(), size_t(m.r) * m.s * 4); }
PDRef* rOutSeed(Mat& m) { return kerOutSeed(m.data(), size_t(m.r) * m.s * 4); }
// Read-only bias/constant vector: fresh staged copy each window so optimizer
// updates are picked up automatically.
PDRef* rScalar(const std::vector<float>& v) {
    size_t bytes = ((v.size() * 4 + 15) & ~size_t(15));
    return kerIn(v.data(), bytes);
}

void mpsAddBias(Mat& y, const std::vector<float>& b) {
    KArgs q{y.c, y.r * y.c, y.s, 0, 0, 0, 0};
    launch1(@"add_bias", @[rOut(y), rScalar(b)], &q, sizeof(q), y.r * y.c);
}

void mpsRelu(Mat& x) {
    KArgs q{x.c, x.r * x.c, x.s, 0, 0, 0, 0};
    launch1(@"relu_fwd", @[rOutSeed(x)], &q, sizeof(q), x.r * x.c);
}

void mpsReluBwd(const Mat& pre, const Mat& gout, Mat& gin) {
    struct P { int cols, n, ps, gs; } p{pre.c, pre.r * pre.c, pre.s, gout.s};
    launch1(@"relu_bwd", @[rIn(pre), rIn(gout), rOut(gin)],
            &p, sizeof(p), pre.r * pre.c);
}

void mpsGateAdd(Mat& gp, const Mat& gi, const Mat& gh,
                const std::vector<float>& bi, const std::vector<float>& bh) {
    struct P { int H, B; } p{gp.c, gp.r};
    launch1(@"gate_add",
            @[rOut(gp), rIn(gi), rIn(gh), rScalar(bi), rScalar(bh)],
            &p, sizeof(p), gp.r * gp.c);
}

void mpsLstmCellFwd(const Mat& gp, const float* cp, Mat& hOut, Mat& cOut,
                    int hidden) {
    struct P { int h, B, hasCp; } p{hidden, gp.r, cp ? 1 : 0};
    PDRef* cpRef = cp ? kerIn(cp, size_t(gp.r) * hidden * 4)
                   : [PDRef with:[g_device newBufferWithLength:16 options:0] off:0];
    launch2(@"lstm_cell_fwd",
            @[rIn(gp), cpRef, rOut(hOut), rOut(cOut)],
            &p, sizeof(p), gp.r, hidden);
}

void mpsLstmCellBwd(const Mat& gh, const Mat& gp, const Mat& cNow,
                    const float* cp, Mat& dh, Mat& dc, Mat& dg, int hidden) {
    struct P { int h, B, hasCp; } p{hidden, gp.r, cp ? 1 : 0};
    PDRef* cpRef = cp ? kerIn(cp, size_t(gp.r) * hidden * 4)
                   : [PDRef with:[g_device newBufferWithLength:16 options:0] off:0];
    launch2(@"lstm_cell_bwd",
            @[rIn(gh), rIn(gp), rIn(cNow), cpRef,
              rIn(dh), rIn(dc), rOut(dg)],
            &p, sizeof(p), gp.r, hidden);
}

void mpsConcat(Mat& z, const Mat& a, int n1, const Mat& b, int n2) {
    struct P { int n1, n2, B, zs, as, bs; }
        p{n1, n2, z.r, z.s, a.s, b.s};
    launch1(@"concat2", @[rOut(z), rIn(a), rIn(b)],
            &p, sizeof(p), z.r * (n1 + n2));
}

void mpsSplit(const Mat& z, int n1, Mat& a, Mat& b, int n2) {
    struct P { int n1, n2, B, zs, as, bs; }
        p{n1, n2, z.r, z.s, a.s, b.s};
    launch1(@"split2", @[rIn(z), rOut(a), rOut(b)],
            &p, sizeof(p), z.r * (n1 + n2));
}

void mpsZeroAndLast(Mat& all, const Mat& gh, int B, int T, int h) {
    struct P { int B, T, h; } p{B, T, h};
    launch1(@"zero_and_last", @[rOut(all), rIn(gh)],
            &p, sizeof(p), B * T * h);
}

void mpsMaskDyn(Mat& logits, const Mat& ds, const Mat& mask, int N) {
    struct P { int N, B, ls, ms; } p{N, logits.r, logits.s, mask.s};
    launch1(@"mask_dyn", @[rOut(logits), rIn(ds), rIn(mask)],
            &p, sizeof(p), logits.r * N);
}

void mpsAddTo(float* dst, void** gSlot, int dstCols, const Mat& src) {
    struct P { int rows, cols, dc, sc; } p{src.r, src.c, dstCols, src.s};
    size_t rounded = (size_t(src.r) * dstCols * 4 + 15) & ~size_t(15);
    PDRef* rd = (__bridge PDRef*)mpsGradCache(gSlot, dst, rounded);
    launch1(@"add_to", @[rd, rIn(src)],
            &p, sizeof(p), src.r * src.c);
}

void mpsBiasGradAdd(const Mat& g, float* db, void** dbSlot) {
    struct P { int B, o, gs; } p{g.r, g.c, g.s};
    size_t rounded = (size_t(g.c) * 4 + 15) & ~size_t(15);
    PDRef* rd = (__bridge PDRef*)mpsGradCache(dbSlot, db, rounded);
    launch1(@"bias_grad_add", @[rIn(g), rd],
            &p, sizeof(p), g.c);
}

void mpsSlice(Mat& dst, const Mat& src, int off, int h) {
    struct P { int B, h, off, ds, ss; } p{src.r, h, off, dst.s, src.s};
    launch1(@"slice_cols", @[rOut(dst), rIn(src)],
            &p, sizeof(p), src.r * h);
}

void mpsFlatten(Mat& dst, const Mat& src, int N) {
    struct P { int B, N, ss; } p{src.r, N, src.s};
    launch1(@"flatten_rows", @[rOut(dst), rIn(src)],
            &p, sizeof(p), src.r * N);
}

void mpsCopyMat(Mat& dst, const Mat& src) {
    struct P { int rows, cols, ds, ss; } p{src.r, src.c, dst.s, src.s};
    launch1(@"copy_mat", @[rOut(dst), rIn(src)],
            &p, sizeof(p), src.r * src.c);
}

void mpsAdam(std::vector<float>& w, std::vector<float>& dw,
             std::vector<float>& m, std::vector<float>& v, int n, float lr,
             float beta1, float beta2, float eps, float bc1, float bc2) {
    struct P {
        int n;
        float lr, b1, b2, eps, bc1, bc2;
    } p{n, lr, beta1, beta2, eps, bc1, bc2};
    size_t bytes = (size_t(n) + 3 & ~size_t(3)) * 4;
    ensureCB();
    closeEncoder();
    Ctx& cx = ctx();
    id<MTLComputeCommandEncoder> en = [cx.cb computeCommandEncoder];
    [en setComputePipelineState:pipeline(@"adam_step")];
    [en setBuffer:scalarRmw(w.data(), bytes)->buf offset:0 atIndex:0];
    [en setBuffer:scalarRmw(dw.data(), bytes)->buf offset:0 atIndex:1];
    [en setBuffer:scalarRmw(m.data(), bytes)->buf offset:0 atIndex:2];
    [en setBuffer:scalarRmw(v.data(), bytes)->buf offset:0 atIndex:3];
    [en setBytes:&p length:sizeof(p) atIndex:4];
    NSUInteger tw = 128;
    [en dispatchThreadgroups:MTLSizeMake((n + tw - 1) / tw, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
    [en endEncoding];
    cx.enc = nil;
    ++cx.nCommit;
}

}  // namespace nn
