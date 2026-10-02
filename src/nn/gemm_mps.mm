// Metal Performance Shaders GEMM backend (Apple only).
//
// MPS runs sgemm on the system default GPU. Input/output matrices are copied
// into MTL shared buffers each call; on Apple Silicon this is unified memory
// and the copies are plain fast memcpys. Per-thread caches reuse kernel
// objects and buffers (every call is synchronous, so reuse is safe).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "nn/gemm.h"

namespace nn {

namespace {

id<MTLDevice> g_device = nil;

struct Ctx {
    id<MTLCommandQueue> queue = nil;
    NSMutableDictionary<NSString*, MPSMatrixMultiplication*>* ops = nil;
    // pool of 4 buffers per byte size (A/B/C of one call must not alias);
    // calls are synchronous, so older slots are safe to overwrite.
    NSMutableDictionary<NSNumber*, NSMutableArray<id<MTLBuffer>>*>* bufs = nil;
    NSUInteger ringPos = 0;
};

Ctx& ctx() {
    static thread_local Ctx c;
    return c;
}

id<MTLBuffer> getBuffer(size_t bytes, const void* src, bool copyIn) {
    Ctx& c = ctx();
    if (c.bufs == nil) c.bufs = [NSMutableDictionary new];
    NSNumber* key = @(bytes);
    NSMutableArray<id<MTLBuffer>>* ring = [c.bufs objectForKey:key];
    if (ring == nil) {
        ring = [NSMutableArray arrayWithCapacity:4];
        for (int i = 0; i < 4; ++i)
            [ring addObject:[g_device newBufferWithLength:bytes
                                                  options:MTLResourceStorageModeShared]];
        [c.bufs setObject:ring forKey:key];
    }
    id<MTLBuffer> buf = [ring objectAtIndex:c.ringPos % 4];
    c.ringPos++;
    if (copyIn) memcpy(buf.contents, src, bytes);
    return buf;
}

MPSMatrixMultiplication* getOp(BOOL tA, BOOL tB, NSUInteger M, NSUInteger N,
                               NSUInteger K) {
    Ctx& c = ctx();
    if (c.ops == nil) c.ops = [NSMutableDictionary new];
    NSString* key = [NSString stringWithFormat:@"%d_%d_%lu_%lu_%lu", tA, tB,
                                               (unsigned long)M, (unsigned long)N,
                                               (unsigned long)K];
    MPSMatrixMultiplication* op = [c.ops objectForKey:key];
    if (op == nil) {
        op = [[MPSMatrixMultiplication alloc] initWithDevice:g_device
                                               transposeLeft:tA
                                              transposeRight:tB
                                                  resultRows:M
                                               resultColumns:N
                                             interiorColumns:K
                                                        alpha:1.0
                                                         beta:0.0];
        [c.ops setObject:op forKey:key];
    }
    return op;
}

}  // namespace

bool mpsInit() {
    static bool tried = false;
    if (!tried) {
        tried = true;
        @autoreleasepool {
            g_device = MTLCreateSystemDefaultDevice();
            if (g_device != nil) MPSSupportsMTLDevice(g_device);
        }
    }
    return g_device != nil;
}

bool mpsAvailable() {
    @autoreleasepool {
        if (!mpsInit()) return false;
        Ctx& c = ctx();
        if (c.queue == nil) c.queue = [g_device newCommandQueue];
        return c.queue != nil;
    }
}

void sgemmMps(char tA, char tB, int M, int N, int K,
              const float* A, int lda, const float* B, int ldb,
              float* C, int ldc) {
    @autoreleasepool {
        Ctx& c = ctx();
        if (c.queue == nil) c.queue = [g_device newCommandQueue];

        NSUInteger ar = (tA == 'N') ? M : K, ac = (tA == 'N') ? K : M;
        NSUInteger br = (tB == 'N') ? K : N, bc = (tB == 'N') ? N : K;

        id<MTLBuffer> bA = getBuffer(ar * ac * 4, A, true);
        id<MTLBuffer> bB = getBuffer(br * bc * 4, B, true);
        id<MTLBuffer> bC = getBuffer(M * N * 4, nullptr, false);

        MPSMatrixDescriptor* dA =
            [MPSMatrixDescriptor matrixDescriptorWithRows:ar columns:ac
                  rowBytes:ac * 4 dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* dB =
            [MPSMatrixDescriptor matrixDescriptorWithRows:br columns:bc
                  rowBytes:bc * 4 dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* dC =
            [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N
                  rowBytes:N * 4 dataType:MPSDataTypeFloat32];
        MPSMatrix* mA = [[MPSMatrix alloc] initWithBuffer:bA descriptor:dA];
        MPSMatrix* mB = [[MPSMatrix alloc] initWithBuffer:bB descriptor:dB];
        MPSMatrix* mC = [[MPSMatrix alloc] initWithBuffer:bC descriptor:dC];

        MPSMatrixMultiplication* op =
            getOp((tA == 'T'), (tB == 'T'), M, N, K);

        id<MTLCommandBuffer> cb = [c.queue commandBuffer];
        [op encodeToCommandBuffer:cb leftMatrix:mA rightMatrix:mB resultMatrix:mC];
        [cb commit];
        [cb waitUntilCompleted];

        if (ldc == N) {
            memcpy(C, bC.contents, M * N * 4);
        } else {
            const float* src = static_cast<const float*>(bC.contents);
            for (int i = 0; i < M; ++i)
                memcpy(C + size_t(i) * ldc, src + size_t(i) * N, N * 4);
        }
    }
}

}  // namespace nn
