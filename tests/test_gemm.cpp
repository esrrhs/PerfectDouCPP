// CPU vs GPU GEMM parity (only meaningful on Apple with Metal/MPS).
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "nn/gemm.h"

using namespace nn;

static float maxDiff(char tA, char tB, int M, int N, int K) {
    int ap = (tA == 'N' ? M : K), ac = (tA == 'N' ? K : M);
    int bp = (tB == 'N' ? K : N), bc = (tB == 'N' ? N : K);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> d(-0.5f, 0.5f);
    std::vector<float> A(ap * ac), B(bp * bc), Cc(M * N), Cg(M * N);
    for (auto& v : A) v = d(rng);
    for (auto& v : B) v = d(rng);
    gemmSetGpu(false);
    sgemm(tA, tB, M, N, K, A.data(), ac, B.data(), bc, Cc.data(), N);
    gemmSetGpu(true);
    sgemm(tA, tB, M, N, K, A.data(), ac, B.data(), bc, Cg.data(), N);
    float md = 0;
    for (size_t i = 0; i < Cc.size(); ++i)
        md = std::max(md, std::abs(Cc[i] - Cg[i]));
    gemmSetGpu(false);
    return md;
}

int main() {
    gemmInit();
    std::printf("GPU available: %s\n", gemmHasGpu() ? "yes" : "no");
    if (!gemmHasGpu()) {
        std::printf("SKIP (no Metal/MPS)\n");
        return 0;
    }
    // large enough to exceed the GPU threshold (>=1e6 MACs)
    float d1 = maxDiff('N', 'N', 256, 256, 4274);
    float d2 = maxDiff('T', 'N', 256, 256, 256);
    float d3 = maxDiff('N', 'T', 256, 256, 4274);
    std::printf("max diff NN=%.2e TN=%.2e NT=%.2e\n", d1, d2, d3);
    bool ok = d1 < 1e-2 && d2 < 1e-2 && d3 < 1e-2;
    std::printf("%s\n", ok ? "GEMM PARITY OK" : "GEMM PARITY FAIL");
    return ok ? 0 : 1;
}
