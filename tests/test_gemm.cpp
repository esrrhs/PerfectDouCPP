// CPU vs GPU GEMM parity (Apple Metal or Windows D3D12).
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <gtest/gtest.h>

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

TEST(GemmTest, Parity) {
    gemmInit();
    if (!gemmHasGpu()) {
        GTEST_SKIP() << "No GPU available, skipping GEMM GPU parity test";
    }
    // large enough to exceed the GPU threshold (>=1e6 MACs)
    float d1 = maxDiff('N', 'N', 256, 256, 4274);
    float d2 = maxDiff('T', 'N', 256, 256, 256);
    float d3 = maxDiff('N', 'T', 256, 256, 4274);
    EXPECT_LT(d1, 1e-2f);
    EXPECT_LT(d2, 1e-2f);
    EXPECT_LT(d3, 1e-2f);
}
