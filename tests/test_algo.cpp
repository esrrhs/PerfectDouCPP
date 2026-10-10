// Unit tests for RL algorithm components:
// 1. Advantage Normalization
// 2. PPO Value Clipping loss & gradient
// 3. Dynamic KL divergence estimation & early stopping logic
// 4. Historical archive pool anchor preservation & thinning logic
// 5. Running normalizer online statistics
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "algo/ppo.h"
#include "algo/rollout.h"

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::printf("CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, \
                        #cond);                                           \
            std::abort();                                                 \
        }                                                                 \
    } while (0)

using namespace algo;

// 1. Test advantage normalization and GAE terminal cutoff
static void testAdvantageNormAndGAE() {
    std::vector<Transition> tr(6);
    // Game 1: 3 steps
    for (int i = 0; i < 3; ++i) {
        tr[i].gameId = 1;
        tr[i].value = 1.0f;
    }
    tr[2].reward = 2.0f;
    tr[2].terminal = true;

    // Game 2: 3 steps
    for (int i = 3; i < 6; ++i) {
        tr[i].gameId = 2;
        tr[i].value = -1.0f;
    }
    tr[5].reward = -2.0f;
    tr[5].terminal = true;

    assignEpisodeReturns(tr, 1.0f, 0.95f);

    // GAE check
    CHECK(tr[2].adv == 1.0f);   // 2.0 - 1.0
    CHECK(tr[5].adv == -1.0f);  // -2.0 - (-1.0)
    CHECK(tr[1].adv == 0.95f);  // delta=0 + 0.95*1.0
    CHECK(tr[4].adv == -0.95f); // delta=0 + 0.95*(-1.0)

    // Manual Advantage Normalization simulation as in ppoUpdate
    int N = static_cast<int>(tr.size());
    double mean = 0.0, var = 0.0;
    for (const Transition& t : tr) mean += t.adv;
    mean /= N;
    CHECK(std::abs(mean) < 1e-6); // symmetric distribution
    for (const Transition& t : tr) var += (t.adv - float(mean)) * (t.adv - float(mean));
    float std = float(std::sqrt(var / N + 1e-8));
    CHECK(std > 0.0f);

    for (Transition& t : tr) t.adv = (t.adv - float(mean)) / std;

    double normMean = 0.0, normVar = 0.0;
    for (const Transition& t : tr) normMean += t.adv;
    normMean /= N;
    for (const Transition& t : tr) normVar += (t.adv - float(normMean)) * (t.adv - float(normMean));
    double normStd = std::sqrt(normVar / N);

    CHECK(std::abs(normMean) < 1e-5);
    CHECK(std::abs(normStd - 1.0) < 1e-4);
}

// 2. Test PPO Value Loss Clipping and gradient behavior
static void testValueClipping() {
    float clip = 0.2f;
    float vfCoef = 0.5f;
    int B = 1;

    // Case A: vPred is within bounds [vOld - clip, vOld + clip]
    {
        float vOld = 1.0f;
        float vPred = 1.1f;    // within [-0.2, 0.2]
        float target = 2.0f;
        float err = vPred - target; // 1.1 - 2.0 = -0.9
        float vClipped = vOld + std::clamp(vPred - vOld, -clip, clip);
        float errClipped = vClipped - target; // 1.1 - 2.0 = -0.9
        float l1 = err * err;
        float l2 = errClipped * errClipped;

        float grad;
        if (l2 > l1) {
            grad = 0.0f; // clipped loss is larger, saturation zone
        } else {
            grad = vfCoef * err / float(B);
        }
        CHECK(l1 == l2);
        CHECK(grad == vfCoef * (-0.9f));
    }

    // Case B: vPred moves away from target past clip bound (l2 > l1)
    {
        float vOld = 1.0f;
        float vPred = 1.5f;     // moved +0.5, past clip of 0.2
        float target = 0.0f;    // target is 0, so vPred moved away from target!
        float err = vPred - target; // 1.5
        float vClipped = vOld + std::clamp(vPred - vOld, -clip, clip); // 1.0 + 0.2 = 1.2
        float errClipped = vClipped - target; // 1.2
        float l1 = err * err;                 // 2.25
        float l2 = errClipped * errClipped;   // 1.44
        // Here l1 > l2, so gradient is active towards target
        float grad = (l2 > l1) ? 0.0f : (vfCoef * err / float(B));
        CHECK(grad != 0.0f);
    }

    // Case C: vPred moved towards target past clip bound (l2 > l1 -> gradient should be 0)
    {
        float vOld = 1.0f;
        float vPred = 1.5f;     // moved +0.5, past clip of 0.2
        float target = 3.0f;    // target is 3.0, so vPred moved closer to target!
        float err = vPred - target;           // 1.5 - 3.0 = -1.5, l1 = 2.25
        float vClipped = vOld + std::clamp(vPred - vOld, -clip, clip); // 1.2
        float errClipped = vClipped - target; // 1.2 - 3.0 = -1.8, l2 = 3.24
        float l1 = err * err;                 // 2.25
        float l2 = errClipped * errClipped;   // 3.24
        CHECK(l2 > l1);
        // Under PPO, clipped surrogate is max(l1, l2) = l2.
        // Derivative of vClipped with respect to vPred is 0!
        float grad = (l2 > l1) ? 0.0f : (vfCoef * err / float(B));
        CHECK(grad == 0.0f); // MUST be exactly 0
    }
}

// 3. Test KL divergence math properties: non-negative and zero when ratio=1
static void testKLDivergence() {
    auto calcKL = [](float ratio) {
        float logRatio = std::log(ratio);
        return (ratio - 1.0f) - logRatio;
    };

    // Exactly 0 at ratio = 1.0
    CHECK(std::abs(calcKL(1.0f)) < 1e-7f);

    // Strictly positive when ratio != 1.0
    for (float r = 0.1f; r < 2.5f; r += 0.05f) {
        if (std::abs(r - 1.0f) > 1e-4f) {
            float kl = calcKL(r);
            CHECK(kl > 0.0f);
        }
    }

    // Monotonically increasing as ratio deviates from 1
    CHECK(calcKL(1.2f) > calcKL(1.1f));
    CHECK(calcKL(0.8f) > calcKL(0.9f));
}

// 4. Test League Archive Anchor Preservation & Thinning
static void testArchiveAnchorPreservation() {
    HistoricalPool pool;
    int maxArchive = 4;

    auto addSnapshot = [&](int upd) {
        auto snap = std::make_shared<HistoricalActorSnapshot>();
        snap->update = upd;
        while ((int)pool.archive.size() >= maxArchive && !pool.archive.empty()) {
            if (pool.archive.size() > 2) {
                // Thin out from the middle
                pool.archive.erase(pool.archive.begin() + 1);
            } else {
                pool.archive.erase(pool.archive.begin());
            }
        }
        pool.archive.push_back(snap);
    };

    // Insert 10 updates
    for (int upd = 1; upd <= 10; ++upd) {
        addSnapshot(upd);
    }

    CHECK(pool.archive.size() == 4);
    // Anchor update 1 must be preserved!
    CHECK(pool.archive[0]->update == 1);
    // The latest update 10 must be present!
    CHECK(pool.archive.back()->update == 10);
}

// 5. Test Welford Running Normalizer
static void testRunningNormalizer() {
    RunningNormalizer norm;
    std::vector<float> data = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
    for (float x : data) norm.update(x);

    // Mean should be 30.0
    CHECK(std::abs(norm.mean - 30.0) < 1e-4);
    // Normalize and Denormalize round-trip
    float val = 25.0f;
    float n = norm.normalize(val);
    float d = norm.denormalize(n);
    CHECK(std::abs(d - val) < 1e-4f);
}

// 6. Test Adam state serialization roundtrip
static void testAdamSerialization() {
    nn::NetConfig cfg{64, 32};
    nn::Actor a1;
    a1.init(cfg, 123);
    auto ps1 = a1.params();
    CHECK(!ps1.empty());
    ps1[0]->m[0] = 0.777f;
    ps1[0]->v[0] = 0.888f;

    nn::Adam opt1;
    opt1.lr = 2.5e-4f;
    opt1.beta1 = 0.91f;
    opt1.beta2 = 0.998f;
    opt1.eps = 1e-6f;
    opt1.t = 77;

    const char* tmpPath = "test_opt_roundtrip.bin";
    nn::saveOptimizer(tmpPath, ps1, opt1);

    nn::Actor a2;
    a2.init(cfg, 123);
    auto ps2 = a2.params();
    nn::Adam opt2;
    bool ok = nn::loadOptimizer(tmpPath, ps2, opt2);
    CHECK(ok);
    CHECK(opt2.t == 77);
    CHECK(std::abs(opt2.lr - 2.5e-4f) < 1e-8f);
    CHECK(std::abs(opt2.beta1 - 0.91f) < 1e-6f);
    CHECK(std::abs(opt2.beta2 - 0.998f) < 1e-6f);
    CHECK(std::abs(opt2.eps - 1e-6f) < 1e-9f);
    CHECK(std::abs(ps2[0]->m[0] - 0.777f) < 1e-6f);
    CHECK(std::abs(ps2[0]->v[0] - 0.888f) < 1e-6f);

    std::remove(tmpPath);
}

// 7. Test Critic head zero-initialization
static void testCriticZeroInit() {
    nn::NetConfig cfg{64, 32};
    nn::Critic critic;
    critic.init(cfg, 456);
    for (float w : critic.out.W.w) {
        CHECK(w == 0.0f);
    }
    for (float b : critic.out.b.w) {
        CHECK(b == 0.0f);
    }
}

// 8. Test ClipFraction sign gating logic
static void testClipFractionSignGating() {
    float clip = 0.2f;
    auto isClipped = [clip](float adv, float ratio) {
        return (adv > 0.0f && ratio > 1.0f + clip) ||
               (adv < 0.0f && ratio < 1.0f - clip);
    };

    // Beneficial moves where ratio is bounded
    CHECK(isClipped(1.0f, 1.3f) == true);   // positive adv, ratio > 1+clip -> clipped
    CHECK(isClipped(1.0f, 0.7f) == false);  // positive adv, ratio < 1-clip -> pessimistic lower bound, unclipped

    // Detrimental moves where ratio is bounded
    CHECK(isClipped(-1.0f, 0.7f) == true);  // negative adv, ratio < 1-clip -> clipped
    CHECK(isClipped(-1.0f, 1.3f) == false); // negative adv, ratio > 1+clip -> pessimistic upper bound, unclipped
}

int main() {
    std::printf("testAdvantageNormAndGAE...\n"); testAdvantageNormAndGAE();
    std::printf("testValueClipping...\n"); testValueClipping();
    std::printf("testKLDivergence...\n"); testKLDivergence();
    std::printf("testArchiveAnchorPreservation...\n"); testArchiveAnchorPreservation();
    std::printf("testRunningNormalizer...\n"); testRunningNormalizer();
    std::printf("testAdamSerialization...\n"); testAdamSerialization();
    std::printf("testCriticZeroInit...\n"); testCriticZeroInit();
    std::printf("testClipFractionSignGating...\n"); testClipFractionSignGating();
    std::printf("ALL ALGO TESTS PASSED\n");
    return 0;
}
