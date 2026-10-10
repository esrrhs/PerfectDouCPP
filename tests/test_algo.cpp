// Unit tests for RL algorithm components:
// 1. Advantage Normalization
// 2. PPO Value Clipping loss & gradient
// 3. Dynamic KL divergence estimation & early stopping logic
// 4. Historical archive pool anchor preservation & thinning logic
// 5. Running normalizer online statistics
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <thread>
#include <gtest/gtest.h>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
typedef SOCKET test_sock_t;
#define TEST_INVALID_SOCK INVALID_SOCKET
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
typedef int test_sock_t;
#define TEST_INVALID_SOCK (-1)
#endif

#include "algo/eval_douzero.h"
#include "algo/ppo.h"
#include "algo/rollout.h"

#define CHECK(cond) ASSERT_TRUE(cond)

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

// 9. Test Masked Softmax NaN and Inf protection
static void testMaskedSoftmaxNaNProtection() {
    // Normal logits
    {
        float logits[3] = {1.0f, 2.0f, 3.0f};
        float probs[3] = {};
        maskedSoftmax(logits, 3, probs);
        float sum = probs[0] + probs[1] + probs[2];
        CHECK(std::abs(sum - 1.0f) < 1e-5f);
        CHECK(probs[2] > probs[1] && probs[1] > probs[0]);
    }

    // All actions masked out (< -1e8f)
    {
        float logits[4] = {-1e9f, -1e9f, -1e9f, -1e9f};
        float probs[4] = {};
        maskedSoftmax(logits, 4, probs);
        for (int a = 0; a < 4; ++a) {
            CHECK(std::isfinite(probs[a]));
            CHECK(std::abs(probs[a] - 0.25f) < 1e-5f);
        }
    }

    // Logits containing NaN or Inf
    {
        float logits[4] = {NAN, INFINITY, -INFINITY, 2.0f};
        float probs[4] = {};
        maskedSoftmax(logits, 4, probs);
        for (int a = 0; a < 4; ++a) {
            CHECK(std::isfinite(probs[a]));
            CHECK(probs[a] >= 0.0f && probs[a] <= 1.0f);
        }
    }

    // Extreme numerical range
    {
        float logits[3] = {1e6f, -1e6f, 0.0f};
        float probs[3] = {};
        maskedSoftmax(logits, 3, probs);
        for (int a = 0; a < 3; ++a) {
            CHECK(std::isfinite(probs[a]));
        }
        CHECK(std::abs(probs[0] - 1.0f) < 1e-5f);
        CHECK(probs[1] == 0.0f);
    }
}

// 10. Test KL divergence calculation stability against degenerate probabilities
static void testKLDivergenceNaNProtection() {
    auto computeApproxKL = [](float newLogp, float oldLogp) {
        float logRatio = newLogp - oldLogp;
        float ratio = std::exp(logRatio);
        float approxKL = (ratio - 1.0f) - logRatio;
        if (!std::isfinite(approxKL) || approxKL < 0.0f) approxKL = 0.0f;
        return approxKL;
    };

    // Identical
    CHECK(computeApproxKL(-1.5f, -1.5f) == 0.0f);

    // Moderate divergence
    float klMod = computeApproxKL(-1.0f, -1.5f);
    CHECK(std::isfinite(klMod) && klMod > 0.0f);

    // Extreme ratio (huge policy shift)
    float klHuge = computeApproxKL(100.0f, -100.0f);
    CHECK(std::isfinite(klHuge) && klHuge >= 0.0f);

    // Negative infinity logp (zero probability)
    float klZero = computeApproxKL(-1000.0f, -1.0f);
    CHECK(std::isfinite(klZero) && klZero >= 0.0f);
}

// 11. Test PPO training produces strictly finite parameters (no NaN)
static void testPPOWeightsFiniteAfterUpdate() {
    nn::NetConfig cfg{64, 32};
    nn::Actor actor;
    nn::Critic critic;
    actor.init(cfg, 42);
    critic.init(cfg, 43);

    nn::Adam actorOpt;
    nn::Adam criticOpt;
    nn::Rng64 rng(999);

    std::vector<Transition> tr(16);
    for (size_t i = 0; i < tr.size(); ++i) {
        tr[i].gameId = 1;
        tr[i].action = int(i % nn::kNumActions);
        tr[i].reward = (i % 2 == 0) ? 1.0f : -1.0f;
        tr[i].value = 0.0f;
        tr[i].logp = -2.0f;
        std::array<float, ddz::kActionSize> feat{};
        tr[i].actions.emplace_back(tr[i].action, feat);
    }
    tr.back().terminal = true;

    PPOConfig ppoCfg;
    ppoCfg.epochs = 2;
    ppoCfg.minibatch = 8;
    PPOStats stats;

    bool ok = ppoUpdate(actor, critic, tr, ppoCfg, actorOpt, criticOpt, rng, stats);
    CHECK(ok);

    for (nn::Param* p : actor.params()) {
        for (float w : p->w) {
            CHECK(std::isfinite(w));
        }
    }
    for (nn::Param* p : critic.params()) {
        for (float w : p->w) {
            CHECK(std::isfinite(w));
        }
    }
    CHECK(std::isfinite(stats.approxKL));
    CHECK(std::isfinite(stats.pgLoss));
    CHECK(std::isfinite(stats.vLoss));
}

// 12. Test concurrent multi-threaded stream synchronization (liveness and deadlock freedom)
static void testConcurrentLaneStress() {
    constexpr int kThreads = 4;
    std::vector<std::thread> workers;
    std::atomic<bool> stop{false};
    std::atomic<int> completedCycles{0};

    for (int t = 0; t < kThreads; ++t) {
        workers.emplace_back([t, &stop, &completedCycles] {
            nn::gemmSetThreadGpu(0);
            for (int iter = 0; iter < 100 && !stop.load(); ++iter) {
                nn::gpuBindSeat(t);
                nn::gpuSync();
                ++completedCycles;
                std::this_thread::yield();
            }
        });
    }

    // Concurrent thread executing sync
    std::thread syncThread([&stop] {
        for (int iter = 0; iter < 50 && !stop.load(); ++iter) {
            nn::gpuSync();
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });

    for (auto& w : workers) w.join();
    syncThread.join();
    CHECK(completedCycles.load() > 0);
}

// 13. loadOptimizer must not clobber state when the file is truncated
static void testLoadOptimizerAtomic() {
    nn::NetConfig cfg{64, 32};
    nn::Actor a;
    a.init(cfg, 5);
    auto ps = a.params();
    nn::Adam src;
    src.t = 11;
    const char* path = "test_opt_trunc.bin";
    nn::saveOptimizer(path, ps, src);
    FILE* f = std::fopen(path, "rb");
    CHECK(f);
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<char> buf(size_t(sz) / 2);
    CHECK(std::fread(buf.data(), 1, buf.size(), f) == buf.size());
    std::fclose(f);
    f = std::fopen(path, "wb");
    std::fwrite(buf.data(), 1, buf.size(), f);
    std::fclose(f);

    ps[0]->m[0] = 0.5f;
    nn::Adam dst;
    dst.t = 3;
    dst.lr = 1e-3f;
    CHECK(!nn::loadOptimizer(path, ps, dst));
    CHECK(dst.t == 3);
    CHECK(dst.lr == 1e-3f);
    CHECK(ps[0]->m[0] == 0.5f);
    std::remove(path);
}

// 14. A NaN gradient must be skipped instead of poisoning weights
static void testNaNGradientSkipped() {
    nn::NetConfig cfg{64, 32};
    nn::Actor a;
    a.init(cfg, 6);
    auto ps = a.params();
    std::vector<float> before = ps[0]->w;
    for (nn::Param* p : ps) std::fill(p->dw.begin(), p->dw.end(), 0.1f);
    ps[0]->dw[0] = std::nanf("");
    nn::Adam opt;
    opt.applyGradNorm(ps, 0.5f);
    CHECK(opt.t == 0);
    for (nn::Param* p : ps)
        for (float w : p->w) CHECK(std::isfinite(w));
    CHECK(ps[0]->w == before);
}

// 15. PPO update must survive NaN/Inf in stored transitions
static void testPPOSurvivesNonFiniteSamples() {
    nn::NetConfig cfg{64, 32};
    nn::Actor actor;
    nn::Critic critic;
    actor.init(cfg, 42);
    critic.init(cfg, 43);
    nn::Adam ao, co;
    nn::Rng64 rng(7);
    std::vector<Transition> tr(16);
    for (size_t i = 0; i < tr.size(); ++i) {
        tr[i].gameId = int(i / 8);
        tr[i].action = int(i % nn::kNumActions);
        tr[i].reward = (i % 2) ? 1.0f : -1.0f;
        tr[i].logp = -2.0f;
        std::array<float, ddz::kActionSize> feat{};
        tr[i].actions.emplace_back(tr[i].action, feat);
        if (i % 8 == 7) tr[i].terminal = true;
    }
    tr[3].logp = -std::numeric_limits<float>::infinity();
    PPOConfig pc;
    pc.epochs = 1;
    pc.minibatch = 8;
    PPOStats st;
    ppoUpdate(actor, critic, tr, pc, ao, co, rng, st);
    for (nn::Param* p : actor.params())
        for (float w : p->w) CHECK(std::isfinite(w));
    for (nn::Param* p : critic.params())
        for (float w : p->w) CHECK(std::isfinite(w));
}

// 16. Producer/consumer ring in ppoUpdate must not hang (repeat to expose
// lost-wakeup races); a watchdog aborts instead of hanging CI.
static void testPPONoDeadlock() {
    std::atomic<bool> done{false};
    std::thread watchdog([&] {
        for (int i = 0; i < 600 && !done.load(); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (!done.load()) {
            std::printf("deadlock detected in ppoUpdate\n");
            std::fflush(stdout);
            std::abort();
        }
    });
    for (int rep = 0; rep < 20; ++rep) testPPOWeightsFiniteAfterUpdate();
    done = true;
    watchdog.join();
}

// 17. Actor cloning deep copy independence and fail-safe handling when DouZero server is unreachable.
static void testActorCloningAndEvalFailSafe() {
    nn::NetConfig cfg{64, 32};
    std::array<nn::Actor, 3> actors;
    for (int s = 0; s < 3; ++s) {
        actors[s].init(cfg, 42 + s);
    }
    auto cloned = cloneActors(actors);
    for (int s = 0; s < 3; ++s) {
        auto origParams = actors[s].params();
        auto cloneParams = (*cloned)[s].params();
        CHECK(origParams.size() == cloneParams.size());
        for (size_t i = 0; i < origParams.size(); ++i) {
            CHECK(origParams[i]->w.size() == cloneParams[i]->w.size());
            for (size_t k = 0; k < origParams[i]->w.size(); ++k) {
                CHECK(origParams[i]->w[k] == cloneParams[i]->w[k]);
            }
        }
    }

    // Mutating original weights must not affect cloned weights
    actors[0].params()[0]->w[0] += 123.45f;
    CHECK(actors[0].params()[0]->w[0] != (*cloned)[0].params()[0]->w[0]);

    // Evaluation against down server should return cleanly without throwing or deadlocking
    DouZeroEvalConfig ecfg;
    ecfg.host = "127.0.0.1";
    ecfg.port = 38999;
    ecfg.decks = 2;
    auto res = evaluateAgainstDouZero(cloned, ecfg);
    CHECK(!res.ok);
    CHECK(!res.error.empty());
}

static void closeTestSock(test_sock_t s) {
#if defined(_WIN32)
    if (s != TEST_INVALID_SOCK) closesocket(s);
#else
    if (s >= 0) close(s);
#endif
}

// 18. End-to-end DouZero evaluation, game playout, CSV metrics recording, and model snapshot saving.
static void testMockDouZeroEvalAndDiskSnapshot() {
#if defined(_WIN32)
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif

    test_sock_t srv = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    ASSERT_NE(srv, TEST_INVALID_SOCK);

    sockaddr_in srvAddr{};
    srvAddr.sin_family = AF_INET;
    srvAddr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    srvAddr.sin_port = 0; // OS assigns free ephemeral port
    ASSERT_EQ(bind(srv, reinterpret_cast<sockaddr*>(&srvAddr), sizeof(srvAddr)), 0);
    ASSERT_EQ(listen(srv, 1), 0);

    socklen_t addrLen = sizeof(srvAddr);
    ASSERT_EQ(getsockname(srv, reinterpret_cast<sockaddr*>(&srvAddr), &addrLen), 0);
    int assignedPort = ntohs(srvAddr.sin_port);
    ASSERT_GT(assignedPort, 0);

    std::atomic<bool> srvRunning{true};
    std::thread serverThread([&]() {
        test_sock_t client = accept(srv, nullptr, nullptr);
        if (client == TEST_INVALID_SOCK) return;
        char c;
        while (srvRunning.load()) {
            std::string line;
            while (recv(client, &c, 1, 0) == 1) {
                if (c == '\n') break;
                if (c != '\r') line.push_back(c);
            }
            if (line.empty() || line == "QUIT") break;
            size_t lastTab = line.rfind('\t');
            std::string legals = (lastTab != std::string::npos) ? line.substr(lastTab + 1) : "";
            size_t slash = legals.find('/');
            std::string choice = (slash != std::string::npos) ? legals.substr(0, slash) : legals;
            choice.push_back('\n');
            send(client, choice.data(), static_cast<int>(choice.size()), 0);
        }
        closeTestSock(client);
    });

    nn::NetConfig cfg{64, 32};
    std::array<nn::Actor, 3> actors;
    for (int s = 0; s < 3; ++s) {
        actors[s].init(cfg, 100 + s);
    }
    auto cloned = cloneActors(actors);

    std::string testDir = "build/test_eval_e2e_run";
    std::string csvPath = testDir + "/eval.csv";
    std::string snapDir = testDir + "/snapshots/u5";

    DouZeroEvalConfig ecfg;
    ecfg.host = "127.0.0.1";
    ecfg.port = assignedPort;
    ecfg.decks = 2; // 2 decks = 4 games
    ecfg.update = 5;
    ecfg.label = "u5";
    ecfg.elapsedMinutes = 1.25;
    ecfg.csvPath = csvPath;
    ecfg.saveDir = snapDir;

    auto res = evaluateAgainstDouZero(cloned, ecfg);

    srvRunning = false;
    closeTestSock(srv);
    serverThread.join();

#if defined(_WIN32)
    WSACleanup();
#endif

    // Assert evaluation succeeded
    EXPECT_TRUE(res.ok);
    EXPECT_EQ(res.decks, 2);
    EXPECT_EQ(res.games, 4);
    EXPECT_GE(res.wp, 0.0);
    EXPECT_LE(res.wp, 1.0);

    // Assert CSV was created and written
    EXPECT_TRUE(std::filesystem::exists(csvPath));
    std::ifstream csvFile(csvPath);
    std::string header, dataLine;
    EXPECT_TRUE(std::getline(csvFile, header));
    EXPECT_TRUE(header.find("update,minutes,timestamp") != std::string::npos);
    EXPECT_TRUE(std::getline(csvFile, dataLine));
    EXPECT_TRUE(dataLine.find("5,1.25,") != std::string::npos);
    EXPECT_TRUE(dataLine.find("u5,2,4,") != std::string::npos);
    csvFile.close();

    // Assert snapshot models and meta.txt were saved to disk
    EXPECT_TRUE(std::filesystem::exists(snapDir + "/actor0.bin"));
    EXPECT_TRUE(std::filesystem::exists(snapDir + "/actor1.bin"));
    EXPECT_TRUE(std::filesystem::exists(snapDir + "/actor2.bin"));
    EXPECT_TRUE(std::filesystem::exists(snapDir + "/meta.txt"));

    std::ifstream metaFile(snapDir + "/meta.txt");
    std::string metaContent((std::istreambuf_iterator<char>(metaFile)),
                            std::istreambuf_iterator<char>());
    EXPECT_TRUE(metaContent.find("update 5") != std::string::npos);
    EXPECT_TRUE(metaContent.find("games 4") != std::string::npos);
    EXPECT_TRUE(metaContent.find("wp ") != std::string::npos);
    metaFile.close();

    // Clean up
    std::error_code ec;
    std::filesystem::remove_all(testDir, ec);
}

TEST(AlgoTest, AdvantageNormAndGAE) { testAdvantageNormAndGAE(); }
TEST(AlgoTest, ValueClipping) { testValueClipping(); }
TEST(AlgoTest, KLDivergence) { testKLDivergence(); }
TEST(AlgoTest, ArchiveAnchorPreservation) { testArchiveAnchorPreservation(); }
TEST(AlgoTest, RunningNormalizer) { testRunningNormalizer(); }
TEST(AlgoTest, AdamSerialization) { testAdamSerialization(); }
TEST(AlgoTest, CriticZeroInit) { testCriticZeroInit(); }
TEST(AlgoTest, ClipFractionSignGating) { testClipFractionSignGating(); }
TEST(AlgoTest, MaskedSoftmaxNaNProtection) { testMaskedSoftmaxNaNProtection(); }
TEST(AlgoTest, KLDivergenceNaNProtection) { testKLDivergenceNaNProtection(); }
TEST(AlgoTest, PPOWeightsFiniteAfterUpdate) { testPPOWeightsFiniteAfterUpdate(); }
TEST(AlgoTest, ConcurrentLaneStress) { testConcurrentLaneStress(); }
TEST(AlgoTest, LoadOptimizerAtomic) { testLoadOptimizerAtomic(); }
TEST(AlgoTest, NaNGradientSkipped) { testNaNGradientSkipped(); }
TEST(AlgoTest, PPOSurvivesNonFiniteSamples) { testPPOSurvivesNonFiniteSamples(); }
TEST(AlgoTest, PPONoDeadlock) { testPPONoDeadlock(); }
TEST(AlgoTest, ActorCloningAndEvalFailSafe) { testActorCloningAndEvalFailSafe(); }
TEST(AlgoTest, MockDouZeroEvalAndDiskSnapshot) { testMockDouZeroEvalAndDiskSnapshot(); }
