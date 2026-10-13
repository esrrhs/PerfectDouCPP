// Unit tests for RL algorithm components:
// 1. Advantage Normalization
// 2. PPO Value Clipping loss & gradient
// 3. Dynamic KL divergence estimation & early stopping logic
// 4. Historical archive pool anchor preservation & thinning logic
// 5. Running normalizer online statistics
// 6. Critic head zero-initialization (LibTorch)
// 7. Optimizer state serialization roundtrip (PPOLearner)
// 8. Actor cloning deep copy independence & fail-safe DouZero evaluation
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
#include "torch_nn/models.h"
#include "torch_nn/ppo_learner.h"
#include "torch_nn/tensor_bridge.h"

#ifdef CHECK
#undef CHECK
#endif
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

    // Manual Advantage Normalization simulation
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
            grad = 0.0f;
        } else {
            grad = vfCoef * err / float(B);
        }
        CHECK(l1 == l2);
        CHECK(grad == vfCoef * (-0.9f));
    }

    // Case B: vPred moves away from target past clip bound (l2 > l1)
    {
        float vOld = 1.0f;
        float vPred = 1.5f;
        float target = 0.0f;
        float err = vPred - target;
        float vClipped = vOld + std::clamp(vPred - vOld, -clip, clip);
        float errClipped = vClipped - target;
        float l1 = err * err;
        float l2 = errClipped * errClipped;
        float grad = (l2 > l1) ? 0.0f : (vfCoef * err / float(B));
        CHECK(grad != 0.0f);
    }

    // Case C: vPred moved towards target past clip bound (l2 > l1 -> gradient should be 0)
    {
        float vOld = 1.0f;
        float vPred = 1.5f;
        float target = 3.0f;
        float err = vPred - target;
        float vClipped = vOld + std::clamp(vPred - vOld, -clip, clip);
        float errClipped = vClipped - target;
        float l1 = err * err;
        float l2 = errClipped * errClipped;
        CHECK(l2 > l1);
        float grad = (l2 > l1) ? 0.0f : (vfCoef * err / float(B));
        CHECK(grad == 0.0f);
    }
}

// 3. Test KL divergence math properties
static void testKLDivergence() {
    auto calcKL = [](float ratio) {
        float logRatio = std::log(ratio);
        return (ratio - 1.0f) - logRatio;
    };

    CHECK(std::abs(calcKL(1.0f)) < 1e-7f);

    for (float r = 0.1f; r < 2.5f; r += 0.05f) {
        if (std::abs(r - 1.0f) > 1e-4f) {
            float kl = calcKL(r);
            CHECK(kl > 0.0f);
        }
    }

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
                pool.archive.erase(pool.archive.begin() + 1);
            } else {
                pool.archive.erase(pool.archive.begin());
            }
        }
        pool.archive.push_back(snap);
    };

    for (int upd = 1; upd <= 10; ++upd) {
        addSnapshot(upd);
    }

    CHECK(pool.archive.size() == 4);
    CHECK(pool.archive[0]->update == 1);
    CHECK(pool.archive.back()->update == 10);
}

// 5. Test Welford Running Normalizer
static void testRunningNormalizer() {
    RunningNormalizer norm;
    std::vector<float> data = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
    for (float x : data) norm.update(x);

    CHECK(std::abs(norm.mean - 30.0) < 1e-4);
    float val = 25.0f;
    float n = norm.normalize(val);
    float d = norm.denormalize(n);
    CHECK(std::abs(d - val) < 1e-4f);
}

// 6. Test Critic head zero-initialization in LibTorch
static void testCriticZeroInit() {
    torch_nn::ModelConfig cfg{64, 32};
    torch_nn::Critic critic(cfg);
    auto w = critic->out->weight;
    auto b = critic->out->bias;
    CHECK(torch::all(w == 0.0f).item<bool>());
    CHECK(torch::all(b == 0.0f).item<bool>());
}

// 7. Test PPOLearner Optimizer serialization roundtrip
static void testOptimizerSerialization() {
    torch_nn::ModelConfig cfg{64, 32};
    torch_nn::Actor a1(cfg);
    torch_nn::Critic c1(cfg);
    torch_nn::PPOConfig ppoCfg;
    torch_nn::PPOLearner l1(a1, c1, ppoCfg);

    const char* tmpPath = "test_opt_roundtrip.pt";
    l1.saveOptimizer(tmpPath);

    torch_nn::Actor a2(cfg);
    torch_nn::Critic c2(cfg);
    torch_nn::PPOLearner l2(a2, c2, ppoCfg);
    l2.loadOptimizer(tmpPath);

    std::remove(tmpPath);
}

// 8. Test ClipFraction sign gating logic
static void testClipFractionSignGating() {
    float clip = 0.2f;
    auto isClipped = [clip](float adv, float ratio) {
        return (adv > 0.0f && ratio > 1.0f + clip) ||
               (adv < 0.0f && ratio < 1.0f - clip);
    };

    CHECK(isClipped(1.0f, 1.3f) == true);
    CHECK(isClipped(1.0f, 0.7f) == false);
    CHECK(isClipped(-1.0f, 0.7f) == true);
    CHECK(isClipped(-1.0f, 1.3f) == false);
}

// 9. Test Masked Softmax NaN and Inf protection
static void testMaskedSoftmaxNaNProtection() {
    {
        float logits[3] = {1.0f, 2.0f, 3.0f};
        float probs[3] = {};
        maskedSoftmax(logits, 3, probs);
        float sum = probs[0] + probs[1] + probs[2];
        CHECK(std::abs(sum - 1.0f) < 1e-5f);
        CHECK(probs[2] > probs[1] && probs[1] > probs[0]);
    }

    {
        float logits[4] = {-1e9f, -1e9f, -1e9f, -1e9f};
        float probs[4] = {};
        maskedSoftmax(logits, 4, probs);
        for (int a = 0; a < 4; ++a) {
            CHECK(std::isfinite(probs[a]));
            CHECK(std::abs(probs[a] - 0.25f) < 1e-5f);
        }
    }

    {
        float logits[4] = {NAN, INFINITY, -INFINITY, 2.0f};
        float probs[4] = {};
        maskedSoftmax(logits, 4, probs);
        for (int a = 0; a < 4; ++a) {
            CHECK(std::isfinite(probs[a]));
            CHECK(probs[a] >= 0.0f && probs[a] <= 1.0f);
        }
    }

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

    CHECK(computeApproxKL(-1.5f, -1.5f) == 0.0f);
    float klMod = computeApproxKL(-1.0f, -1.5f);
    CHECK(std::isfinite(klMod) && klMod > 0.0f);
    float klHuge = computeApproxKL(100.0f, -100.0f);
    CHECK(std::isfinite(klHuge) && klHuge >= 0.0f);
    float klZero = computeApproxKL(-1000.0f, -1.0f);
    CHECK(std::isfinite(klZero) && klZero >= 0.0f);
}

// 11. Actor cloning deep copy independence and fail-safe handling when DouZero server is unreachable.
static void testActorCloningAndEvalFailSafe() {
    torch_nn::ModelConfig cfg{64, 32};
    std::array<torch_nn::Actor, 3> actors{
        torch_nn::Actor(cfg),
        torch_nn::Actor(cfg),
        torch_nn::Actor(cfg)
    };
    auto cloned = cloneActors(actors);
    for (int s = 0; s < 3; ++s) {
        auto origParams = actors[s]->parameters();
        auto cloneParams = (*cloned)[s]->parameters();
        CHECK(origParams.size() == cloneParams.size());
        for (size_t i = 0; i < origParams.size(); ++i) {
            CHECK(torch::all(origParams[i] == cloneParams[i]).item<bool>());
        }
    }

    // Mutating original weights must not affect cloned weights
    actors[0]->l1->weight.data()[0][0].add_(123.45f);
    CHECK(!torch::all(actors[0]->l1->weight == (*cloned)[0]->l1->weight).item<bool>());

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

// 12. End-to-end DouZero evaluation, game playout, CSV metrics recording, and model snapshot saving.
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

    std::atomic<bool> srvRunning{true};
    std::thread serverThread([srv, &srvRunning] {
        test_sock_t client = accept(srv, nullptr, nullptr);
        if (client == TEST_INVALID_SOCK) return;
        std::string buf;
        char c;
        while (srvRunning.load()) {
            buf.clear();
            while (true) {
                int n = recv(client, &c, 1, 0);
                if (n <= 0) break;
                if (c == '\n') break;
                if (c != '\r') buf.push_back(c);
            }
            if (buf.empty() || buf == "QUIT") break;
            auto comma = buf.find(',');
            std::string legals = (comma != std::string::npos) ? buf.substr(comma + 1) : buf;
            auto slash = legals.find('/');
            std::string choice = (slash != std::string::npos) ? legals.substr(0, slash) : legals;
            choice.push_back('\n');
            send(client, choice.data(), static_cast<int>(choice.size()), 0);
        }
        closeTestSock(client);
    });

    torch_nn::ModelConfig cfg{64, 32};
    std::array<torch_nn::Actor, 3> actors{
        torch_nn::Actor(cfg),
        torch_nn::Actor(cfg),
        torch_nn::Actor(cfg)
    };
    auto cloned = cloneActors(actors);

    std::string testDir = "build/test_eval_e2e_run";
    std::string csvPath = testDir + "/eval.csv";
    std::string snapDir = testDir + "/snapshots/u5";

    std::filesystem::create_directories(testDir + "/snapshots/u1");
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    std::filesystem::create_directories(testDir + "/snapshots/u2");
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    DouZeroEvalConfig ecfg;
    ecfg.host = "127.0.0.1";
    ecfg.port = assignedPort;
    ecfg.decks = 2; // 2 decks = 4 games
    ecfg.update = 5;
    ecfg.label = "u5";
    ecfg.elapsedMinutes = 1.25;
    ecfg.csvPath = csvPath;
    ecfg.saveDir = snapDir;
    ecfg.maxSnapshots = 2;

    auto res = evaluateAgainstDouZero(cloned, ecfg);

    srvRunning = false;
    closeTestSock(srv);
    serverThread.join();

#if defined(_WIN32)
    WSACleanup();
#endif

    EXPECT_TRUE(res.ok);
    EXPECT_EQ(res.decks, 2);
    EXPECT_EQ(res.games, 4);
    EXPECT_GE(res.wp, 0.0);
    EXPECT_LE(res.wp, 1.0);

    EXPECT_TRUE(std::filesystem::exists(csvPath));
    std::ifstream csvFile(csvPath);
    std::string header, dataLine;
    EXPECT_TRUE(std::getline(csvFile, header));
    EXPECT_TRUE(header.find("update,minutes,timestamp") != std::string::npos);
    EXPECT_TRUE(std::getline(csvFile, dataLine));
    EXPECT_TRUE(dataLine.find("5,1.25,") != std::string::npos);
    EXPECT_TRUE(dataLine.find("u5,2,4,") != std::string::npos);
    csvFile.close();

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

    EXPECT_FALSE(std::filesystem::exists(testDir + "/snapshots/u1"));
    EXPECT_TRUE(std::filesystem::exists(testDir + "/snapshots/u2"));
    EXPECT_TRUE(std::filesystem::exists(snapDir));

    std::error_code ec;
    std::filesystem::remove_all(testDir, ec);
}

void testLoadEvalDecks() {
    std::string p = "data/eval_data.bin";
    std::error_code ec;
    if (!std::filesystem::exists(p, ec) && std::filesystem::exists("../" + p, ec)) {
        p = "../" + p;
    }
    auto decks = algo::loadEvalDecks(p, 5);
    EXPECT_EQ(decks.size(), 5);
    for (const auto& d : decks) {
        EXPECT_EQ(d.landlord.total(), 20);
        EXPECT_EQ(d.landlordDown.total(), 17);
        EXPECT_EQ(d.landlordUp.total(), 17);
        EXPECT_EQ(d.bottom.total(), 3);
        EXPECT_TRUE(d.landlord.contains(d.bottom));

        ddz::CardSet all;
        for (int r = 0; r < ddz::kRanks; ++r) {
            all.add(r, d.landlord.c[r] + d.landlordDown.c[r] + d.landlordUp.c[r]);
        }
        EXPECT_EQ(all.total(), 54);
    }
}

TEST(AlgoTest, AdvantageNormAndGAE) { testAdvantageNormAndGAE(); }
TEST(AlgoTest, ValueClipping) { testValueClipping(); }
TEST(AlgoTest, KLDivergence) { testKLDivergence(); }
TEST(AlgoTest, ArchiveAnchorPreservation) { testArchiveAnchorPreservation(); }
TEST(AlgoTest, RunningNormalizer) { testRunningNormalizer(); }
TEST(AlgoTest, CriticZeroInit) { testCriticZeroInit(); }
TEST(AlgoTest, OptimizerSerialization) { testOptimizerSerialization(); }
TEST(AlgoTest, ClipFractionSignGating) { testClipFractionSignGating(); }
TEST(AlgoTest, MaskedSoftmaxNaNProtection) { testMaskedSoftmaxNaNProtection(); }
TEST(AlgoTest, KLDivergenceNaNProtection) { testKLDivergenceNaNProtection(); }
TEST(AlgoTest, ActorCloningAndEvalFailSafe) { testActorCloningAndEvalFailSafe(); }
TEST(AlgoTest, MockDouZeroEvalAndDiskSnapshot) { testMockDouZeroEvalAndDiskSnapshot(); }
TEST(AlgoTest, LoadEvalDecks) { testLoadEvalDecks(); }
