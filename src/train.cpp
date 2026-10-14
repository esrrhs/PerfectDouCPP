// PerfectDou CPP — self-play PPO training entry point (LibTorch Async Buffer Architecture).
//
// Asynchronous Replay Buffer architecture:
//   * CPU Rollout workers stream episodes into a high-capacity ring replay buffer
//   * GPU PPOLearner continuously consumes mini-batches from the buffer without blocking
//   * Non-blocking snapshot mechanism updates worker actors periodically (controlled policy lag)
//   * Completely decouples CPU simulation from GPU tensor training
//
// Example:
//   perfectdou_train --updates 200 --games 256 --threads 8 --out ckpt
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <powrprof.h>
#endif

#include "algo/eval_douzero.h"
#include "algo/ppo.h"
#include "algo/rollout.h"
#include "ddz/oracle.h"
#include "torch_nn/models.h"
#include "torch_nn/ppo_learner.h"
#include "torch_nn/tensor_bridge.h"

namespace {

struct Args {
    int updates = 100000;
    int startUpdate = 1;
    int games = 768;
    int threads = std::clamp(static_cast<int>(std::thread::hardware_concurrency()), 1, 16);
    int hidden = 256;
    int lstmHidden = 128;
    int epochs = 2;
    int minibatch = 2048;
    int snapshotEvery = 50;
    float lr = 3e-4f;
    float clip = 0.2f;
    float ent = 0.03f;
    float gamma = 1.0f;
    float lambda = 0.95f;
    float targetKL = 0.03f;
    bool oracleReward = true;      // Paper Sec 4.4 Oracle step difference reward
    float oracleCoef = 0.2f;
    float landlordWinMul = 2.0f;
    float landlordLossMul = 2.0f;
    int poolSize = 16;
    int archiveSize = 16;
    int poolEvery = 20;
    float histProb = 0.2f;
    uint64_t seed = 1;
    std::string out = "ckpt_long";
    std::string resume;
    std::string backend = "cpu";   // cuda | mps | cpu (explicit, no auto-detection)
    int evalEvery = 200;           // eval vs DouZero every 200 updates
    int evalDecks = 1000;          // 1000 decks = 2000 games (swapped seats)
    int evalPort = 18765;          // DouZero TCP server port
    std::string evalHost = "127.0.0.1";
    std::string evalCsv;           // empty = default to <out>/eval_vs_douzero.csv
    std::string evalSaveDir;       // empty = default to <out>/eval_snapshots
    std::string evalData = "data/eval_data.bin"; // path to eval_data.bin or eval_data.pkl
    bool evalSave = true;          // whether to save evaluated models to disk
    int evalMaxSnapshots = 0;      // 0 = keep all, > 0 = keep at most N recent snapshots on disk
    bool evalAsync = true;         // evaluate in background thread
};

const char* argValue(int argc, char** argv, const char* key, const char* def) {
    std::string k = key;
    for (int i = 1; i + 1 < argc; ++i)
        if (k == argv[i]) return argv[i + 1];
    return def;
}
bool hasFlag(int argc, char** argv, const char* key) {
    for (int i = 1; i < argc; ++i)
        if (key == std::string(argv[i])) return true;
    return false;
}

void parseArgs(int argc, char** argv, Args& a) {
    a.updates = std::atoi(argValue(argc, argv, "--updates", std::to_string(a.updates).c_str()));
    a.startUpdate = std::atoi(argValue(argc, argv, "--start-update", "1"));
    a.games = std::atoi(argValue(argc, argv, "--games", std::to_string(a.games).c_str()));
    a.threads = std::atoi(argValue(argc, argv, "--threads",
                                  std::to_string(a.threads).c_str()));
    a.hidden = std::atoi(argValue(argc, argv, "--hidden", std::to_string(a.hidden).c_str()));
    a.lstmHidden = std::atoi(argValue(argc, argv, "--lstm-hidden", std::to_string(a.lstmHidden).c_str()));
    a.epochs = std::atoi(argValue(argc, argv, "--epochs", std::to_string(a.epochs).c_str()));
    a.minibatch = std::atoi(argValue(argc, argv, "--mb", std::to_string(a.minibatch).c_str()));
    a.snapshotEvery = std::atoi(argValue(argc, argv, "--snapshot-every", std::to_string(a.snapshotEvery).c_str()));
    a.lr = float(std::atof(argValue(argc, argv, "--lr", "3e-4")));
    a.clip = float(std::atof(argValue(argc, argv, "--clip", "0.2")));
    a.ent = float(std::atof(argValue(argc, argv, "--ent", "0.03")));
    a.gamma = float(std::atof(argValue(argc, argv, "--gamma", "1.0")));
    a.lambda = float(std::atof(argValue(argc, argv, "--lambda", "0.95")));
    a.targetKL = float(std::atof(argValue(argc, argv, "--target-kl", "0.03")));
    if (hasFlag(argc, argv, "--oracle-reward")) a.oracleReward = true;
    if (hasFlag(argc, argv, "--no-oracle-reward")) a.oracleReward = false;
    a.oracleCoef = float(std::atof(argValue(argc, argv, "--oracle-coef", "0.2")));
    a.landlordWinMul = float(std::atof(argValue(argc, argv, "--landlord-win-mul", "2.0")));
    a.landlordLossMul = float(std::atof(argValue(argc, argv, "--landlord-loss-mul", "2.0")));
    a.poolSize = std::atoi(argValue(argc, argv, "--pool-size", "16"));
    a.archiveSize = std::atoi(argValue(argc, argv, "--archive-size", "16"));
    a.poolEvery = std::atoi(argValue(argc, argv, "--pool-every", "20"));
    a.histProb = float(std::atof(argValue(argc, argv, "--historical-prob", "0.2")));
    a.seed = std::atoll(argValue(argc, argv, "--seed", "1"));
    a.out = argValue(argc, argv, "--out", a.out.c_str());
    a.backend = argValue(argc, argv, "--backend", a.backend.c_str());
    const char* res = argValue(argc, argv, "--resume", "");
    if (*res) a.resume = res;
    a.evalEvery = std::atoi(argValue(argc, argv, "--eval-every", std::to_string(a.evalEvery).c_str()));
    if (hasFlag(argc, argv, "--no-eval")) {
        a.evalEvery = 0;
    }
    a.evalDecks = std::atoi(argValue(argc, argv, "--eval-decks", std::to_string(a.evalDecks).c_str()));
    a.evalPort = std::atoi(argValue(argc, argv, "--eval-port", "18765"));
    a.evalHost = argValue(argc, argv, "--eval-host", "127.0.0.1");
    a.evalCsv = argValue(argc, argv, "--eval-csv", "");
    a.evalSaveDir = argValue(argc, argv, "--eval-save-dir", "");
    a.evalData = argValue(argc, argv, "--eval-data", a.evalData.c_str());
    a.evalMaxSnapshots = std::atoi(argValue(argc, argv, "--eval-max-snapshots", "0"));
    if (hasFlag(argc, argv, "--no-eval-save")) {
        a.evalSave = false;
    }
    if (hasFlag(argc, argv, "--no-eval-async") || hasFlag(argc, argv, "--eval-sync")) {
        a.evalAsync = false;
    }
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    parseArgs(argc, argv, args);
    if (args.threads <= 0) args.threads = 1;

    torch::Device torchDevice = torch::kCPU;
    std::string deviceName = "CPU";
    if (args.backend == "cuda") {
        if (!torch::cuda::is_available()) {
            std::fprintf(stderr,
                         "Backend 'cuda' requested but CUDA is not available in this LibTorch build. "
                         "Pass --backend mps or --backend cpu explicitly.\n");
            return 1;
        }
        torchDevice = torch::kCUDA;
        deviceName = "GPU (CUDA)";
    } else if (args.backend == "mps") {
        if (!torch::mps::is_available()) {
            std::fprintf(stderr,
                         "Backend 'mps' requested but MPS is not available (requires macOS 12.3+ on "
                         "Apple Silicon with an MPS-enabled LibTorch). Pass --backend cpu explicitly.\n");
            return 1;
        }
        torchDevice = torch::kMPS;
        deviceName = "GPU (MPS)";
    } else if (args.backend == "cpu") {
        torchDevice = torch::kCPU;
        deviceName = "CPU";
    } else {
        std::fprintf(stderr,
                     "Unknown or unsupported backend '%s'. Valid values: cuda | mps | cpu "
                     "(no auto-detection).\n",
                     args.backend.c_str());
        return 1;
    }

    std::cout << "PerfectDou CPP training (Synchronous On-Policy Architecture)\n";
    std::cout << "  Compute Device: " << deviceName << "\n";
    if (!args.resume.empty() && !hasFlag(argc, argv, "--start-update")) {
        std::string metaPath = args.resume + "/meta.txt";
        std::ifstream metaFile(metaPath);
        if (metaFile.is_open()) {
            std::string key;
            int val = 0;
            while (metaFile >> key >> val) {
                if (key == "update" || key == "updates") {
                    args.startUpdate = val + 1;
                }
            }
        }
    }

    std::cout << "  updates=" << args.updates
              << (args.startUpdate > 1 ? " (start=" + std::to_string(args.startUpdate) + ")" : "")
              << " games/chunk=" << args.games
              << " threads=" << args.threads
              << " mb=" << args.minibatch
              << " epochs=" << args.epochs
              << " hidden=" << args.hidden
              << " lstm=" << args.lstmHidden
              << " lr=" << args.lr << " (fixed)"
              << " clip=" << args.clip
              << " ent=" << args.ent << " (fixed)"
              << " gae(gamma=" << args.gamma << ",lambda=" << args.lambda << ")"
              << " oracle=" << (args.oracleReward ? ("yes(coef=" + std::to_string(args.oracleCoef) + ")") : "no")
              << " targetKL=" << args.targetKL
              << " league(recent=" << args.poolSize << ",archive=" << args.archiveSize
              << ",every=" << args.poolEvery
              << ",hist=" << args.histProb << ")"
              << " target=terminal-adp"
              << " seed=" << args.seed << std::endl;

    if (args.evalEvery > 0) {
        if (args.evalCsv.empty()) {
            args.evalCsv = args.out + "/eval_vs_douzero.csv";
        }
        if (args.evalSave && args.evalSaveDir.empty()) {
            args.evalSaveDir = args.out + "/eval_snapshots";
        }
        std::cout << "  eval vs DouZero: every=" << args.evalEvery
                  << " decks=" << args.evalDecks << " (" << (args.evalDecks * 2) << " games)"
                  << " target=" << args.evalHost << ":" << args.evalPort
                  << " mode=" << (args.evalAsync ? "async" : "sync")
                  << " data=" << (args.evalData.empty() ? "random" : args.evalData)
                  << " csv=" << args.evalCsv;
        if (args.evalSave) {
            std::cout << " snapshots=" << args.evalSaveDir;
            if (args.evalMaxSnapshots > 0) {
                std::cout << " (keep max " << args.evalMaxSnapshots << ")";
            }
        }
        std::cout << std::endl;
    }
    std::thread evalThread;

    torch_nn::ModelConfig torchModelCfg;
    torchModelCfg.hidden = args.hidden;
    torchModelCfg.lstmHidden = args.lstmHidden;

    torch_nn::PPOConfig torchPpoCfg;
    torchPpoCfg.lr = args.lr;
    torchPpoCfg.clip = args.clip;
    torchPpoCfg.entCoef = args.ent;
    torchPpoCfg.epochs = args.epochs;
    torchPpoCfg.minibatch = args.minibatch;
    torchPpoCfg.targetKL = args.targetKL;
    torchPpoCfg.criticMaxGradNorm = 5.0f;

    std::array<torch_nn::Actor, 3> actor{
        torch_nn::Actor(torchModelCfg),
        torch_nn::Actor(torchModelCfg),
        torch_nn::Actor(torchModelCfg)
    };
    std::array<torch_nn::Critic, 3> critic{
        torch_nn::Critic(torchModelCfg),
        torch_nn::Critic(torchModelCfg),
        torch_nn::Critic(torchModelCfg)
    };

    // Dedicated CPU model instances for multithreaded CPU rollout simulation
    std::array<torch_nn::Actor, 3> cpuActor{
        torch_nn::Actor(torchModelCfg),
        torch_nn::Actor(torchModelCfg),
        torch_nn::Actor(torchModelCfg)
    };
    std::array<torch_nn::Critic, 3> cpuCritic{
        torch_nn::Critic(torchModelCfg),
        torch_nn::Critic(torchModelCfg),
        torch_nn::Critic(torchModelCfg)
    };

    auto syncWeightsToCpu = [&]() {
        torch::NoGradGuard no_grad;
        for (int s = 0; s < 3; ++s) {
            auto sa = actor[s]->parameters();
            auto da = cpuActor[s]->parameters();
            for (size_t i = 0; i < sa.size(); ++i) da[i].copy_(sa[i]);
            auto sc = critic[s]->parameters();
            auto dc = cpuCritic[s]->parameters();
            for (size_t i = 0; i < sc.size(); ++i) dc[i].copy_(sc[i]);
            cpuActor[s]->eval();
            cpuCritic[s]->eval();
        }
    };

    if (!args.resume.empty()) {
        for (int s = 0; s < 3; ++s) {
            std::string actPath = args.resume + "/actor" + std::to_string(s) + ".pt";
            std::string critPath = args.resume + "/critic" + std::to_string(s) + ".pt";
            if (std::filesystem::exists(actPath)) {
                actor[s]->load(actPath);
            }
            if (std::filesystem::exists(critPath)) {
                critic[s]->load(critPath);
            }
        }
    }

    std::vector<std::unique_ptr<torch_nn::PPOLearner>> learners;
    for (int s = 0; s < 3; ++s) {
        learners.push_back(std::make_unique<torch_nn::PPOLearner>(
            actor[s], critic[s], torchPpoCfg, torchDevice));
        if (!args.resume.empty()) {
            std::string optPath = args.resume + "/opt" + std::to_string(s) + ".pt";
            if (std::filesystem::exists(optPath)) {
                learners[s]->loadOptimizer(optPath);
            }
        }
    }

    syncWeightsToCpu();

    std::shared_ptr<algo::HistoricalPool> histPool = std::make_shared<algo::HistoricalPool>();

    auto pushHistPool = [&](std::array<torch_nn::Actor, 3>& srcActor, int currentUpdate) {
        auto snap = std::make_shared<algo::HistoricalActorSnapshot>();
        snap->update = currentUpdate;
        for (int s = 0; s < 3; ++s) {
            snap->actor[s] = torch_nn::Actor(torchModelCfg);
            torch::NoGradGuard no_grad;
            auto srcParams = srcActor[s]->parameters();
            auto dstParams = snap->actor[s]->parameters();
            for (size_t i = 0; i < srcParams.size(); ++i) {
                dstParams[i].copy_(srcParams[i]);
            }
            snap->actor[s]->eval();
        }
        int k = args.poolEvery > 0 ? (currentUpdate / args.poolEvery) : 0;
        bool isPowerOfTwo = (k > 0) && ((k & (k - 1)) == 0);
        bool isCentennial = (currentUpdate % 100 == 0);
        if (args.archiveSize > 0 && (isPowerOfTwo || isCentennial || histPool->archive.empty())) {
            while ((int)histPool->archive.size() >= args.archiveSize && !histPool->archive.empty()) {
                if (histPool->archive.size() > 2) {
                    histPool->archive.erase(histPool->archive.begin() + 1);
                } else {
                    histPool->archive.erase(histPool->archive.begin());
                }
            }
            histPool->archive.push_back(snap);
        }

        if (args.poolSize > 0) {
            while ((int)histPool->recent.size() >= args.poolSize && !histPool->recent.empty()) {
                histPool->recent.erase(histPool->recent.begin());
            }
            histPool->recent.push_back(std::move(snap));
        }
    };

    auto saveHistoricalPool = [&](const std::string& dir) {
        if (!histPool || histPool->empty()) return;
        std::string poolDir = dir + "/hist_pool";
        std::filesystem::create_directories(poolDir);
        std::ofstream meta(poolDir + "/pool_meta.txt");
        if (!meta.is_open()) return;

        meta << "recent_count " << histPool->recent.size() << "\n";
        for (size_t i = 0; i < histPool->recent.size(); ++i) {
            const auto& snap = histPool->recent[i];
            meta << "recent " << i << " " << snap->update << "\n";
            for (int s = 0; s < 3; ++s) {
                snap->actor[s]->save(poolDir + "/recent_" + std::to_string(i) + "_actor" + std::to_string(s) + ".pt");
            }
        }

        meta << "archive_count " << histPool->archive.size() << "\n";
        for (size_t i = 0; i < histPool->archive.size(); ++i) {
            const auto& snap = histPool->archive[i];
            meta << "archive " << i << " " << snap->update << "\n";
            for (int s = 0; s < 3; ++s) {
                snap->actor[s]->save(poolDir + "/archive_" + std::to_string(i) + "_actor" + std::to_string(s) + ".pt");
            }
        }
    };

    auto loadHistoricalPool = [&](const std::string& dir) {
        std::string metaPath = dir + "/hist_pool/pool_meta.txt";
        std::ifstream meta(metaPath);
        if (!meta.is_open()) return;

        std::string tag;
        while (meta >> tag) {
            if (tag == "recent") {
                size_t idx = 0;
                int snapUpd = 0;
                meta >> idx >> snapUpd;
                auto snap = std::make_shared<algo::HistoricalActorSnapshot>();
                snap->update = snapUpd;
                for (int s = 0; s < 3; ++s) {
                    snap->actor[s] = torch_nn::Actor(torchModelCfg);
                    std::string actPath = dir + "/hist_pool/recent_" + std::to_string(idx) + "_actor" + std::to_string(s) + ".pt";
                    if (std::filesystem::exists(actPath)) {
                        snap->actor[s]->load(actPath);
                    }
                    snap->actor[s]->eval();
                }
                histPool->recent.push_back(snap);
            } else if (tag == "archive") {
                size_t idx = 0;
                int snapUpd = 0;
                meta >> idx >> snapUpd;
                auto snap = std::make_shared<algo::HistoricalActorSnapshot>();
                snap->update = snapUpd;
                for (int s = 0; s < 3; ++s) {
                    snap->actor[s] = torch_nn::Actor(torchModelCfg);
                    std::string actPath = dir + "/hist_pool/archive_" + std::to_string(idx) + "_actor" + std::to_string(s) + ".pt";
                    if (std::filesystem::exists(actPath)) {
                        snap->actor[s]->load(actPath);
                    }
                    snap->actor[s]->eval();
                }
                histPool->archive.push_back(snap);
            } else {
                int dummy = 0;
                meta >> dummy;
            }
        }
        std::cout << "  Restored historical pool: recent=" << histPool->recent.size()
                  << " archive=" << histPool->archive.size() << "\n";
    };

    if (!args.resume.empty()) {
        loadHistoricalPool(args.resume);
    }

    auto saveAll = [&](const std::string& dir, int currentUpd = 0) {
        std::filesystem::create_directories(dir);
        for (int s = 0; s < 3; ++s) {
            actor[s]->save(dir + "/actor" + std::to_string(s) + ".pt");
            critic[s]->save(dir + "/critic" + std::to_string(s) + ".pt");
            learners[s]->saveOptimizer(dir + "/opt" + std::to_string(s) + ".pt");
        }
        if (currentUpd > 0) {
            std::ofstream metaFile(dir + "/meta.txt");
            if (metaFile.is_open()) {
                metaFile << "update " << currentUpd << "\n"
                         << "seed " << args.seed << "\n"
                         << "lr " << args.lr << "\n"
                         << "ent " << args.ent << "\n"
                         << "oracle_reward " << (args.oracleReward ? 1 : 0) << "\n"
                         << "oracle_coef " << args.oracleCoef << "\n";
            }
            saveHistoricalPool(dir);
        }
    };

    // Ensure output directory exists from update 0
    std::filesystem::create_directories(args.out);
    saveAll(args.out, 0);

    // warm up the shared (read-only) oracle DP table before spawning workers
    ddz::minSteps(ddz::CardSet::parse("34567"));

#if defined(_WIN32)
    REASON_CONTEXT awakeReason = {};
    awakeReason.Version = POWER_REQUEST_CONTEXT_VERSION;
    awakeReason.Flags = POWER_REQUEST_CONTEXT_SIMPLE_STRING;
    awakeReason.Reason.SimpleReasonString =
        const_cast<wchar_t*>(L"PerfectDou training");
    HANDLE awake = PowerCreateRequest(&awakeReason);
    if (awake) {
        PowerSetRequest(awake, PowerRequestSystemRequired);
        PowerSetRequest(awake, PowerRequestExecutionRequired);
    }
    SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED |
                            ES_AWAYMODE_REQUIRED);
#endif

    std::array<algo::PPOStats, 3> ps{};

    for (int upd = args.startUpdate; upd <= args.updates; ++upd) {
        auto wall0 = std::chrono::steady_clock::now();

        // 1. Synchronous CPU self-play rollout with current models and league pool
        algo::ModelSet models{{&cpuActor[0], &cpuActor[1], &cpuActor[2]},
                              {&cpuCritic[0], &cpuCritic[1], &cpuCritic[2]}};
        algo::RolloutConfig rc;
        rc.gamesPerUpdate = args.games;
        rc.threads = args.threads;
        rc.seed = args.seed + uint64_t(upd) * 7919ULL;
        rc.historicalProb = args.histProb;
        rc.historicalPool = histPool.get();
        rc.oracleReward = args.oracleReward;
        rc.oracleCoef = args.oracleCoef;
        rc.landlordWinMul = args.landlordWinMul;
        rc.landlordLossMul = args.landlordLossMul;

        std::array<std::vector<algo::Transition>, 3> streams;
        algo::RolloutStats rs;
        algo::collectRollout(models, rc, streams, rs);

        // 2. Synchronous GPU training updates
        std::array<size_t, 3> actualSamples{};
        auto tLearn0 = std::chrono::steady_clock::now();
        for (int s = 0; s < 3; ++s) {
            const auto& seatSamples = streams[s];
            actualSamples[s] = seatSamples.size();
            if (seatSamples.empty()) continue;

            try {
                auto batchTensors = torch_nn::transitionsToBatch(
                    seatSamples, args.gamma, args.lambda, torchDevice);
                learners[s]->setLr(args.lr);
                learners[s]->setEntCoef(args.ent);

                auto torchStats = learners[s]->update(batchTensors);
                ps[s].epochsCompleted = torchStats.epochsCompleted;
                ps[s].entropy = torchStats.entropy;
                ps[s].lastEpochKL = torchStats.kl;
                ps[s].clipFraction = torchStats.clipFraction;
                ps[s].pgLoss = torchStats.pgLoss;
                ps[s].vLoss = torchStats.vLoss;
            } catch (const std::exception& e) {
                std::fprintf(stderr, "Seat %d update error: %s\n", s, e.what());
            }
        }
        double learnSecs = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - tLearn0)
                               .count();

        // 3. Sync freshly trained weights from GPU/learner back to CPU models for rollout
        syncWeightsToCpu();

        double wallSecs = std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - wall0)
                              .count();

        double wp = double(rs.landlordWins) / std::max(1, rs.games);
        double adp = double(rs.landlordScore) / std::max(1, rs.games);
        double bpg = double(rs.bombs) / std::max(1, rs.games);
        double mpg = double(rs.moves) / std::max(1, rs.games);
        std::string pureStr;
        if (rs.pureSelfPlayGames > 0) {
            char pbuf[64];
            std::snprintf(pbuf, sizeof(pbuf), " (pure WP %.3f ADP %7.2f [%d g])",
                          double(rs.pureSelfPlayWins) / rs.pureSelfPlayGames,
                          double(rs.pureSelfPlayScore) / rs.pureSelfPlayGames,
                          rs.pureSelfPlayGames);
            pureStr = pbuf;
        }

        std::printf(
            "upd %4d | wall %.1fs learn %.2fs "
            "games %d mb %d ep %d,%d,%d/%d | WP %.3f ADP %7.2f%s bomb/g %.2f "
            "moves/g %.1f | lr %.2e ent %5.3f/%5.3f/%5.3f vL %7.2f/%7.2f/%7.2f "
            "kl %.4f/%.4f/%.4f cf %.2f/%.2f/%.2f | train_n %zu/%zu/%zu\n",
            upd, wallSecs, learnSecs,
            rs.games, args.minibatch,
            ps[0].epochsCompleted, ps[1].epochsCompleted, ps[2].epochsCompleted,
            args.epochs, wp, adp, pureStr.c_str(), bpg, mpg,
            args.lr, ps[0].entropy, ps[1].entropy, ps[2].entropy,
            ps[0].vLoss, ps[1].vLoss, ps[2].vLoss,
            ps[0].lastEpochKL, ps[1].lastEpochKL, ps[2].lastEpochKL,
            ps[0].clipFraction, ps[1].clipFraction, ps[2].clipFraction,
            actualSamples[0], actualSamples[1], actualSamples[2]);
        std::fflush(stdout);

        if (args.poolEvery > 0 && upd % args.poolEvery == 0) {
            pushHistPool(cpuActor, upd);
        }
        if (args.snapshotEvery > 0 &&
            (upd % args.snapshotEvery == 0 || upd == args.updates)) {
            saveAll(args.out, upd);
        }

        if (args.evalEvery > 0 &&
            (upd % args.evalEvery == 0 || upd == args.updates)) {
            if (evalThread.joinable()) {
                evalThread.join();
            }
            auto cloned = algo::cloneActors(cpuActor);
            algo::DouZeroEvalConfig evalCfg;
            evalCfg.host = args.evalHost;
            evalCfg.port = args.evalPort;
            evalCfg.decks = args.evalDecks;
            evalCfg.seed = static_cast<uint64_t>(upd * 10007 + 1);
            evalCfg.csvPath = args.evalCsv;
            evalCfg.dataPath = args.evalData;
            evalCfg.label = "u" + std::to_string(upd);
            evalCfg.saveDir = args.evalSave ? (args.evalSaveDir + "/" + evalCfg.label) : "";
            evalCfg.maxSnapshots = args.evalMaxSnapshots;
            evalCfg.update = upd;
            evalCfg.elapsedMinutes = wallSecs / 60.0;
            evalCfg.verbose = false;

            auto runEval = [cloned = std::move(cloned), evalCfg]() {
                auto res = algo::evaluateAgainstDouZero(cloned, evalCfg);
                if (res.ok) {
                    std::printf("[eval %s] done games %d | WP %.3f ADP %.3f | landlord WP %.3f ADP %.3f | peasant WP %.3f ADP %.3f\n",
                                evalCfg.label.c_str(), res.games, res.wp, res.adp,
                                res.wpLandlord, res.adpLandlord, res.wpPeasant, res.adpPeasant);
                } else {
                    std::printf("[eval %s] skipped: %s\n", evalCfg.label.c_str(), res.error.c_str());
                }
                std::fflush(stdout);
            };

            if (args.evalAsync) {
                evalThread = std::thread(std::move(runEval));
            } else {
                runEval();
            }
        }
    }

    if (evalThread.joinable()) evalThread.join();

#if defined(_WIN32)
    if (awake) {
        PowerClearRequest(awake, PowerRequestExecutionRequired);
        PowerClearRequest(awake, PowerRequestSystemRequired);
        CloseHandle(awake);
    }
    SetThreadExecutionState(ES_CONTINUOUS);
#endif

    saveAll(args.out, args.updates);
    std::cout << "models saved to " << args.out << "/\n";
    return 0;
}
