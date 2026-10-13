// PerfectDou CPP — self-play PPO training entry point (LibTorch Engine).
//
// Modern LibTorch-based PPO training:
//   * three seat models (landlord / landlord_down / landlord_up), self-play
//   * imperfect-information actor over concrete legal plays
//   * perfect-information critic (sees all hands, PTIE / perfect information distillation)
//   * PPO with GAE bootstrapped value targets and terminal ADP reward
//   * LibTorch C++ autograd, optimizers, and CUDA/CPU tensor acceleration
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
    int threads = 16;
    int hidden = 1024;
    int lstmHidden = 256;
    int epochs = 2;
    int minibatch = 2048;
    int snapshotEvery = 50;
    int buffer = 4;
    float lr = 3e-4f;
    float clip = 0.2f;
    float ent = 0.03f;
    float gamma = 1.0f;
    float lambda = 0.95f;
    float targetKL = 0.03f;
    bool clipVf = false;
    float vfClip = 32.0f;
    float shapingCap = 0.05f;
    float landlordWinMul = 2.5f;
    float landlordLossMul = 1.5f;
    bool independentPeasants = true;
    std::string campMode = "alt";  // "all" = train all together, "alt" = alternate landlord/peasant, "landlord" = landlord only, "peasant" = peasant only
    bool lrDecay = true;
    bool entDecay = true;
    int poolSize = 16;
    int archiveSize = 16;
    int poolEvery = 20;
    float histProb = 0.2f;
    float ruleProb = 0.1f;
    uint64_t seed = 1;
    std::string out = "ckpt_long";
    std::string resume;
    std::string backend = "auto";  // auto | cuda | cpu
    int evalEvery = 200;           // eval vs DouZero every 200 updates
    int evalDecks = 1000;          // 1000 decks = 2000 games (swapped seats)
    int evalPort = 18765;          // DouZero TCP server port
    std::string evalHost = "127.0.0.1";
    std::string evalCsv;           // empty = default to <out>/eval_vs_douzero.csv
    std::string evalSaveDir;       // empty = default to <out>/eval_snapshots
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
    a.buffer = std::atoi(argValue(argc, argv, "--buffer", std::to_string(a.buffer).c_str()));
    a.lr = float(std::atof(argValue(argc, argv, "--lr", "3e-4")));
    a.clip = float(std::atof(argValue(argc, argv, "--clip", "0.2")));
    a.ent = float(std::atof(argValue(argc, argv, "--ent", "0.03")));
    a.gamma = float(std::atof(argValue(argc, argv, "--gamma", "1.0")));
    a.lambda = float(std::atof(argValue(argc, argv, "--lambda", "0.95")));
    a.targetKL = float(std::atof(argValue(argc, argv, "--target-kl", "0.03")));
    a.vfClip = float(std::atof(argValue(argc, argv, "--vf-clip", "32.0")));
    a.shapingCap = float(std::atof(argValue(argc, argv, "--shaping-cap", "0.05")));
    a.landlordWinMul = float(std::atof(argValue(argc, argv, "--landlord-win-mul", "2.5")));
    a.landlordLossMul = float(std::atof(argValue(argc, argv, "--landlord-loss-mul", "1.5")));
    if (hasFlag(argc, argv, "--no-independent-peasants") || hasFlag(argc, argv, "--coupled-peasants")) {
        a.independentPeasants = false;
    }
    a.campMode = argValue(argc, argv, "--camp-mode", a.campMode.c_str());
    if (hasFlag(argc, argv, "--clip-vf")) a.clipVf = true;
    if (hasFlag(argc, argv, "--no-clip-vf")) a.clipVf = false;
    if (hasFlag(argc, argv, "--no-lr-decay")) a.lrDecay = false;
    if (hasFlag(argc, argv, "--no-ent-decay")) a.entDecay = false;
    a.poolSize = std::atoi(argValue(argc, argv, "--pool-size", "16"));
    a.archiveSize = std::atoi(argValue(argc, argv, "--archive-size", "16"));
    a.poolEvery = std::atoi(argValue(argc, argv, "--pool-every", "20"));
    a.histProb = float(std::atof(argValue(argc, argv, "--historical-prob", "0.2")));
    a.ruleProb = float(std::atof(argValue(argc, argv, "--rule-prob", "0.1")));
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
    if (args.backend == "cuda") {
        if (!torch::cuda::is_available()) {
            std::fprintf(stderr, "CUDA requested but not available in LibTorch! Falling back to CPU.\n");
        } else {
            torchDevice = torch::kCUDA;
        }
    } else if (args.backend == "auto") {
        if (torch::cuda::is_available()) {
            torchDevice = torch::kCUDA;
        }
    }

    std::cout << "PerfectDou CPP training (LibTorch Architecture)\n";
    std::cout << "  Compute Device: " << (torchDevice == torch::kCUDA ? "GPU (CUDA)" : "CPU") << "\n";
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
              << " games/update=" << args.games
              << " buffer=" << args.buffer
              << " threads=" << args.threads
              << " mb=" << args.minibatch
              << " epochs=" << args.epochs
              << " hidden=" << args.hidden
              << " lstm=" << args.lstmHidden
              << " lr=" << args.lr << (args.lrDecay ? " (cosine)" : " (fixed)")
              << " clip=" << args.clip
              << " ent=" << args.ent << (args.entDecay ? " (cosine)" : " (fixed)")
              << " gae(gamma=" << args.gamma << ",lambda=" << args.lambda << ")"
              << " targetKL=" << args.targetKL << (args.clipVf ? " clipVf" : "")
              << " league(recent=" << args.poolSize << ",archive=" << args.archiveSize
              << ",every=" << args.poolEvery
              << ",hist=" << args.histProb << ",rule=" << args.ruleProb
              << (args.independentPeasants ? ",indepPeasants" : ",coupledPeasants") << ")"
              << " campMode=" << args.campMode
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
    torchPpoCfg.clipVf = args.clipVf;
    torchPpoCfg.vfClip = args.vfClip;

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
                metaFile << "update " << currentUpd << "\n";
            }
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

    std::shared_ptr<const algo::HistoricalPool> histPool = std::make_shared<algo::HistoricalPool>();
    std::mutex histPoolMu;
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
        std::lock_guard<std::mutex> lock(histPoolMu);
        auto oldPool = std::atomic_load(&histPool);
        auto newPool = std::make_shared<algo::HistoricalPool>(oldPool ? *oldPool : algo::HistoricalPool{});
        // 1. Check if snapshot qualifies for long-term geometric/exponential archive
        int k = args.poolEvery > 0 ? (currentUpdate / args.poolEvery) : 0;
        bool isPowerOfTwo = (k > 0) && ((k & (k - 1)) == 0);
        bool isCentennial = (currentUpdate % 100 == 0);
        if (args.archiveSize > 0 && (isPowerOfTwo || isCentennial || newPool->archive.empty())) {
            while ((int)newPool->archive.size() >= args.archiveSize && !newPool->archive.empty()) {
                if (newPool->archive.size() > 2) {
                    newPool->archive.erase(newPool->archive.begin() + 1);
                } else {
                    newPool->archive.erase(newPool->archive.begin());
                }
            }
            newPool->archive.push_back(snap);
        }

        // 2. Add to recent rolling FIFO pool
        if (args.poolSize > 0) {
            while ((int)newPool->recent.size() >= args.poolSize && !newPool->recent.empty()) {
                newPool->recent.erase(newPool->recent.begin());
            }
            newPool->recent.push_back(std::move(snap));
        }
        std::atomic_store(&histPool, std::shared_ptr<const algo::HistoricalPool>(newPool));
    };

    // Inference copies are published by pointer. Rollout holds one snapshot
    // for a whole chunk; learner updates live training models.
    struct InferPack {
        std::array<torch_nn::Actor, 3> actor;
        std::array<torch_nn::Critic, 3> critic;
    };
    auto makePack = [&]() {
        auto p = std::make_shared<InferPack>();
        torch::NoGradGuard no_grad;
        for (int s = 0; s < 3; ++s) {
            p->actor[s] = torch_nn::Actor(torchModelCfg);
            p->critic[s] = torch_nn::Critic(torchModelCfg);
            auto sa = actor[s]->parameters();
            auto da = p->actor[s]->parameters();
            for (size_t i = 0; i < sa.size(); ++i) da[i].copy_(sa[i]);
            auto sc = critic[s]->parameters();
            auto dc = p->critic[s]->parameters();
            for (size_t i = 0; i < sc.size(); ++i) dc[i].copy_(sc[i]);
            p->actor[s]->eval();
            p->critic[s]->eval();
        }
        return p;
    };
    std::shared_ptr<InferPack> live = makePack();

    struct Sample {
        std::array<std::vector<algo::Transition>, 3> streams;
        algo::RolloutStats stats{};
        double rollSecs = 0;
        int trainCamp = -1;
    };
    struct Buffer {
        std::mutex mu;
        std::condition_variable cv;
        std::deque<Sample> q;
        int cap = 2;
        bool stop = false;
    } buf;
    buf.cap = std::max(1, args.buffer);

    std::thread producer([&] {
        for (int chunk = args.startUpdate - 1; chunk < args.updates; ++chunk) {
            {
                std::lock_guard<std::mutex> lock(buf.mu);
                if (buf.stop) return;
            }
            auto snap = std::atomic_load(&live);
            algo::ModelSet models{{&snap->actor[0], &snap->actor[1], &snap->actor[2]},
                                  {&snap->critic[0], &snap->critic[1], &snap->critic[2]}};
            algo::RolloutConfig rc;
            rc.gamesPerUpdate = args.games;
            rc.threads = args.threads;
            rc.seed = args.seed + uint64_t(chunk + 1) * 7919ULL;
            rc.historicalProb = args.histProb;
            rc.ruleProb = args.ruleProb;
            rc.shapingCap = args.shapingCap;
            rc.landlordWinMul = args.landlordWinMul;
            int currentCamp = -1;
            if (args.campMode == "landlord") {
                currentCamp = 0;
            } else if (args.campMode == "peasant") {
                currentCamp = 1;
            } else if (args.campMode == "alt") {
                currentCamp = (chunk % 2);  // even = landlord (0), odd = peasant (1)
            }
            rc.trainCamp = currentCamp;
            auto snapPool = std::atomic_load(&histPool);
            rc.historicalPool = snapPool.get();
            Sample sample;
            sample.trainCamp = currentCamp;
            auto t0 = std::chrono::steady_clock::now();
            algo::collectRollout(models, rc, sample.streams, sample.stats);
            sample.rollSecs = std::chrono::duration<double>(
                                  std::chrono::steady_clock::now() - t0)
                                  .count();
            {
                std::unique_lock<std::mutex> lock(buf.mu);
                buf.cv.wait(lock, [&] {
                    return buf.stop || (int)buf.q.size() < buf.cap;
                });
                if (buf.stop) return;
                double rollSecs = sample.rollSecs;
                auto n0 = sample.stats.transitions[0];
                auto n1 = sample.stats.transitions[1];
                auto n2 = sample.stats.transitions[2];
                buf.q.push_back(std::move(sample));
                int queued = (int)buf.q.size();
                std::printf(
                    "buf chunk %d | rollout %.1fs | q %d/%d | games %d | n %lld/%lld/%lld\n",
                    chunk + 1, rollSecs, queued, buf.cap, args.games,
                    n0, n1, n2);
                std::fflush(stdout);
            }
            buf.cv.notify_all();
        }
    });

    std::array<algo::PPOStats, 3> ps{};

    for (int upd = args.startUpdate; upd <= args.updates; ++upd) {
        auto wall0 = std::chrono::steady_clock::now();
        Sample sample;
        int qReady = 0;
        int qLeft = 0;
        {
            std::unique_lock<std::mutex> lock(buf.mu);
            buf.cv.wait(lock, [&] { return buf.stop || !buf.q.empty(); });
            if (buf.q.empty()) break;
            qReady = (int)buf.q.size();
            sample = std::move(buf.q.front());
            buf.q.pop_front();
            qLeft = (int)buf.q.size();
        }
        buf.cv.notify_all();
        auto streams = std::move(sample.streams);
        algo::RolloutStats rs = sample.stats;
        double rollSecs = sample.rollSecs;
        auto t1 = std::chrono::steady_clock::now();

        float progress = args.updates > 1
                             ? float(upd - 1) / float(args.updates - 1)
                             : 0.0f;
        float currentLr = args.lr;
        if (args.lrDecay) {
            const float minLr = 1e-5f;
            currentLr = minLr + 0.5f * (args.lr - minLr) *
                                    (1.0f + std::cos(progress * 3.141592653589793f));
        }
        float currentEnt = args.ent;
        if (args.entDecay) {
            const float minEnt = 0.01f;
            currentEnt = minEnt + 0.5f * (args.ent - minEnt) *
                                       (1.0f + std::cos(progress * 3.141592653589793f));
        }

        // Determine which seats to train this update
        unsigned targetMask = 7u;
        if (sample.trainCamp == 0) {
            targetMask = (1u << 0);  // Landlord only
        } else if (sample.trainCamp == 1) {
            targetMask = (1u << 1) | (1u << 2);  // Peasant only
        }

        for (int s = 0; s < 3; ++s) {
            if ((targetMask & (1u << s)) == 0) continue;
            try {
                auto batchTensors = torch_nn::transitionsToBatch(
                    streams[s], args.gamma, args.lambda, torchDevice);
                learners[s]->setLr(currentLr);
                learners[s]->setEntCoef(currentEnt);

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

        // Update live snapshot for rollout workers
        std::atomic_store(&live, makePack());

        double learnSecs = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - t1)
                               .count();
        double wallSecs = std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - wall0)
                              .count();
        int qEnd = 0;
        {
            std::lock_guard<std::mutex> lock(buf.mu);
            qEnd = (int)buf.q.size();
        }
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
            "upd %4d | q %d/%d left %d end %d | wall %.1fs rollout %.1fs learn %.1fs "
            "games %d mb %d ep %d,%d,%d/%d threads %d | WP %.3f ADP %7.2f%s bomb/g %.2f "
            "moves/g %.1f | lr %.2e ent %5.3f/%5.3f/%5.3f vL %7.2f/%7.2f/%7.2f "
            "kl %.4f/%.4f/%.4f cf %.2f/%.2f/%.2f | ret %7.1f/%7.1f/%7.1f | n %lld/%lld/%lld\n",
            upd, qReady, buf.cap, qLeft, qEnd, wallSecs, rollSecs, learnSecs,
            rs.games, args.minibatch,
            ps[0].epochsCompleted, ps[1].epochsCompleted, ps[2].epochsCompleted,
            args.epochs, args.threads, wp, adp, pureStr.c_str(), bpg, mpg,
            currentLr, ps[0].entropy, ps[1].entropy, ps[2].entropy,
            ps[0].vLoss, ps[1].vLoss, ps[2].vLoss,
            ps[0].lastEpochKL, ps[1].lastEpochKL, ps[2].lastEpochKL,
            ps[0].clipFraction, ps[1].clipFraction, ps[2].clipFraction,
            ps[0].meanRet, ps[1].meanRet, ps[2].meanRet,
            rs.transitions[0], rs.transitions[1], rs.transitions[2]);
        std::fflush(stdout);

        if (args.poolEvery > 0 && upd % args.poolEvery == 0) {
            pushHistPool(actor, upd);
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
            auto cloned = algo::cloneActors(actor);
            algo::DouZeroEvalConfig evalCfg;
            evalCfg.host = args.evalHost;
            evalCfg.port = args.evalPort;
            evalCfg.decks = args.evalDecks;
            evalCfg.seed = static_cast<uint64_t>(upd * 10007 + 1);
            evalCfg.csvPath = args.evalCsv;
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

    {
        std::lock_guard<std::mutex> lock(buf.mu);
        buf.stop = true;
    }
    buf.cv.notify_all();
    if (producer.joinable()) producer.join();

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
