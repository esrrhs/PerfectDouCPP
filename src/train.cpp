// PerfectDou CPP — self-play PPO training entry point.
//
// Faithful re-implementation of the PerfectDou card-play training:
//   * three seat models (landlord / landlord_down / landlord_up), self-play
//   * imperfect-information actor over concrete legal plays
//   * perfect-information critic (sees all hands, PTIE / perfect information
//     distillation through the advantage)
//   * PPO, TD(lambda) bootstrapped value targets with terminal ADP reward (paper aligned architecture)
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

namespace {

struct Args {
    int updates = 200;
    int startUpdate = 1;
    int games = 256;
    int threads = int(std::thread::hardware_concurrency());
    int hidden = 256;
    int lstmHidden = 128;
    int epochs = 4;
    int minibatch = 1024;
    int snapshotEvery = 20;
    int buffer = 1;
    float lr = 3e-4f;
    float clip = 0.2f;
    float ent = 0.03f;
    float gamma = 1.0f;
    float lambda = 0.95f;
    float targetKL = 0.03f;
    bool clipVf = false;
    float vfClip = 32.0f;
    float shapingCap = 0.05f;
    bool lrDecay = true;
    bool entDecay = true;
    int poolSize = 16;
    int archiveSize = 16;
    int poolEvery = 20;
    float histProb = 0.2f;
    float ruleProb = 0.1f;
    uint64_t seed = 1;
    std::string out = "ckpt";
    std::string resume;
    std::string backend = "auto";  // auto | cuda | cpu
    int evalEvery = 0;             // 0 = disabled, > 0 = eval vs DouZero every N updates (recommended: 200~500)
    int evalDecks = 100;           // number of decks per eval (100 decks = 200 games)
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
    a.updates = std::atoi(argValue(argc, argv, "--updates", "200"));
    a.startUpdate = std::atoi(argValue(argc, argv, "--start-update", "1"));
    a.games = std::atoi(argValue(argc, argv, "--games", "256"));
    a.threads = std::atoi(argValue(argc, argv, "--threads",
                                  std::to_string(a.threads).c_str()));
    a.hidden = std::atoi(argValue(argc, argv, "--hidden", "256"));
    a.lstmHidden = std::atoi(argValue(argc, argv, "--lstm-hidden", "128"));
    a.epochs = std::atoi(argValue(argc, argv, "--epochs", "4"));
    a.minibatch = std::atoi(argValue(argc, argv, "--mb", "1024"));
    a.snapshotEvery = std::atoi(argValue(argc, argv, "--snapshot-every", "20"));
    a.buffer = std::atoi(argValue(argc, argv, "--buffer", "1"));
    a.lr = float(std::atof(argValue(argc, argv, "--lr", "3e-4")));
    a.clip = float(std::atof(argValue(argc, argv, "--clip", "0.2")));
    a.ent = float(std::atof(argValue(argc, argv, "--ent", "0.03")));
    a.gamma = float(std::atof(argValue(argc, argv, "--gamma", "1.0")));
    a.lambda = float(std::atof(argValue(argc, argv, "--lambda", "0.95")));
    a.targetKL = float(std::atof(argValue(argc, argv, "--target-kl", "0.03")));
    a.vfClip = float(std::atof(argValue(argc, argv, "--vf-clip", "32.0")));
    a.shapingCap = float(std::atof(argValue(argc, argv, "--shaping-cap", "0.05")));
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
    a.out = argValue(argc, argv, "--out", "ckpt");
    a.backend = argValue(argc, argv, "--backend", "auto");
    const char* res = argValue(argc, argv, "--resume", "");
    if (*res) a.resume = res;
    a.evalEvery = std::atoi(argValue(argc, argv, "--eval-every", "0"));
    if (a.evalEvery == 0 && hasFlag(argc, argv, "--eval")) {
        a.evalEvery = 500;
    }
    a.evalDecks = std::atoi(argValue(argc, argv, "--eval-decks", "100"));
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

    if (args.backend == "gpu") {
        std::fprintf(stderr,
                     "--backend gpu was removed (pure D3D12 GEMM hangs on some "
                     "NVIDIA drivers). Use --backend cuda or --backend cpu.\n");
        return 1;
    }
    if (args.backend != "auto" && args.backend != "cuda" &&
        args.backend != "cpu") {
        std::fprintf(stderr,
                     "unknown --backend %s (expected auto|cuda|cpu)\n",
                     args.backend.c_str());
        return 1;
    }

    nn::gemmInit();
#if defined(_WIN32)
    // Windows GPU path is CUDA-only. auto falls back to CPU when cuBLAS is
    // unavailable (no toolkit / not built with PD_HAVE_CUBLAS).
    const bool wantCuda = args.backend == "cuda" || args.backend == "auto";
    if (args.backend == "cpu") {
        nn::gemmSetGpu(false);
    } else if (wantCuda) {
        _putenv_s("PD_CUBLAS", "1");
        nn::gemmSetGpu(true);
        if (!nn::gemmGpuEnabled()) {
            if (args.backend == "cuda") {
                std::fprintf(stderr,
                             "--backend cuda unavailable: %s\n"
                             "Install CUDA Toolkit, reconfigure/rebuild, then "
                             "retry (or use --backend cpu).\n",
                             nn::gemmCudaError());
                return 1;
            }
            nn::gemmSetGpu(false);
        }
    }
#else
    if (args.backend == "cuda") {
        std::fprintf(stderr, "--backend cuda is Windows-only; use --backend cpu\n");
        return 1;
    }
    nn::gemmSetGpu(false);
#endif
    std::cout << "PerfectDou CPP training\n";
    const char* backend = nn::gemmGpuEnabled() ? nn::gemmGpuLabel() :
#ifdef __APPLE__
                          "CPU (Accelerate)";
#else
                          "CPU";
#endif
    std::cout << "  GEMM backend: " << backend << "\n";
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
              << ",hist=" << args.histProb << ",rule=" << args.ruleProb << ")"
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

    nn::NetConfig cfg{args.hidden, args.lstmHidden};
    std::array<nn::Actor, 3> actor;
    std::array<nn::Critic, 3> critic;
    std::array<nn::Adam, 3> aOpt, cOpt;
    for (int s = 0; s < 3; ++s) {
        actor[s].init(cfg, args.seed * 100 + s * 17 + 1);
        critic[s].init(cfg, args.seed * 100 + s * 17 + 53);
        aOpt[s].lr = args.lr;
        cOpt[s].lr = args.lr;
        if (!args.resume.empty()) {
            actor[s].load((args.resume + "/actor" + std::to_string(s) + ".bin").c_str());
            critic[s].load((args.resume + "/critic" + std::to_string(s) + ".bin").c_str());
            std::string optA = args.resume + "/opt_actor" + std::to_string(s) + ".bin";
            std::string optC = args.resume + "/opt_critic" + std::to_string(s) + ".bin";
            if (std::filesystem::exists(optA)) {
                nn::loadOptimizer(optA.c_str(), actor[s].params(), aOpt[s]);
            }
            if (std::filesystem::exists(optC)) {
                nn::loadOptimizer(optC.c_str(), critic[s].params(), cOpt[s]);
            }
        }
    }

    algo::PPOConfig ppo;
    ppo.clip = args.clip;
    ppo.entCoef = args.ent;
    ppo.epochs = args.epochs;
    ppo.minibatch = args.minibatch;
    ppo.gamma = args.gamma;
    ppo.lambda = args.lambda;
    ppo.targetKL = args.targetKL;
    ppo.clipVf = args.clipVf;
    ppo.vfClip = args.vfClip;

    auto saveAll = [&](const std::string& dir, int currentUpd = 0) {
        std::filesystem::create_directories(dir);
        for (int s = 0; s < 3; ++s) {
            actor[s].save((dir + "/actor" + std::to_string(s) + ".bin").c_str());
            critic[s].save((dir + "/critic" + std::to_string(s) + ".bin").c_str());
            nn::saveOptimizer((dir + "/opt_actor" + std::to_string(s) + ".bin").c_str(),
                             actor[s].params(), aOpt[s]);
            nn::saveOptimizer((dir + "/opt_critic" + std::to_string(s) + ".bin").c_str(),
                             critic[s].params(), cOpt[s]);
        }
        if (currentUpd > 0) {
            std::ofstream metaFile(dir + "/meta.txt");
            if (metaFile.is_open()) {
                metaFile << "update " << currentUpd << "\n";
            }
        }
    };

    // warm up the shared (read-only) oracle DP table before spawning workers
    ddz::minSteps(ddz::CardSet::parse("34567"));

    // Idle modern standby powers the discrete GPU down; the driver then
    // removes the device. Keep the system in the working state until exit.
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
    auto pushHistPool = [&](std::array<nn::Actor, 3>& srcActor, int currentUpdate) {
        auto snap = std::make_shared<algo::HistoricalActorSnapshot>();
        snap->update = currentUpdate;
        for (int s = 0; s < 3; ++s) {
            snap->actor[s].init(cfg, 1);
            auto da = snap->actor[s].params();
            auto sa = srcActor[s].params();
            for (size_t i = 0; i < sa.size(); ++i) da[i]->w = sa[i]->w;
            snap->actor[s].prepareInference();
        }
        std::lock_guard<std::mutex> lock(histPoolMu);
        auto oldPool = std::atomic_load(&histPool);
        auto newPool = std::make_shared<algo::HistoricalPool>(oldPool ? *oldPool : algo::HistoricalPool{});
        // 1. Check if snapshot qualifies for long-term geometric/exponential archive
        int k = args.poolEvery > 0 ? (currentUpdate / args.poolEvery) : 0;
        bool isPowerOfTwo = (k > 0) && ((k & (k - 1)) == 0);
        bool isCentennial = (currentUpdate % 100 == 0);
        if (args.archiveSize > 0 && (isPowerOfTwo || isCentennial || newPool->archive.empty())) {
            // When full, protect the earliest foundational anchor models (e.g. index 0/1)
            // and thin out the intermediate models rather than dropping the oldest.
            while ((int)newPool->archive.size() >= args.archiveSize && !newPool->archive.empty()) {
                if (newPool->archive.size() > 2) {
                    // Thin out from the middle
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

    if (std::getenv("PD_SINGLE_THREAD")) {
        std::fprintf(stderr, "single-thread: rollout and learn on this thread\n");
        std::fflush(stderr);
        args.threads = 1;
        std::array<std::vector<algo::Transition>, 3> streams;
        std::array<algo::PPOStats, 3> ps{};
        for (int upd = args.startUpdate; upd <= args.updates; ++upd) {
            algo::ModelSet models{{&actor[0], &actor[1], &actor[2]},
                                  {&critic[0], &critic[1], &critic[2]}};
            algo::RolloutConfig rc;
            rc.gamesPerUpdate = args.games;
            rc.threads = 1;
            rc.seed = args.seed + uint64_t(upd) * 7919ULL;
            rc.historicalProb = args.histProb;
            rc.ruleProb = args.ruleProb;
            rc.shapingCap = args.shapingCap;
            auto snapPool = std::atomic_load(&histPool);
            rc.historicalPool = snapPool.get();
            algo::RolloutStats rs;
            auto wall0 = std::chrono::steady_clock::now();
            // collectRollout appends. A fresh Sample in the pipelined path
            // starts empty; this loop reuses the same vectors.
            for (auto& seat : streams) seat.clear();
            algo::collectRollout(models, rc, streams, rs);
            nn::gemmSetThreadGpu(-1);
            double rollSecs = std::chrono::duration<double>(
                                  std::chrono::steady_clock::now() - wall0)
                                  .count();
            float progress = args.updates > 1
                                 ? float(upd - 1) / float(args.updates - 1)
                                 : 0.0f;
            float currentLr = args.lr;
            if (args.lrDecay) {
                const float minLr = 1e-5f;
                currentLr = minLr + 0.5f * (args.lr - minLr) *
                                        (1.0f + std::cos(progress * 3.141592653589793f));
            }
            if (args.entDecay) {
                const float minEnt = 0.01f;
                ppo.entCoef = minEnt + 0.5f * (args.ent - minEnt) *
                                           (1.0f + std::cos(progress * 3.141592653589793f));
            }
            for (int s = 0; s < 3; ++s) {
                aOpt[s].lr = currentLr;
                cOpt[s].lr = currentLr;
            }

            auto t1 = std::chrono::steady_clock::now();
            auto learnSeats = [&](unsigned mask) {
                unsigned failed = 0;
                for (int s = 0; s < 3; ++s) {
                    if ((mask & (1u << s)) == 0) continue;
                    std::fprintf(stderr, "seat %d update %d\n", s, upd);
                    std::fflush(stderr);
                    nn::Rng64 ur(args.seed * 100000 + upd * 31 + s);
                    if (!algo::ppoUpdate(actor[s], critic[s], streams[s], ppo,
                                         aOpt[s], cOpt[s], ur, ps[s]))
                        failed |= 1u << s;
                }
                return failed;
            };
            unsigned failed = learnSeats(7u);
            for (int attempt = 1; failed != 0 && attempt <= 2; ++attempt) {
                std::fprintf(stderr,
                             "GPU device lost during update %d (seats 0x%x), "
                             "recreating and retrying\n",
                             upd, failed);
                nn::gpuReleaseThread();
                if (!nn::gpuRecreate()) {
                    std::fprintf(stderr, "D3D12 device recreate failed\n");
                    failed = 7u;
                    break;
                }
                failed = learnSeats(failed);
            }
            if (failed) {
                std::fprintf(stderr,
                             "GPU device lost again during update %d, stopping\n",
                             upd);
                return 1;
            }
            double learnSecs = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - t1)
                                   .count();
            double wallSecs = std::chrono::duration<double>(
                                  std::chrono::steady_clock::now() - wall0)
                                  .count();
            double wp = double(rs.landlordWins) / std::max(1, rs.games);
            double adp = double(rs.landlordScore) / std::max(1, rs.games);
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
                "upd %4d | wall %.1fs rollout %.1fs learn %.1fs ep %d,%d,%d/%d threads 1 | "
                "WP %.3f ADP %7.2f%s | lr %.2e ent %5.3f/%5.3f/%5.3f | "
                "kl %.4f/%.4f/%.4f cf %.2f/%.2f/%.2f | n %lld/%lld/%lld\n",
                upd, wallSecs, rollSecs, learnSecs,
                ps[0].epochsCompleted, ps[1].epochsCompleted, ps[2].epochsCompleted,
                args.epochs, wp, adp, pureStr.c_str(), currentLr, ps[0].entropy,
                ps[1].entropy, ps[2].entropy,
                ps[0].lastEpochKL, ps[1].lastEpochKL, ps[2].lastEpochKL,
                ps[0].clipFraction, ps[1].clipFraction, ps[2].clipFraction,
                rs.transitions[0], rs.transitions[1], rs.transitions[2]);
            std::fflush(stdout);
            if (args.poolEvery > 0 && upd % args.poolEvery == 0) {
                pushHistPool(actor, upd);
            }
            if (args.snapshotEvery > 0 &&
                (upd % args.snapshotEvery == 0 || upd == args.updates)) {
                saveAll(args.out, upd);
            }
        }
        saveAll(args.out, args.updates);
        std::cout << "models saved to " << args.out << "/\n";
        return 0;
    }

    // One GPU context per seat for the whole run. Destroying a D3D12 compute
    // queue takes several seconds on this driver, and a queue cannot be
    // handed to another thread, so the learners stay alive across updates.
    const int nLearn = std::getenv("PD_ONE_LEARNER") ? 1 : 3;
    struct SeatJob {
        std::mutex mu;
        std::condition_variable cv;
        int generation = 0;
        int done = 0;
        int upd = 0;
        int phase = 0;  // 0 = learn, 1 = release this thread's GPU context
        unsigned seatMask = 0;
        unsigned failed = 0;
        bool stop = false;
    } job;
    std::array<std::vector<algo::Transition>, 3> streams;
    std::array<algo::PPOStats, 3> ps;
    std::vector<std::thread> learners;
    for (int s = 0; s < nLearn; ++s) {
        learners.emplace_back([&, s] {
            // Pure D3D stays on the owner thread: a second compute queue
            // removes the device. CUDA learners record on this thread and
            // share the one queue created inside ensureThread.
            if (!std::getenv("PD_CUBLAS")) nn::gemmSetThreadGpu(0);
            int seen = 0;
            while (true) {
                int upd = 0;
                int phase = 0;
                unsigned mask = 0;
                {
                    std::unique_lock<std::mutex> lock(job.mu);
                    job.cv.wait(lock, [&] {
                        return job.stop || job.generation > seen;
                    });
                    if (job.stop && job.generation == seen) return;
                    upd = job.upd;
                    phase = job.phase;
                    mask = job.seatMask;
                    seen = job.generation;
                }
                if (phase == 1) {
                    nn::gpuReleaseThread();
                } else if (mask & (1u << s)) {
                    nn::Rng64 ur(args.seed * 100000 + upd * 31 + s);
                    bool ok = algo::ppoUpdate(actor[s], critic[s], streams[s],
                                              ppo, aOpt[s], cOpt[s], ur, ps[s]);
                    if (!ok) {
                        std::lock_guard<std::mutex> lock(job.mu);
                        job.failed |= 1u << s;
                    }
                }
                {
                    std::lock_guard<std::mutex> lock(job.mu);
                    ++job.done;
                }
                job.cv.notify_all();
            }
        });
    }

    auto waitLearners = [&] {
        std::unique_lock<std::mutex> lock(job.mu);
        if (!job.cv.wait_for(lock, std::chrono::seconds(180), [&] {
                return job.done == job.generation * nLearn;
            })) {
            std::fprintf(stderr,
                         "FATAL: Timeout (180s) in waitLearners! done=%d expected=%d "
                         "generation=%d (deadlock/hang detected)\n",
                         job.done, job.generation * nLearn, job.generation);
            std::fflush(stderr);
            std::abort();
        }
    };
    auto runLearn = [&](unsigned mask) {
        {
            std::lock_guard<std::mutex> lock(job.mu);
            job.phase = 0;
            job.seatMask = mask;
            job.failed = 0;
            ++job.generation;
        }
        job.cv.notify_all();
        waitLearners();
        std::lock_guard<std::mutex> lock(job.mu);
        return job.failed;
    };
    auto releaseGpu = [&] {
        {
            std::lock_guard<std::mutex> lock(job.mu);
            job.phase = 1;
            ++job.generation;
        }
        job.cv.notify_all();
        waitLearners();
    };

    // Inference copies are published by pointer. Rollout holds one snapshot
    // for a whole chunk; the learner keeps writing the training models.
    struct InferPack {
        std::array<nn::Actor, 3> actor;
        std::array<nn::Critic, 3> critic;
    };
    auto copySeat = [](nn::Actor& dstA, nn::Critic& dstC, nn::Actor& srcA,
                       nn::Critic& srcC) {
        auto da = dstA.params();
        auto sa = srcA.params();
        for (size_t i = 0; i < sa.size(); ++i) da[i]->w = sa[i]->w;
        auto dc = dstC.params();
        auto sc = srcC.params();
        for (size_t i = 0; i < sc.size(); ++i) dc[i]->w = sc[i]->w;
        dstA.prepareInference();
        dstC.prepareInference();
    };
    auto makePack = [&] {
        auto p = std::make_shared<InferPack>();
        for (int s = 0; s < 3; ++s) {
            p->actor[s].init(cfg, 1);
            p->critic[s].init(cfg, 1);
            copySeat(p->actor[s], p->critic[s], actor[s], critic[s]);
        }
        return p;
    };
    std::shared_ptr<InferPack> live = makePack();

    struct Sample {
        std::array<std::vector<algo::Transition>, 3> streams;
        algo::RolloutStats stats{};
        double rollSecs = 0;
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
            auto snapPool = std::atomic_load(&histPool);
            rc.historicalPool = snapPool.get();
            Sample sample;
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
        streams = std::move(sample.streams);
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
        if (args.entDecay) {
            const float minEnt = 0.01f;
            ppo.entCoef = minEnt + 0.5f * (args.ent - minEnt) *
                                       (1.0f + std::cos(progress * 3.141592653589793f));
        }
        for (int s = 0; s < 3; ++s) {
            aOpt[s].lr = currentLr;
            cOpt[s].lr = currentLr;
        }

        {
            std::lock_guard<std::mutex> lock(job.mu);
            job.upd = upd;
        }
        unsigned failed = runLearn((1u << nLearn) - 1u);
        for (int attempt = 1; failed != 0 && attempt <= 2; ++attempt) {
            std::fprintf(stderr,
                         "GPU device lost during update %d (seats 0x%x), "
                         "recreating and retrying\n",
                         upd, failed);
            releaseGpu();
            if (!nn::gpuRecreate()) {
                std::fprintf(stderr, "D3D12 device recreate failed\n");
                failed = (1u << nLearn) - 1u;
                break;
            }
            failed = runLearn(failed);
        }
        if (failed) {
            std::fprintf(stderr,
                         "GPU device lost again during update %d, stopping\n",
                         upd);
            {
                std::lock_guard<std::mutex> lock(job.mu);
                job.stop = true;
            }
            job.cv.notify_all();
            for (auto& th : learners)
                if (th.joinable()) th.join();
#if defined(_WIN32)
            if (awake) {
                PowerClearRequest(awake, PowerRequestExecutionRequired);
                PowerClearRequest(awake, PowerRequestSystemRequired);
                CloseHandle(awake);
            }
            SetThreadExecutionState(ES_CONTINUOUS);
#endif
            {
                std::lock_guard<std::mutex> lock(buf.mu);
                buf.stop = true;
            }
            buf.cv.notify_all();
            if (producer.joinable()) producer.join();
            if (evalThread.joinable()) evalThread.join();
            return 1;
        }

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
        // bound memory: nothing needed (oracle memo is per rollout worker)
    }

    {
        std::lock_guard<std::mutex> lock(buf.mu);
        buf.stop = true;
    }
    buf.cv.notify_all();
    if (producer.joinable()) producer.join();

    {
        std::lock_guard<std::mutex> lock(job.mu);
        job.stop = true;
    }
    job.cv.notify_all();
    for (auto& th : learners)
        if (th.joinable()) th.join();

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
