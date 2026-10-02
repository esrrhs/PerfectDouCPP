// PerfectDou CPP — self-play PPO training entry point.
//
// Faithful re-implementation of the PerfectDou card-play training:
//   * three seat models (landlord / landlord_down / landlord_up), self-play
//   * imperfect-information actor over the 621 abstract actions
//   * perfect-information critic (sees all hands, PTIE / perfect information
//     distillation through the advantage)
//   * PPO + GAE, oracle distance-to-win shaping plus terminal ADP reward
//
// Example:
//   perfectdou_train --updates 200 --games 256 --threads 8 --out ckpt
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>

#include "algo/ppo.h"
#include "algo/rollout.h"
#include "ddz/oracle.h"

namespace {

struct Args {
    int updates = 200;
    int games = 256;
    int threads = int(std::thread::hardware_concurrency());
    int hidden = 256;
    int lstmHidden = 128;
    int epochs = 4;
    int minibatch = 256;
    int snapshotEvery = 20;
    float lr = 3e-4f;
    float clip = 0.2f;
    float ent = 0.1f;
    float gamma = 1.0f;
    float gae = 0.95f;
    float rewardScale = 50.0f;
    uint64_t seed = 1;
    std::string out = "ckpt";
    std::string resume;
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
    a.games = std::atoi(argValue(argc, argv, "--games", "256"));
    a.threads = std::atoi(argValue(argc, argv, "--threads",
                                  std::to_string(a.threads).c_str()));
    a.hidden = std::atoi(argValue(argc, argv, "--hidden", "256"));
    a.lstmHidden = std::atoi(argValue(argc, argv, "--lstm-hidden", "128"));
    a.epochs = std::atoi(argValue(argc, argv, "--epochs", "4"));
    a.minibatch = std::atoi(argValue(argc, argv, "--mb", "256"));
    a.snapshotEvery = std::atoi(argValue(argc, argv, "--snapshot-every", "20"));
    a.lr = float(std::atof(argValue(argc, argv, "--lr", "3e-4")));
    a.clip = float(std::atof(argValue(argc, argv, "--clip", "0.2")));
    a.ent = float(std::atof(argValue(argc, argv, "--ent", "0.1")));
    a.gamma = float(std::atof(argValue(argc, argv, "--gamma", "1.0")));
    a.gae = float(std::atof(argValue(argc, argv, "--gae", "0.95")));
    a.rewardScale = float(std::atof(argValue(argc, argv, "--reward-scale", "50")));
    a.seed = std::atoll(argValue(argc, argv, "--seed", "1"));
    a.out = argValue(argc, argv, "--out", "ckpt");
    const char* res = argValue(argc, argv, "--resume", "");
    if (*res) a.resume = res;
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    parseArgs(argc, argv, args);
    if (args.threads <= 0) args.threads = 1;

    std::cout << "PerfectDou CPP training\n";
    std::cout << "  updates=" << args.updates << " games/update=" << args.games
              << " threads=" << args.threads << " hidden=" << args.hidden
              << " lstm=" << args.lstmHidden << " lr=" << args.lr
              << " rewardScale=" << args.rewardScale << "\n";

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
        }
    }

    algo::PPOConfig ppo;
    ppo.clip = args.clip;
    ppo.entCoef = args.ent;
    ppo.gamma = args.gamma;
    ppo.lambda = args.gae;
    ppo.epochs = args.epochs;
    ppo.minibatch = args.minibatch;

    auto saveAll = [&](const std::string& dir) {
        std::string cmd = "mkdir -p " + dir;
        std::system(cmd.c_str());
        for (int s = 0; s < 3; ++s) {
            actor[s].save((dir + "/actor" + std::to_string(s) + ".bin").c_str());
            critic[s].save((dir + "/critic" + std::to_string(s) + ".bin").c_str());
        }
    };

    // warm up the shared (read-only) oracle DP table before spawning workers
    ddz::minSteps(ddz::CardSet::parse("34567"));

    for (int upd = 1; upd <= args.updates; ++upd) {
        auto t0 = std::chrono::steady_clock::now();
        algo::ModelSet models{{&actor[0], &actor[1], &actor[2]},
                              {&critic[0], &critic[1], &critic[2]}};
        algo::RolloutConfig rc;
        rc.gamesPerUpdate = args.games;
        rc.threads = args.threads;
        rc.rewardScale = args.rewardScale;
        rc.seed = args.seed + uint64_t(upd) * 7919ULL;

        std::array<std::vector<algo::Transition>, 3> streams;
        algo::RolloutStats rs;
        algo::collectRollout(models, rc, streams, rs);

        double secs = std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - t0)
                          .count();

        std::array<algo::PPOStats, 3> ps;
        // The three seat models are independent: update them in parallel.
        std::vector<std::thread> learners;
        for (int s = 0; s < 3; ++s) {
            learners.emplace_back([&, s] {
                nn::Rng64 ur(args.seed * 100000 + upd * 31 + s);
                algo::ppoUpdate(actor[s], critic[s], streams[s], ppo, aOpt[s],
                                cOpt[s], ur, ps[s]);
                actor[s].prepareInference();
                critic[s].prepareInference();
            });
        }
        for (auto& th : learners) th.join();

        double wp = double(rs.landlordWins) / std::max(1, rs.games);
        double adp = double(rs.landlordScore) / std::max(1, rs.games);
        double bpg = double(rs.bombs) / std::max(1, rs.games);
        double mpg = double(rs.moves) / std::max(1, rs.games);
        std::printf(
            "upd %4d | rollout %.1fs games %d WP %.3f ADP %7.2f bomb/g %.2f "
            "moves/g %.1f | ent %5.3f/%5.3f/%5.3f vL %7.2f/%7.2f/%7.2f "
            "ret %7.1f/%7.1f/%7.1f | n %lld/%lld/%lld\n",
            upd, secs, rs.games, wp, adp, bpg, mpg,
            ps[0].entropy, ps[1].entropy, ps[2].entropy,
            ps[0].vLoss, ps[1].vLoss, ps[2].vLoss,
            ps[0].meanRet, ps[1].meanRet, ps[2].meanRet,
            rs.transitions[0], rs.transitions[1], rs.transitions[2]);
        std::fflush(stdout);

        if (args.snapshotEvery > 0 &&
            (upd % args.snapshotEvery == 0 || upd == args.updates)) {
            saveAll(args.out);
        }
        // bound memory: nothing needed (oracle memo is per rollout worker)
    }

    saveAll(args.out);
    std::cout << "models saved to " << args.out << "/\n";
    return 0;
}
