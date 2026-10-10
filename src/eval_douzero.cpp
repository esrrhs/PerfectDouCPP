// Play saved PerfectDou actors against a DouZero server.
// The training process is not involved. Inference stays on the CPU.
//
// perfectdou_eval --models DIR --decks N --port 18765 --label u1100
//                 --csv build-win/curve/curve.csv
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "algo/eval_douzero.h"
#include "nn/gemm.h"
#include "nn/net.h"

namespace {

std::string argValue(int argc, char** argv, const char* key, const char* def) {
    for (int i = 1; i < argc - 1; ++i)
        if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
    return def;
}

}  // namespace

int main(int argc, char** argv) {
    std::string models = argValue(argc, argv, "--models", "");
    int decks = std::atoi(argValue(argc, argv, "--decks", "200").c_str());
    std::string host = argValue(argc, argv, "--host", "127.0.0.1");
    int port = std::atoi(argValue(argc, argv, "--port", "18765").c_str());
    std::string label = argValue(argc, argv, "--label", "snap");
    std::string csv = argValue(argc, argv, "--csv", "build-win/curve/curve.csv");
    uint64_t seed = std::strtoull(argValue(argc, argv, "--seed", "1").c_str(), nullptr, 10);
    if (models.empty() || decks <= 0) {
        std::fprintf(stderr,
                     "usage: perfectdou_eval --models DIR --decks N [--port P] [--host H] "
                     "[--csv FILE] [--label L] [--seed S]\n");
        return 2;
    }

    nn::gemmInit();
    nn::gemmSetGpu(false);
    nn::gemmSetThreadGpu(0);

    std::array<nn::Actor, 3> actor;
    for (int s = 0; s < 3; ++s) {
        actor[s].load((models + "/actor" + std::to_string(s) + ".bin").c_str());
        actor[s].prepareInference();
    }

    algo::DouZeroEvalConfig cfg;
    cfg.host = host;
    cfg.port = port;
    cfg.decks = decks;
    cfg.seed = seed;
    cfg.csvPath = csv;
    cfg.label = label;
    cfg.verbose = true;

    auto res = algo::evaluateAgainstDouZero(actor, cfg);
    if (!res.ok) {
        std::fprintf(stderr, "evaluation failed: %s\n", res.error.c_str());
        return 1;
    }

    std::printf(
        "done games %d WP %.3f ADP %.3f | landlord WP %.3f ADP %.3f | peasant WP %.3f ADP %.3f\n",
        res.games, res.wp, res.adp, res.wpLandlord, res.adpLandlord, res.wpPeasant, res.adpPeasant);
    return 0;
}
