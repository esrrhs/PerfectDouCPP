// PPO update. Supports GAE bootstrapped value targets (default lambda=0.95),
// reducing to the seat's final Monte Carlo ADP when lambda=1.0.
#pragma once

#include <vector>

#include "algo/rollout.h"
#include "nn/net.h"

namespace algo {

struct PPOConfig {
    float clip = 0.2f;       // Policy ratio clipping
    float vfClip = 5.0f;     // Value clipping range (ADP returns span roughly +-2..48)
    float entCoef = 0.03f;
    float vfCoef = 0.5f;
    int epochs = 4;
    int minibatch = 1024;
    float maxGradNorm = 0.5f;
    float criticMaxGradNorm = 5.0f;  // Critic gradients are on ADP scale, so allow a larger norm
    float gamma = 1.0f;
    float lambda = 0.95f;    // GAE lambda (0.95 for variance reduction, 1.0 for pure Monte Carlo)
    float targetKL = 0.03f;  // Early stopping threshold (0 to disable)
    bool clipVf = true;      // Value loss clipping enabled
};

struct PPOStats {
    double pgLoss = 0;
    double vLoss = 0;
    double entropy = 0;
    double meanRet = 0;
    double meanAdv = 0;
    double approxKL = 0;
    double lastEpochKL = 0;
    double clipFraction = 0;
    int epochsCompleted = 0;
};

// Running mean and variance normalizer with Welford algorithm
struct RunningNormalizer {
    double count = 0.0;
    double mean = 0.0;
    double M2 = 0.0;

    void update(float val) {
        count += 1.0;
        double delta = double(val) - mean;
        mean += delta / count;
        double delta2 = double(val) - mean;
        M2 += delta * delta2;
    }

    float normalize(float val) const {
        double var = (count > 1.0) ? (M2 / (count - 1.0)) : 1.0;
        float std = float(std::sqrt(std::max(var, 1e-6)));
        return float((double(val) - mean) / std);
    }

    float denormalize(float val) const {
        double var = (count > 1.0) ? (M2 / (count - 1.0)) : 1.0;
        float std = float(std::sqrt(std::max(var, 1e-6)));
        return float(double(val) * std + mean);
    }
};

// Computes GAE advantages and returns for each transition in the trajectory stream.
// When lambda == 1.0f and gamma == 1.0f, this smoothly matches the full Monte Carlo ADP return.
void assignEpisodeReturns(std::vector<Transition>& tr, float gamma = 1.0f,
                          float lambda = 0.95f);

// One PPO update of one seat's actor + critic over its rollout stream.
// False means the GPU device was removed before Adam wrote host weights, so
// the update can be repeated after the device is recreated.
bool ppoUpdate(nn::Actor& actor, nn::Critic& critic,
               std::vector<Transition>& tr, const PPOConfig& cfg,
               nn::Adam& actorOpt, nn::Adam& criticOpt, nn::Rng64& rng,
               PPOStats& stats);

}  // namespace algo
