// PPO update with GAE (paper Sec. 4.3 / Table 7):
//   clipped policy gradient, entropy bonus 0.1, value-function MSE against
//   GAE returns of the perfect-information critic.
#pragma once

#include <vector>

#include "algo/rollout.h"
#include "nn/net.h"

namespace algo {

struct PPOConfig {
    float clip = 0.2f;
    float entCoef = 0.1f;
    float vfCoef = 0.5f;
    int epochs = 4;
    int minibatch = 256;
    float gamma = 1.0f;
    float lambda = 0.95f;
    float maxGradNorm = 0.5f;
};

struct PPOStats {
    double pgLoss = 0;
    double vLoss = 0;
    double entropy = 0;
    double meanRet = 0;
    double meanAdv = 0;
    double meanAbsOldLogp = 0;
};

// Computes GAE returns/advantages in place (transitions must be grouped by
// game through their terminal flags / gameId).
void computeGAE(std::vector<Transition>& tr, float gamma, float lambda);

// One PPO update of one seat's actor + critic over its rollout stream.
// False means the GPU device was removed before Adam wrote host weights, so
// the update can be repeated after the device is recreated.
bool ppoUpdate(nn::Actor& actor, nn::Critic& critic,
               std::vector<Transition>& tr, const PPOConfig& cfg,
               nn::Adam& actorOpt, nn::Adam& criticOpt, nn::Rng64& rng,
               PPOStats& stats);

}  // namespace algo
