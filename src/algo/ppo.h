// PPO update. The critic target of every decision is that seat's final ADP,
// the same quantity DouZero regresses onto and the quantity we evaluate.
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
    int minibatch = 1024;
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

// Sets every decision in a game to that seat's total reward. After rollout
// the only reward is the terminal ADP, so each return equals the final score.
void assignEpisodeReturns(std::vector<Transition>& tr);

// One PPO update of one seat's actor + critic over its rollout stream.
// False means the GPU device was removed before Adam wrote host weights, so
// the update can be repeated after the device is recreated.
bool ppoUpdate(nn::Actor& actor, nn::Critic& critic,
               std::vector<Transition>& tr, const PPOConfig& cfg,
               nn::Adam& actorOpt, nn::Adam& criticOpt, nn::Rng64& rng,
               PPOStats& stats);

}  // namespace algo
