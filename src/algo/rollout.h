// Batched self-play rollout collection.
//
// A pool of games is advanced in lockstep. At every tick all games waiting on
// the same seat are evaluated in one batched forward pass. Each seat keeps
// its own transition stream for PPO/GAE.
#pragma once

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include "ddz/features.h"
#include "ddz/game.h"
#include "nn/net.h"

namespace algo {

struct Transition {
    std::array<uint8_t, ddz::kImpBin> imp{};
    std::array<float, ddz::kImpScalars> scalar{};
    std::array<uint8_t, ddz::kExtraBin> extra{};
    std::array<float, ddz::kExtraScalars> extraScalar{};
    std::array<uint64_t, 10> mask{};  // 621 action bits
    // dynamic features for the legal actions (abstract id -> 7 values)
    std::vector<std::pair<int, std::array<float, ddz::kActionDyn>>> dyn;
    int action = -1;
    int gameId = -1;
    bool terminal = false;
    float logp = 0.0f;
    float value = 0.0f;
    float reward = 0.0f;
    float adv = 0.0f;
    float ret = 0.0f;
};

struct RolloutStats {
    int games = 0;
    int landlordWins = 0;
    long long landlordScore = 0;  // ADP accumulated for the landlord camp
    long long bombs = 0;
    long long moves = 0;
    std::array<long long, 3> transitions{};
};

struct ModelSet {
    std::array<nn::Actor*, 3> actor{};
    std::array<nn::Critic*, 3> critic{};
};

struct RolloutConfig {
    int gamesPerUpdate = 256;
    int threads = 4;
    float rewardScale = 50.0f;  // l in the paper; 0 disables oracle shaping
    uint64_t seed = 1;
};

// Runs self-play games using the current models.
void collectRollout(const ModelSet& models, const RolloutConfig& cfg,
                    std::array<std::vector<Transition>, 3>& out,
                    RolloutStats& stats);

// Assembles the network input batch for a set of transitions.
struct Batch {
    nn::Mat xImp, seq, mask, dynFeat, extra;
};
void buildBatch(const std::vector<Transition*>& tr, nn::Mat& xImp,
                nn::Mat& seq, nn::Mat& mask, nn::Mat& dynFeat,
                nn::Mat& extra);

}  // namespace algo
