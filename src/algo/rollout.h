// Batched self-play rollout collection.
//
// A pool of games is advanced in lockstep. At every tick all games waiting on
// the same seat are evaluated in one batched forward pass. Each seat keeps
// its own transition stream for PPO/GAE.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
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
    // Released-model action representation (abstract id -> 12x15 + 6).
    std::vector<std::pair<int, std::array<float, ddz::kActionSize>>> actions;
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
    int pureSelfPlayGames = 0;
    int pureSelfPlayWins = 0;
    long long pureSelfPlayScore = 0;
    long long bombs = 0;
    long long moves = 0;
    std::array<long long, 3> transitions{};
};

struct ModelSet {
    std::array<nn::Actor*, 3> actor{};
    std::array<nn::Critic*, 3> critic{};
};

struct HistoricalActorSnapshot {
    std::array<nn::Actor, 3> actor;
    int update = 0;  // Update index at which this snapshot was created
};

// Hierarchical League Pool for long-term (multi-billion sample) training.
// Maintains both recent high-frequency rolling snapshots (recent tactical evolution)
// and exponentially/geometrically spaced archive snapshots (preventing catastrophic forgetting).
struct HistoricalPool {
    std::vector<std::shared_ptr<HistoricalActorSnapshot>> recent;
    std::vector<std::shared_ptr<HistoricalActorSnapshot>> archive;

    bool empty() const {
        return recent.empty() && archive.empty();
    }
};

struct RolloutConfig {
    int gamesPerUpdate = 256;
    int threads = 4;
    uint64_t seed = 1;
    // League opponent ratios
    float historicalProb = 0.2f;  // Probability of facing a historical model snapshot
    float ruleProb = 0.1f;        // Probability of facing heuristic rule agent
    // Pointer to read-only historical pool (if available)
    const HistoricalPool* historicalPool = nullptr;
    // Terminal shaping cap (nudge towards fewer opponent cards/clearing hand, small to prevent distorting ADP)
    float shapingCap = 0.05f;
};

// Runs self-play / league games using the current models, historical pool, and rule agent.
void collectRollout(const ModelSet& models, const RolloutConfig& cfg,
                    std::array<std::vector<Transition>, 3>& out,
                    RolloutStats& stats);

// Assembles the network input batch for a set of transitions.
struct Batch {
    nn::Mat xImp, seq, actionFeat, actionSample, actionId, actionOffset, extra;
};
// Float features for one stream, built once and reused across PPO epochs.
void cacheTransitionFeatures(const std::vector<Transition>& tr,
                             std::vector<float>& xImp, std::vector<float>& seq,
                             std::vector<float>& extra);
void buildBatch(const std::vector<Transition*>& tr, nn::Mat& xImp,
                nn::Mat& seq, nn::Mat& actionFeat, nn::Mat& actionSample,
                nn::Mat& actionId, nn::Mat& actionOffset, nn::Mat& extra);
// `base` is the stream the cached rows were built from. Each pointer in
// `tr` must address an element of that stream.
void buildBatchFromCache(const std::vector<Transition*>& tr,
                         const Transition* base, const float* xAll, int xStride,
                         const float* seqAll, const float* eAll, int eStride,
                         nn::Mat& xImp, nn::Mat& seq, nn::Mat& actionFeat,
                         nn::Mat& actionSample, nn::Mat& actionId,
                         nn::Mat& actionOffset, nn::Mat& extra);

}  // namespace algo
