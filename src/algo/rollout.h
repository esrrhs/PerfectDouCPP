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
#include "torch_nn/models.h"

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
    std::array<torch_nn::Actor*, 3> actor{};
    std::array<torch_nn::Critic*, 3> critic{};
};

struct HistoricalActorSnapshot {
    std::array<torch_nn::Actor, 3> actor;
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
    // Standard symmetric ADP payoff for landlord (2.0x base mult for zero-sum game)
    float landlordWinMul = 2.0f;
    float landlordLossMul = 2.0f;
    // Sample peasant players independently (Seat 1 and Seat 2 can have different models/rules)
    bool independentPeasants = true;
};

// Runs self-play / league games using the current models, historical pool, and rule agent.
void collectRollout(const ModelSet& models, const RolloutConfig& cfg,
                    std::array<std::vector<Transition>, 3>& out,
                    RolloutStats& stats);

}  // namespace algo
