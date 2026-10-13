#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "ddz/cards.h"
#include "torch_nn/models.h"

namespace algo {

using ActorGroup = std::array<torch_nn::Actor, 3>;

struct DouZeroEvalConfig {
    std::string host = "127.0.0.1";
    int port = 18765;
    int decks = 100;             // 100 decks = 200 games (statistically robust WP/ADP)
    uint64_t seed = 0;           // 0 means auto seed
    std::string csvPath = "";    // if not empty, append to CSV
    std::string saveDir = "";    // if not empty, save cloned models to disk for backtracking
    std::string dataPath = "";   // path to eval_data.bin or eval_data.pkl; if empty, uses default or random
    int maxSnapshots = 0;        // 0 = keep all, >0 = retain most recent N snapshots to protect disk space
    std::string label = "";      // label or tag
    int update = 0;              // training update step
    double elapsedMinutes = 0.0; // elapsed minutes since training start
    bool verbose = false;        // print progress during play
};

struct EvalDeck {
    ddz::CardSet landlord;
    ddz::CardSet landlordDown;
    ddz::CardSet landlordUp;
    ddz::CardSet bottom;
};

// Loads pre-generated decks from a binary dataset file (or auto-converts from .pkl if needed).
std::vector<EvalDeck> loadEvalDecks(const std::string& path, int maxDecks = 0);

struct DouZeroEvalResult {
    bool ok = false;
    int decks = 0;
    int games = 0;
    double wp = 0.0;
    double adp = 0.0;
    double wpLandlord = 0.0;
    double adpLandlord = 0.0;
    double wpPeasant = 0.0;
    double adpPeasant = 0.0;
    std::string error;
};

// Deep copies 3 actor models into a heap-allocated ActorGroup.
std::shared_ptr<ActorGroup> cloneActors(const std::array<torch_nn::Actor, 3>& src);

// Evaluates the given actors against DouZero TCP server.
DouZeroEvalResult evaluateAgainstDouZero(const std::array<torch_nn::Actor, 3>& actors,
                                        const DouZeroEvalConfig& cfg);

DouZeroEvalResult evaluateAgainstDouZero(std::shared_ptr<const ActorGroup> actors,
                                        const DouZeroEvalConfig& cfg);

}  // namespace algo
