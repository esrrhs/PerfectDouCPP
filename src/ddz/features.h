// Node / action feature engineering, following PerfectDou paper Tables 1, 5, 6.
//
// Imperfect node features (observable by the acting player):
//   23 card matrices of 12 x 15 = 4140 binaries
//     hand, unplayed cards, my/prev/next played cards, the 3 bottom cards,
//     last 15 moves, prev player's last move, next player's last move
//   + 6 scalars (min play-out steps, three hand sizes, bomb count, control)
// Additional perfect features (critic only):
//   2 card matrices (prev hand, next hand) + 2 min-step scalars = 362
//
// Action features: 12 x 15 matrix of the abstract action's main cards plus
// 7 dynamic scalars (paper Table 6).
#pragma once

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include "ddz/action_space.h"
#include "ddz/cards.h"
#include "ddz/game.h"

namespace ddz {

constexpr int kCardMat = 12 * kRanks;  // 180
constexpr int kHistoryLen = 15;
constexpr int kImpMatrices = 23;
constexpr int kImpBin = kImpMatrices * kCardMat;  // 4140
constexpr int kImpScalars = 6;
constexpr int kImpSize = kImpBin + kImpScalars;  // 4146
constexpr int kExtraBin = 2 * kCardMat;          // 360
constexpr int kExtraScalars = 2;
constexpr int kExtraSize = kExtraBin + kExtraScalars;  // 362
constexpr int kActionDyn = 7;

struct EncodedState {
    std::array<uint8_t, kImpBin> imp{};
    std::array<float, kImpScalars> scalar{};
    std::array<uint8_t, kExtraBin> extra{};
    std::array<float, kExtraScalars> extraScalar{};
};

struct LegalOption {
    int abstractId;
    CardSet concrete;
    std::array<float, kActionDyn> dyn{};
};

// Encodes the node for the player whose turn it currently is.
EncodedState encodeState(const Game& g);

// Builds legal abstract options for the player whose turn it is, decoding
// each abstract action to the concrete move that will actually be played.
std::vector<LegalOption> legalOptions(const Game& g);

// Writes the static 12x15 matrix of an abstract action into `out`.
void actionCardMatrix(int abstractId, std::array<uint8_t, kCardMat>& out);

// Writes the generic 12x15 card matrix for any card set.
void cardMatrix(const CardSet& cs, uint8_t* out);

}  // namespace ddz
