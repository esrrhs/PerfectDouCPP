// Node / action feature engineering, following PerfectDou paper Tables 1, 5, 6.
//
// Imperfect node features (observable by the acting player):
//   39 card matrices of 12 x 15
//     own hand, the other two players' cards, my/prev/next played cards,
//     the 3 bottom cards, own/prev/next last move, last 30 moves
//   + one-hot hand counts (20+20), one-hot bomb count (15), control flag
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
constexpr int kHistoryLen = 30;
constexpr int kHistoryGroups = 10;  // 10 LSTM steps x 3 moves
constexpr int kStaticMatrices = 9;
// Nine current-state matrices plus the last 30 moves. The ninth static
// matrix is the current player's own last move.
constexpr int kImpMatrices = kStaticMatrices + kHistoryLen;
constexpr int kImpBin = kImpMatrices * kCardMat;  // 7020
constexpr int kHandHot = 20;   // DouZero: one-hot of cards left, index = count-1
constexpr int kBombHot = 15;   // DouZero: one-hot of bombs shown, including zero
constexpr int kImpScalars = kHandHot + kHandHot + kBombHot + 1;
constexpr int kImpSize = kImpBin + kImpScalars;  // stored transition feature
constexpr int kNodeBin = kStaticMatrices * kCardMat;  // 1620
constexpr int kNodeSize = kNodeBin + kImpScalars;     // 1676
constexpr int kExtraBin = 2 * kCardMat;          // 360
constexpr int kExtraScalars = 2;
constexpr int kExtraSize = kExtraBin + kExtraScalars;  // 362
constexpr int kActionExtra = 6;
constexpr int kActionSize = kCardMat + kActionExtra;  // 186

struct EncodedState {
    std::array<uint8_t, kImpBin> imp{};
    std::array<float, kImpScalars> scalar{};
    std::array<uint8_t, kExtraBin> extra{};
    std::array<float, kExtraScalars> extraScalar{};
};

struct LegalOption {
    int abstractId;
    CardSet concrete;
    std::array<float, kActionSize> feature{};
};

// Encodes the node for the player whose turn it currently is.
EncodedState encodeState(const Game& g);

// One option per concrete legal play. abstractId is the local slot 0..n-1.
std::vector<LegalOption> legalOptions(const Game& g);

// Writes the static 12x15 matrix of an abstract action into `out`.
void actionCardMatrix(int abstractId, std::array<uint8_t, kCardMat>& out);

// Writes the generic 12x15 card matrix for any card set.
void cardMatrix(const CardSet& cs, uint8_t* out);

}  // namespace ddz
