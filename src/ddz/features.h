// Node / action feature engineering, following PerfectDou paper Tables 1, 5, 6.
//
// Imperfect node features (observable by the acting player):
//   24 card matrices of 12 x 15
//     own hand, the other two players' cards, my/prev/next played cards,
//     the 3 bottom cards, own/prev/next last move, last 15 moves
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
constexpr int kHistoryLen = 15;
constexpr int kHistoryGroups = 5;  // official model: 5 LSTM steps x 3 moves
constexpr int kStaticMatrices = 9;
// The released model has 24 imperfect matrices: nine current-state matrices
// and the last 15 moves.  The ninth static matrix is the current player's
// own last move (the paper's table only lists the two opponents' last moves).
constexpr int kImpMatrices = kStaticMatrices + kHistoryLen;
constexpr int kImpBin = kImpMatrices * kCardMat;  // 4320
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

// Builds legal abstract options for the player whose turn it is, decoding
// each abstract action to the concrete move that will actually be played.
std::vector<LegalOption> legalOptions(const Game& g);

// Writes the static 12x15 matrix of an abstract action into `out`.
void actionCardMatrix(int abstractId, std::array<uint8_t, kCardMat>& out);

// Writes the generic 12x15 card matrix for any card set.
void cardMatrix(const CardSet& cs, uint8_t* out);

}  // namespace ddz
