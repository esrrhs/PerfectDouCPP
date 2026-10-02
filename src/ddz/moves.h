// Move type detection, move generation and legal-action filtering.
// Faithful C++ port of the DouZero / PerfectDou engine:
//   perfectdou/env/move_detector.py
//   perfectdou/env/move_generator.py
//   perfectdou/env/move_selector.py
//
// Tencent DouDizhu rule set:
//   solo, pair, triple, triple+solo, triple+pair, straight (>=5, no 2/jokers),
//   pair straight (>=3), plane (>=2 consecutive triples, may carry the same
//   number of solos or pairs), four+two solos, four+two pairs, bomb, rocket.
#pragma once

#include <vector>

#include "ddz/cards.h"

namespace ddz {

enum MoveType {
    MT_PASS = 0,
    MT_SINGLE = 1,
    MT_PAIR = 2,
    MT_TRIPLE = 3,
    MT_BOMB = 4,
    MT_ROCKET = 5,
    MT_TRIPLE_ONE = 6,       // 3 + 1
    MT_TRIPLE_TWO = 7,       // 3 + pair
    MT_STRAIGHT = 8,         // chain of solos
    MT_PAIR_STRAIGHT = 9,    // chain of pairs
    MT_PLANE = 10,           // chain of triples (no wings)
    MT_PLANE_SOLO = 11,      // chain of triples + solos
    MT_PLANE_PAIR = 12,      // chain of triples + pairs
    MT_FOUR_TWO = 13,        // four + two solos
    MT_FOUR_TWO_PAIR = 14,   // four + two pairs
    MT_WRONG = 15,
};

constexpr int kMinStraight = 5;   // minimum solos in a straight
constexpr int kMinPairChain = 3;  // minimum pairs in a pair straight
constexpr int kMinPlane = 2;      // minimum triples in a plane

struct MoveInfo {
    int type = MT_WRONG;
    int rank = -1;  // rank index of the main card (first/lowest rank for chains)
    int len = 1;    // chain length
};

MoveInfo detectMove(const CardSet& m);

// Generates every combination that can be formed from a hand (used when leading).
std::vector<CardSet> genAllMoves(const CardSet& hand);

// Legal moves against `toBeat`; when `toBeat` is empty the player leads freely.
// The returned list always contains the pass move when following a non-empty
// rival play.
std::vector<CardSet> legalMoves(const CardSet& hand, const CardSet& toBeat);

// Returns true if `m` is a bomb (four of a kind) or the rocket (both jokers).
bool isBombLike(const CardSet& m);

}  // namespace ddz
