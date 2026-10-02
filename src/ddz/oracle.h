// Oracle: minimum number of moves to play out all cards of a hand.
//
// Implements paper Appendix E.1 (Algorithm 1): a count-based dynamic program
// for all non-chain plays combined with a memoized DFS that enumerates
// straights, pair straights and planes (with wings). Used for:
//   - the shaped "distance to win" reward (Sec. 4.4)
//   - perfect/imperfect node features and action features.
#pragma once

#include <cstdint>

#include "ddz/cards.h"

namespace ddz {

// Minimum play-out steps for the hand. Results are memoized globally.
int minSteps(const CardSet& hand);

// Clears the global memoization table (e.g. between training iterations to
// bound memory).
void clearOracleCache();

// Current cache size (number of memoized states).
size_t oracleCacheSize();

}  // namespace ddz
