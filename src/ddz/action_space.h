// The 621 abstract actions used by the PerfectDou policy network.
//
// Concrete legal plays (27,472 possible combinations) are mapped to abstract
// actions: kickers of planes / four-with-kickers are abstracted away and are
// recovered by a decoding function (paper Appendix E.2, Algorithm 2, which
// follows the old RLCard implementation).
#pragma once

#include <string>
#include <vector>

#include "ddz/cards.h"
#include "ddz/moves.h"

namespace ddz {

constexpr int kAbstractActions = 621;

struct AbstractAction {
    int id;
    int kind;          // MoveType (MT_STRAIGHT, MT_PLANE_SOLO, ...), MT_PASS for pass
    int rank;          // main-card rank (lowest rank for chains), -1 for pass
    int len;           // chain length (1 otherwise)
    bool hasKicker;    // true for planes/four-with-kickers
    CardSet main;      // main cards (without kickers)
    std::string name;  // human readable template, e.g. "333444**"
};

const std::vector<AbstractAction>& abstractTable();

// Maps a concrete move to its abstract action id.
// Requires the MoveInfo of the concrete move (pass it in to avoid recompute).
int concreteToAbstract(const CardSet& concrete, const MoveInfo& info);

// Number of cards of a concrete play of an abstract action.
int abstractSize(const AbstractAction& a);

// Given the hand, the concrete legal moves and the chosen abstract action id,
// returns the concrete move to play (Algorithm 2 kicker decoding).
CardSet decodeConcrete(int abstractId,
                       const std::vector<CardSet>& legalConcrete,
                       const CardSet& hand);

}  // namespace ddz
