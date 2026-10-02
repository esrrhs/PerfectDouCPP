// Card-play phase game engine (bidding is omitted, as in the PerfectDou
// paper / DouZero evaluation: the landlord is dealt 20 cards and leads).
//
// Seat order (play order): 0 = landlord, 1 = landlord_down, 2 = landlord_up.
#pragma once

#include <array>
#include <vector>

#include "ddz/cards.h"
#include "ddz/moves.h"

namespace ddz {

constexpr int kLandlord = 0;
constexpr int kDown = 1;
constexpr int kUp = 2;

struct Game {
    std::array<CardSet, 3> hand;
    std::array<CardSet, 3> played;
    std::array<CardSet, 3> lastMove;  // last move per seat (cleared on pass)
    CardSet bottom;                   // the three face-up landlord cards

    std::vector<CardSet> seq;  // full action sequence including passes
    int turn = 0;              // acting seat
    int lastPlayer = 0;        // last seat that made a non-pass move
    int bombCount = 0;         // number of bombs / rockets shown
    bool over = false;
    int winner = -1;           // 0 = landlord, 1 = peasants

    void deal(Rng& rng);

    // The play that must currently be beaten; empty means free lead.
    CardSet toBeat() const;
    std::vector<CardSet> legal() const;

    // Applies the move (must be legal); handles pass, bomb counting, win test.
    void step(const CardSet& m);

    CardSet unplayed() const {  // cards still held by all players
        CardSet u;
        for (const CardSet& h : hand)
            for (int r = 0; r < kRanks; ++r) u.add(r, h.c[r]);
        return u;
    }
    int prevSeat() const { return (turn + 2) % 3; }
    int nextSeat() const { return (turn + 1) % 3; }

    // ADP payoff from the seat's perspective (base 2 for landlord, 1 for
    // each peasant, doubled per bomb/rocket).
    double payoff(int seat) const;
};

}  // namespace ddz
