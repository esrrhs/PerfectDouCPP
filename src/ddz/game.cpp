#include "ddz/game.h"

#include <algorithm>

namespace ddz {

namespace {
CardSet sortedFromIndices(const std::array<int, kDeckSize>& deck, int lo, int hi) {
    CardSet m;
    for (int i = lo; i < hi; ++i) m.add(deck[i]);
    return m;
}
}  // namespace

void Game::deal(Rng& rng) {
    *this = Game{};
    auto deck = shuffledDeck(rng);
    // Same allocation as DouZero: landlord gets indices [0,20), the bottom is
    // [17,20), landlord_up [20,37), landlord_down [37,54).
    hand[kLandlord] = sortedFromIndices(deck, 0, 20);
    hand[kUp] = sortedFromIndices(deck, 20, 37);
    hand[kDown] = sortedFromIndices(deck, 37, 54);
    bottom = sortedFromIndices(deck, 17, 20);
    turn = 0;
    lastPlayer = 0;
}

CardSet Game::toBeat() const {
    if (seq.empty()) return CardSet{};
    if (!seq.back().empty()) return seq.back();
    // The last action was a pass: either two consecutive passes (free lead)
    // or the action before the pass must be beaten.
    if (seq.size() >= 2 && seq[seq.size() - 2].empty()) return CardSet{};
    if (seq.size() >= 2) return seq[seq.size() - 2];
    return CardSet{};
}

std::vector<CardSet> Game::legal() const {
    return legalMoves(hand[turn], toBeat());
}

void Game::step(const CardSet& m) {
    if (over) return;
    if (!m.empty()) {
        lastPlayer = turn;
        if (isBombLike(m)) ++bombCount;
        hand[turn].sub(m);
        // The released PerfectDou environment exposes the still-unplayed
        // subset of the three public bottom cards.
        if (turn == kLandlord) {
            CardSet used;
            for (int r = 0; r < kRanks; ++r)
                used.add(r, std::min(bottom.c[r], m.c[r]));
            bottom.sub(used);
        }
        for (int r = 0; r < kRanks; ++r) played[turn].add(r, m.c[r]);
        if (hand[turn].total() == 0) {
            over = true;
            winner = (turn == kLandlord) ? 0 : 1;
        }
    }
    lastMove[turn] = m;
    seq.push_back(m);
    if (!over) turn = (turn + 1) % 3;
}

float Game::shaping(int seat, float cap) const {
    bool won = (winner == 0) == (seat == kLandlord);
    if (won) {
        float enemyPlayed =
            seat == kLandlord
                ? 0.5f * float(played[kDown].total() + played[kUp].total())
                : float(played[kLandlord].total());
        return -cap * enemyPlayed / 20.0f;
    }
    return -cap * float(hand[seat].total()) / 20.0f;
}

double Game::payoff(int seat) const {
    double mult = 1.0;
    for (int i = 0; i < bombCount; ++i) mult *= 2.0;
    bool landlordWins = winner == 0;
    if (seat == kLandlord) return landlordWins ? 2.0 * mult : -2.0 * mult;
    return landlordWins ? -mult : mult;
}

}  // namespace ddz
