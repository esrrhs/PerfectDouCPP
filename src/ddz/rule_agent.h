// Lightweight heuristic rule agent for DouDizhu (League opponent).
// Provides an anchor baseline with fundamental DouDizhu card-play logic:
//   1. Free leading: prefer shedding chain/pair/solo combos from smallest ranks, save bombs.
//   2. Following rival: beat with smallest non-bomb legal play, don't bomb unless opponent is low.
//   3. Teammate cooperation: don't beat peasant partner's high cards, assist partner.
#pragma once

#include <vector>

#include "ddz/cards.h"
#include "ddz/game.h"
#include "ddz/moves.h"

namespace ddz {

class RuleAgent {
public:
    // Selects the concrete move from legal moves according to heuristic card-play rules.
    static CardSet selectMove(const Game& g, int seat,
                             const std::vector<CardSet>& legal) {
        if (legal.empty()) return CardSet();
        if (legal.size() == 1) return legal.front();

        const CardSet& toBeat = g.lastMove[(g.turn + 2) % 3].total() > 0
                                    ? g.lastMove[(g.turn + 2) % 3]
                                    : g.lastMove[(g.turn + 1) % 3];

        bool isLeading = toBeat.empty() || g.lastPlayer == seat;

        // If leading freely:
        if (isLeading) {
            // Find non-bomb moves
            std::vector<CardSet> nonBombs;
            for (const auto& m : legal) {
                if (!m.empty() && !isBombLike(m)) nonBombs.push_back(m);
            }
            if (!nonBombs.empty()) {
                // Pick the non-bomb move with lowest main rank, preferring multi-card combos
                size_t bestIdx = 0;
                int bestScore = 9999;
                for (size_t i = 0; i < nonBombs.size(); ++i) {
                    MoveInfo info = detectMove(nonBombs[i]);
                    // Score = rank * 10 - cards_count (lower is better, shedding more cards earlier)
                    int score = info.rank * 10 - nonBombs[i].total();
                    if (score < bestScore) {
                        bestScore = score;
                        bestIdx = i;
                    }
                }
                return nonBombs[bestIdx];
            }
            // If only bombs left, play smallest bomb
            return legal.front();
        }

        // Following rival move:
        int rivalSeat = g.lastPlayer;
        bool isTeammate = (seat != kLandlord) && (rivalSeat != kLandlord);

        // If rival is actually our peasant partner:
        if (isTeammate) {
            MoveInfo partnerInfo = detectMove(toBeat);
            // If partner played a high card (rank >= 10: 2, Joker, Ace, King), pass to let partner keep control
            if (partnerInfo.rank >= 10 || isBombLike(toBeat)) {
                // Pass if legal
                for (const auto& m : legal) {
                    if (m.empty()) return m;
                }
            }
        }

        // Check enemy hand count to decide if urgent to bomb
        int minEnemyHand = 20;
        if (seat == kLandlord) {
            minEnemyHand = std::min(g.hand[kDown].total(), g.hand[kUp].total());
        } else {
            minEnemyHand = g.hand[kLandlord].total();
        }

        // Separate non-bomb and bomb legal moves
        std::vector<CardSet> nonBombs;
        std::vector<CardSet> bombs;
        bool hasPass = false;
        for (const auto& m : legal) {
            if (m.empty()) {
                hasPass = true;
            } else if (isBombLike(m)) {
                bombs.push_back(m);
            } else {
                nonBombs.push_back(m);
            }
        }

        // Prefer smallest non-bomb move that beats rival
        if (!nonBombs.empty()) {
            size_t bestIdx = 0;
            int bestRank = 9999;
            for (size_t i = 0; i < nonBombs.size(); ++i) {
                MoveInfo info = detectMove(nonBombs[i]);
                if (info.rank < bestRank) {
                    bestRank = info.rank;
                    bestIdx = i;
                }
            }
            return nonBombs[bestIdx];
        }

        // If only bombs can beat it: only bomb if enemy is threatening to win (<= 3 cards)
        if (!bombs.empty()) {
            if (minEnemyHand <= 3 || !hasPass) {
                return bombs.front();  // smallest bomb
            }
        }

        // Otherwise pass
        for (const auto& m : legal) {
            if (m.empty()) return m;
        }
        return legal.front();
    }
};

}  // namespace ddz
