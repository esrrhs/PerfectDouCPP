// Dump lead moves and follow-moves for random hands, one case per line.
// Used to diff the C++ engine against DouZero's Python move generator.
#include <cstdio>
#include <string>
#include <vector>

#include "ddz/cards.h"
#include "ddz/moves.h"

using namespace ddz;

static void printMove(const CardSet& m) {
    std::string s;
    for (int r = 0; r < kRanks; ++r)
        for (int k = 0; k < m.c[r]; ++k) {
            if (!s.empty()) s.push_back(',');
            s += std::to_string(rankValue(r));
        }
    std::printf("%s\n", s.c_str());
}

static CardSet handFromCounts(const int* c) {
    CardSet h;
    for (int r = 0; r < kRanks; ++r) h.add(r, c[r]);
    return h;
}

int main() {
    int n = 0;
    if (std::scanf("%d", &n) != 1) return 1;
    for (int i = 0; i < n; ++i) {
        char kind[8];
        if (std::scanf("%7s", kind) != 1) return 1;
        int c[kRanks];
        for (int r = 0; r < kRanks; ++r)
            if (std::scanf("%d", &c[r]) != 1) return 1;
        CardSet hand = handFromCounts(c);
        std::vector<CardSet> moves;
        if (kind[0] == 'L') {
            moves = genAllMoves(hand);
        } else {
            int b[kRanks];
            for (int r = 0; r < kRanks; ++r)
                if (std::scanf("%d", &b[r]) != 1) return 1;
            moves = legalMoves(hand, handFromCounts(b));
        }
        std::printf("%d\n", (int)moves.size());
        for (const CardSet& m : moves) printMove(m);
    }
    return 0;
}
