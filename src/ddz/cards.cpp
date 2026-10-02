#include "ddz/cards.h"

#include <algorithm>

namespace ddz {

namespace {
// Index -> DouZero numeric values (see cards.h).
constexpr int kValues[kRanks] = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 20, 30};
constexpr char kChars[kRanks] = {'3', '4', '5', '6', '7', '8', '9', 'T', 'J',
                                 'Q', 'K', 'A', '2', 'B', 'R'};
}  // namespace

int rankValue(int rankIndex) { return kValues[rankIndex]; }

int valueRank(int value) {
    switch (value) {
        case 17: return 12;
        case 20: return 13;
        case 30: return 14;
        default: return value - 3;  // 3..14 -> 0..11
    }
}

char rankChar(int rankIndex) { return kChars[rankIndex]; }

int charRank(char ch) {
    for (int r = 0; r < kRanks; ++r)
        if (kChars[r] == ch) return r;
    return -1;
}

std::string CardSet::str() const {
    std::string s;
    for (int r = 0; r < kRanks; ++r)
        s.append(std::string(c[r], kChars[r]));
    return s;
}

CardSet CardSet::parse(std::string_view s) {
    CardSet m;
    for (char ch : s) {
        int r = charRank(ch);
        if (r >= 0) m.c[r]++;
    }
    return m;
}

std::array<int, kDeckSize> shuffledDeck(Rng& rng) {
    std::array<int, kDeckSize> deck;
    for (int i = 0; i < kNormalRanks; ++i)
        for (int k = 0; k < 4; ++k) deck[i * 4 + k] = i;
    deck[52] = 13;  // black joker
    deck[53] = 14;  // red joker
    // Fisher-Yates
    for (int i = kDeckSize - 1; i > 0; --i) {
        int j = rng.below(i + 1);
        std::swap(deck[i], deck[j]);
    }
    return deck;
}

}  // namespace ddz
