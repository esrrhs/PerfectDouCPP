// Card primitives for DouDizhu.
//
// Ranks are indexed 0..14:
//   0=3 1=4 2=5 3=6 4=7 5=8 6=9 7=T 8=J 9=Q 10=K 11=A 12=2 13=B(black joker) 14=R(red joker)
//
// For move generation / type detection we additionally use the DouZero numeric
// card values {3..14, 17, 20, 30}: consecutive rank *values* automatically stop at
// A->2 (gap 3), 2->B (gap 3), B->R (gap 10), which is exactly the Tencent rule
// that straights/planes never include rank 2 or the jokers.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

namespace ddz {

constexpr int kRanks = 15;          // number of rank values
constexpr int kNormalRanks = 13;    // 3..2
constexpr int kDeckSize = 54;

// Rank index -> DouZero numeric card value.
int rankValue(int rankIndex);
// DouZero numeric card value -> rank index.
int valueRank(int value);

char rankChar(int rankIndex);
int  charRank(char c);

// A multiset of cards represented as counts of the 15 ranks.
// The empty set (all zero) denotes "pass".
struct CardSet {
    std::array<int8_t, kRanks> c{};

    int total() const {
        int s = 0;
        for (int x : c) s += x;
        return s;
    }
    bool empty() const { return total() == 0; }

    void add(int rank, int n = 1) { c[rank] = static_cast<int8_t>(c[rank] + n); }
    void sub(const CardSet& o) {
        for (int r = 0; r < kRanks; ++r) c[r] = static_cast<int8_t>(c[r] - o.c[r]);
    }
    bool contains(const CardSet& o) const {
        for (int r = 0; r < kRanks; ++r)
            if (c[r] < o.c[r]) return false;
        return true;
    }
    void clear() { c.fill(0); }

    bool operator==(const CardSet& o) const { return c == o.c; }
    bool operator!=(const CardSet& o) const { return !(*this == o); }

    std::string str() const;             // e.g. "33344BR" (ascending rank order)
    static CardSet parse(std::string_view s);

    // Stable 64-bit key (2 bits per rank, counts are in 0..4).
    uint64_t key() const {
        uint64_t k = 0;
        for (int r = 0; r < kRanks; ++r) k |= uint64_t(c[r]) << (2 * r);
        return k;
    }
};

// A shuffled deck has four copies of ranks 0..12 and one copy of each joker.
// Returns the 54 rank indices in shuffled order.
class Rng {
public:
    explicit Rng(uint64_t seed = 0) { state_ = seed ? seed : 0x9e3779b97f4a7c15ULL; }
    uint64_t nextU64() {
        // splitmix64
        uint64_t z = (state_ += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    // Uniform integer in [0, n).
    int below(int n) { return int(nextU64() % uint64_t(n)); }
    uint64_t state() const { return state_; }

private:
    uint64_t state_;
};

std::array<int, kDeckSize> shuffledDeck(Rng& rng);

}  // namespace ddz
