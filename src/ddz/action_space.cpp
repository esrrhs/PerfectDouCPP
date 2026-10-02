#include "ddz/action_space.h"

#include <algorithm>
#include <unordered_map>

namespace ddz {

namespace {

constexpr int kSolo0 = 0;        // 15
constexpr int kPair0 = 15;       // 13
constexpr int kTrio0 = 28;       // 13
constexpr int kTrioOne0 = 41;    // 13 * 14
constexpr int kTrioTwo0 = 223;   // 13 * 12
constexpr int kStraight0 = 379;  // 36
constexpr int kPairChain0 = 415; // 52
constexpr int kPlane0 = 467;     // 45
constexpr int kPlaneSolo0 = 512; // 38
constexpr int kPlanePair0 = 550; // 30
constexpr int kFourTwo0 = 580;   // 13
constexpr int kFourTwoPair0 = 593; // 13
constexpr int kBomb0 = 606;      // 13
constexpr int kRocketId = 619;
constexpr int kPassId = 620;

// number of chains of a given minimal length: starts for length L = 13 - L
// (12 consecutive ranks 3..A)
int chainOffset(int base, int len, int minLen) {
    int off = 0;
    for (int L = minLen; L < len; ++L) off += 13 - L;
    return base + off;
}

std::string rankName(int r) { return std::string(1, rankChar(r)); }

CardSet makeChainMain(int start, int len, int repeat) {
    CardSet m;
    for (int i = 0; i < len; ++i) m.add(start + i, repeat);
    return m;
}

struct Builder {
    std::vector<AbstractAction> v;

    void push(int kind, int rank, int len, bool kicker, const CardSet& main,
              std::string name) {
        AbstractAction a;
        a.id = static_cast<int>(v.size());
        a.kind = kind;
        a.rank = rank;
        a.len = len;
        a.hasKicker = kicker;
        a.main = main;
        a.name = std::move(name);
        v.push_back(std::move(a));
    }

    void build() {
        // (1) solo
        for (int r = 0; r < kRanks; ++r) {
            CardSet m;
            m.add(r);
            push(MT_SINGLE, r, 1, false, m, rankName(r));
        }
        // (2) pair
        for (int r = 0; r < kNormalRanks; ++r) {
            CardSet m;
            m.add(r, 2);
            push(MT_PAIR, r, 1, false, m, rankName(r) + rankName(r));
        }
        // (3) trio
        for (int r = 0; r < kNormalRanks; ++r) {
            CardSet m;
            m.add(r, 3);
            push(MT_TRIPLE, r, 1, false, m, std::string(3, rankChar(r)));
        }
        // (4) trio + solo (trio-major order, 182)
        for (int t = 0; t < kNormalRanks; ++t) {
            int kickerIndex = 0;
            for (int k = 0; k < kRanks; ++k) {
                if (k == t) continue;
                CardSet m;
                m.add(t, 3);
                m.add(k);
                push(MT_TRIPLE_ONE, t, 1, false, m,
                     std::string(3, rankChar(t)) + rankChar(k));
                ++kickerIndex;
            }
            (void)kickerIndex;
        }
        // (5) trio + pair (156)
        for (int t = 0; t < kNormalRanks; ++t) {
            for (int p = 0; p < kNormalRanks; ++p) {
                if (p == t) continue;
                CardSet m;
                m.add(t, 3);
                m.add(p, 2);
                push(MT_TRIPLE_TWO, t, 1, false, m,
                     std::string(3, rankChar(t)) + std::string(2, rankChar(p)));
            }
        }
        // (6) straight (36)
        for (int len = 5; len <= 12; ++len)
            for (int start = 0; start + len <= 12; ++start) {
                CardSet m = makeChainMain(start, len, 1);
                std::string nm;
                for (int i = 0; i < len; ++i) nm += rankChar(start + i);
                push(MT_STRAIGHT, start, len, false, m, nm);
            }
        // (7) pair straight (52)
        for (int len = 3; len <= 10; ++len)
            for (int start = 0; start + len <= 12; ++start) {
                CardSet m = makeChainMain(start, len, 2);
                std::string nm;
                for (int i = 0; i < len; ++i) nm += std::string(2, rankChar(start + i));
                push(MT_PAIR_STRAIGHT, start, len, false, m, nm);
            }
        // (8) plane without wings (45, length 2..6)
        for (int len = 2; len <= 6; ++len)
            for (int start = 0; start + len <= 12; ++start) {
                CardSet m = makeChainMain(start, len, 3);
                std::string nm;
                for (int i = 0; i < len; ++i) nm += std::string(3, rankChar(start + i));
                push(MT_PLANE, start, len, false, m, nm);
            }
        // (9) plane + solo (38 abstract, length 2..5)
        for (int len = 2; len <= 5; ++len)
            for (int start = 0; start + len <= 12; ++start) {
                CardSet m = makeChainMain(start, len, 3);
                std::string nm;
                for (int i = 0; i < len; ++i) nm += std::string(3, rankChar(start + i));
                nm += std::string(len, '*');
                push(MT_PLANE_SOLO, start, len, true, m, nm);
            }
        // (10) plane + pair (30 abstract, length 2..4)
        for (int len = 2; len <= 4; ++len)
            for (int start = 0; start + len <= 12; ++start) {
                CardSet m = makeChainMain(start, len, 3);
                std::string nm;
                for (int i = 0; i < len; ++i) nm += std::string(3, rankChar(start + i));
                nm += std::string(2 * len, '*');
                push(MT_PLANE_PAIR, start, len, true, m, nm);
            }
        // (11) four + two solos (13)
        for (int r = 0; r < kNormalRanks; ++r) {
            CardSet m;
            m.add(r, 4);
            push(MT_FOUR_TWO, r, 1, true, m, std::string(4, rankChar(r)) + "**");
        }
        // (12) four + two pairs (13)
        for (int r = 0; r < kNormalRanks; ++r) {
            CardSet m;
            m.add(r, 4);
            push(MT_FOUR_TWO_PAIR, r, 1, true, m,
                 std::string(4, rankChar(r)) + "****");
        }
        // (13) bomb
        for (int r = 0; r < kNormalRanks; ++r) {
            CardSet m;
            m.add(r, 4);
            push(MT_BOMB, r, 1, false, m, std::string(4, rankChar(r)));
        }
        // (14) rocket
        {
            CardSet m;
            m.add(13);
            m.add(14);
            push(MT_ROCKET, -1, 1, false, m, "BR");
        }
        // (15) pass
        push(MT_PASS, -1, 1, false, CardSet{}, "pass");
    }
};

const std::vector<AbstractAction>& table() {
    static const std::vector<AbstractAction> t = [] {
        Builder b;
        b.build();
        return std::move(b.v);
    }();
    return t;
}

int chainId(int base, int minLen, int start, int len) {
    return chainOffset(base, len, minLen) + start;
}

}  // namespace

const std::vector<AbstractAction>& abstractTable() { return table(); }

int concreteToAbstract(const CardSet& concrete, const MoveInfo& info) {
    switch (info.type) {
        case MT_PASS: return kPassId;
        case MT_SINGLE: return kSolo0 + info.rank;
        case MT_PAIR: return kPair0 + info.rank;
        case MT_TRIPLE: return kTrio0 + info.rank;
        case MT_BOMB: return kBomb0 + info.rank;
        case MT_ROCKET: return kRocketId;
        case MT_TRIPLE_ONE: {
            int kicker = -1;
            for (int r = 0; r < kRanks; ++r)
                if (r != info.rank && concrete.c[r] > 0) kicker = r;
            int idx = 0;
            for (int k = 0; k < kRanks; ++k) {
                if (k == info.rank) continue;
                if (k == kicker) return kTrioOne0 + info.rank * 14 + idx;
                ++idx;
            }
            return -1;
        }
        case MT_TRIPLE_TWO: {
            int pair = -1;
            for (int r = 0; r < kNormalRanks; ++r)
                if (r != info.rank && concrete.c[r] == 2) pair = r;
            int idx = 0;
            for (int k = 0; k < kNormalRanks; ++k) {
                if (k == info.rank) continue;
                if (k == pair) return kTrioTwo0 + info.rank * 12 + idx;
                ++idx;
            }
            return -1;
        }
        case MT_STRAIGHT: return chainId(kStraight0, 5, info.rank, info.len);
        case MT_PAIR_STRAIGHT: return chainId(kPairChain0, 3, info.rank, info.len);
        case MT_PLANE: return chainId(kPlane0, 2, info.rank, info.len);
        case MT_PLANE_SOLO: return chainId(kPlaneSolo0, 2, info.rank, info.len);
        case MT_PLANE_PAIR: return chainId(kPlanePair0, 2, info.rank, info.len);
        case MT_FOUR_TWO: return kFourTwo0 + info.rank;
        case MT_FOUR_TWO_PAIR: return kFourTwoPair0 + info.rank;
        default: return -1;
    }
}

int abstractSize(const AbstractAction& a) {
    switch (a.kind) {
        case MT_SINGLE: case MT_BOMB: case MT_PASS: return 1;
        case MT_PAIR: return 2;
        case MT_TRIPLE: return 3;
        case MT_TRIPLE_ONE: return 4;
        case MT_TRIPLE_TWO: case MT_STRAIGHT: return a.len == 1 ? 5 : a.len;
        case MT_PAIR_STRAIGHT: return 2 * a.len;
        case MT_PLANE: return 3 * a.len;
        case MT_PLANE_SOLO: return 4 * a.len;
        case MT_PLANE_PAIR: return 5 * a.len;
        case MT_FOUR_TWO: return 6;
        case MT_FOUR_TWO_PAIR: return 8;
        case MT_ROCKET: return 2;
        default: return a.main.total();
    }
}

CardSet decodeConcrete(int abstractId,
                       const std::vector<CardSet>& legalConcrete,
                       const CardSet& hand) {
    const AbstractAction& a = table()[abstractId];
    const CardSet& mainCards = a.main;

    // Directly unique: non-kicker actions have exactly one concrete realization.
    if (!a.hasKicker) {
        for (const CardSet& m : legalConcrete)
            if (m == mainCards) return m;
        return mainCards;
    }

    // Collect concrete candidates with this abstract id and their kickers.
    struct Cand {
        CardSet move;
        std::string kicker;
    };
    std::vector<Cand> candidates;
    for (const CardSet& m : legalConcrete) {
        MoveInfo mi = detectMove(m);
        if (concreteToAbstract(m, mi) != abstractId) continue;
        // kickers = multiset difference, expressed as a sorted string
        std::string kicker;
        for (int r = 0; r < kRanks; ++r) {
            int n = m.c[r] - mainCards.c[r];
            kicker.append(std::string(std::max(0, n), rankChar(r)));
        }
        candidates.push_back({m, kicker});
    }
    if (candidates.empty()) return mainCards;  // defensive, should not happen
    if (candidates.size() == 1) return candidates.front().move;

    // Algorithm 2: prefer the kicker that is contained in the fewest playable
    // combinations of the hand, breaking ties by low rank.
    std::vector<CardSet> playable = genAllMoves(hand);
    int best = 0;
    double bestScore = 1e18;
    for (size_t i = 0; i < candidates.size(); ++i) {
        int n = 0;
        for (const CardSet& p : playable) {
            if (p.str().find(candidates[i].kicker) != std::string::npos) ++n;
        }
        double rankSum = 0;
        for (char ch : candidates[i].kicker) rankSum += charRank(ch);
        double score = double(n) + 0.1 * rankSum;
        if (score < bestScore) {
            bestScore = score;
            best = static_cast<int>(i);
        }
    }
    return candidates[best].move;
}

}  // namespace ddz
