#include "ddz/moves.h"

#include <algorithm>
#include <unordered_set>

namespace ddz {

namespace {

// sorted numeric values contained in the move
std::vector<int> valuesOf(const CardSet& m) {
    std::vector<int> v;
    for (int r = 0; r < kRanks; ++r)
        for (int k = 0; k < m.c[r]; ++k) v.push_back(rankValue(r));
    return v;
}

bool continuous(const std::vector<int>& v) {
    for (size_t i = 1; i < v.size(); ++i)
        if (v[i] - v[i - 1] != 1) return false;
    return true;
}

// combinations of `num` POSITIONS of vals; equal-valued positions create
// duplicate multisets which are filtered by the caller (itertools.groupby).
template <class Fn>
void forEachCombination(const std::vector<int>& vals, int num, Fn&& fn) {
    int n = static_cast<int>(vals.size());
    if (num > n || num <= 0) return;
    std::vector<int> idx(num);
    for (int i = 0; i < num; ++i) idx[i] = i;
    while (true) {
        fn(idx);
        int i = num - 1;
        while (i >= 0 && idx[i] == i + n - num) --i;
        if (i < 0) break;
        ++idx[i];
        for (int j = i + 1; j < num; ++j) idx[j] = idx[j - 1] + 1;
    }
}

CardSet fromRanks(const std::vector<int>& ranks, int repeat) {
    CardSet m;
    for (int r : ranks) m.add(r, repeat);
    return m;
}

CardSet fromValues(const std::vector<int>& vals) {
    CardSet m;
    for (int v : vals) m.add(valueRank(v), 1);
    return m;
}

// ---------------------------------------------------------------------------
// Move generator (port of MovesGener)
// ---------------------------------------------------------------------------
class MoveGenerator {
public:
    explicit MoveGenerator(const CardSet& hand) : hand_(hand) {
        for (int r = 0; r < kRanks; ++r)
            if (hand.c[r] > 0) presentRanks_.push_back(r);
    }

    std::vector<CardSet> genAll() {
        std::vector<CardSet> out;
        auto append = [&](std::vector<CardSet> v) {
            for (auto& m : v) out.push_back(std::move(m));
            v.clear();
        };
        append(genSingles());
        append(genPairs());
        append(genTriples());
        append(genBombs());
        append(genRocket());
        append(genTripleSolo());
        append(genTriplePair());
        append(genSerials(1, kMinStraight, 0));    // straights
        append(genSerials(2, kMinPairChain, 0));   // pair straights
        append(genSerials(3, kMinPlane, 0));       // planes without wings
        append(genPlaneSolo(0));
        append(genPlanePair(0));
        append(genFourTwo());
        append(genFourTwoPair());
        dedupe(out);
        return out;
    }

    // Matching generators (with a fixed chain length when following a play).
    std::vector<CardSet> genByType(int type, int len = 0) {
        std::vector<CardSet> out;
        switch (type) {
            case MT_SINGLE: out = genSingles(); break;
            case MT_PAIR: out = genPairs(); break;
            case MT_TRIPLE: out = genTriples(); break;
            case MT_BOMB: {
                out = genBombs();
                auto rocket = genRocket();
                out.insert(out.end(), rocket.begin(), rocket.end());
                break;
            }
            case MT_TRIPLE_ONE: out = genTripleSolo(); break;
            case MT_TRIPLE_TWO: out = genTriplePair(); break;
            case MT_STRAIGHT: out = genSerials(1, kMinStraight, len); break;
            case MT_PAIR_STRAIGHT: out = genSerials(2, kMinPairChain, len); break;
            case MT_PLANE: out = genSerials(3, kMinPlane, len); break;
            case MT_PLANE_SOLO: out = genPlaneSolo(len); break;
            case MT_PLANE_PAIR: out = genPlanePair(len); break;
            case MT_FOUR_TWO: out = genFourTwo(); break;
            case MT_FOUR_TWO_PAIR: out = genFourTwoPair(); break;
            default: break;
        }
        dedupe(out);
        return out;
    }

private:
    CardSet hand_;
    std::vector<int> presentRanks_;

    static void dedupe(std::vector<CardSet>& moves) {
        std::unordered_set<uint64_t> seen;
        seen.reserve(moves.size() * 2);
        size_t w = 0;
        for (auto& m : moves) {
            if (seen.insert(m.key()).second) moves[w++] = std::move(m);
        }
        moves.resize(w);
    }

    std::vector<CardSet> genSingles() const {
        std::vector<CardSet> r;
        for (int rnk : presentRanks_) {
            CardSet m;
            m.add(rnk);
            r.push_back(m);
        }
        return r;
    }
    std::vector<CardSet> genPairs() const {
        std::vector<CardSet> r;
        for (int rnk = 0; rnk < kRanks; ++rnk)
            if (hand_.c[rnk] >= 2) {
                CardSet m;
                m.add(rnk, 2);
                r.push_back(m);
            }
        return r;
    }
    std::vector<CardSet> genTriples() const {
        std::vector<CardSet> r;
        for (int rnk = 0; rnk < kRanks; ++rnk)
            if (hand_.c[rnk] >= 3) {
                CardSet m;
                m.add(rnk, 3);
                r.push_back(m);
            }
        return r;
    }
    std::vector<CardSet> genBombs() const {
        std::vector<CardSet> r;
        for (int rnk = 0; rnk < kNormalRanks; ++rnk)
            if (hand_.c[rnk] == 4) {
                CardSet m;
                m.add(rnk, 4);
                r.push_back(m);
            }
        return r;
    }
    std::vector<CardSet> genRocket() const {
        std::vector<CardSet> r;
        if (hand_.c[13] && hand_.c[14]) {
            CardSet m;
            m.add(13);
            m.add(14);
            r.push_back(m);
        }
        return r;
    }
    std::vector<CardSet> genTripleSolo() const {
        std::vector<CardSet> r;
        for (int t = 0; t < kRanks; ++t) {
            if (hand_.c[t] < 3) continue;
            for (int k : presentRanks_) {
                if (k == t) continue;
                CardSet m;
                m.add(t, 3);
                m.add(k, 1);
                r.push_back(m);
            }
        }
        return r;
    }
    std::vector<CardSet> genTriplePair() const {
        std::vector<CardSet> r;
        for (int t = 0; t < kRanks; ++t) {
            if (hand_.c[t] < 3) continue;
            for (int p = 0; p < kNormalRanks; ++p) {
                if (p == t || hand_.c[p] < 2) continue;
                CardSet m;
                m.add(t, 3);
                m.add(p, 2);
                r.push_back(m);
            }
        }
        return r;
    }

    // repeat = 1 solos / 2 pairs / 3 triples; fixedLen 0 means every length.
    std::vector<CardSet> genSerials(int repeat, int minLen, int fixedLen) const {
        std::vector<int> ranks;
        for (int r = 0; r < kRanks; ++r) {
            int need = repeat;
            if (hand_.c[r] >= need) ranks.push_back(r);
            // Numeric-value continuity automatically excludes rank 2 / jokers:
            // this works because rank values are 3..14,17,20,30.
        }
        std::vector<std::pair<int, int>> runs;  // (start index in ranks, length)
        {
            int i = 0;
            while (i < static_cast<int>(ranks.size())) {
                int j = i;
                while (j + 1 < static_cast<int>(ranks.size()) &&
                       rankValue(ranks[j + 1]) - rankValue(ranks[j]) == 1)
                    ++j;
                runs.emplace_back(i, j - i + 1);
                i = j + 1;
            }
        }
        std::vector<CardSet> out;
        for (auto [start, longest] : runs) {
            if (longest < minLen) continue;
            if (fixedLen > 0) {
                if (longest < fixedLen) continue;
                for (int idx = 0; idx + fixedLen <= longest; ++idx) {
                    std::vector<int> chain(ranks.begin() + start + idx,
                                           ranks.begin() + start + idx + fixedLen);
                    out.push_back(fromRanks(chain, repeat));
                }
            } else {
                for (int len = minLen; len <= longest; ++len)
                    for (int idx = 0; idx + len <= longest; ++idx) {
                        std::vector<int> chain(ranks.begin() + start + idx,
                                               ranks.begin() + start + idx + len);
                        out.push_back(fromRanks(chain, repeat));
                    }
            }
        }
        return out;
    }

    std::vector<CardSet> genPlaneSolo(int fixedLen) const {
        std::vector<CardSet> planes = genSerials(3, kMinPlane, fixedLen);
        std::vector<CardSet> out;
        for (const CardSet& pl : planes) {
            std::vector<int> rest;  // actual cards (values), trio ranks removed
            int chainLen = 0;
            for (int r = 0; r < kRanks; ++r) {
                if (pl.c[r] == 3) {
                    ++chainLen;
                    continue;
                }
                for (int k = 0; k < hand_.c[r]; ++k) rest.push_back(rankValue(r));
            }
            forEachCombination(rest, chainLen, [&](const std::vector<int>& idx) {
                std::vector<int> vals;
                for (int i : idx) vals.push_back(rest[i]);
                CardSet m = pl;
                for (int v : vals) m.add(valueRank(v), 1);
                out.push_back(m);
            });
        }
        return out;
    }

    std::vector<CardSet> genPlanePair(int fixedLen) const {
        std::vector<CardSet> planes = genSerials(3, kMinPlane, fixedLen);
        std::vector<int> pairRanks;
        for (int r = 0; r < kNormalRanks; ++r)
            if (hand_.c[r] >= 2) pairRanks.push_back(r);
        std::vector<CardSet> out;
        for (const CardSet& pl : planes) {
            std::vector<int> candidates;
            int chainLen = 0;
            for (int r = 0; r < kRanks; ++r) {
                if (pl.c[r] == 3) {
                    ++chainLen;
                    continue;
                }
                if (hand_.c[r] >= 2 && r < kNormalRanks) candidates.push_back(r);
            }
            forEachCombination(candidates, chainLen, [&](const std::vector<int>& idx) {
                CardSet m = pl;
                for (int i : idx) m.add(candidates[i], 2);
                out.push_back(m);
            });
        }
        return out;
    }

    std::vector<CardSet> genFourTwo() const {
        std::vector<CardSet> out;
        for (int f = 0; f < kNormalRanks; ++f) {
            if (hand_.c[f] != 4) continue;
            std::vector<int> rest;  // actual cards, the four-of-kind removed
            for (int r = 0; r < kRanks; ++r) {
                int n = hand_.c[r] - (r == f ? 4 : 0);
                for (int k = 0; k < n; ++k) rest.push_back(rankValue(r));
            }
            forEachCombination(rest, 2, [&](const std::vector<int>& idx) {
                CardSet m;
                m.add(f, 4);
                m.add(valueRank(rest[idx[0]]), 1);
                m.add(valueRank(rest[idx[1]]), 1);
                out.push_back(m);
            });
        }
        return out;
    }

    std::vector<CardSet> genFourTwoPair() const {
        std::vector<CardSet> out;
        for (int f = 0; f < kNormalRanks; ++f) {
            if (hand_.c[f] != 4) continue;
            std::vector<int> pairRanks;
            for (int r = 0; r < kNormalRanks; ++r)
                if (r != f && hand_.c[r] >= 2) pairRanks.push_back(r);
            forEachCombination(pairRanks, 2, [&](const std::vector<int>& idx) {
                CardSet m;
                m.add(f, 4);
                m.add(pairRanks[idx[0]], 2);
                m.add(pairRanks[idx[1]], 2);
                out.push_back(m);
            });
        }
        return out;
    }
};

}  // namespace

MoveInfo detectMove(const CardSet& m) {
    MoveInfo info;
    int size = m.total();

    if (size == 0) {
        info.type = MT_PASS;
        return info;
    }
    if (size == 1) {
        for (int r = 0; r < kRanks; ++r)
            if (m.c[r]) {
                info.type = MT_SINGLE;
                info.rank = r;
                return info;
            }
    }
    if (size == 2) {
        for (int r = 0; r < kNormalRanks; ++r)
            if (m.c[r] == 2) {
                info.type = MT_PAIR;
                info.rank = r;
                return info;
            }
        if (m.c[13] == 1 && m.c[14] == 1) {
            info.type = MT_ROCKET;
            return info;
        }
        return info;  // WRONG
    }
    if (size == 3) {
        for (int r = 0; r < kRanks; ++r)
            if (m.c[r] == 3) {
                info.type = MT_TRIPLE;
                info.rank = r;
                return info;
            }
        return info;
    }
    if (size == 4) {
        for (int r = 0; r < kNormalRanks; ++r)
            if (m.c[r] == 4) {
                info.type = MT_BOMB;
                info.rank = r;
                return info;
            }
        // exactly two distinct ranks: triple + solo
        int distinct = 0, triRank = -1, soloRank = -1;
        for (int r = 0; r < kRanks; ++r) {
            if (m.c[r] == 3) { triRank = r; ++distinct; }
            else if (m.c[r] == 1) { soloRank = r; ++distinct; }
            else if (m.c[r] != 0) return info;
        }
        if (distinct == 2 && triRank >= 0) {
            info.type = MT_TRIPLE_ONE;
            info.rank = triRank;  // rank of the triple (sorted[1] in DouZero)
            return info;
        }
        return info;
    }

    std::vector<int> vals = valuesOf(m);
    if (continuous(vals)) {
        info.type = MT_STRAIGHT;
        info.rank = valueRank(vals.front());
        info.len = size;
        return info;
    }

    if (size == 5) {
        int distinct = 0, triRank = -1;
        for (int r = 0; r < kRanks; ++r) {
            if (m.c[r] == 3) { triRank = r; ++distinct; }
            else if (m.c[r] == 2) ++distinct;
            else if (m.c[r] != 0) return info;
        }
        if (distinct == 2 && triRank >= 0) {
            info.type = MT_TRIPLE_TWO;
            info.rank = triRank;
            return info;
        }
        return info;
    }

    // count histogram keyed by multiplicity: how many ranks have k copies
    int byCount[5] = {0, 0, 0, 0, 0};
    int distinct = 0;
    for (int r = 0; r < kRanks; ++r)
        if (m.c[r] > 0) {
            ++byCount[m.c[r]];
            ++distinct;
        }

    if (size == 6) {
        bool shape = (distinct == 2 || distinct == 3) && byCount[4] == 1 &&
                     (byCount[2] == 1 || byCount[1] == 2);
        if (shape) {
            info.type = MT_FOUR_TWO;
            for (int r = 0; r < kRanks; ++r)
                if (m.c[r] == 4) { info.rank = r; break; }
            return info;
        }
    }
    if (size == 8) {
        bool shape = ((distinct == 3 || distinct == 2) && byCount[4] == 1 &&
                      byCount[2] == 2) ||
                     byCount[4] == 2;
        if (shape) {
            info.type = MT_FOUR_TWO_PAIR;
            for (int r = kRanks - 1; r >= 0; --r)
                if (m.c[r] == 4) { info.rank = r; break; }
            return info;
        }
    }

    std::vector<int> present;
    for (int r = 0; r < kRanks; ++r)
        if (m.c[r] > 0) present.push_back(r);

    // all pairs, consecutive ranks -> pair straight
    if (distinct == byCount[2]) {
        std::vector<int> pairVals;
        for (int r : present) pairVals.push_back(rankValue(r));
        if (continuous(pairVals)) {
            info.type = MT_PAIR_STRAIGHT;
            info.rank = present.front();
            info.len = distinct;
            return info;
        }
    }
    // all triples, consecutive -> plane
    if (distinct == byCount[3]) {
        std::vector<int> trioVals;
        for (int r : present) trioVals.push_back(rankValue(r));
        if (continuous(trioVals)) {
            info.type = MT_PLANE;
            info.rank = present.front();
            info.len = distinct;
            return info;
        }
    }

    // planes with wings
    if (byCount[3] >= kMinPlane) {
        std::vector<int> triRanks, soloRanks, pairRanks;
        bool bad = false;
        for (int r = 0; r < kRanks; ++r) {
            if (m.c[r] == 3) triRanks.push_back(r);
            else if (m.c[r] == 1) soloRanks.push_back(r);
            else if (m.c[r] == 2) pairRanks.push_back(r);
            else if (m.c[r] != 0) { bad = true; break; }
        }
        if (!bad) {
            std::vector<int> triVals;
            for (int r : triRanks) triVals.push_back(rankValue(r));
            if (continuous(triVals)) {
                int nTri = static_cast<int>(triRanks.size());
                if (nTri == static_cast<int>(soloRanks.size()) +
                                2 * static_cast<int>(pairRanks.size())) {
                    info.type = MT_PLANE_SOLO;
                    info.rank = triRanks.front();
                    info.len = nTri;
                    return info;
                }
                if (nTri == static_cast<int>(pairRanks.size()) &&
                    distinct == 2 * nTri) {
                    info.type = MT_PLANE_PAIR;
                    info.rank = triRanks.front();
                    info.len = nTri;
                    return info;
                }
            }
            // DouZero edge case: four consecutive ranks each with 3 cards may
            // be interpreted as a length-3 plane (the 4th triple supplies the
            // three solo kickers).
            if (triRanks.size() == 4) {
                auto checkSub = [&](int from) {
                    std::vector<int> sub(triRanks.begin() + from,
                                         triRanks.begin() + from + 3);
                    std::vector<int> sv;
                    for (int r : sub) sv.push_back(rankValue(r));
                    return continuous(sv);
                };
                if (checkSub(1)) {
                    info.type = MT_PLANE_SOLO;
                    info.rank = triRanks[1];
                    info.len = 3;
                    return info;
                }
                if (checkSub(0)) {
                    info.type = MT_PLANE_SOLO;
                    info.rank = triRanks[0];
                    info.len = 3;
                    return info;
                }
            }
        }
    }
    return info;  // WRONG
}

bool isBombLike(const CardSet& m) {
    MoveInfo i = detectMove(m);
    return i.type == MT_BOMB || i.type == MT_ROCKET;
}

std::vector<CardSet> genAllMoves(const CardSet& hand) {
    return MoveGenerator(hand).genAll();
}

namespace {

// beat filters (port of move_selector.py)
bool beats(const CardSet& move, const CardSet& rival, int type) {
    MoveInfo a = detectMove(move);
    MoveInfo b = detectMove(rival);
    if (a.type == MT_ROCKET) return true;
    if (a.type == MT_BOMB && type != MT_BOMB) return true;
    switch (type) {
        case MT_SINGLE:
        case MT_PAIR:
        case MT_TRIPLE:
        case MT_BOMB:
        case MT_STRAIGHT:
        case MT_PAIR_STRAIGHT:
        case MT_PLANE:
            return a.type == type && a.len == b.len && a.rank > b.rank;
        case MT_TRIPLE_ONE:
        case MT_TRIPLE_TWO:
        case MT_FOUR_TWO:
        case MT_FOUR_TWO_PAIR:
            return a.type == type && a.rank > b.rank;
        case MT_PLANE_SOLO:
        case MT_PLANE_PAIR: {
            if (a.type != type || a.len != b.len) return false;
            int ra = -1, rb = -1;
            for (int r = 0; r < kRanks; ++r) {
                if (move.c[r] == 3) ra = r;  // highest trio rank
                if (rival.c[r] == 3) rb = r;
            }
            return ra > rb;
        }
        default:
            return false;
    }
}

}  // namespace

std::vector<CardSet> legalMoves(const CardSet& hand, const CardSet& toBeat) {
    if (toBeat.empty()) {
        // free lead: any combination, no pass
        return genAllMoves(hand);
    }
    MoveInfo rival = detectMove(toBeat);
    if (rival.type == MT_ROCKET) {
        return {CardSet{}};  // nothing beats the rocket
    }
    MoveGenerator gen(hand);
    std::vector<CardSet> moves;
    if (rival.type == MT_BOMB) {
        // only a bigger bomb or the rocket
        for (CardSet& m : gen.genByType(MT_BOMB))
            if (beats(m, toBeat, MT_BOMB)) moves.push_back(std::move(m));
    } else {
        for (CardSet& m : gen.genByType(rival.type, rival.len))
            if (beats(m, toBeat, rival.type)) moves.push_back(std::move(m));
        // bombs / rocket beat every ordinary category
        for (CardSet& m : gen.genByType(MT_BOMB))
            if (beats(m, toBeat, rival.type)) moves.push_back(std::move(m));
    }
    moves.push_back(CardSet{});  // pass
    return moves;
}

}  // namespace ddz
