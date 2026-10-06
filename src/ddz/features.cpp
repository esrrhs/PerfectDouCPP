#include "ddz/features.h"

#include <algorithm>
#include <unordered_set>

#include "ddz/oracle.h"

namespace ddz {

namespace {

// row offsets inside the 12 x 15 card matrix
constexpr int R_COUNT0 = 0;  // count rows 0..3
constexpr int R_SOLO = 4;
constexpr int R_PAIR = 5;
constexpr int R_TRIO = 6;
constexpr int R_BOMB = 7;
constexpr int R_ROCKET = 8;
constexpr int R_SOLOCHAIN = 9;
constexpr int R_PAIRCHAIN = 10;
constexpr int R_TRIOCHAIN = 11;

bool inChain(const int8_t* cnt, int r, int minLen, int need) {
    // expand left/right over consecutive numeric ranks with enough copies;
    // only ranks 3..A (0..11) participate.
    int lo = r, hi = r;
    while (lo - 1 >= 0 && cnt[lo - 1] >= need &&
           rankValue(lo) - rankValue(lo - 1) == 1)
        --lo;
    while (hi + 1 < kNormalRanks && cnt[hi + 1] >= need &&
           rankValue(hi + 1) - rankValue(hi) == 1)
        ++hi;
    return hi - lo + 1 >= minLen;
}

}  // namespace

void cardMatrix(const CardSet& cs, uint8_t* out) {
    std::fill(out, out + kCardMat, 0);
    // count rows 0..3
    for (int r = 0; r < kRanks; ++r) {
        for (int k = 0; k < cs.c[r]; ++k)
            out[(R_COUNT0 + k) * kRanks + r] = 1;
        if (cs.c[r] >= 1) out[R_SOLO * kRanks + r] = 1;
        if (cs.c[r] >= 2) out[R_PAIR * kRanks + r] = 1;
        if (cs.c[r] >= 3) out[R_TRIO * kRanks + r] = 1;
        if (r < kNormalRanks && cs.c[r] == 4) out[R_BOMB * kRanks + r] = 1;
    }
    if (cs.c[13] && cs.c[14]) {
        out[R_ROCKET * kRanks + 13] = 1;
        out[R_ROCKET * kRanks + 14] = 1;
    }
    for (int r = 0; r < kNormalRanks; ++r) {
        if (inChain(cs.c.data(), r, kMinStraight, 1))
            out[R_SOLOCHAIN * kRanks + r] = 1;
        if (inChain(cs.c.data(), r, kMinPairChain, 2))
            out[R_PAIRCHAIN * kRanks + r] = 1;
        if (inChain(cs.c.data(), r, kMinPlane, 3))
            out[R_TRIOCHAIN * kRanks + r] = 1;
    }
}

void actionCardMatrix(int abstractId, std::array<uint8_t, kCardMat>& out) {
    const AbstractAction& a = abstractTable()[abstractId];
    cardMatrix(a.main, out.data());
}

EncodedState encodeState(const Game& g) {
    EncodedState e;
    int seat = g.turn;
    int prev = g.prevSeat();
    int nxt = g.nextSeat();

    auto put = [&](int slot, const CardSet& cs) {
        cardMatrix(cs, e.imp.data() + slot * kCardMat);
    };
    CardSet others;
    for (int r = 0; r < kRanks; ++r)
        others.add(r, g.hand[prev].c[r] + g.hand[nxt].c[r]);
    put(0, g.hand[seat]);
    put(1, others);
    put(2, g.played[seat]);
    put(3, g.played[prev]);
    put(4, g.played[nxt]);
    put(5, g.bottom);
    put(6, g.lastMove[seat]);
    put(7, g.lastMove[prev]);
    put(8, g.lastMove[nxt]);

    // last 15 moves, chronological, zero-padded at the front
    int hlen = std::min<int>(kHistoryLen, g.seq.size());
    for (int i = 0; i < hlen; ++i) {
        const CardSet& m = g.seq[g.seq.size() - hlen + i];
        put(kStaticMatrices + (kHistoryLen - hlen) + i, m);
    }

    auto oneHot = [](float* dst, int n, int value) {
        if (value >= 0 && value < n) dst[value] = 1.0f;
    };
    oneHot(e.scalar.data(), kHandHot, g.hand[prev].total() - 1);
    oneHot(e.scalar.data() + kHandHot, kHandHot, g.hand[nxt].total() - 1);
    oneHot(e.scalar.data() + 2 * kHandHot, kBombHot,
           std::min(g.bombCount, kBombHot - 1));
    e.scalar[kImpScalars - 1] = (g.lastPlayer == seat) ? 1.0f : 0.0f;

    // perfect information extras
    cardMatrix(g.hand[prev], e.extra.data());
    cardMatrix(g.hand[nxt], e.extra.data() + kCardMat);
    e.extraScalar[0] = float(minSteps(g.hand[prev])) / 20.0f;
    e.extraScalar[1] = float(minSteps(g.hand[nxt])) / 20.0f;
    return e;
}

namespace {

bool isLargest(const AbstractAction& a) {
    switch (a.kind) {
        case MT_ROCKET: return true;
        case MT_SINGLE: return a.rank == 14;  // red joker
        case MT_BOMB:
        case MT_PAIR:
        case MT_TRIPLE:
        case MT_TRIPLE_ONE:
        case MT_TRIPLE_TWO:
        case MT_FOUR_TWO:
        case MT_FOUR_TWO_PAIR:
            return a.rank == 12;  // rank 2
        case MT_STRAIGHT:
        case MT_PAIR_STRAIGHT:
        case MT_PLANE:
        case MT_PLANE_SOLO:
        case MT_PLANE_PAIR:
            return a.rank + a.len - 1 == 11;  // ends with A
        default:
            return false;
    }
}

}  // namespace

std::vector<LegalOption> legalOptions(const Game& g) {
    std::vector<CardSet> concrete = g.legal();
    int seat = g.turn;
    int prev = g.prevSeat();
    int nxt = g.nextSeat();

    std::vector<LegalOption> out;
    std::unordered_set<int> seen;
    for (const CardSet& m : concrete) {
        MoveInfo info = detectMove(m);
        int id = concreteToAbstract(m, info);
        // The DouZero generator can produce a few degenerate combinations that
        // its own detector labels WRONG (not legal under Tencent rules); skip.
        if (id >= 0) seen.insert(id);
        // The official 27,472 -> 621 mapping is one-to-many for ambiguous
        // planes (e.g. four consecutive trios can also be a shorter plane
        // with trio kickers). Preserve all official abstract choices.
        for (const AbstractAction& a : abstractTable())
            if (a.hasKicker && abstractMatches(a, m)) seen.insert(a.id);
    }
    std::vector<int> ids(seen.begin(), seen.end());
    std::sort(ids.begin(), ids.end());
    for (int id : ids) {
        CardSet chosen = decodeConcrete(id, concrete, g.hand[seat]);
        LegalOption o;
        o.abstractId = id;
        o.concrete = chosen;
        const AbstractAction& a = abstractTable()[id];
        int size = chosen.total();
        // The network has to see the kickers that will actually be played.
        std::array<uint8_t, kCardMat> card{};
        cardMatrix(chosen, card.data());
        for (int j = 0; j < kCardMat; ++j) o.feature[j] = float(card[j]);
        float* ex = o.feature.data() + kCardMat;
        ex[0] = (a.kind == MT_BOMB || a.kind == MT_ROCKET) ? 1.0f : 0.0f;
        ex[1] = isLargest(a) ? 1.0f : 0.0f;
        ex[2] = (size == g.hand[nxt].total()) ? 1.0f : 0.0f;
        ex[3] = (size == g.hand[prev].total()) ? 1.0f : 0.0f;
        CardSet after = g.hand[seat];
        after.sub(chosen);
        ex[4] = float(minSteps(after)) / 20.0f;
        ex[5] = 1.0f;  // validity marker, needed for the all-zero pass matrix
        out.push_back(std::move(o));
    }
    return out;
}

}  // namespace ddz
