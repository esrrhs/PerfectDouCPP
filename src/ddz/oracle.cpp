#include "ddz/oracle.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "ddz/moves.h"

namespace ddz {

namespace {

// ---------------------------------------------------------------------------
// F(a,b,c,d): minimum moves for a joker-free hand described by group counts
// (a singles, b pairs, c triples, d bombs). Rank identities are abstracted
// away; chains are NOT considered here, they are handled by the outer DFS.
// ---------------------------------------------------------------------------

constexpr int kMaxA = 22;
constexpr int kMaxB = 15;
constexpr int kMaxC = 9;
constexpr int kMaxD = 6;
int8_t F[kMaxA][kMaxB][kMaxC][kMaxD];

int fDP(int a, int b, int c, int d);

// Enumerate ways to take `need` loose cards (need <= 2) from pools of group
// counts. Broken groups downgrade: a used-1 trio leaves a pair, a used-1 bomb
// leaves a trio, etc. Callback receives the new group counts.
template <class Fn>
void enumLooseKickers(int a, int b, int c, int d, int need, Fn&& fn) {
    // Take `s` cards from the singles pool.
    for (int s = 0; s <= std::min(need, a); ++s) {
        int r1 = need - s;
        if (r1 == 0) { fn(a - s, b, c, d); continue; }
        // Pairs: choose bp broken pairs and pu cards from them (1..2 each).
        for (int bp = 0; bp <= std::min(b, r1); ++bp) {
            for (int pu = std::max(bp, 0); pu <= std::min(2 * bp, r1); ++pu) {
                int r2 = r1 - pu;
                // number of broken pairs leaving 1 single = 2bp - pu
                int pairLeftSingles = 2 * bp - pu;
                if (r2 == 0) {
                    fn(a - s + pairLeftSingles, b - bp, c, d);
                    continue;
                }
                // Trios: bt broken, tu cards (1..3 each).
                for (int bt = 0; bt <= std::min(c, r2); ++bt) {
                    for (int tu = bt; tu <= std::min(3 * bt, r2); ++tu) {
                        int r3 = r2 - tu;
                        // distribute `tu` cards among bt broken trios, each 1..3:
                        // leftovers: used-1 trio -> pair, used-2 trio -> single.
                        // t1 + t2 (+t3) = bt, t1 + 2 t2 + 3 t3 = tu.
                        for (int t2 = 0; t2 <= bt; ++t2) {
                            for (int t1 = 0; t1 + t2 <= bt; ++t1) {
                                int t3 = bt - t1 - t2;
                                if (t1 + 2 * t2 + 3 * t3 != tu) continue;
                                int r4 = r3;
                                if (r4 == 0) {
                                    fn(a - s + pairLeftSingles + t2,
                                       b - bp + t1, c - bt, d);
                                    continue;
                                }
                                // Bombs: bd broken, bu cards (1..4 each).
                                for (int bd = 0; bd <= std::min(d, r4); ++bd) {
                                    for (int bu = bd; bu <= std::min(4 * bd, r4); ++bu) {
                                        if (bu != r4) continue;
                                        // distribute bu among bd bombs 1..4:
                                        // u1->trio, u2->pair, u3->single
                                        for (int u3 = 0; u3 <= bd; ++u3)
                                            for (int u2 = 0; u2 + u3 <= bd; ++u2)
                                                for (int u1 = 0; u1 + u2 + u3 <= bd; ++u1) {
                                                    int u4 = bd - u1 - u2 - u3;
                                                    if (u1 + 2 * u2 + 3 * u3 + 4 * u4 != bu)
                                                        continue;
                                                    fn(a - s + pairLeftSingles + t2 + u3,
                                                       b - bp + t1 + u2,
                                                       c - bt + u1, d - bd);
                                                }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

bool g_fReady = false;
std::once_flag g_fFlag;

// read helper used while filling the table (predecessors have fewer cards and
// are therefore already computed)
inline int fGet(int a, int b, int c, int d) {
    if (a < 0 || b < 0 || c < 0 || d < 0) return 100;
    if (a >= kMaxA || b >= kMaxB || c >= kMaxC || d >= kMaxD)
        return a + b + c + d;
    int v = F[a][b][c][d];
    return v < 0 ? a + b + c + d : v;
}

void initFTable() {
    std::memset(F, -1, sizeof(F));
    F[0][0][0][0] = 0;
    for (int total = 1; total <= 20; ++total) {
        for (int a = 0; a < kMaxA; ++a)
            for (int b = 0; b < kMaxB; ++b)
                for (int c = 0; c < kMaxC; ++c)
                    for (int d = 0; d < kMaxD; ++d) {
                        if (a + 2 * b + 3 * c + 4 * d != total) continue;
                        int best = a + b + c + d;
                        if (a >= 1) best = std::min(best, 1 + fGet(a - 1, b, c, d));
                        if (b >= 1) best = std::min(best, 1 + fGet(a, b - 1, c, d));
                        if (c >= 1) best = std::min(best, 1 + fGet(a, b, c - 1, d));
                        if (d >= 1) best = std::min(best, 1 + fGet(a, b, c, d - 1));
                        if (c >= 1)
                            enumLooseKickers(a, b, c - 1, d, 1,
                                             [&](int na, int nb, int nc, int nd) {
                                                 best = std::min(best, 1 + fGet(na, nb, nc, nd));
                                             });
                        if (c >= 1) {
                            if (b >= 1) best = std::min(best, 1 + fGet(a, b - 1, c - 1, d));
                            if (c >= 2) best = std::min(best, 1 + fGet(a + 1, b, c - 2, d));
                            if (d >= 1) best = std::min(best, 1 + fGet(a, b, c - 1, d - 1));
                        }
                        if (d >= 1)
                            enumLooseKickers(a, b, c, d - 1, 2,
                                             [&](int na, int nb, int nc, int nd) {
                                                 best = std::min(best, 1 + fGet(na, nb, nc, nd));
                                             });
                        if (d >= 1) {
                            if (b >= 2) best = std::min(best, 1 + fGet(a, b - 2, c, d - 1));
                            if (b >= 1 && c >= 1)
                                best = std::min(best, 1 + fGet(a + 1, b - 1, c - 1, d - 1));
                            if (c >= 2) best = std::min(best, 1 + fGet(a + 2, b, c - 2, d - 1));
                        }
                        F[a][b][c][d] = static_cast<int8_t>(best);
                    }
    }
    g_fReady = true;
}

int fDP(int a, int b, int c, int d) {
    if (a < 0 || b < 0 || c < 0 || d < 0) return 100;
    if (!g_fReady) std::call_once(g_fFlag, initFTable);
    if (a >= kMaxA || b >= kMaxB || c >= kMaxC || d >= kMaxD)
        return a + b + c + d;
    int v = F[a][b][c][d];
    return v < 0 ? a + b + c + d : v;
}

// ---------------------------------------------------------------------------
// Memoized DFS over explicit rank counts. The cache is thread local: rollout
// workers are short lived, so memory is released when they join.
// ---------------------------------------------------------------------------

// The map lives on the heap. A thread_local unordered_map is destroyed from
// the TLS callback after MinGW's pthread runtime has already freed that
// storage, which is the 0xC0000374 seen when a rollout worker exits.
struct Memo {
    std::unordered_map<uint64_t, int> m;
};

Memo*& memoPtr() {
    static thread_local Memo* p = nullptr;
    return p;
}

std::unordered_map<uint64_t, int>& cache() {
    Memo*& p = memoPtr();
    if (!p) p = new Memo;
    return p->m;
}

void releaseMemo() {
    Memo*& p = memoPtr();
    delete p;
    p = nullptr;
}

uint64_t packKey(const std::array<int8_t, kRanks>& cnt) {
    uint64_t k = 0;
    for (int r = 0; r < kRanks; ++r) k |= uint64_t(cnt[r]) << (3 * r);
    return k;
}

// lower bound ignoring chains
int lowerBound(const std::array<int8_t, kRanks>& cnt) {
    int a = 0, b = 0, c = 0, d = 0;
    for (int r = 0; r < kNormalRanks; ++r) {
        if (cnt[r] == 1) ++a;
        else if (cnt[r] == 2) ++b;
        else if (cnt[r] == 3) ++c;
        else if (cnt[r] == 4) ++d;
    }
    int jokers = (cnt[13] ? 1 : 0) + (cnt[14] ? 1 : 0);
    if (cnt[13] && cnt[14]) jokers = 1;  // rocket is a single move
    return fDP(a, b, c, d) + jokers;
}

// Enumerate every multiset of `need` cards available in cnt (rank by rank).
template <class Fn>
void enumWingMultisets(const std::array<int8_t, kRanks>& cnt, int rank, int need,
                       std::array<int8_t, kRanks>& pick, Fn&& fn) {
    if (need == 0) { fn(pick); return; }
    if (rank == kRanks) return;
    int mx = std::min(int(cnt[rank]), need);
    for (int take = 0; take <= mx; ++take) {
        pick[rank] = static_cast<int8_t>(take);
        enumWingMultisets(cnt, rank + 1, need - take, pick, fn);
    }
    pick[rank] = 0;
}

int solve(std::array<int8_t, kRanks> cnt) {
    uint64_t key = packKey(cnt);
    auto it = cache().find(key);
    if (it != cache().end()) return it->second;

    int best = 100;
    // Jokers: rocket is one move, otherwise the joker is played as a solo
    // (the DFS enumeration below still explores using them as plane kickers).
    if (cnt[13] && cnt[14]) {
        auto rem = cnt;
        rem[13] = rem[14] = 0;
        best = std::min(best, 1 + solve(rem));
    } else if (cnt[13] || cnt[14]) {
        auto rem = cnt;
        rem[13] = rem[14] = 0;
        best = std::min(best, 1 + solve(rem));
    }

    int a = 0, b = 0, c = 0, d = 0;
    for (int r = 0; r < kNormalRanks; ++r) {
        if (cnt[r] == 1) ++a;
        else if (cnt[r] == 2) ++b;
        else if (cnt[r] == 3) ++c;
        else if (cnt[r] == 4) ++d;
    }
    // jokers: rocket is one move, otherwise each joker is a loose solo
    int jokerGroups = (cnt[13] ? 1 : 0) + (cnt[14] ? 1 : 0);
    if (cnt[13] && cnt[14]) jokerGroups = 1;
    best = std::min(best, fDP(a, b, c, d) + jokerGroups);

    if (best > 1) {
        // runs of consecutive ranks (3..A only) with enough copies.
        auto enumerateChains = [&](int repeat, int minLen, bool trios) {
            int i = 0;
            while (i < kNormalRanks) {
                if (cnt[i] < repeat) { ++i; continue; }
                int j = i;
                while (j + 1 < kNormalRanks && cnt[j + 1] >= repeat &&
                       rankValue(j + 1) - rankValue(j) == 1)
                    ++j;
                int runLen = j - i + 1;
                for (int len = minLen; len <= runLen; ++len) {
                    for (int s = i; s + len <= j + 1; ++s) {
                        auto rem = cnt;
                        for (int x = 0; x < len; ++x) rem[s + x] -= repeat;
                        if (1 + lowerBound(rem) < best)
                            best = std::min(best, 1 + solve(rem));
                        if (trios) {
                            // plane + L solo kickers
                            if (4 * len <= 20) {
                                std::array<int8_t, kRanks> pick{};
                                enumWingMultisets(rem, 0, len, pick,
                                                  [&](const std::array<int8_t, kRanks>& w) {
                                                      auto after = rem;
                                                      for (int r = 0; r < kRanks; ++r)
                                                          after[r] -= w[r];
                                                      if (1 + lowerBound(after) < best)
                                                          best = std::min(best, 1 + solve(after));
                                                  });
                            }
                            // plane + L pair kickers (distinct ranks)
                            if (5 * len <= 20) {
                                std::vector<int> pairRanks;
                                for (int r = 0; r < kNormalRanks; ++r)
                                    if (rem[r] >= 2) pairRanks.push_back(r);
                                int np = static_cast<int>(pairRanks.size());
                                if (np >= len) {
                                    std::vector<int> sel(len);
                                    std::function<void(int, int)> combos = [&](int idx, int start) {
                                        if (idx == len) {
                                            auto after = rem;
                                            for (int r : sel) after[r] -= 2;
                                            if (1 + lowerBound(after) < best)
                                                best = std::min(best, 1 + solve(after));
                                            return;
                                        }
                                        for (int x = start; x + (len - idx - 1) < np; ++x) {
                                            sel[idx] = pairRanks[x];
                                            combos(idx + 1, x + 1);
                                        }
                                    };
                                    combos(0, 0);
                                }
                            }
                        }
                    }
                }
                i = j + 1;
            }
        };
        enumerateChains(1, kMinStraight, false);
        enumerateChains(2, kMinPairChain, false);
        enumerateChains(3, kMinPlane, true);
    }

    cache().emplace(key, best);
    return best;
}

}  // namespace

int minSteps(const CardSet& hand) {
    std::array<int8_t, kRanks> cnt{};
    for (int r = 0; r < kRanks; ++r) cnt[r] = hand.c[r];
    return solve(cnt);
}

void clearOracleCache() { releaseMemo(); }

size_t oracleCacheSize() { return cache().size(); }

}  // namespace ddz
