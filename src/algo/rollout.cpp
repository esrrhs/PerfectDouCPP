#include "algo/rollout.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <thread>

#include "ddz/oracle.h"
#include "ddz/rule_agent.h"

namespace algo {

static_assert(nn::kImpInput == ddz::kNodeSize, "actor input must match node features");
static_assert(ddz::kHistoryGroups * 3 == ddz::kHistoryLen, "history grouping");
static_assert(nn::kLstmSteps == ddz::kHistoryGroups, "LSTM steps follow history");

namespace {

void writeDense(const Transition& t, float* xImp, float* seqSample, float* extra) {
    for (int j = 0; j < ddz::kNodeBin; ++j) xImp[j] = float(t.imp[j]);
    std::copy(t.scalar.begin(), t.scalar.end(), xImp + ddz::kNodeBin);
    for (int k = 0; k < ddz::kHistoryLen; ++k) {
        const uint8_t* src =
            t.imp.data() + (ddz::kStaticMatrices + k) * ddz::kCardMat;
        float* dst = seqSample + k * ddz::kCardMat;
        for (int j = 0; j < ddz::kCardMat; ++j) dst[j] = float(src[j]);
    }
    for (int j = 0; j < ddz::kExtraBin; ++j) extra[j] = float(t.extra[j]);
    std::copy(t.extraScalar.begin(), t.extraScalar.end(), extra + ddz::kExtraBin);
}

void gatherActions(const std::vector<Transition*>& tr, nn::Mat& actionFeat,
                   nn::Mat& actionSample, nn::Mat& actionId,
                   nn::Mat& actionOffset) {
    int B = static_cast<int>(tr.size());
    int L = 0;
    for (const Transition* t : tr) L += static_cast<int>(t->actions.size());
    actionFeat.resize(L, ddz::kActionSize);
    actionSample.resize(L, 1);
    actionId.resize(L, 1);
    actionOffset.resize(B + 1, 1);
    int row = 0;
    for (int i = 0; i < B; ++i) {
        actionOffset.row(i)[0] = float(row);
        float fi = float(i);
        for (const auto& [id, feature] : tr[i]->actions) {
            std::memcpy(actionFeat.row(row), feature.data(), ddz::kActionSize * sizeof(float));
            actionSample.row(row)[0] = fi;
            actionId.row(row)[0] = float(id);
            ++row;
        }
    }
    actionOffset.row(B)[0] = float(row);
}

void sizeBatch(int B, nn::Mat& xImp, nn::Mat& seq, nn::Mat& extra) {
    xImp.resize(B, ddz::kNodeSize);
    seq.resize(B * ddz::kHistoryGroups, 3 * ddz::kCardMat);
    extra.resize(B, ddz::kExtraSize);
}

}  // namespace

void cacheTransitionFeatures(const std::vector<Transition>& tr,
                             std::vector<float>& xImp, std::vector<float>& seq,
                             std::vector<float>& extra) {
    int N = static_cast<int>(tr.size());
    int xs = nn::padStride(ddz::kNodeSize);
    int es = nn::padStride(ddz::kExtraSize);
    xImp.assign(size_t(N) * xs, 0.0f);
    seq.assign(size_t(N) * ddz::kHistoryLen * ddz::kCardMat, 0.0f);
    extra.assign(size_t(N) * es, 0.0f);
    for (int i = 0; i < N; ++i) {
        writeDense(tr[i], xImp.data() + size_t(i) * xs,
                   seq.data() + size_t(i) * ddz::kHistoryLen * ddz::kCardMat,
                   extra.data() + size_t(i) * es);
    }
}

void buildBatch(const std::vector<Transition*>& tr, nn::Mat& xImp, nn::Mat& seq,
                nn::Mat& actionFeat, nn::Mat& actionSample, nn::Mat& actionId,
                nn::Mat& actionOffset, nn::Mat& extra) {
    int B = static_cast<int>(tr.size());
    sizeBatch(B, xImp, seq, extra);
    gatherActions(tr, actionFeat, actionSample, actionId, actionOffset);
    for (int i = 0; i < B; ++i) {
        float seqSample[ddz::kHistoryLen * ddz::kCardMat];
        writeDense(*tr[i], xImp.row(i), seqSample, extra.row(i));
        for (int k = 0; k < ddz::kHistoryGroups; ++k)
            std::memcpy(seq.row(k * B + i), seqSample + k * 3 * ddz::kCardMat,
                        3 * ddz::kCardMat * sizeof(float));
    }
}

void buildBatchFromCache(const std::vector<Transition*>& tr,
                         const Transition* base, const float* xAll, int xStride,
                         const float* seqAll, const float* eAll, int eStride,
                         nn::Mat& xImp, nn::Mat& seq, nn::Mat& actionFeat,
                         nn::Mat& actionSample, nn::Mat& actionId,
                         nn::Mat& actionOffset, nn::Mat& extra) {
    int B = static_cast<int>(tr.size());
    sizeBatch(B, xImp, seq, extra);
    gatherActions(tr, actionFeat, actionSample, actionId, actionOffset);
    constexpr size_t kGroupBytes = 3 * ddz::kCardMat * sizeof(float);
    constexpr size_t kNodeBytes = ddz::kNodeSize * sizeof(float);
    constexpr size_t kExtraBytes = ddz::kExtraSize * sizeof(float);
    for (int i = 0; i < B; ++i) {
        int id = static_cast<int>(tr[i] - base);
        std::memcpy(xImp.row(i), xAll + size_t(id) * xStride, kNodeBytes);
        const float* ss = seqAll + size_t(id) * (ddz::kHistoryLen * ddz::kCardMat);
        for (int k = 0; k < ddz::kHistoryGroups; ++k) {
            std::memcpy(seq.row(k * B + i), ss + k * 3 * ddz::kCardMat, kGroupBytes);
        }
        std::memcpy(extra.row(i), eAll + size_t(id) * eStride, kExtraBytes);
    }
}

namespace {

struct ThreadResult {
    std::array<std::vector<Transition>, 3> seats;
    RolloutStats stats;
};

enum PlayerType : uint8_t {
    PLAYER_LATEST = 0,
    PLAYER_HISTORICAL = 1,
    PLAYER_RULE = 2
};

struct HistModelRef {
    bool isArchive = false;
    int index = -1;

    bool operator<(const HistModelRef& o) const {
        if (isArchive != o.isArchive) return isArchive < o.isArchive;
        return index < o.index;
    }
    bool operator==(const HistModelRef& o) const {
        return isArchive == o.isArchive && index == o.index;
    }
};

struct SeatAssignment {
    PlayerType type = PLAYER_LATEST;
    HistModelRef histRef;
};

void runWorker(const ModelSet& models, const RolloutConfig& cfg, int nGames,
               uint64_t seed, ThreadResult& res) {
    using namespace nn;
    using namespace ddz;

    // Self-play inference stays on CPU: the GPU is reserved for PPO learning,
    // and many worker threads issuing synchronous GPU encodings serialize.
    gemmSetThreadGpu(0);
    // Drop the oracle memo before this thread returns. The pthread TLS
    // teardown on MinGW runs before C++ thread_local destructors.
    struct OracleGuard {
        ~OracleGuard() { clearOracleCache(); }
    } oracleGuard;

    Rng dealRng(seed ^ 0xabcdef123456789ULL);
    Rng sampleRng(seed ^ 0x123456789abcdef0ULL);

    std::vector<Game> games(nGames);
    std::vector<bool> active(nGames, true);
    // index of the last transition of each game per seat
    std::array<std::vector<int>, 3> lastIdx;
    for (int s = 0; s < 3; ++s) lastIdx[s].assign(nGames, -1);

    ActorInfer actorW[3];
    CriticInfer criticW[3];

    std::vector<std::array<SeatAssignment, 3>> gameSeats(nGames);
    bool hasRecent = cfg.historicalPool && !cfg.historicalPool->recent.empty();
    bool hasArchive = cfg.historicalPool && !cfg.historicalPool->archive.empty();
    bool hasHistPool = hasRecent || hasArchive;

    for (int gi = 0; gi < nGames; ++gi) {
        games[gi].deal(dealRng);
        auto sampleHistRef = [&]() -> HistModelRef {
            if (!hasHistPool) return HistModelRef{};
            bool pickArchive = false;
            if (hasRecent && hasArchive) {
                float subR = float((dealRng.nextU64() >> 11) / double(1ULL << 53));
                pickArchive = (subR < 0.3f);
            } else if (hasArchive) {
                pickArchive = true;
            }
            if (pickArchive) {
                int nArchive = static_cast<int>(cfg.historicalPool->archive.size());
                int idx = int(dealRng.nextU64() % uint64_t(nArchive));
                return HistModelRef{true, idx};
            } else {
                int nRecent = static_cast<int>(cfg.historicalPool->recent.size());
                int idx = int(dealRng.nextU64() % uint64_t(nRecent));
                return HistModelRef{false, idx};
            }
        };

        // 1. Landlord camp (Seat 0): independently sampled
        float rLandlord = float((dealRng.nextU64() >> 11) / double(1ULL << 53));
        if (rLandlord < cfg.ruleProb) {
            gameSeats[gi][0].type = PLAYER_RULE;
        } else if (hasHistPool && rLandlord < (cfg.ruleProb + cfg.historicalProb)) {
            gameSeats[gi][0].type = PLAYER_HISTORICAL;
            gameSeats[gi][0].histRef = sampleHistRef();
        } else {
            gameSeats[gi][0].type = PLAYER_LATEST;
        }

        // 2. Peasant camp (Seat 1 & 2): sampled as a unified cooperative team
        float rPeasant = float((dealRng.nextU64() >> 11) / double(1ULL << 53));
        PlayerType peasantType = PLAYER_LATEST;
        HistModelRef peasantHistRef;
        if (rPeasant < cfg.ruleProb) {
            peasantType = PLAYER_RULE;
        } else if (hasHistPool && rPeasant < (cfg.ruleProb + cfg.historicalProb)) {
            peasantType = PLAYER_HISTORICAL;
            peasantHistRef = sampleHistRef();
        } else {
            peasantType = PLAYER_LATEST;
        }
        gameSeats[gi][1].type = peasantType;
        gameSeats[gi][1].histRef = peasantHistRef;
        gameSeats[gi][2].type = peasantType;
        gameSeats[gi][2].histRef = peasantHistRef;

        // Ensure at least one camp per game is PLAYER_LATEST so games always produce training data.
        // If neither camp picked LATEST, flip an entire camp (50% Landlord, 50% Peasant team)
        // so peasant team coordination is strictly preserved.
        if (gameSeats[gi][0].type != PLAYER_LATEST && peasantType != PLAYER_LATEST) {
            if ((dealRng.nextU64() & 1) == 0) {
                gameSeats[gi][0].type = PLAYER_LATEST;
                gameSeats[gi][0].histRef = HistModelRef{};
            } else {
                gameSeats[gi][1].type = PLAYER_LATEST;
                gameSeats[gi][1].histRef = HistModelRef{};
                gameSeats[gi][2].type = PLAYER_LATEST;
                gameSeats[gi][2].histRef = HistModelRef{};
            }
        }
    }

    std::vector<int> latestGis;
    latestGis.reserve(nGames);
    std::vector<int> ruleGis;
    ruleGis.reserve(nGames);
    std::vector<std::pair<int, HistModelRef>> histGis;
    histGis.reserve(nGames);

    std::vector<Transition> trPool(nGames);
    std::vector<const LegalOption*> chosenPool(nGames, nullptr);
    std::vector<std::vector<LegalOption>> optionsPool(nGames);
    std::vector<Transition*> ptrsPool(nGames, nullptr);

    std::vector<Transition> hTrPool(nGames);
    std::vector<std::vector<LegalOption>> hOptsPool(nGames);
    std::vector<Transition*> hPtrsPool(nGames, nullptr);

    Mat xImp, seq, actionFeat, actionSample, actionId, actionOffset, extra;
    Mat hXImp, hSeq, hActionFeat, hActionSample, hActionId, hActionOffset, hExtra;

    auto encodeTransition = [&](const Game& g,
                                const std::vector<LegalOption>& options,
                                Transition& t) {
        EncodedState e = encodeState(g);
        t.imp = e.imp;
        t.scalar = e.scalar;
        t.extra = e.extra;
        t.extraScalar = e.extraScalar;
        t.actions.clear();
        t.actions.reserve(options.size());
        for (const LegalOption& o : options) {
            t.actions.emplace_back(o.abstractId, o.feature);
        }
    };

    auto chooseAction = [&](const Mat& logits, const std::vector<LegalOption>& opts,
                            int batchRow) {
        float mx = -1e30f;
        for (int a = 0; a < kNumActions; ++a)
            mx = std::max(mx, logits.row(batchRow)[a]);
        float probs[kNumActions];
        float sum = 0.0f;
        for (int a = 0; a < kNumActions; ++a) {
            probs[a] = expf(logits.row(batchRow)[a] - mx);
            sum += probs[a];
        }
        float draw = float((sampleRng.nextU64() >> 11) / double(1ULL << 53));
        if (!(sum > 0.0f) || !std::isfinite(sum)) {
            // NaN/Inf logits: fall back to a uniform choice over legal moves
            // so a bad forward pass cannot write NaN into the training stream.
            size_t k = std::min(size_t(draw * opts.size()), opts.size() - 1);
            return std::make_pair(opts[k].abstractId,
                                  -std::log(float(opts.size())));
        }
        int chosenId = opts.front().abstractId;
        float acc = 0.0f;
        for (const LegalOption& o : opts) {
            float p = probs[o.abstractId] / sum;
            acc += p;
            chosenId = o.abstractId;
            if (acc >= draw) break;
        }
        float logp = std::log(std::max(probs[chosenId] / sum, 1e-12f));
        return std::make_pair(chosenId, logp);
    };

    int remaining = nGames;
    auto handleGameOver = [&](int gi, Game& g) {
        active[gi] = false;
        --remaining;
        ++res.stats.games;
        if (g.winner == 0) ++res.stats.landlordWins;
        double mult = std::pow(2.0, g.bombCount);
        long long scoreLL = g.winner == 0
                               ? llround(2.0 * mult)
                               : -llround(2.0 * mult);
        res.stats.landlordScore += scoreLL;

        bool isPure = (gameSeats[gi][0].type == PLAYER_LATEST &&
                       gameSeats[gi][1].type == PLAYER_LATEST &&
                       gameSeats[gi][2].type == PLAYER_LATEST);
        if (isPure) {
            ++res.stats.pureSelfPlayGames;
            if (g.winner == 0) ++res.stats.pureSelfPlayWins;
            res.stats.pureSelfPlayScore += scoreLL;
        }

        for (int s = 0; s < 3; ++s) {
            int idx = lastIdx[s][gi];
            if (idx >= 0) {
                auto& t = res.seats[s][idx];
                t.reward += float(g.payoff(s)) + g.shaping(s, cfg.shapingCap);
                t.terminal = true;
            }
        }
    };

    while (remaining > 0) {
        for (int seat = 0; seat < 3; ++seat) {
            // Group active games waiting on this seat by player type
            latestGis.clear();
            ruleGis.clear();
            histGis.clear();

            for (int gi = 0; gi < nGames; ++gi) {
                if (active[gi] && games[gi].turn == seat) {
                    PlayerType pt = gameSeats[gi][seat].type;
                    if (pt == PLAYER_LATEST) {
                        latestGis.push_back(gi);
                    } else if (pt == PLAYER_RULE) {
                        ruleGis.push_back(gi);
                    } else {
                        histGis.emplace_back(gi, gameSeats[gi][seat].histRef);
                    }
                }
            }

            // 1. Process latest players (batched neural forward, transitions recorded)
            if (!latestGis.empty()) {
                int B = static_cast<int>(latestGis.size());
                for (int i = 0; i < B; ++i) {
                    const Game& g = games[latestGis[i]];
                    legalOptions(g, optionsPool[i]);
                    if ((int)optionsPool[i].size() > kNumActions) {
                        std::fprintf(stderr, "legal moves %d exceed logit width %d\n",
                                     (int)optionsPool[i].size(), kNumActions);
                        std::abort();
                    }
                    encodeTransition(g, optionsPool[i], trPool[i]);
                    trPool[i].gameId = latestGis[i];
                    ptrsPool[i] = &trPool[i];
                    chosenPool[i] = nullptr;
                }

                std::vector<Transition*> ptrs(ptrsPool.begin(), ptrsPool.begin() + B);
                buildBatch(ptrs, xImp, seq, actionFeat, actionSample, actionId,
                           actionOffset, extra);

                const Mat& logits = actorInferForward(*models.actor[seat],
                                                      actorW[seat], xImp, seq,
                                                      actionFeat, actionSample,
                                                      actionId, actionOffset);
                const Mat& values = criticInferForward(*models.critic[seat],
                                                       criticW[seat], xImp, seq,
                                                       extra);
                for (int i = 0; i < B; ++i) {
                    if (optionsPool[i].empty()) {
                        std::fprintf(stderr, "empty options B=%d i=%d seat=%d hand=%s\n",
                                     B, i, seat,
                                     games[latestGis[i]].hand[seat].str().c_str());
                        std::abort();
                    }
                    auto [chosenId, logp] = chooseAction(logits, optionsPool[i], i);
                    trPool[i].action = chosenId;
                    trPool[i].logp = logp;
                    trPool[i].value = std::isfinite(values.row(i)[0]) ? values.row(i)[0] : 0.0f;
                    for (const LegalOption& o : optionsPool[i]) {
                        if (o.abstractId == chosenId) chosenPool[i] = &o;
                    }
                }

                for (int i = 0; i < B; ++i) {
                    int gi = latestGis[i];
                    Game& g = games[gi];
                    const CardSet& concrete = chosenPool[i]->concrete;

                    int storedIndex = static_cast<int>(res.seats[seat].size());
                    res.seats[seat].push_back(std::move(trPool[i]));
                    lastIdx[seat][gi] = storedIndex;

                    g.step(concrete);
                    res.stats.moves += 1;
                    if (isBombLike(concrete)) res.stats.bombs += 1;

                    if (g.over) {
                        handleGameOver(gi, g);
                    }
                }
            }

            // 2. Process historical model players (inference only, not recorded into training stream)
            if (!histGis.empty()) {
                // Group by snapshot reference to batch inference per historical model
                std::sort(histGis.begin(), histGis.end(),
                          [](const auto& a, const auto& b) {
                              return a.second < b.second;
                          });

                size_t start = 0;
                while (start < histGis.size()) {
                    size_t end = start + 1;
                    while (end < histGis.size() && histGis[end].second == histGis[start].second) {
                        ++end;
                    }
                    const HistModelRef& mRef = histGis[start].second;
                    int subB = static_cast<int>(end - start);
                    const nn::Actor& histActor = mRef.isArchive
                                                    ? cfg.historicalPool->archive[mRef.index]->actor[seat]
                                                    : cfg.historicalPool->recent[mRef.index]->actor[seat];
                    ActorInfer histInfer;

                    for (int i = 0; i < subB; ++i) {
                        int gi = histGis[start + i].first;
                        const Game& g = games[gi];
                        legalOptions(g, hOptsPool[i]);
                        encodeTransition(g, hOptsPool[i], hTrPool[i]);
                        hPtrsPool[i] = &hTrPool[i];
                    }
                    std::vector<Transition*> ptrs(hPtrsPool.begin(), hPtrsPool.begin() + subB);
                    buildBatch(ptrs, hXImp, hSeq, hActionFeat, hActionSample, hActionId,
                               hActionOffset, hExtra);

                    const Mat& logits = actorInferForward(histActor, histInfer,
                                                          hXImp, hSeq, hActionFeat,
                                                          hActionSample, hActionId,
                                                          hActionOffset);

                    for (int i = 0; i < subB; ++i) {
                        int gi = histGis[start + i].first;
                        Game& g = games[gi];
                        auto [chosenId, logp] = chooseAction(logits, hOptsPool[i], i);
                        (void)logp;
                        CardSet concrete;
                        for (const LegalOption& o : hOptsPool[i]) {
                            if (o.abstractId == chosenId) {
                                concrete = o.concrete;
                                break;
                            }
                        }

                        g.step(concrete);
                        res.stats.moves += 1;
                        if (isBombLike(concrete)) res.stats.bombs += 1;

                        if (g.over) {
                            handleGameOver(gi, g);
                        }
                    }
                    start = end;
                }
            }

            // 3. Process heuristic rule agents (fast C++ heuristic rule evaluation)
            for (int gi : ruleGis) {
                Game& g = games[gi];
                std::vector<CardSet> legalMoves = g.legal();
                CardSet concrete = RuleAgent::selectMove(g, seat, legalMoves);

                g.step(concrete);
                res.stats.moves += 1;
                if (isBombLike(concrete)) res.stats.bombs += 1;

                if (g.over) {
                    handleGameOver(gi, g);
                }
            }
        }
    }
    for (int s = 0; s < 3; ++s)
        res.stats.transitions[s] = (long long)res.seats[s].size();
}

}  // namespace

void collectRollout(const ModelSet& models, const RolloutConfig& cfg,
                    std::array<std::vector<Transition>, 3>& out,
                    RolloutStats& stats) {
    int T = std::max(1, std::min(cfg.threads, cfg.gamesPerUpdate));
    std::vector<int> split(T, cfg.gamesPerUpdate / T);
    for (int i = 0; i < cfg.gamesPerUpdate % T; ++i) ++split[i];

    std::vector<ThreadResult> results(T);
    std::vector<std::thread> threads;
    for (int t = 0; t < T; ++t) {
        if (split[t] == 0) continue;
        uint64_t seed = cfg.seed + 0x9e3779b97f4a7c15ULL * uint64_t(t + 1);
        if (T == 1) {
            runWorker(models, cfg, split[t], seed, results[t]);
            continue;
        }
        threads.emplace_back(runWorker, std::cref(models), std::cref(cfg),
                             split[t], seed, std::ref(results[t]));
    }
    for (auto& th : threads) th.join();

    stats = RolloutStats{};
    int gameOffset = 0;
    for (int t = 0; t < T; ++t) {
        if (split[t] == 0) continue;
        auto& r = results[t];
        stats.games += r.stats.games;
        stats.landlordWins += r.stats.landlordWins;
        stats.landlordScore += r.stats.landlordScore;
        stats.pureSelfPlayGames += r.stats.pureSelfPlayGames;
        stats.pureSelfPlayWins += r.stats.pureSelfPlayWins;
        stats.pureSelfPlayScore += r.stats.pureSelfPlayScore;
        stats.bombs += r.stats.bombs;
        stats.moves += r.stats.moves;
        for (int s = 0; s < 3; ++s) {
            stats.transitions[s] += r.stats.transitions[s];
            for (Transition tr : r.seats[s]) {
                if (tr.gameId >= 0) tr.gameId += gameOffset;
                out[s].push_back(std::move(tr));
            }
        }
        gameOffset += r.stats.games;
    }
}

}  // namespace algo
