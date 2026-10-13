#include "algo/rollout.h"

#include <algorithm>
#include <condition_variable>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>

#include "ddz/oracle.h"
#include "ddz/rule_agent.h"
#include "torch_nn/models.h"

namespace algo {

static_assert(torch_nn::kImpInput == ddz::kNodeSize, "actor input must match node features");
static_assert(ddz::kHistoryGroups * 3 == ddz::kHistoryLen, "history grouping");
static_assert(torch_nn::kLstmSteps == ddz::kHistoryGroups, "LSTM steps follow history");

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
    using namespace ddz;

    torch::NoGradGuard no_grad;

    struct OracleGuard {
        ~OracleGuard() { clearOracleCache(); }
    } oracleGuard;

    Rng dealRng(seed ^ 0xabcdef123456789ULL);
    Rng sampleRng(seed ^ 0x123456789abcdef0ULL);

    std::vector<Game> games(nGames);
    std::vector<bool> active(nGames, true);
    std::array<std::vector<int>, 3> lastIdx;
    for (int s = 0; s < 3; ++s) lastIdx[s].assign(nGames, -1);

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

        auto sampleSeatPlayer = [&]() -> std::pair<PlayerType, HistModelRef> {
            float r = float((dealRng.nextU64() >> 11) / double(1ULL << 53));
            if (r < cfg.ruleProb) {
                return {PLAYER_RULE, HistModelRef{}};
            } else if (hasHistPool && r < (cfg.ruleProb + cfg.historicalProb)) {
                return {PLAYER_HISTORICAL, sampleHistRef()};
            } else {
                return {PLAYER_LATEST, HistModelRef{}};
            }
        };

        // 1. Landlord camp (Seat 0)
        if (cfg.trainCamp == 0) {
            gameSeats[gi][0].type = PLAYER_LATEST;
            gameSeats[gi][0].histRef = HistModelRef{};
        } else if (cfg.trainCamp == 1) {
            auto [lType, lHistRef] = sampleSeatPlayer();
            gameSeats[gi][0].type = lType;
            gameSeats[gi][0].histRef = lHistRef;
        } else {
            auto [lType, lHistRef] = sampleSeatPlayer();
            gameSeats[gi][0].type = lType;
            gameSeats[gi][0].histRef = lHistRef;
        }

        // 2. Peasant camp (Seat 1 & 2)
        if (cfg.trainCamp == 1) {
            gameSeats[gi][1].type = PLAYER_LATEST;
            gameSeats[gi][1].histRef = HistModelRef{};
            gameSeats[gi][2].type = PLAYER_LATEST;
            gameSeats[gi][2].histRef = HistModelRef{};
        } else if (cfg.trainCamp == 0) {
            if (cfg.independentPeasants) {
                auto [p1Type, p1HistRef] = sampleSeatPlayer();
                gameSeats[gi][1].type = p1Type;
                gameSeats[gi][1].histRef = p1HistRef;

                auto [p2Type, p2HistRef] = sampleSeatPlayer();
                gameSeats[gi][2].type = p2Type;
                gameSeats[gi][2].histRef = p2HistRef;
            } else {
                auto [pType, pHistRef] = sampleSeatPlayer();
                gameSeats[gi][1].type = pType;
                gameSeats[gi][1].histRef = pHistRef;
                gameSeats[gi][2].type = pType;
                gameSeats[gi][2].histRef = pHistRef;
            }
        } else {
            if (cfg.independentPeasants) {
                auto [p1Type, p1HistRef] = sampleSeatPlayer();
                gameSeats[gi][1].type = p1Type;
                gameSeats[gi][1].histRef = p1HistRef;

                auto [p2Type, p2HistRef] = sampleSeatPlayer();
                gameSeats[gi][2].type = p2Type;
                gameSeats[gi][2].histRef = p2HistRef;
            } else {
                auto [pType, pHistRef] = sampleSeatPlayer();
                gameSeats[gi][1].type = pType;
                gameSeats[gi][1].histRef = pHistRef;
                gameSeats[gi][2].type = pType;
                gameSeats[gi][2].histRef = pHistRef;
            }
        }

        if (gameSeats[gi][0].type != PLAYER_LATEST &&
            gameSeats[gi][1].type != PLAYER_LATEST &&
            gameSeats[gi][2].type != PLAYER_LATEST) {
            if (cfg.trainCamp == 0) {
                gameSeats[gi][0].type = PLAYER_LATEST;
                gameSeats[gi][0].histRef = HistModelRef{};
            } else if (cfg.trainCamp == 1) {
                gameSeats[gi][1].type = PLAYER_LATEST;
                gameSeats[gi][1].histRef = HistModelRef{};
                gameSeats[gi][2].type = PLAYER_LATEST;
                gameSeats[gi][2].histRef = HistModelRef{};
            } else if (cfg.independentPeasants) {
                int pickSeat = int(dealRng.nextU64() % 3);
                gameSeats[gi][pickSeat].type = PLAYER_LATEST;
                gameSeats[gi][pickSeat].histRef = HistModelRef{};
            } else {
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

    std::vector<Transition> hTrPool(nGames);
    std::vector<std::vector<LegalOption>> hOptsPool(nGames);

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

    auto chooseAction = [&](const float* logitsRow, const std::vector<LegalOption>& opts) {
        float mx = -1e30f;
        for (const LegalOption& o : opts) {
            mx = std::max(mx, logitsRow[o.abstractId]);
        }
        float sum = 0.0f;
        for (const LegalOption& o : opts) {
            sum += std::exp(logitsRow[o.abstractId] - mx);
        }
        float draw = float((sampleRng.nextU64() >> 11) / double(1ULL << 53));
        if (!(sum > 0.0f) || !std::isfinite(sum)) {
            size_t k = std::min(size_t(draw * opts.size()), opts.size() - 1);
            return std::make_pair(opts[k].abstractId, -std::log(float(opts.size())));
        }
        int chosenId = opts.front().abstractId;
        float acc = 0.0f;
        for (const LegalOption& o : opts) {
            float p = std::exp(logitsRow[o.abstractId] - mx) / sum;
            acc += p;
            chosenId = o.abstractId;
            if (acc >= draw) break;
        }
        float pChosen = std::exp(logitsRow[chosenId] - mx) / sum;
        float logp = std::log(std::max(pChosen, 1e-12f));
        return std::make_pair(chosenId, logp);
    };

    int remaining = nGames;
    auto handleGameOver = [&](int gi, Game& g) {
        active[gi] = false;
        --remaining;
        ++res.stats.games;
        if (g.winner == 0) ++res.stats.landlordWins;
        double mult = std::pow(2.0, g.bombCount);
        long long scoreLL = g.winner == 0 ? llround(2.0 * mult) : -llround(2.0 * mult);
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
                float r = 0.0f;
                if (s == ddz::kLandlord) {
                    bool landlordWon = (g.winner == 0);
                    r = landlordWon ? (cfg.landlordWinMul * float(mult))
                                    : (-cfg.landlordLossMul * float(mult));
                } else {
                    r = float(g.payoff(s));
                }
                t.reward += r + g.shaping(s, cfg.shapingCap);
                t.terminal = true;
            }
        }
    };

    while (remaining > 0) {
        for (int seat = 0; seat < 3; ++seat) {
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

            // 1. Process latest players (batched LibTorch forward)
            if (!latestGis.empty()) {
                int64_t B = static_cast<int64_t>(latestGis.size());
                int64_t totalOptions = 0;
                for (int64_t i = 0; i < B; ++i) {
                    const Game& g = games[latestGis[i]];
                    legalOptions(g, optionsPool[i]);
                    encodeTransition(g, optionsPool[i], trPool[i]);
                    trPool[i].gameId = latestGis[i];
                    chosenPool[i] = nullptr;
                    totalOptions += static_cast<int64_t>(optionsPool[i].size());
                }

                torch::Tensor xImpT = torch::empty({B, torch_nn::kImpInput}, torch::kFloat32);
                torch::Tensor seqT = torch::empty({B, torch_nn::kLstmSteps, torch_nn::kLstmIn}, torch::kFloat32);
                torch::Tensor extraT = torch::empty({B, torch_nn::kExtraInput}, torch::kFloat32);

                torch::Tensor actFeatT = torch::empty({totalOptions, torch_nn::kActionInput}, torch::kFloat32);
                torch::Tensor actSampleT = torch::empty({totalOptions}, torch::kLong);
                torch::Tensor actIdT = torch::empty({totalOptions}, torch::kLong);

                float* pXImp = xImpT.data_ptr<float>();
                float* pSeq = seqT.data_ptr<float>();
                float* pExtra = extraT.data_ptr<float>();
                float* pActFeat = actFeatT.data_ptr<float>();
                int64_t* pActSample = actSampleT.data_ptr<int64_t>();
                int64_t* pActId = actIdT.data_ptr<int64_t>();

                int64_t actOff = 0;
                for (int64_t i = 0; i < B; ++i) {
                    float seqSample[ddz::kHistoryLen * ddz::kCardMat];
                    writeDense(trPool[i], pXImp + i * torch_nn::kImpInput, seqSample, pExtra + i * torch_nn::kExtraInput);
                    for (int k = 0; k < ddz::kHistoryGroups; ++k) {
                        std::memcpy(pSeq + (i * ddz::kHistoryGroups + k) * (3 * ddz::kCardMat),
                                    seqSample + k * 3 * ddz::kCardMat, 3 * ddz::kCardMat * sizeof(float));
                    }
                    for (const auto& [aid, afeat] : trPool[i].actions) {
                        std::memcpy(pActFeat + actOff * torch_nn::kActionInput, afeat.data(), torch_nn::kActionInput * sizeof(float));
                        pActSample[actOff] = i;
                        pActId[actOff] = aid;
                        ++actOff;
                    }
                }

                torch::Tensor logitsT = (*models.actor[seat])->forward(xImpT, seqT, actFeatT, actSampleT, actIdT);
                torch::Tensor valuesT = (*models.critic[seat])->forward(xImpT, seqT, extraT).squeeze(-1);

                const float* pLogits = logitsT.data_ptr<float>();
                const float* pValues = valuesT.data_ptr<float>();

                for (int64_t i = 0; i < B; ++i) {
                    auto [chosenId, logp] = chooseAction(pLogits + i * torch_nn::kNumActions, optionsPool[i]);
                    trPool[i].action = chosenId;
                    trPool[i].logp = logp;
                    float v = pValues[i];
                    trPool[i].value = std::isfinite(v) ? v : 0.0f;
                    for (const LegalOption& o : optionsPool[i]) {
                        if (o.abstractId == chosenId) chosenPool[i] = &o;
                    }
                }

                for (int64_t i = 0; i < B; ++i) {
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

            // 2. Process historical opponents
            if (!histGis.empty()) {
                size_t start = 0;
                while (start < histGis.size()) {
                    size_t end = start + 1;
                    while (end < histGis.size() && histGis[end].second == histGis[start].second) {
                        ++end;
                    }
                    const HistModelRef& mRef = histGis[start].second;
                    int64_t subB = static_cast<int64_t>(end - start);
                    const torch_nn::Actor& histActor = mRef.isArchive
                                                           ? cfg.historicalPool->archive[mRef.index]->actor[seat]
                                                           : cfg.historicalPool->recent[mRef.index]->actor[seat];

                    int64_t totalOptions = 0;
                    for (int64_t i = 0; i < subB; ++i) {
                        int gi = histGis[start + i].first;
                        const Game& g = games[gi];
                        legalOptions(g, hOptsPool[i]);
                        encodeTransition(g, hOptsPool[i], hTrPool[i]);
                        totalOptions += static_cast<int64_t>(hOptsPool[i].size());
                    }

                    torch::Tensor xImpT = torch::empty({subB, torch_nn::kImpInput}, torch::kFloat32);
                    torch::Tensor seqT = torch::empty({subB, torch_nn::kLstmSteps, torch_nn::kLstmIn}, torch::kFloat32);
                    torch::Tensor actFeatT = torch::empty({totalOptions, torch_nn::kActionInput}, torch::kFloat32);
                    torch::Tensor actSampleT = torch::empty({totalOptions}, torch::kLong);
                    torch::Tensor actIdT = torch::empty({totalOptions}, torch::kLong);

                    float* pXImp = xImpT.data_ptr<float>();
                    float* pSeq = seqT.data_ptr<float>();
                    float* pActFeat = actFeatT.data_ptr<float>();
                    int64_t* pActSample = actSampleT.data_ptr<int64_t>();
                    int64_t* pActId = actIdT.data_ptr<int64_t>();

                    int64_t actOff = 0;
                    for (int64_t i = 0; i < subB; ++i) {
                        float seqSample[ddz::kHistoryLen * ddz::kCardMat];
                        float dummyExtra[ddz::kExtraSize];
                        writeDense(hTrPool[i], pXImp + i * torch_nn::kImpInput, seqSample, dummyExtra);
                        for (int k = 0; k < ddz::kHistoryGroups; ++k) {
                            std::memcpy(pSeq + (i * ddz::kHistoryGroups + k) * (3 * ddz::kCardMat),
                                        seqSample + k * 3 * ddz::kCardMat, 3 * ddz::kCardMat * sizeof(float));
                        }
                        for (const auto& [aid, afeat] : hTrPool[i].actions) {
                            std::memcpy(pActFeat + actOff * torch_nn::kActionInput, afeat.data(), torch_nn::kActionInput * sizeof(float));
                            pActSample[actOff] = i;
                            pActId[actOff] = aid;
                            ++actOff;
                        }
                    }

                    torch::Tensor logitsT = histActor->forward(xImpT, seqT, actFeatT, actSampleT, actIdT);
                    const float* pLogits = logitsT.data_ptr<float>();

                    for (int64_t i = 0; i < subB; ++i) {
                        int gi = histGis[start + i].first;
                        Game& g = games[gi];
                        auto [chosenId, logp] = chooseAction(pLogits + i * torch_nn::kNumActions, hOptsPool[i]);
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

            // 3. Process heuristic rule agents
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

void runOnPool(const ModelSet& models, const RolloutConfig& cfg,
               const std::vector<int>& split, std::vector<ThreadResult>& results) {
    struct State {
        std::mutex mu;
        std::condition_variable cv;
        std::mutex api;
        int generation = 0;
        int finished = 0;
        int active = 0;
        const ModelSet* models = nullptr;
        const RolloutConfig* cfg = nullptr;
        const std::vector<int>* split = nullptr;
        std::vector<ThreadResult>* results = nullptr;
        std::vector<std::thread> threads;
    };
    static State* st = new State;
    const int T = static_cast<int>(split.size());
    std::lock_guard<std::mutex> api(st->api);
    {
        std::lock_guard<std::mutex> lk(st->mu);
        while (static_cast<int>(st->threads.size()) < T) {
            int index = static_cast<int>(st->threads.size());
            st->threads.emplace_back([index] {
                int seen = 0;
                for (;;) {
                    const ModelSet* jobModels = nullptr;
                    const RolloutConfig* jobCfg = nullptr;
                    int n = 0;
                    uint64_t seed = 0;
                    ThreadResult* res = nullptr;
                    int gen = 0;
                    {
                        std::unique_lock<std::mutex> lk(st->mu);
                        st->cv.wait(lk, [&] { return st->generation != seen; });
                        gen = st->generation;
                        if (index < st->active) {
                            jobModels = st->models;
                            jobCfg = st->cfg;
                            n = (*st->split)[index];
                            seed = jobCfg->seed +
                                   0x9e3779b97f4a7c15ULL * uint64_t(index + 1);
                            res = &(*st->results)[index];
                        }
                    }
                    if (n > 0) runWorker(*jobModels, *jobCfg, n, seed, *res);
                    {
                        std::lock_guard<std::mutex> lk(st->mu);
                        if (index < st->active && st->generation == gen) {
                            if (++st->finished == st->active) st->cv.notify_all();
                        }
                        seen = gen;
                    }
                }
            });
        }
    }
    {
        std::unique_lock<std::mutex> lk(st->mu);
        st->models = &models;
        st->cfg = &cfg;
        st->split = &split;
        st->results = &results;
        st->finished = 0;
        st->active = T;
        ++st->generation;
        st->cv.notify_all();
        st->cv.wait(lk, [&] { return st->finished == T; });
    }
}

}  // namespace

void collectRollout(const ModelSet& models, const RolloutConfig& cfg,
                    std::array<std::vector<Transition>, 3>& out,
                    RolloutStats& stats) {
    int T = std::max(1, std::min(cfg.threads, cfg.gamesPerUpdate));
    std::vector<int> split(T, cfg.gamesPerUpdate / T);
    for (int i = 0; i < cfg.gamesPerUpdate % T; ++i) ++split[i];

    std::vector<ThreadResult> results(T);
    if (T == 1) {
        if (split[0] > 0) {
            uint64_t seed = cfg.seed + 0x9e3779b97f4a7c15ULL;
            runWorker(models, cfg, split[0], seed, results[0]);
        }
    } else {
        runOnPool(models, cfg, split, results);
    }

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
