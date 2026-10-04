#include "algo/rollout.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <thread>

#include "ddz/oracle.h"

namespace algo {

namespace {

void writeDense(const Transition& t, float* xImp, float* seqSample, float* extra) {
    for (int j = 0; j < ddz::kImpBin; ++j) xImp[j] = float(t.imp[j]);
    std::copy(t.scalar.begin(), t.scalar.end(), xImp + ddz::kImpBin);
    for (int k = 0; k < ddz::kHistoryLen; ++k) {
        const uint8_t* src = t.imp.data() + (6 + k) * ddz::kCardMat;
        float* dst = seqSample + k * ddz::kCardMat;
        for (int j = 0; j < ddz::kCardMat; ++j) dst[j] = float(src[j]);
    }
    for (int j = 0; j < ddz::kExtraBin; ++j) extra[j] = float(t.extra[j]);
    std::copy(t.extraScalar.begin(), t.extraScalar.end(), extra + ddz::kExtraBin);
}

void scatterMaskDyn(const std::vector<Transition*>& tr, nn::Mat& mask,
                    nn::Mat& dynFeat) {
    std::fill(dynFeat.d.begin(), dynFeat.d.end(), 0.0f);
    std::fill(mask.d.begin(), mask.d.end(), 0.0f);
    int B = static_cast<int>(tr.size());
    for (int i = 0; i < B; ++i) {
        const Transition& t = *tr[i];
        for (int w = 0; w < 10; ++w) {
            uint64_t bits = t.mask[w];
            while (bits) {
                int b = __builtin_ctzll(bits);
                mask.row(i)[w * 64 + b] = 1.0f;
                bits &= bits - 1;
            }
        }
        for (const auto& [id, d] : t.dyn) {
            float* df = dynFeat.row(i * ddz::kAbstractActions + id);
            for (int j = 0; j < ddz::kActionDyn; ++j) df[j] = d[j];
        }
    }
}

void sizeBatch(int B, nn::Mat& xImp, nn::Mat& seq, nn::Mat& mask, nn::Mat& dynFeat,
               nn::Mat& extra) {
    xImp.resize(B, ddz::kImpSize);
    seq.resize(B * ddz::kHistoryLen, ddz::kCardMat);
    mask.resize(B, ddz::kAbstractActions);
    dynFeat.resize(B * ddz::kAbstractActions, ddz::kActionDyn);
    extra.resize(B, ddz::kExtraSize);
}

}  // namespace

void cacheTransitionFeatures(const std::vector<Transition>& tr,
                             std::vector<float>& xImp, std::vector<float>& seq,
                             std::vector<float>& extra) {
    int N = static_cast<int>(tr.size());
    int xs = nn::padStride(ddz::kImpSize);
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
                nn::Mat& mask, nn::Mat& dynFeat, nn::Mat& extra) {
    int B = static_cast<int>(tr.size());
    sizeBatch(B, xImp, seq, mask, dynFeat, extra);
    scatterMaskDyn(tr, mask, dynFeat);
    for (int i = 0; i < B; ++i) {
        float seqSample[ddz::kHistoryLen * ddz::kCardMat];
        writeDense(*tr[i], xImp.row(i), seqSample, extra.row(i));
        for (int k = 0; k < ddz::kHistoryLen; ++k)
            std::copy(seqSample + k * ddz::kCardMat,
                      seqSample + (k + 1) * ddz::kCardMat, seq.row(k * B + i));
    }
}

void buildBatchFromCache(const std::vector<Transition*>& tr,
                         const Transition* base, const float* xAll, int xStride,
                         const float* seqAll, const float* eAll, int eStride,
                         nn::Mat& xImp, nn::Mat& seq, nn::Mat& mask,
                         nn::Mat& dynFeat, nn::Mat& extra) {
    int B = static_cast<int>(tr.size());
    sizeBatch(B, xImp, seq, mask, dynFeat, extra);
    scatterMaskDyn(tr, mask, dynFeat);
    for (int i = 0; i < B; ++i) {
        int id = static_cast<int>(tr[i] - base);
        std::copy(xAll + size_t(id) * xStride,
                  xAll + size_t(id) * xStride + ddz::kImpSize, xImp.row(i));
        const float* ss = seqAll + size_t(id) * ddz::kHistoryLen * ddz::kCardMat;
        for (int k = 0; k < ddz::kHistoryLen; ++k)
            std::copy(ss + k * ddz::kCardMat, ss + (k + 1) * ddz::kCardMat,
                      seq.row(k * B + i));
        std::copy(eAll + size_t(id) * eStride,
                  eAll + size_t(id) * eStride + ddz::kExtraSize, extra.row(i));
    }
}

namespace {

struct ThreadResult {
    std::array<std::vector<Transition>, 3> seats;
    RolloutStats stats;
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

    for (int gi = 0; gi < nGames; ++gi) games[gi].deal(dealRng);

    auto encodeTransition = [&](const Game& g,
                                const std::vector<LegalOption>& options) {
        EncodedState e = encodeState(g);
        Transition t;
        t.imp = e.imp;
        t.scalar = e.scalar;
        t.extra = e.extra;
        t.extraScalar = e.extraScalar;
        for (const LegalOption& o : options) {
            t.mask[o.abstractId >> 6] |= uint64_t(1) << (o.abstractId & 63);
            t.dyn.emplace_back(o.abstractId, o.dyn);
        }
        return t;
    };

    int remaining = nGames;
    while (remaining > 0) {
        for (int seat = 0; seat < 3; ++seat) {
            std::vector<int> idxs;
            for (int gi = 0; gi < nGames; ++gi)
                if (active[gi] && games[gi].turn == seat) idxs.push_back(gi);
            if (idxs.empty()) continue;
            int B = static_cast<int>(idxs.size());

            std::vector<Transition> tr(B);
            std::vector<const LegalOption*> chosen(B, nullptr);
            std::vector<std::vector<LegalOption>> options(B);
            std::vector<std::array<float, 3>> steps(B);
            for (int i = 0; i < B; ++i) {
                const Game& g = games[idxs[i]];
                options[i] = legalOptions(g);
                tr[i] = encodeTransition(g, options[i]);
                tr[i].gameId = idxs[i];
                int prev = g.prevSeat();
                int nxt = g.nextSeat();
                steps[i][seat] = tr[i].scalar[0] * 20.0f;
                steps[i][prev] = tr[i].extraScalar[0] * 20.0f;
                steps[i][nxt] = tr[i].extraScalar[1] * 20.0f;
            }

            std::vector<Transition*> ptrs(B);
            for (int i = 0; i < B; ++i) ptrs[i] = &tr[i];
            Mat xImp, seq, mask, dynFeat, extra;
            buildBatch(ptrs, xImp, seq, mask, dynFeat, extra);

            const Mat& logits = actorInferForward(*models.actor[seat],
                                                  actorW[seat], xImp, seq,
                                                  mask, dynFeat);
            const Mat& values = criticInferForward(*models.critic[seat],
                                                   criticW[seat], xImp, seq,
                                                   extra);
            for (int i = 0; i < B; ++i) {
                if (options[i].empty()) {
                    std::fprintf(stderr, "empty options B=%d i=%d seat=%d hand=%s\n",
                                 B, i, seat,
                                 games[idxs[i]].hand[seat].str().c_str());
                    std::abort();
                }
                // masked softmax + categorical sampling
                float mx = -1e30f;
                for (int a = 0; a < kAbstractActions; ++a)
                    mx = std::max(mx, logits.row(i)[a]);
                float probs[kAbstractActions];
                float sum = 0.0f;
                for (int a = 0; a < kAbstractActions; ++a) {
                    probs[a] = std::exp(logits.row(i)[a] - mx);
                    sum += probs[a];
                }
                float draw = float((sampleRng.nextU64() >> 11) /
                                   double(1ULL << 53));
                int chosenId = options[i].front().abstractId;
                float acc = 0.0f;
                for (const LegalOption& o : options[i]) {
                    float p = probs[o.abstractId] / sum;
                    acc += p;
                    chosenId = o.abstractId;
                    if (acc >= draw) break;
                }
                tr[i].action = chosenId;
                tr[i].logp = std::log(std::max(probs[chosenId] / sum, 1e-12f));
                tr[i].value = values.row(i)[0];
                for (const LegalOption& o : options[i])
                    if (o.abstractId == chosenId) chosen[i] = &o;
            }

            for (int i = 0; i < B; ++i) {
                int gi = idxs[i];
                Game& g = games[gi];
                const CardSet& concrete = chosen[i]->concrete;

                // oracle shaped reward (distance-to-win advantage delta)
                std::array<float, 3> after = steps[i];
                CardSet nh = g.hand[seat];
                nh.sub(concrete);
                after[seat] = float(minSteps(nh));
                float advB = steps[i][0] - std::min(steps[i][1], steps[i][2]);
                float advA = after[0] - std::min(after[1], after[2]);
                float dAdv = advA - advB;
                if (cfg.rewardScale > 0.0f)
                    tr[i].reward = (seat == kLandlord ? -1.0f : 0.5f) * dAdv *
                                   cfg.rewardScale;

                int storedIndex = static_cast<int>(res.seats[seat].size());
                res.seats[seat].push_back(std::move(tr[i]));
                lastIdx[seat][gi] = storedIndex;

                g.step(concrete);
                res.stats.moves += 1;
                if (isBombLike(concrete)) res.stats.bombs += 1;

                if (g.over) {
                    active[gi] = false;
                    --remaining;
                    ++res.stats.games;
                    res.stats.moves += 0;
                    if (g.winner == 0) ++res.stats.landlordWins;
                    double mult = std::pow(2.0, g.bombCount);
                    long long scoreLL = g.winner == 0
                                           ? llround(2.0 * mult)
                                           : -llround(2.0 * mult);
                    res.stats.landlordScore += scoreLL;
                    // each seat receives the terminal ADP on its last decision
                    for (int s = 0; s < 3; ++s) {
                        int idx = lastIdx[s][gi];
                        if (idx >= 0) {
                            auto& t = res.seats[s][idx];
                            t.reward += float(g.payoff(s));
                            t.terminal = true;
                        }
                    }
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
