#include "algo/ppo.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>

namespace algo {

void computeGAE(std::vector<Transition>& tr, float gamma, float lambda) {
    // group indices by game (chronological within each game)
    std::unordered_map<int, std::vector<size_t>> groups;
    for (size_t i = 0; i < tr.size(); ++i) groups[tr[i].gameId].push_back(i);

    for (auto& [id, is] : groups) {
        (void)id;
        float gae = 0.0f;
        float nextValue = 0.0f;  // bootstrap with 0 at terminal
        for (int k = static_cast<int>(is.size()) - 1; k >= 0; --k) {
            Transition& t = tr[is[k]];
            float delta = t.reward + gamma * nextValue - t.value;
            gae = delta + gamma * lambda * gae;
            t.adv = gae;
            t.ret = gae + t.value;
            nextValue = t.value;
        }
    }
}

namespace {

void maskedSoftmax(const float* logits, int n, float* probs) {
    float mx = -1e30f;
    for (int a = 0; a < n; ++a) mx = std::max(mx, logits[a]);
    float sum = 0.0f;
    for (int a = 0; a < n; ++a) {
        probs[a] = logits[a] < -1e8f ? 0.0f : std::exp(logits[a] - mx);
        sum += probs[a];
    }
    float inv = 1.0f / sum;
    for (int a = 0; a < n; ++a) probs[a] *= inv;
}

}  // namespace

void ppoUpdate(nn::Actor& actor, nn::Critic& critic,
               std::vector<Transition>& tr, const PPOConfig& cfg,
               nn::Adam& actorOpt, nn::Adam& criticOpt, nn::Rng64& rng,
               PPOStats& stats) {
    int N = static_cast<int>(tr.size());
    if (N == 0) return;

    computeGAE(tr, cfg.gamma, cfg.lambda);

    double mean = 0.0, var = 0.0;
    for (const Transition& t : tr) mean += t.adv;
    mean /= N;
    for (const Transition& t : tr) var += (t.adv - float(mean)) * (t.adv - float(mean));
    float std = float(std::sqrt(var / N + 1e-8));
    for (Transition& t : tr) t.adv = (t.adv - float(mean)) / std;

    double sumRet = 0;
    for (const Transition& t : tr) sumRet += t.ret;
    stats.meanRet = sumRet / N;
    stats.meanAdv = mean;

    std::vector<int> order(N);
    std::iota(order.begin(), order.end(), 0);

    actor.zeroGrad();
    critic.zeroGrad();

    double pgLossSum = 0, vLossSum = 0, entSum = 0;
    int mbCount = 0;

    for (int ep = 0; ep < cfg.epochs; ++ep) {
        // Fisher-Yates shuffle
        for (int i = N - 1; i > 0; --i) {
            int j = int(rng.nextU64() % uint64_t(i + 1));
            std::swap(order[i], order[j]);
        }
        for (int lo = 0; lo < N; lo += cfg.minibatch) {
            int hi = std::min(lo + cfg.minibatch, N);
            int B = hi - lo;
            std::vector<Transition*> mb(B);
            for (int i = 0; i < B; ++i) mb[i] = &tr[order[lo + i]];

            nn::Mat xImp, seq, mask, dynFeat, extra;
            buildBatch(mb, xImp, seq, mask, dynFeat, extra);

            // ---- actor ----
            nn::Mat& logits = actor.forward(xImp, seq, mask, dynFeat);
            nn::Mat dLogits(B, nn::kNumActions);
            std::fill(dLogits.d.begin(), dLogits.d.end(), 0.0f);
            for (int i = 0; i < B; ++i) {
                float probs[nn::kNumActions];
                maskedSoftmax(logits.row(i), nn::kNumActions, probs);
                int act = mb[i]->action;
                float newLogp = std::log(std::max(probs[act], 1e-12f));
                float ratio = std::exp(newLogp - mb[i]->logp);
                float s = mb[i]->adv;
                float rc = std::clamp(ratio, 1.0f - cfg.clip, 1.0f + cfg.clip);
                float l1 = ratio * s, l2 = rc * s;
                float surrogate = std::min(l1, l2);
                pgLossSum += surrogate;

                float entropy = 0.0f;
                for (int a = 0; a < nn::kNumActions; ++a)
                    if (probs[a] > 1e-12f) entropy -= probs[a] * std::log(probs[a]);
                entSum += entropy;

                // We minimize -surrogate + (no entropy in the policy term).
                // dLoss/dratio is -g in the interior and 0 when clipped.
                float g;
                if (l1 <= l2)
                    g = ratio * s;  // interior: gradient flows
                else
                    g = 0.0f;       // clipped region
                // softmax backprop: the policy term behaves like
                // L = -s*log p_act, so prob-space upstream v is -s/p_act on
                // the chosen action and 0 elsewhere (Σ p v = -s).
                float scale = 1.0f / float(N);
                float pa = probs[act];
                float sumW = g * scale +
                             cfg.entCoef * scale * (1.0f - entropy);
                for (int a = 0; a < nn::kNumActions; ++a) {
                    if (probs[a] <= 0.0f) continue;
                    float W = (a == act ? -g / std::max(pa, 1e-8f) * scale
                                       : 0.0f) +
                              cfg.entCoef * scale * (std::log(probs[a]) + 1.0f);
                    dLogits.row(i)[a] = probs[a] * (W - sumW);
                }
            }
            actor.backward(dLogits);

            // ---- critic ----
            nn::Mat& values = critic.forward(xImp, seq, extra);
            nn::Mat dValue(B, 1);
            for (int i = 0; i < B; ++i) {
                float err = values.row(i)[0] - mb[i]->ret;
                vLossSum += 0.5 * err * err;
                dValue.row(i)[0] = cfg.vfCoef * err / float(N);
            }
            critic.backward(dValue);
            ++mbCount;
        }
    }

    actorOpt.applyGradNorm(actor.params(), cfg.maxGradNorm);
    criticOpt.applyGradNorm(critic.params(), cfg.maxGradNorm);

    double samples = double(N) * cfg.epochs;
    stats.pgLoss = -pgLossSum / samples;
    stats.vLoss = vLossSum / samples;
    stats.entropy = entSum / samples;
    stats.meanAbsOldLogp = 0;
    (void)mbCount;
}

}  // namespace algo
