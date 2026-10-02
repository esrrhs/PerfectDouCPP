// Numerical gradient checks for actor / critic (validates the whole backward
// path including the LSTM).
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "nn/net.h"

using namespace nn;

static void fillRandom(Mat& m, std::mt19937& rng, float scale = 0.3f) {
    std::uniform_real_distribution<float> d(-scale, scale);
    for (float& v : m.d) v = d(rng);
}

int main() {
    NetConfig cfg{16, 8};
    const int B = 2;
    Actor actor;
    actor.init(cfg, 42);
    Critic critic;
    critic.init(cfg, 99);

    std::mt19937 rng(7);
    Mat xImp(B, kImpInput), seq(B * kLstmSteps, kLstmIn), mask(B, kNumActions),
        dynFeat(B * kNumActions, kActionDyn), extra(B, kExtraInput);
    fillRandom(xImp, rng, 0.1f);
    fillRandom(seq, rng, 0.1f);
    fillRandom(extra, rng, 0.1f);
    for (int i = 0; i < B; ++i) {
        for (int j = 0; j < kNumActions; ++j) mask.row(i)[j] = 0.0f;
        mask.row(i)[5] = 1.0f;
        mask.row(i)[620] = 1.0f;
        for (int j = 0; j < kNumActions; ++j)
            for (int k = 0; k < kActionDyn; ++k)
                dynFeat.d[size_t(i) * kNumActions * kActionDyn +
                          j * kActionDyn + k] = 0.0f;
        dynFeat.d[size_t(i) * kNumActions * kActionDyn + 5 * kActionDyn + 6] =
            0.1f;
    }
    std::vector<int> acts = {5, 620};

    auto actorLoss = [&]() {
        Mat& lg = actor.forward(xImp, seq, mask, dynFeat);
        double l = 0;
        for (int i = 0; i < B; ++i) {
            float mx = -1e30f;
            for (int a = 0; a < kNumActions; ++a) mx = std::max(mx, lg.row(i)[a]);
            float z = 0;
            for (int a = 0; a < kNumActions; ++a)
                z += std::exp(lg.row(i)[a] - mx);
            l -= lg.row(i)[acts[i]] - mx - std::log(z);
        }
        return l / B;
    };

    // analytic gradient
    actor.zeroGrad();
    actorLoss();
    Mat dLogits(B, kNumActions);
    {
        Mat& lg = actor.logits;
        for (int i = 0; i < B; ++i) {
            float probs[kNumActions], sum = 0;
            float mx = -1e30f;
            for (int a = 0; a < kNumActions; ++a) mx = std::max(mx, lg.row(i)[a]);
            for (int a = 0; a < kNumActions; ++a) {
                probs[a] = std::exp(lg.row(i)[a] - mx);
                sum += probs[a];
            }
            for (int a = 0; a < kNumActions; ++a) probs[a] /= sum;
            for (int a = 0; a < kNumActions; ++a)
                dLogits.row(i)[a] = -((a == acts[i] ? 1.0f : 0.0f) - probs[a]) / B;
        }
    }
    actor.backward(dLogits);

    int checked = 0, failed = 0;
    auto checkParam = [&](const char* name, Param& p) {
        std::uniform_int_distribution<int> di(0, int(p.w.size()) - 1);
        for (int t = 0; t < 8; ++t) {
            size_t idx = di(rng);
            float eps = 1e-3f;
            float orig = p.w[idx];
            p.w[idx] = orig + eps;
            actor.prepareInference();
            double lp = actorLoss();
            p.w[idx] = orig - eps;
            actor.prepareInference();
            double lm = actorLoss();
            p.w[idx] = orig;
            actor.prepareInference();
            double num = (lp - lm) / (2 * eps);
            double ana = p.dw[idx];
            bool ok = std::abs(ana) < 1e-4 ? std::abs(num - ana) < 2e-5
                                           : std::abs(num - ana) /
                                                     (std::abs(num) +
                                                      std::abs(ana) + 1e-9) <
                                                 2e-2;
            if (!ok) {
                std::printf("  ACTOR %s idx=%zu ana=%f num=%f FAIL\n", name,
                            idx, ana, num);
                ++failed;
            }
            ++checked;
        }
    };
    checkParam("l1.W", actor.l1.W);
    checkParam("l4.W", actor.l4.W);
    checkParam("head.W", actor.head.W);
    checkParam("lstm.Wi", actor.lstm.Wi);
    checkParam("lstm.Wh", actor.lstm.Wh);
    checkParam("lstm.bi", actor.lstm.bi);
    checkParam("dyn.W", actor.dyn.W);

    // ---- critic ----
    auto criticLoss = [&]() {
        Mat& v = critic.forward(xImp, seq, extra);
        double l = 0;
        for (int i = 0; i < B; ++i) l += 0.5 * (v.row(i)[0] - 0.7f) *
                                                  (v.row(i)[0] - 0.7f);
        return l / B;
    };
    critic.zeroGrad();
    Mat& v = critic.forward(xImp, seq, extra);
    Mat dV(B, 1);
    for (int i = 0; i < B; ++i) dV.row(i)[0] = (v.row(i)[0] - 0.7f) / B;
    critic.backward(dV);
    auto checkCritic = [&](const char* name, Param& p) {
        std::uniform_int_distribution<int> di(0, int(p.w.size()) - 1);
        for (int t = 0; t < 8; ++t) {
            size_t idx = di(rng);
            float eps = 1e-3f, orig = p.w[idx];
            p.w[idx] = orig + eps;
            critic.prepareInference();
            double lp = criticLoss();
            p.w[idx] = orig - eps;
            critic.prepareInference();
            double lm = criticLoss();
            p.w[idx] = orig;
            critic.prepareInference();
            double num = (lp - lm) / (2 * eps), ana = p.dw[idx];
            bool ok = std::abs(ana) < 1e-4 ? std::abs(num - ana) < 2e-5
                                           : std::abs(num - ana) /
                                                     (std::abs(num) +
                                                      std::abs(ana) + 1e-9) <
                                                 2e-2;
            if (!ok) {
                std::printf("  CRITIC %s idx=%zu ana=%f num=%f FAIL\n",
                            name, idx, ana, num);
                ++failed;
            }
            ++checked;
        }
    };
    checkCritic("i1.W", critic.i1.W);
    checkCritic("c1.W", critic.c1.W);
    checkCritic("out.W", critic.out.W);
    checkCritic("p1.W", critic.p1.W);
    checkCritic("lstm.Wh", critic.lstm.Wh);

    std::printf("gradient check: %d params, %d failures\n", checked, failed);
    return failed == 0 ? 0 : 1;
}
