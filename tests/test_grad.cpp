// Numerical gradient checks for actor / critic (validates the whole backward
// path including the LSTM).
#include <atomic>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include "nn/gemm.h"
#include "nn/net.h"

using namespace nn;

static void fillRandom(Mat& m, std::mt19937& rng, float scale = 0.3f) {
    std::uniform_real_distribution<float> d(-scale, scale);
    for (int i = 0; i < m.r; ++i)
        for (int j = 0; j < m.c; ++j) m.row(i)[j] = d(rng);
}

// ---- CPU vs GPU analytic parity over the full actor/critic graph ----------
struct GradSnap {
    std::vector<std::vector<float>> dw;
};

static void snapGrads(const std::vector<Param*>& ps, GradSnap& s) {
    s.dw.resize(ps.size());
    for (size_t k = 0; k < ps.size(); ++k) {
        s.dw[k].resize(ps[k]->size());
        for (int i = 0; i < ps[k]->size(); ++i) s.dw[k][i] = ps[k]->dw[i];
    }
}

static const char* kActorParamNames[] = {
    "lstm.Wi", "lstm.bi", "lstm.Wh", "lstm.bh",
    "l1.W", "l1.b", "l2.W", "l2.b", "l3.W", "l3.b",
    "l4.W", "l4.b", "head.W", "head.b", "dyn.W", "dyn.b"};
static const char* kCriticParamNames[] = {
    "lstm.Wi", "lstm.bi", "lstm.Wh", "lstm.bh",
    "i1.W", "i1.b", "i2.W", "i2.b", "i3.W", "i3.b",
    "p1.W", "p1.b", "p2.W", "p2.b", "c1.W", "c1.b",
    "c2.W", "c2.b", "out.W", "out.b"};

static void cmpGrads(const char* tag, const std::vector<Param*>& ps,
                     const GradSnap& ref, int& failed) {
    const char** names = (tag[0] == 'a') ? kActorParamNames
                                         : kCriticParamNames;
    for (size_t k = 0; k < ps.size(); ++k) {
        int bad = 0, shown = 0;
        double maxAbsCpu = 0, maxAbsGpu = 0;
        for (int i = 0; i < ps[k]->size(); ++i) {
            float a = ref.dw[k][i], b = ps[k]->dw[i];
            maxAbsCpu = std::max(maxAbsCpu, double(std::abs(a)));
            maxAbsGpu = std::max(maxAbsGpu, double(std::abs(b)));
            double den = std::max(std::abs(a), std::abs(b));
            double rel = den > 1e-5 ? std::abs(a - b) / den : std::abs(a - b);
            if (rel > 0.08) {
                if (shown < 3)
                    std::printf("  GPU %s %s idx=%d cpu=%f gpu=%f\n",
                                tag, names[k], i, a, b);
                ++shown;
                ++bad;
            }
        }
        if (bad)
            std::printf("  GPU %s %s: %d/%d mismatched, max|cpu|=%.3e "
                        "max|gpu|=%.3e\n",
                        tag, names[k], bad, ps[k]->size(),
                        maxAbsCpu, maxAbsGpu);
        failed += bad;
    }
}

static void softmaxGrad(const Mat& lg, const std::vector<int>& acts, Mat& dl) {
    int B = lg.r;
    for (int i = 0; i < B; ++i) {
        float probs[kNumActions], sum = 0, mx = -1e30f;
        for (int a = 0; a < kNumActions; ++a) mx = std::max(mx, lg.row(i)[a]);
        for (int a = 0; a < kNumActions; ++a) {
            probs[a] = std::exp(lg.row(i)[a] - mx);
            sum += probs[a];
        }
        for (int a = 0; a < kNumActions; ++a) probs[a] /= sum;
        for (int a = 0; a < kNumActions; ++a)
            dl.row(i)[a] = -((a == acts[i] ? 1.0f : 0.0f) - probs[a]) / B;
    }
}

static float maxOutDiff(const Mat& a, const Mat& b) {
    float m = 0;
    for (int i = 0; i < a.r; ++i)
        for (int j = 0; j < a.c; ++j)
            m = std::max(m, std::abs(a.row(i)[j] - b.row(i)[j]));
    return m;
}

static void flushGrads(const std::vector<Param*>& ps) {
    for (Param* p : ps) {
        size_t bytes = size_t(padStride(p->rows * p->cols)) * 4;
        gpuFlushGrad(p->devG, p->dw.data(), bytes);
    }
}

static void invalidateWeights(const std::vector<Param*>& ps) {
    for (Param* p : ps) gpuDropCache(&p->devW);
}

// Runs forward+backward on the given backend; returns output snapshots.
static void runNet(Actor& actor, Critic& critic, const Mat& xImp,
                   const Mat& seq, const Mat& mask, const Mat& dynFeat,
                   const Mat& extra, const std::vector<int>& acts,
                   bool useGpu, Mat& logitsOut, Mat& valueOut,
                   GradSnap& ag, GradSnap& cg) {
    if (useGpu) gemmSetGpu(true);
    else gemmSetGpu(false);
    int B = xImp.r;

    actor.zeroGrad();
    Mat& lg = actor.forward(xImp, seq, mask, dynFeat);
    if (useGpu) {
        gpuMarkHost(lg.data());
        gpuWaitEx(true);
    }
    Mat dLogits(B, kNumActions);
    softmaxGrad(lg, acts, dLogits);
    actor.backward(dLogits);
    if (useGpu) { gpuWait(); flushGrads(actor.params()); }
    logitsOut.resize(B, kNumActions);
    for (int i = 0; i < B; ++i)
        for (int a = 0; a < kNumActions; ++a)
            logitsOut.row(i)[a] = lg.row(i)[a];
    snapGrads(actor.params(), ag);

    critic.zeroGrad();
    Mat& v = critic.forward(xImp, seq, extra);
    if (useGpu) {
        gpuMarkHost(v.data());
        gpuWaitEx(true);
    }
    Mat dV(B, 1);
    for (int i = 0; i < B; ++i) dV.row(i)[0] = (v.row(i)[0] - 0.7f) / B;
    critic.backward(dV);
    if (useGpu) { gpuWait(); flushGrads(critic.params()); }
    valueOut.resize(B, 1);
    for (int i = 0; i < B; ++i) valueOut.row(i)[0] = v.row(i)[0];
    snapGrads(critic.params(), cg);
}

static int gpuParity() {
    if (!gemmHasGpu()) {
        std::printf("GPU parity: SKIP (no Metal)\n");
        return 0;
    }
    NetConfig cfg{16, 8};
    const int B = 2;
    Actor actor;
    actor.init(cfg, 42);
    Critic critic;
    critic.init(cfg, 99);

    std::mt19937 rng(7);
    // head.W is zero-initialized, which would make the whole MLP trunk
    // gradient vanish; randomize it so parity covers every backward layer.
    {
        std::uniform_real_distribution<float> hd(-0.05f, 0.05f);
        for (float& w : actor.head.W.w) w = hd(rng);
        actor.head.buildWT();
    }

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
                dynFeat.row(i * kNumActions + j)[k] = 0.0f;
        dynFeat.row(i * kNumActions + 5)[6] = 0.1f;
    }
    std::vector<int> acts = {5, 620};

    Mat lgCpu, vCpu, lgGpu, vGpu;
    GradSnap aCpu, cCpu, aGpu, cGpu;
    runNet(actor, critic, xImp, seq, mask, dynFeat, extra, acts, false,
           lgCpu, vCpu, aCpu, cCpu);
    runNet(actor, critic, xImp, seq, mask, dynFeat, extra, acts, true,
           lgGpu, vGpu, aGpu, cGpu);
    gemmSetGpu(false);

    int failed = 0;
    float dl = maxOutDiff(lgCpu, lgGpu);
    float dv = maxOutDiff(vCpu, vGpu);
    std::printf("GPU parity: max logit diff %.3e, value diff %.3e\n", dl, dv);
    // Deep fp32 graph: small absolute output wobble is fine as long as the
    // analytic gradients agree.
    if (dl > 1e-2 || dv > 2e-2) ++failed;
    cmpGrads("actor", actor.params(), aCpu, failed);
    cmpGrads("critic", critic.params(), cCpu, failed);
    std::printf("GPU parity: %s\n", failed == 0 ? "OK" : "FAIL");
    return failed == 0 ? 0 : 1;
}

// ---- Full training-loop parity: realistic sizes, gradient accumulation
// across multiple minibatches/epochs/windows, and Adam weight updates. ----
struct Batch {
    Mat xImp, seq, mask, dynFeat, extra;
    std::vector<int> acts;
    std::vector<float> tgt;
};

static void makeBatches(std::vector<Batch>& batches, int nBatches, int B,
                        std::mt19937& rng) {
    batches.resize(nBatches);
    for (Batch& bt : batches) {
        bt.acts.resize(B);
        bt.tgt.resize(B);
        fillRandom(bt.xImp = Mat(B, kImpInput), rng, 0.1f);
        fillRandom(bt.seq = Mat(B * kLstmSteps, kLstmIn), rng, 0.1f);
        fillRandom(bt.dynFeat = Mat(B * kNumActions, kActionDyn), rng, 0.05f);
        bt.extra.resize(B, kExtraInput);
        fillRandom(bt.extra, rng, 0.1f);
        bt.mask.resize(B, kNumActions);
        std::uniform_real_distribution<float> legal(0.0f, 1.0f), tv(-1.0f, 1.0f);
        std::uniform_int_distribution<int> act(0, kNumActions - 1);
        for (int i = 0; i < B; ++i) {
            for (int a = 0; a < kNumActions; ++a)
                bt.mask.row(i)[a] = (legal(rng) < 0.02f) ? 1.0f : 0.0f;
            // ensure at least one legal action and pick a legal one
            bt.mask.row(i)[0] = 1.0f;
            int a;
            do { a = act(rng); } while (bt.mask.row(i)[a] < 0.5f);
            bt.acts[i] = a;
            bt.tgt[i] = tv(rng);
        }
    }
}

static void trainStep(Actor& actor, Critic& critic, const Batch& bt,
                      bool useGpu, std::vector<float>* lgTrace = nullptr) {
    int B = bt.xImp.r;
    Mat& lg = actor.forward(bt.xImp, bt.seq, bt.mask, bt.dynFeat);
    if (useGpu) { gpuMarkHost(lg.data()); gpuWaitEx(true); }
    if (lgTrace)
        for (int i = 0; i < lg.r * lg.c; ++i)
            lgTrace->push_back(lg.data()[i]);
    Mat dLogits(B, kNumActions);
    softmaxGrad(lg, bt.acts, dLogits);
    actor.backward(dLogits);
    if (useGpu) gpuWait();

    Mat& v = critic.forward(bt.xImp, bt.seq, bt.extra);
    if (useGpu) { gpuMarkHost(v.data()); gpuWaitEx(true); }
    Mat dValue(B, 1);
    for (int i = 0; i < B; ++i)
        dValue.row(i)[0] = (v.row(i)[0] - bt.tgt[i]) / B;
    critic.backward(dValue);
    if (useGpu) gpuWait();
}

static double maxParamFieldDiff(const std::vector<Param*>& a,
                                const std::vector<Param*>& b, bool grad) {
    double m = 0;
    for (size_t k = 0; k < a.size(); ++k) {
        const std::vector<float>& va = grad ? a[k]->dw : a[k]->w;
        const std::vector<float>& vb = grad ? b[k]->dw : b[k]->w;
        for (size_t i = 0; i < va.size(); ++i)
            m = std::max(m, double(std::abs(va[i] - vb[i])));
    }
    return m;
}

static int trainParity() {
    if (!gemmHasGpu()) {
        std::printf("train parity: SKIP (no Metal)\n");
        return 0;
    }
    const NetConfig cfg{128, 64};
    const int B = 32, nBatches = 4, epochs = 2, rounds = 4;
    std::mt19937 rng(123);
    std::vector<Batch> batches;
    makeBatches(batches, nBatches, B, rng);

    Actor ac, ag;
    Critic cc, cg;
    ac.init(cfg, 7);  ag.init(cfg, 7);
    cc.init(cfg, 9);  cg.init(cfg, 9);
    Adam ao, go, co, cgo;
    ao.lr = go.lr = co.lr = cgo.lr = 3e-4f;

    int failed = 0;
    for (int rd = 0; rd < rounds; ++rd) {
        auto runRound = [&](Actor& a, Critic& c, Adam& ao2, Adam& co2,
                            bool gpu) {
            if (gpu) gemmSetGpu(true); else gemmSetGpu(false);
            a.zeroGrad();
            c.zeroGrad();
            for (int ep = 0; ep < epochs; ++ep)
                for (int bi = 0; bi < nBatches; ++bi)
                    trainStep(a, c, batches[bi], gpu);
            if (gpu) { gpuWait(); flushGrads(a.params()); flushGrads(c.params()); }
            ao2.applyGradNorm(a.params(), 1.0f);
            co2.applyGradNorm(c.params(), 1.0f);
            if (gpu) { invalidateWeights(a.params()); invalidateWeights(c.params()); }
            a.prepareInference();
            c.prepareInference();
        };
        runRound(ac, cc, ao, co, false);
        runRound(ag, cg, go, cgo, true);
        gemmSetGpu(false);
        double dwA = maxParamFieldDiff(ac.params(), ag.params(), true);
        double dwC = maxParamFieldDiff(cc.params(), cg.params(), true);
        double wA = maxParamFieldDiff(ac.params(), ag.params(), false);
        double wC = maxParamFieldDiff(cc.params(), cg.params(), false);
        std::printf("train parity round %d: max|Δdw| %.3e/%.3e, "
                    "max|Δw| %.3e/%.3e\n",
                    rd + 1, dwA, dwC, wA, wC);
        if (dwA > 1e-4 || dwC > 1e-4 || wA > 1e-5 || wC > 1e-5) ++failed;
    }
    std::printf("train parity: %s\n", failed == 0 ? "OK" : "FAIL");
    return failed == 0 ? 0 : 1;
}

// ---- Concurrency isolation: 3 independent model sets trained at the same
// time on 3 GPU threads (as the trainer does) must exactly match the same
// work executed sequentially on one GPU thread. ----------------------------
struct SeatSet {
    Actor actor;
    Critic critic;
    Adam ao, co;
    std::vector<Batch> batches;
};

static void buildSeat(SeatSet& m, int seed) {
    NetConfig cfg{128, 64};
    m.actor.init(cfg, uint64_t(seed) * 100 + 1);
    m.critic.init(cfg, uint64_t(seed) * 100 + 53);
    m.ao.lr = m.co.lr = 3e-4f;
    std::mt19937 rng(seed * 7919 + 13);
    makeBatches(m.batches, 4, 32, rng);
}

static void seatRounds(SeatSet& m, int rounds,
                       std::vector<float>* lgTrace = nullptr,
                       std::vector<float>* outWTrace = nullptr) {
    for (int rd = 0; rd < rounds; ++rd) {
        m.actor.zeroGrad();
        m.critic.zeroGrad();
        for (int ep = 0; ep < 2; ++ep)
            for (const Batch& bt : m.batches)
                trainStep(m.actor, m.critic, bt, true,
                          rd == 0 && lgTrace ? lgTrace : nullptr);
        gpuWait();
        flushGrads(m.actor.params()); flushGrads(m.critic.params());
        // critic out.W/out.b gradients BEFORE the optimizer step
        if (outWTrace) {
            for (float g : m.critic.out.W.dw) outWTrace->push_back(g);
            for (float g : m.critic.out.b.dw) outWTrace->push_back(g);
        }
        m.ao.applyGradNorm(m.actor.params(), 1.0f);
        m.co.applyGradNorm(m.critic.params(), 1.0f);
        invalidateWeights(m.actor.params()); invalidateWeights(m.critic.params());
        m.actor.prepareInference();
        m.critic.prepareInference();
        if (outWTrace) {
            for (float w : m.critic.out.W.w) outWTrace->push_back(w);
            for (float w : m.critic.out.b.w) outWTrace->push_back(w);
        }
    }
}

static int concurrentParity() {
    // Independent model sets must produce identical results whether trained
    // sequentially on the submitting thread or concurrently on fresh worker
    // threads (each worker gets a fresh GPU context). Guards against pointer
    // aliasing / stale device-cache bugs across independent models.
    if (!gemmHasGpu()) {
        std::printf("concurrent parity: SKIP (no Metal)\n");
        return 0;
    }
    const int seats = 3, rounds = 4, nBatches = 4, epochs = 2;
    NetConfig cfg{128, 64};

    struct Set {
        Actor actor; Critic critic; Adam ao, co;
        std::vector<Batch> batches;
    };
    auto buildSet = [&](int seed) {
        Set* m = new Set;
        m->actor.init(cfg, seed * 100 + 1);
        m->critic.init(cfg, seed * 100 + 53);
        m->batches.resize(nBatches);
        std::mt19937 rng(seed * 7919u + 13u);
        makeBatches(m->batches, nBatches, 32, rng);
        return m;
    };
    auto runSet = [&](Set& m, bool gpu) {
        for (int rd = 0; rd < rounds; ++rd) {
            m.actor.zeroGrad();
            m.critic.zeroGrad();
            for (int ep = 0; ep < epochs; ++ep)
                for (Batch& bt : m.batches)
                    trainStep(m.actor, m.critic, bt, gpu);
            if (gpu) {
                gpuWait();
                flushGrads(m.actor.params());
                flushGrads(m.critic.params());
            }
            m.ao.applyGradNorm(m.actor.params(), 1.0f);
            m.co.applyGradNorm(m.critic.params(), 1.0f);
            if (gpu) {
                invalidateWeights(m.actor.params());
                invalidateWeights(m.critic.params());
            }
            m.actor.prepareInference();
            m.critic.prepareInference();
        }
    };

    Set* cpu[seats]; Set* seq[seats]; Set* par[seats];
    for (int s = 0; s < seats; ++s) {
        cpu[s] = buildSet(s + 1);
        seq[s] = buildSet(s + 1);
        par[s] = buildSet(s + 1);
    }

    gemmSetGpu(false);
    for (int s = 0; s < seats; ++s) runSet(*cpu[s], false);

    gemmSetGpu(true);
    for (int s = 0; s < seats; ++s) runSet(*seq[s], true);
    std::vector<std::thread> th;
    for (int s = 0; s < seats; ++s)
        th.emplace_back([&, s] { runSet(*par[s], true); });
    for (auto& t : th) t.join();
    gemmSetGpu(false);

    int failed = 0;
    for (int s = 0; s < seats; ++s) {
        double wa = maxParamFieldDiff(cpu[s]->actor.params(),
                                      seq[s]->actor.params(), false);
        double wc = maxParamFieldDiff(cpu[s]->critic.params(),
                                      seq[s]->critic.params(), false);
        double pa = maxParamFieldDiff(cpu[s]->actor.params(),
                                      par[s]->actor.params(), false);
        double pc = maxParamFieldDiff(cpu[s]->critic.params(),
                                      par[s]->critic.params(), false);
        std::printf("  seat %d vs CPU: sequential actor %.3e critic %.3e | "
                    "parallel actor %.3e critic %.3e\n",
                    s, wa, wc, pa, pc);
        for (double d : {wa, wc, pa, pc})
            if (d > 1e-5) ++failed;
    }
    std::printf("concurrent parity: %s\n", failed == 0 ? "OK" : "FAIL");
    return failed == 0 ? 0 : 1;
}

int main() {
    // Numerical reference runs on the synchronous CPU backend.
    gemmInit();
    gemmSetGpu(false);

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
                dynFeat.row(i * kNumActions + j)[k] = 0.0f;
        dynFeat.row(i * kNumActions + 5)[6] = 0.1f;
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
        std::uniform_int_distribution<int> di(0, p.size() - 1);
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
        std::uniform_int_distribution<int> di(0, p.size() - 1);
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
    int numFail = failed;

    // Same graph on GPU: analytic outputs/gradients must match the CPU ones.
    int gpuFail = gpuParity();

    // Realistic sizes: accumulated gradients over several minibatches and
    // Adam-updated weights must also agree, round after round.
    int trainFail = trainParity();

    // Concurrent GPU learners must not perturb one another.
    int concFail = concurrentParity();
    return (numFail == 0 && gpuFail == 0 && trainFail == 0 &&
            concFail == 0)
               ? 0
               : 1;
}
