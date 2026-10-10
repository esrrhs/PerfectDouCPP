#include "nn/net.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

#include "nn/kernels.h"

namespace nn {

Param::~Param() {
    gpuDropCache(&devW);
    gpuDropCache(&devG);
}

Param& Param::operator=(Param&& o) noexcept {
    if (this != &o) {
        gpuDropCache(&devW);
        gpuDropCache(&devG);
        w = std::move(o.w);
        dw = std::move(o.dw);
        m = std::move(o.m);
        v = std::move(o.v);
        rows = o.rows;
        cols = o.cols;
        devW = o.devW;
        devG = o.devG;
        o.devW = nullptr;
        o.devG = nullptr;
        o.rows = 0;
        o.cols = 0;
    }
    return *this;
}

void Param::zeroGrad() {
    std::fill(dw.begin(), dw.end(), 0.0f);
    // Re-seed the existing device accumulator from the zeroed host gradient.
    gpuStaleCache(&devG);
}

namespace {

void heInit(Param& p, Rng64& rng) {
    float bound = std::sqrt(6.0f / float(p.cols + p.rows));
    for (float& v : p.w) v = rng.uniform(-bound, bound);
}

void writeInt(FILE* f, int32_t v) { std::fwrite(&v, sizeof(v), 1, f); }
int32_t readInt(FILE* f) {
    int32_t v = 0;
    if (std::fread(&v, sizeof(v), 1, f) != 1)
        throw std::runtime_error("short read");
    return v;
}

void saveParam(FILE* f, const Param& p) {
    writeInt(f, p.rows);
    writeInt(f, p.cols);
    std::fwrite(p.w.data(), sizeof(float), p.size(), f);
}
void loadParam(FILE* f, Param& p) {
    int r = readInt(f), c = readInt(f);
    if (r != p.rows || c != p.cols)
        throw std::runtime_error("parameter shape mismatch");
    size_t n = size_t(p.size());
    if (std::fread(p.w.data(), sizeof(float), n, f) != n)
        throw std::runtime_error("short param read");
}

void saveParams(FILE* f, const std::vector<Param*>& ps) {
    writeInt(f, int32_t(ps.size()));
    for (const Param* p : ps) saveParam(f, *p);
}
void loadParams(FILE* f, const std::vector<Param*>& ps) {
    int n = readInt(f);
    if (n != int(ps.size())) throw std::runtime_error("param count mismatch");
    for (Param* p : ps) loadParam(f, *p);
}

}  // namespace

// ---------------------------------------------------------------------------
// Linear
// ---------------------------------------------------------------------------
void Linear::init(int in, int out, Rng64& rng) {
    W.init(out, in);
    b.init(1, out);
    heInit(W, rng);
    std::fill(b.w.begin(), b.w.end(), 0.0f);
    buildWT();
}

void Linear::buildWT() {
    wt.resize(W.cols, W.rows);
    for (int q = 0; q < W.rows; ++q)
        for (int t = 0; t < W.cols; ++t) wt.row(t)[q] = W.w[size_t(q) * W.cols + t];
}

void linearForward(const Mat& x, const Mat& wt, const std::vector<float>& bias,
                   Mat& y) {
    matmul(x, wt, y, /*weightB=*/true);
    kAddBias(y, bias);
}

const Mat& Linear::forward(const Mat& x) {
    inCache.ref(x);
    linearForward(x, wt, b.w, outCache);
    return outCache;
}

const Mat& Linear::forwardAct(const Mat& x) {
    inCache.ref(x);
    outCache.resize(x.r, W.rows);
    GemmOp g{'N', 'N', x.r, W.rows, W.cols,
             x.data(), x.s, wt.data(), wt.s, outCache.data(), outCache.s};
    g.wSlot = const_cast<void**>(&wt.devCache);
    gpuGemm(g, b.w.data(), W.rows, /*epi=*/1);
    return outCache;
}

void Linear::backward(const Mat& gOut, Mat& gIn) {
    const int r = gOut.r, k = W.cols, o = W.rows;
    // dW += gOut^T * inCache and gIn = gOut * W are independent: encode both
    // back to back (single FIFO sequence, one sync at the minibatch edge).
    dwTmp.resize(o, k);
    gIn.resize(r, k);
    GemmOp gDw{'T', 'N', o, k, r, gOut.data(), gOut.s,
               inCache.data(), inCache.s, dwTmp.data(), dwTmp.s};
    GemmOp gDx{'N', 'N', r, k, o, gOut.data(), gOut.s,
               W.w.data(), k, gIn.data(), gIn.s};
    gDx.wSlot = &W.devW;
    gpuGemm(gDw, nullptr, 0, 0);
    gpuGemm(gDx, nullptr, 0, 0);
    kAddTo(W.dw.data(), &W.devG, k, dwTmp);
    kBiasGradAdd(gOut, b.dw.data(), &b.devG);
}

void Linear::zeroGrad() {
    W.zeroGrad();
    b.zeroGrad();
}
void Linear::save(FILE* f) const {
    saveParam(f, W);
    saveParam(f, b);
}
void Linear::load(FILE* f) {
    loadParam(f, W);
    loadParam(f, b);
}

// ---------------------------------------------------------------------------
// LSTM (gates ordered: input, forget, candidate, output)
// ---------------------------------------------------------------------------
void Lstm::init(int inputSize, int hiddenSize, Rng64& rng) {
    n = inputSize;
    h = hiddenSize;
    Wi.init(4 * h, n);
    bi.init(1, 4 * h);
    Wh.init(4 * h, h);
    bh.init(1, 4 * h);
    float boundI = std::sqrt(1.0f / float(n));
    float boundH = std::sqrt(1.0f / float(h));
    for (float& v : Wi.w) v = rng.uniform(-boundI, boundI);
    for (float& v : Wh.w) v = rng.uniform(-boundH, boundH);
    std::fill(bi.w.begin(), bi.w.end(), 0.0f);
    std::fill(bh.w.begin(), bh.w.end(), 0.0f);
    buildWT();
}

void Lstm::buildWT() {
    wTi.resize(n, 4 * h);
    for (int q = 0; q < 4 * h; ++q)
        for (int u = 0; u < n; ++u) wTi.row(u)[q] = Wi.w[size_t(q) * n + u];
    wTh.resize(h, 4 * h);
    for (int q = 0; q < 4 * h; ++q)
        for (int u = 0; u < h; ++u) wTh.row(u)[q] = Wh.w[size_t(q) * h + u];
}

const Mat& Lstm::forward(const Mat& x, int batch, int steps) {
    B = batch;
    T = steps;
    xCache = x;
    int H = 4 * h;
    gpre.resize(B * T, H);
    statesH.resize((T + 1) * B, h);
    statesC.resize(T * B, h);
    outCache.resize(B * T, h);
    // The fused GPU cell writes the zero initial state itself and never reads
    // these host buffers. Filling them is a few MB of memset on every
    // minibatch. The CPU fallback below still needs the zeros.

    // Input-gate products for all steps in a single batched GEMM: (B*T x n) @ (n x 4h) -> (B*T x 4h).
    gateIAll.resize(B * T, H);
    GemmOp giOp{'N', 'N', B * T, H, n,
                xCache.data(), xCache.s,
                wTi.data(), wTi.s,
                gateIAll.data(), gateIAll.s};
    giOp.wSlot = const_cast<void**>(&wTi.devCache);
    gpuGemm(giOp, nullptr, 0, 0);

    // CPU backward reuses these activations. The GPU cell kernel does not
    // write them; invalidate so a later GPU step cannot read a stale cache.
    actReady = !gpuActive();
    if (actReady) {
        actCache.resize(B * T, H);
        tanhC.resize(B * T, h);
    }
    // The recurrent GEMM, bias add and cell share one row of hidden state, so
    // the CUDA path walks all T steps inside one block per batch row.
    bool seq = kLstmSeqFwd(statesH, wTh, gateIAll, bi.w, bh.w, gpre, statesC, B, T, h);
    if (!seq) {
        std::fill(statesH.d.begin(), statesH.d.end(), 0.0f);
        std::fill(statesC.d.begin(), statesC.d.end(), 0.0f);
        ghGate.resize(B, H);
        for (int t = 0; t < T; ++t) {
            Mat hPrev = viewRows(statesH, t * B, B);
            Mat gp = viewRows(gpre, t * B, B);
            GemmOp g{'N', 'N', B, H, h, hPrev.data(), hPrev.s,
                     wTh.data(), wTh.s, ghGate.data(), ghGate.s};
            g.wSlot = const_cast<void**>(&wTh.devCache);
            gpuGemm(g, nullptr, 0, 0);
            Mat gi = viewRows(gateIAll, t * B, B);
            kGateAdd(gp, gi, ghGate, bi.w, bh.w);

            Mat cNow = viewRows(statesC, t * B, B);
            Mat hNext = viewRows(statesH, (t + 1) * B, B);
            const float* cp = t == 0 ? nullptr
                                     : statesC.data() + size_t(t - 1) * B * statesC.s;
            Mat gateV, tanhV;
            Mat* gateP = nullptr;
            Mat* tanhP = nullptr;
            if (actReady) {
                gateV = viewRows(actCache, t * B, B);
                tanhV = viewRows(tanhC, t * B, B);
                gateP = &gateV;
                tanhP = &tanhV;
            }
            kLstmCellFwd(gp, cp, hNext, cNow, h, gateP, tanhP);
        }
    }

    hlLast = viewRows(statesH, T * B, B);
    return outCache;
}

void Lstm::backward(const Mat& ghAll) {
    int H = 4 * h;
    dgAll.resize(B * T, H);
    dhBuf[0].resize(B, h);
    dhBuf[1].resize(B, h);
    dcBuf.resize(B, h);
    std::fill(dhBuf[0].d.begin(), dhBuf[0].d.end(), 0.0f);
    std::fill(dcBuf.d.begin(), dcBuf.d.end(), 0.0f);
    // The running recurrent gradients start zero and are read/written on the
    // device; stage the initial contents into the residency window.
    gpuStageInput(dhBuf[0].data(), B * dhBuf[0].s);
    gpuStageInput(dcBuf.data(), B * dcBuf.s);

    bool seq = kLstmSeqBwd(ghAll, gpre, statesC, Wh.w.data(), &Wh.devW, dgAll, B, T, h);
    if (seq) {
        // sum_t dg_t^T * h_t  and  sum_t dg_t^T * x_t  are two GEMMs with
        // K = B*T. h_0..h_{T-1} are the leading rows of statesH, and xCache
        // is already timestep-major in that same order.
        gwh.resize(H, h);
        GemmOp gWh{'T', 'N', H, h, B * T, dgAll.data(), dgAll.s,
                   statesH.data(), statesH.s, gwh.data(), gwh.s};
        gpuGemm(gWh, nullptr, 0, 0);
        if (gwCache.empty()) gwCache.resize(1);
        gwCache[0].resize(H, n);
        GemmOp gWi{'T', 'N', H, n, B * T, dgAll.data(), dgAll.s,
                   xCache.data(), xCache.s, gwCache[0].data(), gwCache[0].s};
        gpuGemm(gWi, nullptr, 0, 0);
        kAddTo(Wh.dw.data(), &Wh.devG, h, gwh);
        kAddTo(Wi.dw.data(), &Wi.devG, n, gwCache[0]);
        kBiasGradAdd(dgAll, bi.dw.data(), &bi.devG);
        kBiasGradAdd(dgAll, bh.dw.data(), &bh.devG);
        return;
    }
    gwCache.resize(T);
    std::vector<GemmOp> gwOps(T);
    for (int t = T - 1; t >= 0; --t) {
        int idx = (T - 1 - t) & 1;
        Mat gh = viewRows(ghAll, t * B, B);
        Mat gp = viewRows(gpre, t * B, B);
        Mat cNow = viewRows(statesC, t * B, B);
        Mat dg = viewRows(dgAll, t * B, B);
        const float* cp = t == 0 ? nullptr
                                 : statesC.data() + size_t(t - 1) * B * statesC.s;
        Mat gateV, tanhV;
        const Mat* gateP = nullptr;
        const Mat* tanhP = nullptr;
        if (actReady) {
            gateV = viewRows(actCache, t * B, B);
            tanhV = viewRows(tanhC, t * B, B);
            gateP = &gateV;
            tanhP = &tanhV;
        }
        kLstmCellBwd(gh, gp, cNow, cp, dhBuf[idx], dcBuf, dg, h, gateP, tanhP);

        Mat hPrev = viewRows(statesH, t * B, B);
        gwh.resize(H, h);
        GemmOp gWh{'T', 'N', H, h, B, dg.data(), dg.s,
                   hPrev.data(), hPrev.s, gwh.data(), gwh.s};
        GemmOp gDh{'N', 'N', B, h, H, dg.data(), dg.s,
                   Wh.w.data(), h, dhBuf[1 - idx].data(), dhBuf[1 - idx].s};
        gDh.wSlot = &Wh.devW;
        gpuGemm(gWh, nullptr, 0, 0);
        gpuGemm(gDh, nullptr, 0, 0);
        kAddTo(Wh.dw.data(), &Wh.devG, h, gwh);
        kBiasGradAdd(dg, bi.dw.data(), &bi.devG);
        kBiasGradAdd(dg, bh.dw.data(), &bh.devG);

        gwCache[t].resize(H, n);
        gwOps[t] = GemmOp{'T', 'N', H, n, B, dg.data(), dg.s,
                          xCache.data() + size_t(t) * B * xCache.s, xCache.s,
                          gwCache[t].data(), gwCache[t].s};
    }
    gpuCommitGemms(T, gwOps.data());
    for (int t = T - 1; t >= 0; --t)
        kAddTo(Wi.dw.data(), &Wi.devG, n, gwCache[t]);
}

void Lstm::zeroGrad() {
    Wi.zeroGrad();
    bi.zeroGrad();
    Wh.zeroGrad();
    bh.zeroGrad();
}
void Lstm::save(FILE* f) const {
    saveParam(f, Wi);
    saveParam(f, bi);
    saveParam(f, Wh);
    saveParam(f, bh);
}
void Lstm::load(FILE* f) {
    loadParam(f, Wi);
    loadParam(f, bi);
    loadParam(f, Wh);
    loadParam(f, bh);
}

// ---------------------------------------------------------------------------
// Actor
// ---------------------------------------------------------------------------
void Actor::init(const NetConfig& c, uint64_t seed) {
    cfg = c;
    Rng64 rng(seed);
    lstm.init(kLstmIn, c.lstmHidden, rng);
    l1.init(kImpInput + c.lstmHidden + kActionInput, c.hidden, rng);
    l2.init(c.hidden, c.hidden, rng);
    l3.init(c.hidden, c.hidden, rng);
    l4.init(c.hidden, c.hidden * 2, rng);
    head.init(c.hidden * 2, 1, rng);
    std::fill(head.W.w.begin(), head.W.w.end(), 0.0f);
    std::fill(head.b.w.begin(), head.b.w.end(), 0.0f);
    prepareInference();
}

void Actor::prepareInference() {
    lstm.buildWT();
    l1.buildWT();
    l2.buildWT();
    l3.buildWT();
    l4.buildWT();
    head.buildWT();
}

Mat& Actor::forward(const Mat& xImp, const Mat& seq, const Mat& actionFeat,
                    const Mat& actionSample, const Mat& actionId,
                    const Mat& actionOffset) {
    int B = xImp.r;
    lstm.forward(seq, B, kLstmSteps);
    node.resize(B, kImpInput + cfg.lstmHidden);
    kConcat(node, xImp, kImpInput, lstm.hlLast, cfg.lstmHidden);
    joint.resize(actionFeat.r, node.c + kActionInput);
    kRaggedConcat(joint, node, actionFeat, actionSample);
    actionSampleCache.ref(actionSample);
    actionIdCache.ref(actionId);
    actionOffsetCache.ref(actionOffset);
    f1.ref(l1.forwardAct(joint));
    f2.ref(l2.forwardAct(f1));
    f3.ref(l3.forwardAct(f2));
    feat.ref(l4.forwardAct(f3));
    scores.ref(head.forward(feat));
    kRaggedScatter(logits, scores, actionSample, actionId, B, kNumActions);
    return logits;
}

void Actor::backward(const Mat& dLogits) {
    int B = dLogits.r;
    kRaggedGather(gScores, dLogits, actionSampleCache, actionIdCache);
    head.backward(gScores, gFeat);
    kReluBwd(feat, gFeat, gF3);
    l4.backward(gF3, gP3);
    kReluBwd(f3, gP3, gF2);
    l3.backward(gF2, gP2);
    kReluBwd(f2, gP2, gF1);
    l2.backward(gF1, gP1);
    kReluBwd(f1, gP1, gXImp);
    l1.backward(gXImp, gJoint);

    gNode.resize(B, kImpInput + cfg.lstmHidden);
    kRaggedNodeReduce(gNode, gJoint, actionOffsetCache, gNode.c);

    ghLast.resize(B, cfg.lstmHidden);
    kSlice(ghLast, gNode, kImpInput, cfg.lstmHidden);
    ghAll.resize(B * kLstmSteps, cfg.lstmHidden);
    kZeroAndLast(ghAll, ghLast, B, kLstmSteps, cfg.lstmHidden);
    lstm.backward(ghAll);
}

void Actor::zeroGrad() {
    lstm.zeroGrad();
    l1.zeroGrad();
    l2.zeroGrad();
    l3.zeroGrad();
    l4.zeroGrad();
    head.zeroGrad();
}

std::vector<Param*> Actor::params() {
    return {&lstm.Wi, &lstm.bi, &lstm.Wh, &lstm.bh,
            &l1.W, &l1.b, &l2.W, &l2.b, &l3.W, &l3.b,
            &l4.W, &l4.b, &head.W, &head.b};
}

void Actor::save(const char* path) const {
    FILE* f = std::fopen(path, "wb");
    if (!f) throw std::runtime_error("cannot open file for writing");
    std::fwrite("PDAC", 1, 4, f);
    writeInt(f, cfg.hidden);
    writeInt(f, cfg.lstmHidden);
    saveParams(f, const_cast<Actor*>(this)->params());
    std::fclose(f);
}
void Actor::load(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) throw std::runtime_error("cannot open model file");
    char magic[4] = {0};
    if (std::fread(magic, 1, 4, f) != 4)
        throw std::runtime_error("bad model file");
    NetConfig c;
    c.hidden = readInt(f);
    c.lstmHidden = readInt(f);
    init(c, 1);
    loadParams(f, params());
    std::fclose(f);
    prepareInference();
}

// ---------------------------------------------------------------------------
// Critic
// ---------------------------------------------------------------------------
void Critic::init(const NetConfig& c, uint64_t seed) {
    cfg = c;
    Rng64 rng(seed);
    lstm.init(kLstmIn, c.lstmHidden, rng);
    c1.init(kImpInput + c.lstmHidden + kExtraInput, c.hidden, rng);
    c2.init(c.hidden, c.hidden, rng);
    c3.init(c.hidden, c.hidden, rng);
    c4.init(c.hidden, c.hidden, rng);
    out.init(c.hidden, 1, rng);
    std::fill(out.W.w.begin(), out.W.w.end(), 0.0f);
    std::fill(out.b.w.begin(), out.b.w.end(), 0.0f);
    prepareInference();
}

void Critic::prepareInference() {
    lstm.buildWT();
    c1.buildWT();
    c2.buildWT();
    c3.buildWT();
    c4.buildWT();
    out.buildWT();
}

Mat& Critic::forward(const Mat& xImp, const Mat& seq, const Mat& extra) {
    int B = xImp.r;
    lstm.forward(seq, B, kLstmSteps);
    node.resize(B, kImpInput + cfg.lstmHidden);
    kConcat(node, xImp, kImpInput, lstm.hlLast, cfg.lstmHidden);
    all.resize(B, node.c + kExtraInput);
    kConcat(all, node, node.c, extra, kExtraInput);
    q1.ref(c1.forwardAct(all));
    q2.ref(c2.forwardAct(q1));
    q3.ref(c3.forwardAct(q2));
    q4.ref(c4.forwardAct(q3));
    value.ref(out.forward(q4));
    return value;
}

void Critic::backward(const Mat& dValue) {
    int B = dValue.r;
    out.backward(dValue, gQ4);
    kReluBwd(q4, gQ4, gP4);
    c4.backward(gP4, gQ3);
    kReluBwd(q3, gQ3, gP3);
    c3.backward(gP3, gQ2);
    kReluBwd(q2, gQ2, gP2);
    c2.backward(gP2, gQ1);
    kReluBwd(q1, gQ1, gP1);
    c1.backward(gP1, gAll);

    gNode.resize(B, kImpInput + cfg.lstmHidden);
    gExtra.resize(B, kExtraInput);
    kSplit(gAll, gNode.c, gNode, gExtra, kExtraInput);

    ghLast.resize(B, cfg.lstmHidden);
    kSlice(ghLast, gNode, kImpInput, cfg.lstmHidden);
    ghAll.resize(B * kLstmSteps, cfg.lstmHidden);
    kZeroAndLast(ghAll, ghLast, B, kLstmSteps, cfg.lstmHidden);
    lstm.backward(ghAll);
}

void Critic::zeroGrad() {
    lstm.zeroGrad();
    c1.zeroGrad();
    c2.zeroGrad();
    c3.zeroGrad();
    c4.zeroGrad();
    out.zeroGrad();
}

std::vector<Param*> Critic::params() {
    return {&lstm.Wi, &lstm.bi, &lstm.Wh, &lstm.bh,
            &c1.W, &c1.b, &c2.W, &c2.b, &c3.W, &c3.b,
            &c4.W, &c4.b, &out.W, &out.b};
}

void Critic::save(const char* path) const {
    FILE* f = std::fopen(path, "wb");
    if (!f) throw std::runtime_error("cannot open file for writing");
    std::fwrite("PDCR", 1, 4, f);
    writeInt(f, cfg.hidden);
    writeInt(f, cfg.lstmHidden);
    saveParams(f, const_cast<Critic*>(this)->params());
    std::fclose(f);
}
void Critic::load(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) throw std::runtime_error("cannot open model file");
    char magic[4] = {0};
    if (std::fread(magic, 1, 4, f) != 4)
        throw std::runtime_error("bad model file");
    NetConfig c;
    c.hidden = readInt(f);
    c.lstmHidden = readInt(f);
    init(c, 1);
    loadParams(f, params());
    std::fclose(f);
    prepareInference();
}

// ---------------------------------------------------------------------------
// Thread-safe inference (weights read-only; runs entirely on CPU)
// ---------------------------------------------------------------------------
void lstmInferForward(const Lstm& l, const Mat& x, int B, int T,
                      LstmInfer& w) {
    const int h = l.h, H = 4 * h, n = l.n;
    w.B = B;
    w.T = T;
    w.hp.resize(B, h);
    w.cp.resize(B, h);
    w.hc.resize(B, h);
    w.cc.resize(B, h);
    w.gp.resize(B, H);
    w.hl.resize(B, h);
    std::fill(w.hp.d.begin(), w.hp.d.end(), 0.0f);
    std::fill(w.cp.d.begin(), w.cp.d.end(), 0.0f);
    w.giAll.resize(T);
    std::vector<GemmOp> giOps(T);
    for (int t = 0; t < T; ++t) {
        w.giAll[t].resize(B, H);
        giOps[t] = GemmOp{'N', 'N', B, H, n,
                          x.data() + size_t(t) * B * x.s, x.s,
                          l.wTi.data(), l.wTi.s,
                          w.giAll[t].data(), w.giAll[t].s};
    }
    gpuCommitGemms(T, giOps.data());
    Mat gateH;
    for (int t = 0; t < T; ++t) {
        matmul(w.hp, l.wTh, gateH);
        w.gp.resize(B, H);
        for (int ib = 0; ib < B; ++ib) {
            float* gp = w.gp.row(ib);
            const float* gi = w.giAll[t].row(ib);
            const float* gh = gateH.row(ib);
            for (int q = 0; q < H; ++q)
                gp[q] = gi[q] + gh[q] + l.bi.w[q] + l.bh.w[q];
        }
        kLstmCellFwd(w.gp, w.cp.data(), w.hc, w.cc, h);
        if (t == T - 1)
            for (int ib = 0; ib < B; ++ib)
                std::copy(w.hc.row(ib), w.hc.row(ib) + h, w.hl.row(ib));
        std::swap(w.hp, w.hc);
        std::swap(w.cp, w.cc);
    }
}

const Mat& actorInferForward(const Actor& a, ActorInfer& w, const Mat& xImp,
                             const Mat& seq, const Mat& actionFeat,
                             const Mat& actionSample, const Mat& actionId,
                             const Mat& actionOffset) {
    (void)actionOffset;
    int B = xImp.r;
    lstmInferForward(a.lstm, seq, B, kLstmSteps, w.lstm);
    w.node.resize(B, kImpInput + a.cfg.lstmHidden);
    for (int i = 0; i < B; ++i) {
        std::copy(xImp.row(i), xImp.row(i) + kImpInput, w.node.row(i));
        std::copy(w.lstm.hl.row(i), w.lstm.hl.row(i) + a.cfg.lstmHidden,
                  w.node.row(i) + kImpInput);
    }
    w.joint.resize(actionFeat.r, w.node.c + kActionInput);
    for (int r = 0; r < actionFeat.r; ++r) {
        int s = int(actionSample.row(r)[0]);
        std::copy(w.node.row(s), w.node.row(s) + w.node.c, w.joint.row(r));
        std::copy(actionFeat.row(r), actionFeat.row(r) + kActionInput,
                  w.joint.row(r) + w.node.c);
    }
    linearForward(w.joint, a.l1.wt, a.l1.b.w, w.f1);
    reluFwd(w.f1);
    linearForward(w.f1, a.l2.wt, a.l2.b.w, w.f2);
    reluFwd(w.f2);
    linearForward(w.f2, a.l3.wt, a.l3.b.w, w.f3);
    reluFwd(w.f3);
    linearForward(w.f3, a.l4.wt, a.l4.b.w, w.feat);
    reluFwd(w.feat);
    linearForward(w.feat, a.head.wt, a.head.b.w, w.scores);
    w.logits.resize(B, kNumActions);
    std::fill(w.logits.d.begin(), w.logits.d.end(), -1e9f);
    for (int r = 0; r < actionFeat.r; ++r)
        w.logits.row(int(actionSample.row(r)[0]))[int(actionId.row(r)[0])] =
            w.scores.row(r)[0];
    return w.logits;
}

const Mat& criticInferForward(const Critic& c, CriticInfer& w,
                              const Mat& xImp, const Mat& seq,
                              const Mat& extra) {
    int B = xImp.r;
    lstmInferForward(c.lstm, seq, B, kLstmSteps, w.lstm);
    w.node.resize(B, kImpInput + c.cfg.lstmHidden);
    for (int i = 0; i < B; ++i) {
        std::copy(xImp.row(i), xImp.row(i) + kImpInput, w.node.row(i));
        std::copy(w.lstm.hl.row(i), w.lstm.hl.row(i) + c.cfg.lstmHidden,
                  w.node.row(i) + kImpInput);
    }
    w.all.resize(B, w.node.c + kExtraInput);
    for (int i = 0; i < B; ++i) {
        std::copy(w.node.row(i), w.node.row(i) + w.node.c, w.all.row(i));
        std::copy(extra.row(i), extra.row(i) + kExtraInput,
                  w.all.row(i) + w.node.c);
    }
    linearForward(w.all, c.c1.wt, c.c1.b.w, w.q1);
    reluFwd(w.q1);
    linearForward(w.q1, c.c2.wt, c.c2.b.w, w.q2);
    reluFwd(w.q2);
    linearForward(w.q2, c.c3.wt, c.c3.b.w, w.q3);
    reluFwd(w.q3);
    linearForward(w.q3, c.c4.wt, c.c4.b.w, w.q4);
    reluFwd(w.q4);
    linearForward(w.q4, c.out.wt, c.out.b.w, w.value);
    return w.value;
}

// ---------------------------------------------------------------------------
// Adam
// ---------------------------------------------------------------------------
void Adam::apply(const std::vector<Param*>& ps) {
    ++t;
    float bc1 = 1.0f - std::pow(beta1, float(t));
    float bc2 = 1.0f - std::pow(beta2, float(t));
    for (Param* p : ps)
        kAdam(p->w, p->dw, p->m, p->v, p->size(), lr,
              beta1, beta2, eps, bc1, bc2);
}

void Adam::applyGradNorm(const std::vector<Param*>& ps, float maxNorm) {
    // Gradients are produced asynchronously into per-parameter device
    // accumulators; fence, then flush each slot before scanning on host.
    gpuWait();
    if (!gpuDeviceOk()) return;
    for (Param* p : ps) {
        size_t bytes = size_t(padStride(p->rows * p->cols)) * 4;
        gpuFlushGrad(p->devG, p->dw.data(), bytes);
    }
    double sum = 0.0;
    for (Param* p : ps)
        for (float g : p->dw) sum += double(g) * g;
    float norm = float(std::sqrt(sum));
    if (!std::isfinite(norm)) {
        // A NaN/Inf gradient would poison every weight and Adam moment.
        std::fprintf(stderr, "WARN: non-finite gradient norm, skipping optimizer step\n");
        return;
    }
    if (norm > maxNorm) {
        float scale = maxNorm / (norm + 1e-6f);
        for (Param* p : ps)
            for (float& g : p->dw) g *= scale;
    }
    apply(ps);
    // Weights must be resident before buildWT() / CPU inference read them.
    gpuWait();
    // The step counter advanced, but a lost device does not copy w/m/v back.
    if (!gpuDeviceOk()) --t;
}

void saveOptimizer(const char* path, const std::vector<Param*>& ps, const Adam& opt) {
    FILE* f = std::fopen(path, "wb");
    if (!f) return;
    std::fwrite("PDOP", 1, 4, f);
    std::fwrite(&opt.t, sizeof(opt.t), 1, f);
    std::fwrite(&opt.lr, sizeof(opt.lr), 1, f);
    std::fwrite(&opt.beta1, sizeof(opt.beta1), 1, f);
    std::fwrite(&opt.beta2, sizeof(opt.beta2), 1, f);
    std::fwrite(&opt.eps, sizeof(opt.eps), 1, f);
    int32_t n = int32_t(ps.size());
    std::fwrite(&n, sizeof(n), 1, f);
    for (const Param* p : ps) {
        int sz = p->size();
        std::fwrite(&sz, sizeof(sz), 1, f);
        if (sz > 0) {
            if (p->m.size() == size_t(sz)) {
                std::fwrite(p->m.data(), sizeof(float), sz, f);
            } else {
                std::vector<float> zeros(sz, 0.0f);
                std::fwrite(zeros.data(), sizeof(float), sz, f);
            }
            if (p->v.size() == size_t(sz)) {
                std::fwrite(p->v.data(), sizeof(float), sz, f);
            } else {
                std::vector<float> zeros(sz, 0.0f);
                std::fwrite(zeros.data(), sizeof(float), sz, f);
            }
        }
    }
    std::fclose(f);
}

bool loadOptimizer(const char* path, const std::vector<Param*>& ps, Adam& opt) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    // Parse into temporaries so a truncated or mismatched file leaves the
    // optimizer and the parameters' moments untouched.
    char magic[4] = {0};
    Adam tmp = opt;
    if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "PDOP", 4) != 0 ||
        std::fread(&tmp.t, sizeof(tmp.t), 1, f) != 1 ||
        std::fread(&tmp.lr, sizeof(tmp.lr), 1, f) != 1 ||
        std::fread(&tmp.beta1, sizeof(tmp.beta1), 1, f) != 1 ||
        std::fread(&tmp.beta2, sizeof(tmp.beta2), 1, f) != 1 ||
        std::fread(&tmp.eps, sizeof(tmp.eps), 1, f) != 1) {
        std::fclose(f);
        return false;
    }
    int32_t n = 0;
    if (std::fread(&n, sizeof(n), 1, f) != 1 || n != int32_t(ps.size())) {
        std::fclose(f);
        return false;
    }
    std::vector<std::vector<float>> ms(ps.size()), vs(ps.size());
    for (size_t k = 0; k < ps.size(); ++k) {
        int sz = 0;
        if (std::fread(&sz, sizeof(sz), 1, f) != 1 || sz != ps[k]->size()) {
            std::fclose(f);
            return false;
        }
        if (sz > 0) {
            ms[k].resize(sz);
            vs[k].resize(sz);
            if (std::fread(ms[k].data(), sizeof(float), sz, f) != size_t(sz) ||
                std::fread(vs[k].data(), sizeof(float), sz, f) != size_t(sz)) {
                std::fclose(f);
                return false;
            }
        }
    }
    std::fclose(f);
    for (size_t k = 0; k < ps.size(); ++k) {
        if (ps[k]->size() > 0) {
            ps[k]->m = std::move(ms[k]);
            ps[k]->v = std::move(vs[k]);
        }
    }
    opt = tmp;
    return true;
}

}  // namespace nn
