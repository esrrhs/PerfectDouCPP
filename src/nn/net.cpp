#include "nn/net.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace nn {

namespace {

inline float sigmoidf(float x) {
    if (x < -30.0f) return 0.0f;
    if (x > 30.0f) return 1.0f;
    return 1.0f / (1.0f + std::exp(-x));
}

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
    std::fwrite(p.w.data(), sizeof(float), p.w.size(), f);
}
void loadParam(FILE* f, Param& p) {
    int r = readInt(f), c = readInt(f);
    if (r != p.rows || c != p.cols)
        throw std::runtime_error("parameter shape mismatch");
    if (std::fread(p.w.data(), sizeof(float), p.w.size(), f) != p.w.size())
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

// C = A^T * B, A [r x ka], B [r x kb] -> [ka x kb], a-i-b order for SIMD.
void matmulAtB(const Mat& A, const Mat& B, Mat& C) {
    const int r = A.r, ka = A.c, kb = B.c;
    C.resize(ka, kb);
    std::fill(C.d.begin(), C.d.end(), 0.0f);
    for (int a = 0; a < ka; ++a) {
        float* ca = C.row(a);
        for (int i = 0; i < r; ++i) {
            float va = A.row(i)[a];
            if (va == 0.0f) continue;
            const float* bi = B.row(i);
            for (int b = 0; b < kb; ++b) ca[b] += va * bi[b];
        }
    }
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
        for (int t = 0; t < W.cols; ++t) wt.d[size_t(t) * W.rows + q] = W.w[size_t(q) * W.cols + t];
}

void linearForward(const Mat& x, const Mat& wt, const std::vector<float>& bias,
                   Mat& y) {
    matmul(x, wt, y);
    addBiasRows(y, bias);
}

const Mat& Linear::forward(const Mat& x) {
    inCache = x;
    linearForward(x, wt, b.w, outCache);
    return outCache;
}

void Linear::backward(const Mat& gOut, Mat& gIn) {
    const int r = gOut.r, k = W.cols, o = W.rows;
    // dW += gOut^T * inCache
    Mat dwTmp;
    matmulAtB(gOut, inCache, dwTmp);
    for (size_t i = 0; i < dwTmp.d.size(); ++i) W.dw[i] += dwTmp.d[i];
    for (int i = 0; i < r; ++i)
        for (int q = 0; q < o; ++q) b.dw[q] += gOut.row(i)[q];
    // gIn = gOut * W, i-q-t order (no horizontal reduction, vectorizes)
    gIn.resize(r, k);
    std::fill(gIn.d.begin(), gIn.d.end(), 0.0f);
    for (int i = 0; i < r; ++i) {
        const float* gi = gOut.row(i);
        float* go = gIn.row(i);
        for (int q = 0; q < o; ++q) {
            float g = gi[q];
            if (g == 0.0f) continue;
            const float* wq = &W.w[size_t(q) * k];
            for (int t = 0; t < k; ++t) go[t] += g * wq[t];
        }
    }
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
        for (int u = 0; u < n; ++u)
            wTi.d[size_t(u) * 4 * h + q] = Wi.w[size_t(q) * n + u];
    wTh.resize(h, 4 * h);
    for (int q = 0; q < 4 * h; ++q)
        for (int u = 0; u < h; ++u)
            wTh.d[size_t(u) * 4 * h + q] = Wh.w[size_t(q) * h + u];
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
    std::fill(statesH.d.begin(), statesH.d.end(), 0.0f);
    std::fill(statesC.d.begin(), statesC.d.end(), 0.0f);

    Mat xT, hPrev, gateI, gateH;
    for (int t = 0; t < T; ++t) {
        xT.resize(B, n);
        std::copy(xCache.d.begin() + size_t(t) * B * n,
                  xCache.d.begin() + size_t(t + 1) * B * n, xT.d.begin());
        hPrev.resize(B, h);
        if (t == 0)
            std::fill(hPrev.d.begin(), hPrev.d.end(), 0.0f);
        else
            std::copy(statesH.d.begin() + size_t(t) * B * h,
                      statesH.d.begin() + size_t(t + 1) * B * h,
                      hPrev.d.begin());
        matmul(xT, wTi, gateI);
        matmul(hPrev, wTh, gateH);
        float* gp = gpre.d.data() + size_t(t) * B * H;
        for (int ib = 0; ib < B; ++ib) {
            const float* gi = gateI.row(ib);
            const float* gh = gateH.row(ib);
            float* o = gp + size_t(ib) * H;
            for (int q = 0; q < H; ++q)
                o[q] = gi[q] + gh[q] + bi.w[q] + bh.w[q];
        }
        for (int ib = 0; ib < B; ++ib) {
            const float* hp = statesH.row(t * B + ib);
            const float* cp = t == 0 ? nullptr : statesC.row((t - 1) * B + ib);
            const float* gpp = gpre.row(t * B + ib);
            float* hn = outCache.row(t * B + ib);
            float* cn = statesC.row(t * B + ib);
            for (int u = 0; u < h; ++u) {
                float iv = sigmoidf(gpp[u]);
                float fv = sigmoidf(gpp[h + u]);
                float gv = std::tanh(gpp[2 * h + u]);
                float ov = sigmoidf(gpp[3 * h + u]);
                float pc = cp ? cp[u] : 0.0f;
                float cc = fv * pc + iv * gv;
                cn[u] = cc;
                hn[u] = ov * std::tanh(cc);
            }
            std::copy(hn, hn + h, statesH.row((t + 1) * B + ib));
        }
    }
    return outCache;
}

void Lstm::lastHidden(Mat& out) const {
    out.resize(B, h);
    for (int ib = 0; ib < B; ++ib)
        std::copy(statesH.row(T * B + ib), statesH.row(T * B + ib) + h,
                  out.row(ib));
}

void Lstm::backward(const Mat& ghAll) {
    int H = 4 * h;
    Mat dhNext(B, h), dcNext(B, h);
    std::fill(dhNext.d.begin(), dhNext.d.end(), 0.0f);
    std::fill(dcNext.d.begin(), dcNext.d.end(), 0.0f);

    Mat dg(B, H), dhTmp(B, h), xT(B, n), hPrev(B, h), gw, gwh;
    for (int t = T - 1; t >= 0; --t) {
        for (int ib = 0; ib < B; ++ib) {
            const float* gh = ghAll.row(t * B + ib);
            const float* gpn = gpre.row(t * B + ib);
            const float* cn = statesC.row(t * B + ib);
            const float* cp = t == 0 ? nullptr : statesC.row((t - 1) * B + ib);
            float* dgp = dg.row(ib);
            for (int u = 0; u < h; ++u) {
                float tc = std::tanh(cn[u]);
                float iv = sigmoidf(gpn[u]);
                float fv = sigmoidf(gpn[h + u]);
                float gv = std::tanh(gpn[2 * h + u]);
                float ov = sigmoidf(gpn[3 * h + u]);
                float dhn = gh[u] + dhNext.row(ib)[u];
                float dct = dhn * ov * (1.0f - tc * tc) + dcNext.row(ib)[u];
                dgp[u] = dct * gv * iv * (1.0f - iv);
                dgp[h + u] = dct * (cp ? cp[u] : 0.0f) * fv * (1.0f - fv);
                dgp[2 * h + u] = dct * iv * (1.0f - gv * gv);
                dgp[3 * h + u] = dhn * tc * ov * (1.0f - ov);
                dcNext.row(ib)[u] = dct * fv;
            }
        }
        // parameter gradients
        std::copy(xCache.d.begin() + size_t(t) * B * n,
                  xCache.d.begin() + size_t(t + 1) * B * n, xT.d.begin());
        matmulAtB(dg, xT, gw);
        for (size_t i = 0; i < gw.d.size(); ++i) Wi.dw[i] += gw.d[i];
        if (t == 0)
            std::fill(hPrev.d.begin(), hPrev.d.end(), 0.0f);
        else
            std::copy(statesH.d.begin() + size_t(t) * B * h,
                      statesH.d.begin() + size_t(t + 1) * B * h,
                      hPrev.d.begin());
        matmulAtB(dg, hPrev, gwh);
        for (size_t i = 0; i < gwh.d.size(); ++i) Wh.dw[i] += gwh.d[i];
        for (int ib = 0; ib < B; ++ib)
            for (int q = 0; q < H; ++q) {
                bi.dw[q] += dg.row(ib)[q];
                bh.dw[q] += dg.row(ib)[q];
            }
        // recurrent gradient (input gradient is not needed): dg * Wh^T
        dhTmp.resize(B, h);
        for (int ib = 0; ib < B; ++ib) {
            const float* dgb = dg.row(ib);
            float* ho = dhTmp.row(ib);
            for (int u = 0; u < h; ++u) {
                float s = 0.0f;
                for (int q = 0; q < H; ++q)
                    s += dgb[q] * Wh.w[size_t(q) * h + u];
                ho[u] = s;
            }
        }
        dhNext = dhTmp;
    }
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
    l1.init(kImpInput + c.lstmHidden, c.hidden, rng);
    l2.init(c.hidden, c.hidden, rng);
    l3.init(c.hidden, c.hidden, rng);
    l4.init(c.hidden, c.hidden * 2, rng);
    head.init(c.hidden * 2, kNumActions, rng);
    std::fill(head.W.w.begin(), head.W.w.end(), 0.0f);
    std::fill(head.b.w.begin(), head.b.w.end(), 0.0f);
    dyn.init(kActionDyn, 1, rng);
    std::fill(dyn.W.w.begin(), dyn.W.w.end(), 0.0f);
    std::fill(dyn.b.w.begin(), dyn.b.w.end(), 0.0f);
    prepareInference();
}

void Actor::prepareInference() {
    lstm.buildWT();
    l1.buildWT();
    l2.buildWT();
    l3.buildWT();
    l4.buildWT();
    head.buildWT();
    dyn.buildWT();
}

Mat& Actor::forward(const Mat& xImp, const Mat& seq, const Mat& mask,
                    const Mat& dynFeat) {
    int B = xImp.r;
    lstm.forward(seq, B, kLstmSteps);
    Mat hl;
    lstm.lastHidden(hl);
    z.resize(B, kImpInput + cfg.lstmHidden);
    for (int i = 0; i < B; ++i) {
        std::copy(xImp.row(i), xImp.row(i) + kImpInput, z.row(i));
        std::copy(hl.row(i), hl.row(i) + cfg.lstmHidden,
                  z.row(i) + kImpInput);
    }
    f1 = l1.forward(z);
    reluFwd(f1);
    f2 = l2.forward(f1);
    reluFwd(f2);
    f3 = l3.forward(f2);
    reluFwd(f3);
    feat = l4.forward(f3);
    logits = head.forward(feat);
    const Mat& ds = dyn.forward(dynFeat);
    dynScore.resize(B, kNumActions);
    for (int i = 0; i < B; ++i)
        for (int a = 0; a < kNumActions; ++a)
            dynScore.row(i)[a] = ds.d[size_t(i) * kNumActions + a];
    for (int i = 0; i < B; ++i)
        for (int a = 0; a < kNumActions; ++a) {
            if (mask.row(i)[a] > 0.5f)
                logits.row(i)[a] += dynScore.row(i)[a];
            else
                logits.row(i)[a] = -1e9f;
        }
    return logits;
}

void Actor::backward(const Mat& dLogits) {
    int B = dLogits.r;
    // dynamic action head (gradient w.r.t. its dense input is discarded)
    Mat gDs(B * kNumActions, 1);
    std::copy(dLogits.d.begin(), dLogits.d.end(), gDs.d.begin());
    Mat gDynFeat;
    dyn.backward(gDs, gDynFeat);
    // action head
    Mat gFeat;
    head.backward(dLogits, gFeat);
    // MLP trunk
    Mat gF3, gP3, gF2, gP2, gF1, gP1, gZ;
    l4.backward(gFeat, gF3);
    reluBwd(f3, gF3, gP3);
    l3.backward(gP3, gF2);
    reluBwd(f2, gF2, gP2);
    l2.backward(gP2, gF1);
    reluBwd(f1, gF1, gP1);
    l1.backward(gP1, gZ);
    // last hidden state -> LSTM
    Mat ghLast(B, cfg.lstmHidden);
    for (int i = 0; i < B; ++i)
        std::copy(gZ.row(i) + kImpInput,
                  gZ.row(i) + kImpInput + cfg.lstmHidden, ghLast.row(i));
    Mat ghAll(B * kLstmSteps, cfg.lstmHidden);
    std::fill(ghAll.d.begin(), ghAll.d.end(), 0.0f);
    for (int i = 0; i < B; ++i)
        std::copy(ghLast.row(i), ghLast.row(i) + cfg.lstmHidden,
                  ghAll.row((kLstmSteps - 1) * B + i));
    lstm.backward(ghAll);
}

void Actor::zeroGrad() {
    lstm.zeroGrad();
    l1.zeroGrad();
    l2.zeroGrad();
    l3.zeroGrad();
    l4.zeroGrad();
    head.zeroGrad();
    dyn.zeroGrad();
}

std::vector<Param*> Actor::params() {
    return {&lstm.Wi, &lstm.bi, &lstm.Wh, &lstm.bh,
            &l1.W, &l1.b, &l2.W, &l2.b, &l3.W, &l3.b,
            &l4.W, &l4.b, &head.W, &head.b, &dyn.W, &dyn.b};
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
}

// ---------------------------------------------------------------------------
// Critic
// ---------------------------------------------------------------------------
void Critic::init(const NetConfig& c, uint64_t seed) {
    cfg = c;
    Rng64 rng(seed);
    lstm.init(kLstmIn, c.lstmHidden, rng);
    i1.init(kImpInput + c.lstmHidden, c.hidden, rng);
    i2.init(c.hidden, c.hidden, rng);
    i3.init(c.hidden, c.hidden, rng);
    p1.init(kExtraInput, c.hidden, rng);
    p2.init(c.hidden, c.hidden, rng);
    c1.init(2 * c.hidden, c.hidden, rng);
    c2.init(c.hidden, c.hidden, rng);
    out.init(c.hidden, 1, rng);
    prepareInference();
}

void Critic::prepareInference() {
    lstm.buildWT();
    i1.buildWT();
    i2.buildWT();
    i3.buildWT();
    p1.buildWT();
    p2.buildWT();
    c1.buildWT();
    c2.buildWT();
    out.buildWT();
}

Mat& Critic::forward(const Mat& xImp, const Mat& seq, const Mat& extra) {
    int B = xImp.r;
    lstm.forward(seq, B, kLstmSteps);
    Mat hl;
    lstm.lastHidden(hl);
    z.resize(B, kImpInput + cfg.lstmHidden);
    for (int i = 0; i < B; ++i) {
        std::copy(xImp.row(i), xImp.row(i) + kImpInput, z.row(i));
        std::copy(hl.row(i), hl.row(i) + cfg.lstmHidden,
                  z.row(i) + kImpInput);
    }
    f1 = i1.forward(z);
    reluFwd(f1);
    f2 = i2.forward(f1);
    reluFwd(f2);
    imp = i3.forward(f2);

    pf1 = p1.forward(extra);
    reluFwd(pf1);
    pe = p2.forward(pf1);

    cat.resize(B, 2 * cfg.hidden);
    for (int i = 0; i < B; ++i) {
        std::copy(imp.row(i), imp.row(i) + cfg.hidden, cat.row(i));
        std::copy(pe.row(i), pe.row(i) + cfg.hidden,
                  cat.row(i) + cfg.hidden);
    }
    q1 = c1.forward(cat);
    reluFwd(q1);
    q2 = c2.forward(q1);
    reluFwd(q2);
    value = out.forward(q2);
    return value;
}

void Critic::backward(const Mat& dValue) {
    int B = dValue.r;
    Mat gQ2, gP2, gQ1, gP1, gCat;
    out.backward(dValue, gQ2);
    reluBwd(q2, gQ2, gP2);
    c2.backward(gP2, gQ1);
    Mat gQ0;
    reluBwd(q1, gQ1, gQ0);
    c1.backward(gQ0, gCat);
    Mat gImp(B, cfg.hidden), gPe(B, cfg.hidden);
    for (int i = 0; i < B; ++i) {
        std::copy(gCat.row(i), gCat.row(i) + cfg.hidden, gImp.row(i));
        std::copy(gCat.row(i) + cfg.hidden,
                  gCat.row(i) + 2 * cfg.hidden, gPe.row(i));
    }
    Mat gPf1, gPf0, gExtra;
    p2.backward(gPe, gPf1);
    reluBwd(pf1, gPf1, gPf0);
    p1.backward(gPf0, gExtra);

    Mat giF2, giP3, giF1, giP1, giZ;
    i3.backward(gImp, giF2);
    reluBwd(f2, giF2, giP3);
    i2.backward(giP3, giF1);
    reluBwd(f1, giF1, giP1);
    i1.backward(giP1, giZ);
    Mat ghLast(B, cfg.lstmHidden);
    for (int i = 0; i < B; ++i)
        std::copy(giZ.row(i) + kImpInput,
                  giZ.row(i) + kImpInput + cfg.lstmHidden, ghLast.row(i));
    Mat ghAll(B * kLstmSteps, cfg.lstmHidden);
    std::fill(ghAll.d.begin(), ghAll.d.end(), 0.0f);
    for (int i = 0; i < B; ++i)
        std::copy(ghLast.row(i), ghLast.row(i) + cfg.lstmHidden,
                  ghAll.row((kLstmSteps - 1) * B + i));
    lstm.backward(ghAll);
}

void Critic::zeroGrad() {
    lstm.zeroGrad();
    i1.zeroGrad();
    i2.zeroGrad();
    i3.zeroGrad();
    p1.zeroGrad();
    p2.zeroGrad();
    c1.zeroGrad();
    c2.zeroGrad();
    out.zeroGrad();
}

std::vector<Param*> Critic::params() {
    return {&lstm.Wi, &lstm.bi, &lstm.Wh, &lstm.bh,
            &i1.W, &i1.b, &i2.W, &i2.b, &i3.W, &i3.b,
            &p1.W, &p1.b, &p2.W, &p2.b, &c1.W, &c1.b,
            &c2.W, &c2.b, &out.W, &out.b};
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
}

// ---------------------------------------------------------------------------
// Thread-safe inference (weights read-only)
// ---------------------------------------------------------------------------
namespace {

}  // namespace

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
    Mat gateI, gateH, xT;
    for (int t = 0; t < T; ++t) {
        xT.resize(B, n);
        std::copy(x.d.begin() + size_t(t) * B * n,
                  x.d.begin() + size_t(t + 1) * B * n, xT.d.begin());
        matmul(xT, l.wTi, gateI);
        matmul(w.hp, l.wTh, gateH);
        w.gp.resize(B, H);
        for (int ib = 0; ib < B; ++ib) {
            float* gp = w.gp.row(ib);
            const float* gi = gateI.row(ib);
            const float* gh = gateH.row(ib);
            for (int q = 0; q < H; ++q)
                gp[q] = gi[q] + gh[q] + l.bi.w[q] + l.bh.w[q];
        }
        for (int ib = 0; ib < B; ++ib) {
            const float* gp = w.gp.row(ib);
            const float* cp = w.cp.row(ib);
            float* hc = w.hc.row(ib);
            float* cc = w.cc.row(ib);
            for (int u = 0; u < h; ++u) {
                float iv = sigmoidf(gp[u]);
                float fv = sigmoidf(gp[h + u]);
                float gv = std::tanh(gp[2 * h + u]);
                float ov = sigmoidf(gp[3 * h + u]);
                float c = fv * cp[u] + iv * gv;
                cc[u] = c;
                hc[u] = ov * std::tanh(c);
            }
        }
        if (t == T - 1)
            for (int ib = 0; ib < B; ++ib)
                std::copy(w.hc.row(ib), w.hc.row(ib) + h, w.hl.row(ib));
        std::swap(w.hp, w.hc);
        std::swap(w.cp, w.cc);
    }
}

const Mat& actorInferForward(const Actor& a, ActorInfer& w, const Mat& xImp,
                             const Mat& seq, const Mat& mask,
                             const Mat& dynFeat) {
    int B = xImp.r;
    lstmInferForward(a.lstm, seq, B, kLstmSteps, w.lstm);
    w.z.resize(B, kImpInput + a.cfg.lstmHidden);
    for (int i = 0; i < B; ++i) {
        std::copy(xImp.row(i), xImp.row(i) + kImpInput, w.z.row(i));
        std::copy(w.lstm.hl.row(i), w.lstm.hl.row(i) + a.cfg.lstmHidden,
                  w.z.row(i) + kImpInput);
    }
    linearForward(w.z, a.l1.wt, a.l1.b.w, w.f1);
    reluFwd(w.f1);
    linearForward(w.f1, a.l2.wt, a.l2.b.w, w.f2);
    reluFwd(w.f2);
    linearForward(w.f2, a.l3.wt, a.l3.b.w, w.f3);
    reluFwd(w.f3);
    linearForward(w.f3, a.l4.wt, a.l4.b.w, w.feat);
    linearForward(w.feat, a.head.wt, a.head.b.w, w.logits);
    linearForward(dynFeat, a.dyn.wt, a.dyn.b.w, w.ds);
    for (int i = 0; i < B; ++i)
        for (int j = 0; j < kNumActions; ++j) {
            if (mask.row(i)[j] > 0.5f)
                w.logits.row(i)[j] += w.ds.d[size_t(i) * kNumActions + j];
            else
                w.logits.row(i)[j] = -1e9f;
        }
    return w.logits;
}

const Mat& criticInferForward(const Critic& c, CriticInfer& w,
                              const Mat& xImp, const Mat& seq,
                              const Mat& extra) {
    int B = xImp.r;
    lstmInferForward(c.lstm, seq, B, kLstmSteps, w.lstm);
    w.z.resize(B, kImpInput + c.cfg.lstmHidden);
    for (int i = 0; i < B; ++i) {
        std::copy(xImp.row(i), xImp.row(i) + kImpInput, w.z.row(i));
        std::copy(w.lstm.hl.row(i), w.lstm.hl.row(i) + c.cfg.lstmHidden,
                  w.z.row(i) + kImpInput);
    }
    linearForward(w.z, c.i1.wt, c.i1.b.w, w.f1);
    reluFwd(w.f1);
    linearForward(w.f1, c.i2.wt, c.i2.b.w, w.f2);
    reluFwd(w.f2);
    linearForward(w.f2, c.i3.wt, c.i3.b.w, w.imp);
    linearForward(extra, c.p1.wt, c.p1.b.w, w.pf1);
    reluFwd(w.pf1);
    linearForward(w.pf1, c.p2.wt, c.p2.b.w, w.pe);
    w.cat.resize(B, 2 * c.cfg.hidden);
    for (int i = 0; i < B; ++i) {
        std::copy(w.imp.row(i), w.imp.row(i) + c.cfg.hidden, w.cat.row(i));
        std::copy(w.pe.row(i), w.pe.row(i) + c.cfg.hidden,
                  w.cat.row(i) + c.cfg.hidden);
    }
    linearForward(w.cat, c.c1.wt, c.c1.b.w, w.q1);
    reluFwd(w.q1);
    linearForward(w.q1, c.c2.wt, c.c2.b.w, w.q2);
    reluFwd(w.q2);
    linearForward(w.q2, c.out.wt, c.out.b.w, w.value);
    return w.value;
}

// ---------------------------------------------------------------------------
// Adam
// ---------------------------------------------------------------------------
void Adam::apply(const std::vector<Param*>& ps) {
    ++t;
    float bc1 = 1.0f - std::pow(beta1, float(t));
    float bc2 = 1.0f - std::pow(beta2, float(t));
    for (Param* p : ps) {
        for (size_t i = 0; i < p->w.size(); ++i) {
            float g = p->dw[i];
            p->m[i] = beta1 * p->m[i] + (1.0f - beta1) * g;
            p->v[i] = beta2 * p->v[i] + (1.0f - beta2) * g * g;
            float mh = p->m[i] / bc1;
            float vh = p->v[i] / bc2;
            p->w[i] -= lr * mh / (std::sqrt(vh) + eps);
        }
    }
}

void Adam::applyGradNorm(const std::vector<Param*>& ps, float maxNorm) {
    double sum = 0.0;
    for (Param* p : ps)
        for (float g : p->dw) sum += double(g) * g;
    float norm = float(std::sqrt(sum));
    if (norm > maxNorm) {
        float scale = maxNorm / (norm + 1e-6f);
        for (Param* p : ps)
            for (float& g : p->dw) g *= scale;
    }
    apply(ps);
}

}  // namespace nn
