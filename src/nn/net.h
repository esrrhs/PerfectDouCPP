// Small from-scratch neural network library for PerfectDou training.
//
// Implements just what the actor-critic needs: dense linear layers, ReLU and a
// batched single-layer LSTM, all with manual reverse-mode differentiation.
// No external dependencies.
#pragma once

#include <cstdint>
#include <cstdio>
#include <vector>

#include "nn/mat.h"

namespace nn {

constexpr int kImpInput = 4146;  // 23*180 binaries + 6 scalars
constexpr int kExtraInput = 362;
constexpr int kLstmSteps = 15;
constexpr int kLstmIn = 180;
constexpr int kNumActions = 621;
constexpr int kActionDyn = 7;

struct Param {
    std::vector<float> w;   // values (storage rounded up to 4 floats)
    std::vector<float> dw;  // gradients
    std::vector<float> m;   // Adam first moment
    std::vector<float> v;   // Adam second moment
    int rows = 0, cols = 0;
    // Opaque persistent device caches (weight copy / gradient accumulator),
    // owned here so they die exactly when the parameter does.
    mutable void* devW = nullptr;
    mutable void* devG = nullptr;

    Param() = default;
    Param(const Param&) : w(), dw(), m(), v() {}  // caches never copied
    Param& operator=(const Param& o) {
        w = o.w; dw = o.dw; m = o.m; v = o.v;
        rows = o.rows; cols = o.cols;
        return *this;
    }
    ~Param();

    void init(int r, int c) {
        rows = r;
        cols = c;
        size_t cap = size_t(padStride(r * c));  // 16-byte aligned allocation
        gpuDropCache(&devW);
        gpuDropCache(&devG);
        w.assign(cap, 0.0f);
        dw.assign(cap, 0.0f);
        m.assign(cap, 0.0f);
        v.assign(cap, 0.0f);
    }
    int size() const { return rows * cols; }
    void zeroGrad();
};

class Rng64 {
public:
    explicit Rng64(uint64_t seed) : s_(seed ? seed : 0x9e3779b97f4a7c15ULL) {}
    float uniform(float lo, float hi) {
        double u = double(nextU64() >> 11) / double(1ULL << 53);
        return float(lo + u * (hi - lo));
    }
    uint64_t nextU64() {
        uint64_t z = (s_ += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

private:
    uint64_t s_;
};

struct Linear {
    Param W, b;
    Mat inCache;
    Mat outCache;
    Mat wt;     // transposed W [in x out], rebuilt for fast forward passes
    Mat dwTmp;  // persistent dW workspace
    void init(int in, int out, Rng64& rng);
    void buildWT();
    const Mat& forward(const Mat& x);
    // Forward fused with bias add + ReLU (training graph).
    const Mat& forwardAct(const Mat& x);
    void backward(const Mat& gOut, Mat& gIn);
    void zeroGrad();
    void save(FILE* f) const;
    void load(FILE* f);
};

// y = x * wt + b
void linearForward(const Mat& x, const Mat& wt, const std::vector<float>& bias,
                   Mat& y);

// Single-layer LSTM with sigmoid/tanh gates [input, forget, candidate, output].
struct Lstm {
    int n = 0, h = 0;
    Param Wi, bi, Wh, bh;  // Wi/Wh shaped [4h x n] / [4h x h]
    Mat wTi, wTh;         // transposed weights for fast forward passes
    // caches
    int B = 0, T = 0;
    Mat xCache;       // B*T x n
    Mat gpre;         // B*T x 4h
    Mat statesH;      // (T+1)*B x h, index (t*B+ib)
    Mat statesC;      // T*B x h
    Mat outCache;                 // B*T x h
    Mat gateIAll;                 // B*T x 4h batched input-gate result
    std::vector<Mat> gwCache;     // T reusable dWi products
    Mat dwWhTmp, dwWiTmp;         // batched dWh and dWi products
    // backward workspaces
    Mat dgAll, ghGate, ghRec, dhBuf[2], dcBuf, dgNow, gwh;
    Mat hlLast;  // last hidden state (forward, persistent for GPU lifetime)

    void init(int inputSize, int hiddenSize, Rng64& rng);
    void buildWT();
    // x: B*T rows of length n, time-major (t = row / B)
    const Mat& forward(const Mat& x, int batch, int steps);
    // ghAll: B*T x h (gradients for every output, may be all zero except
    // last time). Input gradients are discarded.
    void backward(const Mat& ghAll);
    void zeroGrad();
    void save(FILE* f) const;
    void load(FILE* f);
};

struct NetConfig {
    int hidden = 256;
    int lstmHidden = 128;
};

// Imperfect-information policy network.
//   LSTM(180 -> 128) over the last 15 moves
//   concat with node features -> MLP [256,256,256,512]
//   621 action logits + dynamic action-feature score, masked
struct Actor {
    NetConfig cfg;
    Lstm lstm;
    Linear l1, l2, l3, l4, head, dyn;

    // forward caches
    Mat z, f1, f2, f3, feat, logits, gDs;
    // backward caches (kept as members so GPU encodes outlive function scope)
    Mat gFeat, gF3, gP3, gF2, gP2, gF1, gP1, gZ, ghLast, ghAll, gDynFeat;

    void init(const NetConfig& c, uint64_t seed);
    void prepareInference();  // rebuild transposed weights after an optimizer step
    // xImp: B x 4146; seq: B*15 x 180; mask: B x 621 (1 legal);
    // dynFeat: B*621 x 7
    Mat& forward(const Mat& xImp, const Mat& seq, const Mat& mask,
                 const Mat& dynFeat);
    void backward(const Mat& dLogits);
    void zeroGrad();
    std::vector<Param*> params();
    void save(const char* path) const;
    void load(const char* path);
};

// Perfect-information value network (shared-shape imperfect trunk plus an
// encoder of the perfect-only features).
struct Critic {
    NetConfig cfg;
    Lstm lstm;
    Linear i1, i2, i3;      // imperfect trunk -> hidden
    Linear p1, p2;          // perfect extras -> hidden
    Linear c1, c2, out;     // concat -> hidden -> hidden -> 1

    Mat z, f1, f2, imp, pf1, pe, cat, q1, q2, value;
    // backward caches (persistent for GPU encodes)
    Mat gQ2, gP2, gQ1, gQ0, gCat, gImp, gPe, gPf1, gPf0, gExtra;
    Mat giF2, giP3, giF1, giP1, giZ, ghLast, ghAll;

    void init(const NetConfig& c, uint64_t seed);
    void prepareInference();
    Mat& forward(const Mat& xImp, const Mat& seq, const Mat& extra);
    void backward(const Mat& dValue);
    void zeroGrad();
    std::vector<Param*> params();
    void save(const char* path) const;
    void load(const char* path);
};

// ---------------------------------------------------------------------------
// Inference-only workspaces (thread local; they never touch the trained
// module's internal forward caches, allowing concurrent reads of weights).
// ---------------------------------------------------------------------------
struct LstmInfer {
    int B = 0, T = 0;
    Mat gp;     // current gate preactivations
    Mat hp, cp; // previous h / c
    Mat hc, cc; // current h / c
    Mat hl;     // last hidden B x h
    std::vector<Mat> giAll;  // T reusable input-gate results
};
void lstmInferForward(const Lstm& l, const Mat& x, int B, int T, LstmInfer& w);

struct ActorInfer {
    LstmInfer lstm;
    Mat z, f1, f2, f3, feat, logits, ds;
};
const Mat& actorInferForward(const Actor& a, ActorInfer& w, const Mat& xImp,
                             const Mat& seq, const Mat& mask,
                             const Mat& dynFeat);

struct CriticInfer {
    LstmInfer lstm;
    Mat z, f1, f2, imp, pf1, pe, cat, q1, q2, value;
};
const Mat& criticInferForward(const Critic& c, CriticInfer& w,
                              const Mat& xImp, const Mat& seq,
                              const Mat& extra);

// Adam optimizer over a parameter set.
struct Adam {
    float lr = 3e-4f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-5f;
    long long t = 0;
    void apply(const std::vector<Param*>& ps);
    void applyGradNorm(const std::vector<Param*>& ps, float maxNorm);
};

}  // namespace nn
