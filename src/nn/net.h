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

constexpr int kImpInput = 1676;  // 9*180 current-state binaries + 56 one-hots
constexpr int kExtraInput = 362;
constexpr int kLstmSteps = 10;
constexpr int kLstmIn = 540;     // three consecutive moves per LSTM step
// Logit width for one decision. Concrete legal moves are scored in slots
// 0..n-1. A live hand reached 519 moves, past the old 512 cap.
constexpr int kNumActions = 2048;
constexpr int kActionInput = 186; // 12*15 action matrix + 6 properties

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
    Param(const Param& o)
        : w(o.w), dw(o.dw), m(o.m), v(o.v), rows(o.rows), cols(o.cols) {}
    Param(Param&& o) noexcept
        : w(std::move(o.w)), dw(std::move(o.dw)), m(std::move(o.m)), v(std::move(o.v)),
          rows(o.rows), cols(o.cols), devW(o.devW), devG(o.devG) {
        o.devW = nullptr;
        o.devG = nullptr;
        o.rows = 0;
        o.cols = 0;
    }
    Param& operator=(const Param& o) {
        if (this != &o) {
            w = o.w; dw = o.dw; m = o.m; v = o.v;
            rows = o.rows; cols = o.cols;
        }
        return *this;
    }
    Param& operator=(Param&& o) noexcept;
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
    Mat actCache;                 // B*T x 4h activated i,f,g,o (CPU forward)
    Mat tanhC;                    // B*T x h tanh(c_t)
    bool actReady = false;
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
// Released-model architecture:
//   LSTM(540 -> 128) over ten groups of three historical moves
//   concat node embedding with every legal 186-D action representation
//   shared MLP [256,256,256,512,1], scattered into concrete-action logits
struct Actor {
    NetConfig cfg;
    Lstm lstm;
    Linear l1, l2, l3, l4, head;

    // forward caches
    Mat node, joint, f1, f2, f3, feat, scores, logits;
    Mat actionSampleCache, actionIdCache, actionOffsetCache;
    // backward caches (kept as members so GPU encodes outlive function scope)
    Mat gScores, gFeat, gF3, gP3, gF2, gP2, gF1, gP1, gJoint;
    Mat gNode, gXImp, ghLast, ghAll;

    void init(const NetConfig& c, uint64_t seed);
    void prepareInference();  // rebuild transposed weights after an optimizer step
    Mat& forward(const Mat& xImp, const Mat& seq, const Mat& actionFeat,
                 const Mat& actionSample, const Mat& actionId,
                 const Mat& actionOffset);
    void backward(const Mat& dLogits);
    void zeroGrad();
    std::vector<Param*> params();
    void save(const char* path) const;
    void load(const char* path);
};

// Perfect-information value network: imperfect node embedding plus the two
// hidden hands/min-step features, followed by the paper's four 256-wide MLPs.
struct Critic {
    NetConfig cfg;
    Lstm lstm;
    Linear c1, c2, c3, c4, out;

    Mat node, all, q1, q2, q3, q4, value;
    // backward caches (persistent for GPU encodes)
    Mat gQ4, gP4, gQ3, gP3, gQ2, gP2, gQ1, gP1, gAll;
    Mat gNode, gExtra, ghLast, ghAll;

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
    Mat node, joint, f1, f2, f3, feat, scores, logits;
};
const Mat& actorInferForward(const Actor& a, ActorInfer& w, const Mat& xImp,
                             const Mat& seq, const Mat& actionFeat,
                             const Mat& actionSample, const Mat& actionId,
                             const Mat& actionOffset);

struct CriticInfer {
    LstmInfer lstm;
    Mat node, all, q1, q2, q3, q4, value;
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

void saveOptimizer(const char* path, const std::vector<Param*>& ps, const Adam& opt);
bool loadOptimizer(const char* path, const std::vector<Param*>& ps, Adam& opt);

}  // namespace nn
