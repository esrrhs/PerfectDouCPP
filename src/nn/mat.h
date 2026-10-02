// Minimal dense matrix / dense-ops primitives (row major, float32).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace nn {

struct Mat {
    int r = 0, c = 0;
    std::vector<float> d;
    Mat() = default;
    Mat(int rows, int cols) : r(rows), c(cols), d(size_t(rows) * cols, 0.0f) {}
    void resize(int rows, int cols) {
        r = rows;
        c = cols;
        d.assign(size_t(rows) * cols, 0.0f);
    }
    float* row(int i) { return d.data() + size_t(i) * c; }
    const float* row(int i) const { return d.data() + size_t(i) * c; }
};

// C = A * B, shapes [r x k] * [k x c] -> [r x c]. i-k-j loop order for cache
// friendliness; weights are stored [out x in], so B is often transposed via
// dedicated helpers in Linear/Lstm.
inline void matmul(const Mat& A, const Mat& B, Mat& C) {
    const int r = A.r, k = A.c, c = B.c;
    C.resize(r, c);
    std::fill(C.d.begin(), C.d.end(), 0.0f);
    for (int i = 0; i < r; ++i) {
        const float* ai = A.row(i);
        float* ci = C.row(i);
        for (int t = 0; t < k; ++t) {
            float a = ai[t];
            if (a == 0.0f) continue;
            const float* bt = B.row(t);
            for (int j = 0; j < c; ++j) ci[j] += a * bt[j];
        }
    }
}

// C = A * W^T where W is stored [k x out] (its rows are contiguous).
// Equivalently C[i,t] = dot(A row i, W row t), which vectorizes well.
inline void matmulWT(const Mat& A, const Mat& W, Mat& C) {
    const int r = A.r, out = A.c, k = W.r;
    C.resize(r, k);
    for (int i = 0; i < r; ++i) {
        const float* ai = A.row(i);
        float* ci = C.row(i);
        for (int t = 0; t < k; ++t) {
            const float* wt = W.row(t);
            float s = 0.0f;
            for (int j = 0; j < out; ++j) s += ai[j] * wt[j];
            ci[t] = s;
        }
    }
}

inline void addBiasRows(Mat& X, const std::vector<float>& b) {
    for (int i = 0; i < X.r; ++i) {
        float* x = X.row(i);
        for (int j = 0; j < X.c; ++j) x[j] += b[j];
    }
}

inline void reluFwd(Mat& X) {
    for (float& v : X.d) v = v > 0.0f ? v : 0.0f;
}

// dy = (pre > 0) * gradOut; can operate in place.
inline void reluBwd(const Mat& pre, const Mat& gradOut, Mat& gradIn) {
    gradIn.resize(gradOut.r, gradOut.c);
    for (size_t i = 0; i < gradOut.d.size(); ++i)
        gradIn.d[i] = pre.d[i] > 0.0f ? gradOut.d[i] : 0.0f;
}

}  // namespace nn
