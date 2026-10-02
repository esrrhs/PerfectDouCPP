// Minimal dense matrix / dense-ops primitives (row major, float32).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "nn/gemm.h"

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

// C = A * B, shapes [M x K] * [K x N] -> [M x N].
inline void matmul(const Mat& A, const Mat& B, Mat& C) {
    const int M = A.r, K = A.c, N = B.c;
    C.resize(M, N);
    sgemm('N', 'N', M, N, K, A.d.data(), A.c, B.d.data(), B.c, C.d.data(), C.c);
}

// C = A * B^T, B physical [N x K] -> [M x N].
inline void matmulABt(const Mat& A, const Mat& B, Mat& C) {
    const int M = A.r, K = A.c, N = B.r;
    C.resize(M, N);
    sgemm('N', 'T', M, N, K, A.d.data(), A.c, B.d.data(), B.c, C.d.data(), C.c);
}

// C = A^T * B, A physical [M x K], B [M x N] -> [K x N].
inline void matmulAtB(const Mat& A, const Mat& B, Mat& C) {
    const int M = A.r, K = A.c, N = B.c;
    C.resize(K, N);
    sgemm('T', 'N', K, N, M, A.d.data(), A.c, B.d.data(), B.c, C.d.data(), C.c);
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
