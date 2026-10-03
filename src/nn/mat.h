// Minimal dense matrix / dense-ops primitives (row major, float32).
//
// Rows are padded up to a multiple of 4 floats. On Apple Silicon the padding
// lets the Metal backend attach a device buffer to the host storage: vector
// 16-byte stores, which must never spill past the allocation's end.
//
// viewRows() returns a non-owning slice into another Mat's storage; the
// caller must keep the parent alive. These views let the LSTM unroll feed
// per-time-step pointers to GPU kernels without copies.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "nn/gemm.h"

namespace nn {

inline int padStride(int cols) { return (cols + 3) & ~3; }

struct Mat {
    int r = 0, c = 0, s = 0;  // rows, cols, row stride (>= c, %4 == 0)
    std::vector<float> d;     // owning storage
    float* ext = nullptr;     // non-owning view pointer (overrides d)
    // Opaque device-side cache of this Mat's contents (id<MTLBuffer> on
    // Apple). Owned here and released on destruction/invalidation, so it can
    // never outlive or alias the host storage.
    mutable void* devCache = nullptr;

    Mat() = default;
    Mat(int rows, int cols) { resize(rows, cols); }
    Mat(const Mat& o) { *this = o; }
    Mat& operator=(const Mat& o) {
        gpuDropCache(&devCache);
        r = o.r; c = o.c; s = o.s; ext = o.ext;
        if (ext == nullptr) d = o.d;
        return *this;
    }
    Mat(Mat&& o) noexcept { *this = std::move(o); }
    Mat& operator=(Mat&& o) noexcept {
        if (this != &o) {
            gpuDropCache(&devCache);
            r = o.r; c = o.c; s = o.s; ext = o.ext;
            d = std::move(o.d);
            gpuDropCache(&o.devCache);
        }
        return *this;
    }
    ~Mat() { gpuDropCache(&devCache); }

    void resize(int rows, int cols) {
        r = rows;
        c = cols;
        s = padStride(cols);
        ext = nullptr;
        gpuDropCache(&devCache);  // host storage (re)allocated
        d.assign(size_t(r) * s, 0.0f);
    }
    // Non-owning view of another Mat's storage (r/c/s shared, data aliased).
    // Lets layer caches keep the exact pointer a kernel bound, so GPU-resident
    // results are read back through the same residency binding.
    void ref(const Mat& o) {
        r = o.r;
        c = o.c;
        s = o.s;
        ext = const_cast<float*>(o.data());
    }
    const float* data() const { return ext ? ext : d.data(); }
    float* data() { return ext ? ext : d.data(); }
    float* row(int i) { return data() + size_t(i) * s; }
    const float* row(int i) const { return data() + size_t(i) * s; }
};

// Non-owning view of `rows` rows starting at row `first` of `src`.
inline Mat viewRows(const Mat& src, int first, int rows) {
    Mat v;
    v.r = rows;
    v.c = src.c;
    v.s = src.s;
    v.ext = const_cast<float*>(src.data()) + size_t(first) * src.s;
    return v;
}

// C = A * B, shapes [M x K] * [K x N] -> [M x N].
inline void matmul(const Mat& A, const Mat& B, Mat& C, bool weightB = false) {
    const int M = A.r, K = A.c, N = B.c;
    C.resize(M, N);
    GemmOp g{'N', 'N', M, N, K, A.data(), A.s, B.data(), B.s, C.data(), C.s};
    if (weightB) g.wSlot = const_cast<void**>(&B.devCache);
    gpuCommitGemms(1, &g);
}

// C = A * B^T, B physical [N x K] -> [M x N].
inline void matmulABt(const Mat& A, const Mat& B, Mat& C) {
    const int M = A.r, K = A.c, N = B.r;
    C.resize(M, N);
    GemmOp g{'N', 'T', M, N, K, A.data(), A.s, B.data(), B.s, C.data(), C.s};
    gpuCommitGemms(1, &g);
}

// C = A^T * B, A physical [M x K], B [M x N] -> [K x N].
inline void matmulAtB(const Mat& A, const Mat& B, Mat& C) {
    const int M = A.r, K = A.c, N = B.c;
    C.resize(K, N);
    GemmOp g{'T', 'N', K, N, M, A.data(), A.s, B.data(), B.s, C.data(), C.s};
    gpuCommitGemms(1, &g);
}

inline void addBiasRows(Mat& X, const std::vector<float>& b) {
    for (int i = 0; i < X.r; ++i) {
        float* x = X.row(i);
        for (int j = 0; j < X.c; ++j) x[j] += b[j];
    }
}

inline void reluFwd(Mat& X) {
    for (int i = 0; i < X.r; ++i) {
        float* x = X.row(i);
        for (int j = 0; j < X.c; ++j) x[j] = x[j] > 0.0f ? x[j] : 0.0f;
    }
}

// dy = (pre > 0) * gradOut; can operate in place.
inline void reluBwd(const Mat& pre, const Mat& gradOut, Mat& gradIn) {
    gradIn.resize(gradOut.r, gradOut.c);
    for (int i = 0; i < gradOut.r; ++i)
        for (int j = 0; j < gradOut.c; ++j)
            gradIn.row(i)[j] = pre.row(i)[j] > 0.0f ? gradOut.row(i)[j] : 0.0f;
}

}  // namespace nn
