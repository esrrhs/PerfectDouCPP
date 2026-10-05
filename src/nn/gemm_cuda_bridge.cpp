#include "nn/gemm_cuda_bridge.h"

#if defined(_WIN32) && defined(PD_HAVE_CUBLAS)

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <d3d12.h>

#include <cuda.h>
#include <cublas_v2.h>
#include <nvrtc.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace nn {
namespace {

using PFN_cuInit = CUresult(CUDAAPI*)(unsigned int);
using PFN_cuGetErrorString = CUresult(CUDAAPI*)(CUresult, const char**);
using PFN_cuDeviceGet = CUresult(CUDAAPI*)(CUdevice*, int);
using PFN_cuDeviceGetAttribute = CUresult(CUDAAPI*)(int*, CUdevice_attribute, CUdevice);
using PFN_cuDevicePrimaryCtxRetain = CUresult(CUDAAPI*)(CUcontext*, CUdevice);
using PFN_cuCtxSetCurrent = CUresult(CUDAAPI*)(CUcontext);
using PFN_cuCtxGetCurrent = CUresult(CUDAAPI*)(CUcontext*);
using PFN_cuImportExternalMemory = CUresult(CUDAAPI*)(CUexternalMemory*, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC*);
using PFN_cuExternalMemoryGetMappedBuffer = CUresult(CUDAAPI*)(CUdeviceptr*, CUexternalMemory, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC*);
using PFN_cuDestroyExternalMemory = CUresult(CUDAAPI*)(CUexternalMemory);
using PFN_cuMemcpyDtoH_v2 = CUresult(CUDAAPI*)(void*, CUdeviceptr, size_t);
using PFN_cuStreamCreate = CUresult(CUDAAPI*)(CUstream*, unsigned int);
using PFN_cuStreamSynchronize = CUresult(CUDAAPI*)(CUstream);
using PFN_cuLaunchHostFunc = CUresult(CUDAAPI*)(CUstream, CUhostFn, void*);
using PFN_cuStreamDestroy_v2 = CUresult(CUDAAPI*)(CUstream);
using PFN_cuModuleLoadData = CUresult(CUDAAPI*)(CUmodule*, const void*);
using PFN_cuModuleGetFunction = CUresult(CUDAAPI*)(CUfunction*, CUmodule, const char*);
using PFN_cuModuleUnload = CUresult(CUDAAPI*)(CUmodule);
using PFN_cuLaunchKernel = CUresult(CUDAAPI*)(CUfunction, unsigned int, unsigned int, unsigned int,
                                            unsigned int, unsigned int, unsigned int,
                                            unsigned int, CUstream, void**, void**);
using PFN_cublasCreate = cublasStatus_t(CUBLASAPI*)(cublasHandle_t*);
using PFN_cublasDestroy = cublasStatus_t(CUBLASAPI*)(cublasHandle_t);
using PFN_cublasSetMathMode = cublasStatus_t(CUBLASAPI*)(cublasHandle_t, cublasMath_t);
using PFN_cublasSetStream = cublasStatus_t(CUBLASAPI*)(cublasHandle_t, cudaStream_t);
using PFN_cublasSgemm = cublasStatus_t(CUBLASAPI*)(cublasHandle_t, cublasOperation_t, cublasOperation_t,
                                                 int, int, int, const float*, const float*, int,
                                                 const float*, int, const float*, float*, int);
using PFN_nvrtcCreateProgram = nvrtcResult(*)(nvrtcProgram*, const char*, const char*, int, const char* const*, const char* const*);
using PFN_nvrtcDestroyProgram = nvrtcResult(*)(nvrtcProgram*);
using PFN_nvrtcCompileProgram = nvrtcResult(*)(nvrtcProgram, int, const char* const*);
using PFN_nvrtcGetPTXSize = nvrtcResult(*)(nvrtcProgram, size_t*);
using PFN_nvrtcGetPTX = nvrtcResult(*)(nvrtcProgram, char*);
using PFN_nvrtcGetCUBINSize = nvrtcResult(*)(nvrtcProgram, size_t*);
using PFN_nvrtcGetCUBIN = nvrtcResult(*)(nvrtcProgram, char*);
using PFN_nvrtcGetProgramLogSize = nvrtcResult(*)(nvrtcProgram, size_t*);
using PFN_nvrtcGetProgramLog = nvrtcResult(*)(nvrtcProgram, char*);

struct Api {
    HMODULE nvcuda = nullptr;
    HMODULE cublas = nullptr;
    HMODULE nvrtc = nullptr;
    PFN_cuInit cuInit = nullptr;
    PFN_cuGetErrorString cuGetErrorString = nullptr;
    PFN_cuDeviceGet cuDeviceGet = nullptr;
    PFN_cuDeviceGetAttribute cuDeviceGetAttribute = nullptr;
    PFN_cuDevicePrimaryCtxRetain cuDevicePrimaryCtxRetain = nullptr;
    PFN_cuCtxSetCurrent cuCtxSetCurrent = nullptr;
    PFN_cuCtxGetCurrent cuCtxGetCurrent = nullptr;
    PFN_cuImportExternalMemory cuImportExternalMemory = nullptr;
    PFN_cuExternalMemoryGetMappedBuffer cuExternalMemoryGetMappedBuffer = nullptr;
    PFN_cuDestroyExternalMemory cuDestroyExternalMemory = nullptr;
    PFN_cuMemcpyDtoH_v2 cuMemcpyDtoH = nullptr;
    PFN_cuStreamCreate cuStreamCreate = nullptr;
    PFN_cuStreamSynchronize cuStreamSynchronize = nullptr;
    PFN_cuLaunchHostFunc cuLaunchHostFunc = nullptr;
    PFN_cuStreamDestroy_v2 cuStreamDestroy = nullptr;
    PFN_cuModuleLoadData cuModuleLoadData = nullptr;
    PFN_cuModuleGetFunction cuModuleGetFunction = nullptr;
    PFN_cuModuleUnload cuModuleUnload = nullptr;
    PFN_cuLaunchKernel cuLaunchKernel = nullptr;
    PFN_cublasCreate cublasCreate = nullptr;
    PFN_cublasDestroy cublasDestroy = nullptr;
    PFN_cublasSetMathMode cublasSetMathMode = nullptr;
    PFN_cublasSetStream cublasSetStream = nullptr;
    PFN_cublasSgemm cublasSgemm = nullptr;
    PFN_nvrtcCreateProgram nvrtcCreateProgram = nullptr;
    PFN_nvrtcDestroyProgram nvrtcDestroyProgram = nullptr;
    PFN_nvrtcCompileProgram nvrtcCompileProgram = nullptr;
    PFN_nvrtcGetPTXSize nvrtcGetPTXSize = nullptr;
    PFN_nvrtcGetPTX nvrtcGetPTX = nullptr;
    PFN_nvrtcGetCUBINSize nvrtcGetCUBINSize = nullptr;
    PFN_nvrtcGetCUBIN nvrtcGetCUBIN = nullptr;
    PFN_nvrtcGetProgramLogSize nvrtcGetProgramLogSize = nullptr;
    PFN_nvrtcGetProgramLog nvrtcGetProgramLog = nullptr;
};

Api g_api;
std::mutex g_mu;
CUcontext g_ctx = nullptr;
CUdevice g_dev = 0;
CUmodule g_mod = nullptr;
bool g_ready = false;
bool g_failed = false;
// One stream per seat. Non-blocking so a DtoH on the NULL stream does not
// wait for the other seats. Each handle stays bound to its stream.
constexpr int kLanes = 8;
struct Lane {
    CUstream stream = nullptr;
    cublasHandle_t blas = nullptr;
    HANDLE done = nullptr;
    bool busy = false;
};
void CUDA_CB onLaneDone(void* user) {
    // No CUDA calls here. The waiting thread blocks on this event, not inside
    // cuStreamSynchronize, so the driver lock stays free for the other seats.
    SetEvent(static_cast<HANDLE>(user));
}
Lane g_lanes[kLanes];
std::atomic<int> g_laneNext{0};
thread_local int t_lane = -1;

int laneIndex() {
    if (t_lane < 0) t_lane = g_laneNext.fetch_add(1, std::memory_order_relaxed) % kLanes;
    return t_lane;
}
char g_err[256] = "cuda bridge not initialised";

struct Map {
    CUexternalMemory mem = nullptr;
    CUdeviceptr ptr = 0;
};
std::unordered_map<ID3D12Resource*, Map> g_maps;
std::unordered_map<std::string, CUfunction> g_fn;

template <typename T>
bool loadOne(HMODULE mod, const char* name, T& out) {
    out = reinterpret_cast<T>(GetProcAddress(mod, name));
    if (!out) {
        std::snprintf(g_err, sizeof(g_err), "missing %s", name);
        return false;
    }
    return true;
}

void setCu(CUresult st, const char* what) {
    const char* s = nullptr;
    if (g_api.cuGetErrorString) g_api.cuGetErrorString(st, &s);
    std::snprintf(g_err, sizeof(g_err), "%s failed (%d%s%s)", what, (int)st,
                  s ? ": " : "", s ? s : "");
}

bool loadApi() {
    if (g_api.nvcuda) return true;
    g_api.nvcuda = LoadLibraryA("nvcuda.dll");
    if (!g_api.nvcuda) {
        std::snprintf(g_err, sizeof(g_err), "nvcuda.dll not found");
        return false;
    }
    const char* cuda = std::getenv("CUDA_PATH");
    if (cuda && cuda[0]) {
        char path[MAX_PATH];
        std::snprintf(path, sizeof(path), "%s\\bin\\x64\\cublas64_13.dll", cuda);
        g_api.cublas = LoadLibraryA(path);
        std::snprintf(path, sizeof(path), "%s\\bin\\x64\\nvrtc64_130_0.dll", cuda);
        g_api.nvrtc = LoadLibraryA(path);
    }
    if (!g_api.cublas) g_api.cublas = LoadLibraryA("cublas64_13.dll");
    if (!g_api.nvrtc) g_api.nvrtc = LoadLibraryA("nvrtc64_130_0.dll");
    if (!g_api.cublas || !g_api.nvrtc) {
        std::snprintf(g_err, sizeof(g_err), "cublas64_13.dll or nvrtc64_130_0.dll not found");
        return false;
    }
    return loadOne(g_api.nvcuda, "cuInit", g_api.cuInit) &&
           loadOne(g_api.nvcuda, "cuGetErrorString", g_api.cuGetErrorString) &&
           loadOne(g_api.nvcuda, "cuDeviceGet", g_api.cuDeviceGet) &&
           loadOne(g_api.nvcuda, "cuDeviceGetAttribute", g_api.cuDeviceGetAttribute) &&
           loadOne(g_api.nvcuda, "cuDevicePrimaryCtxRetain", g_api.cuDevicePrimaryCtxRetain) &&
           loadOne(g_api.nvcuda, "cuCtxSetCurrent", g_api.cuCtxSetCurrent) &&
           loadOne(g_api.nvcuda, "cuCtxGetCurrent", g_api.cuCtxGetCurrent) &&
           loadOne(g_api.nvcuda, "cuImportExternalMemory", g_api.cuImportExternalMemory) &&
           loadOne(g_api.nvcuda, "cuExternalMemoryGetMappedBuffer", g_api.cuExternalMemoryGetMappedBuffer) &&
           loadOne(g_api.nvcuda, "cuDestroyExternalMemory", g_api.cuDestroyExternalMemory) &&
           loadOne(g_api.nvcuda, "cuMemcpyDtoH_v2", g_api.cuMemcpyDtoH) &&
           loadOne(g_api.nvcuda, "cuStreamCreate", g_api.cuStreamCreate) &&
           loadOne(g_api.nvcuda, "cuStreamSynchronize", g_api.cuStreamSynchronize) &&
           loadOne(g_api.nvcuda, "cuLaunchHostFunc", g_api.cuLaunchHostFunc) &&
           loadOne(g_api.nvcuda, "cuStreamDestroy_v2", g_api.cuStreamDestroy) &&
           loadOne(g_api.nvcuda, "cuModuleLoadData", g_api.cuModuleLoadData) &&
           loadOne(g_api.nvcuda, "cuModuleGetFunction", g_api.cuModuleGetFunction) &&
           loadOne(g_api.nvcuda, "cuModuleUnload", g_api.cuModuleUnload) &&
           loadOne(g_api.nvcuda, "cuLaunchKernel", g_api.cuLaunchKernel) &&
           loadOne(g_api.cublas, "cublasCreate_v2", g_api.cublasCreate) &&
           loadOne(g_api.cublas, "cublasDestroy_v2", g_api.cublasDestroy) &&
           loadOne(g_api.cublas, "cublasSetMathMode", g_api.cublasSetMathMode) &&
           loadOne(g_api.cublas, "cublasSetStream_v2", g_api.cublasSetStream) &&
           loadOne(g_api.cublas, "cublasSgemm_v2", g_api.cublasSgemm) &&
           loadOne(g_api.nvrtc, "nvrtcCreateProgram", g_api.nvrtcCreateProgram) &&
           loadOne(g_api.nvrtc, "nvrtcDestroyProgram", g_api.nvrtcDestroyProgram) &&
           loadOne(g_api.nvrtc, "nvrtcCompileProgram", g_api.nvrtcCompileProgram) &&
           loadOne(g_api.nvrtc, "nvrtcGetPTXSize", g_api.nvrtcGetPTXSize) &&
           loadOne(g_api.nvrtc, "nvrtcGetPTX", g_api.nvrtcGetPTX) &&
           loadOne(g_api.nvrtc, "nvrtcGetCUBINSize", g_api.nvrtcGetCUBINSize) &&
           loadOne(g_api.nvrtc, "nvrtcGetCUBIN", g_api.nvrtcGetCUBIN) &&
           loadOne(g_api.nvrtc, "nvrtcGetProgramLogSize", g_api.nvrtcGetProgramLogSize) &&
           loadOne(g_api.nvrtc, "nvrtcGetProgramLog", g_api.nvrtcGetProgramLog);
}

bool enter() {
    if (!g_ready) return false;
    CUresult st = g_api.cuCtxSetCurrent(g_ctx);
    if (st != CUDA_SUCCESS) {
        setCu(st, "cuCtxSetCurrent");
        return false;
    }
    return true;
}

void dropAll() {
    if (g_ctx) g_api.cuCtxSetCurrent(g_ctx);
    for (Lane& ln : g_lanes) {
        if (ln.busy && ln.stream) g_api.cuStreamSynchronize(ln.stream);
        ln.busy = false;
    }
    for (auto& kv : g_maps) {
        if (kv.second.mem) g_api.cuDestroyExternalMemory(kv.second.mem);
    }
    g_maps.clear();
    g_fn.clear();
    if (g_mod) {
        g_api.cuModuleUnload(g_mod);
        g_mod = nullptr;
    }
    for (Lane& ln : g_lanes) {
        if (ln.blas) {
            g_api.cublasDestroy(ln.blas);
            ln.blas = nullptr;
        }
        if (ln.stream) {
            g_api.cuStreamDestroy(ln.stream);
            ln.stream = nullptr;
        }
    }
    g_ready = false;
}

bool ensureInit() {
    if (g_ready) return enter();
    if (g_failed) return false;
    if (!loadApi()) {
        g_failed = true;
        return false;
    }
    if (!g_ctx) {
        CUresult st = g_api.cuInit(0);
        if (st != CUDA_SUCCESS) {
            setCu(st, "cuInit");
            g_failed = true;
            return false;
        }
        st = g_api.cuDeviceGet(&g_dev, 0);
        if (st != CUDA_SUCCESS) {
            setCu(st, "cuDeviceGet");
            g_failed = true;
            return false;
        }
        st = g_api.cuDevicePrimaryCtxRetain(&g_ctx, g_dev);
        if (st != CUDA_SUCCESS) {
            setCu(st, "cuDevicePrimaryCtxRetain");
            g_failed = true;
            return false;
        }
    }
    CUresult st = g_api.cuCtxSetCurrent(g_ctx);
    if (st != CUDA_SUCCESS) {
        setCu(st, "cuCtxSetCurrent");
        g_failed = true;
        return false;
    }
    // Non-blocking: a blocking stream joins the NULL stream, and cuMemcpyDtoH
    // uses that stream, which would stall every seat at every readback.
    // cublasSetStream is checked; the gemm does not stay on NULL.
    for (Lane& ln : g_lanes) {
        if (!ln.done) {
            ln.done = CreateEventA(nullptr, FALSE, FALSE, nullptr);
            if (!ln.done) {
                std::snprintf(g_err, sizeof(g_err), "CreateEvent failed");
                g_failed = true;
                return false;
            }
        }
        st = g_api.cuStreamCreate(&ln.stream, CU_STREAM_NON_BLOCKING);
        if (st != CUDA_SUCCESS) {
            setCu(st, "cuStreamCreate");
            g_failed = true;
            return false;
        }
        cublasStatus_t bs = g_api.cublasCreate(&ln.blas);
        if (bs != CUBLAS_STATUS_SUCCESS) {
            std::snprintf(g_err, sizeof(g_err), "cublasCreate failed (%d)", (int)bs);
            g_failed = true;
            return false;
        }
        g_api.cublasSetMathMode(ln.blas, CUBLAS_PEDANTIC_MATH);
        bs = g_api.cublasSetStream(ln.blas, reinterpret_cast<cudaStream_t>(ln.stream));
        if (bs != CUBLAS_STATUS_SUCCESS) {
            std::snprintf(g_err, sizeof(g_err), "cublasSetStream failed (%d)", (int)bs);
            g_failed = true;
            return false;
        }
    }
    g_ready = true;
    std::snprintf(g_err, sizeof(g_err), "ok");
    return true;
}

bool mapOf(ID3D12Device* device, ID3D12Resource* resource, CUdeviceptr* out) {
    auto it = g_maps.find(resource);
    if (it != g_maps.end()) {
        *out = it->second.ptr;
        return true;
    }
    D3D12_RESOURCE_DESC rd = resource->GetDesc();
    HANDLE handle = nullptr;
    HRESULT hr = device->CreateSharedHandle(resource, nullptr, GENERIC_ALL, nullptr, &handle);
    if (FAILED(hr) || !handle) {
        std::snprintf(g_err, sizeof(g_err), "CreateSharedHandle failed (0x%08lx)", (unsigned long)hr);
        return false;
    }
    CUDA_EXTERNAL_MEMORY_HANDLE_DESC hd = {};
    hd.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
    hd.handle.win32.handle = handle;
    hd.size = (unsigned long long)rd.Width;
    hd.flags = CUDA_EXTERNAL_MEMORY_DEDICATED;
    CUexternalMemory mem = nullptr;
    CUresult st = g_api.cuImportExternalMemory(&mem, &hd);
    CloseHandle(handle);
    if (st != CUDA_SUCCESS) {
        setCu(st, "cuImportExternalMemory");
        return false;
    }
    CUDA_EXTERNAL_MEMORY_BUFFER_DESC bd = {};
    bd.offset = 0;
    bd.size = (unsigned long long)rd.Width;
    CUdeviceptr ptr = 0;
    st = g_api.cuExternalMemoryGetMappedBuffer(&ptr, mem, &bd);
    if (st != CUDA_SUCCESS) {
        setCu(st, "cuExternalMemoryGetMappedBuffer");
        g_api.cuDestroyExternalMemory(mem);
        return false;
    }
    g_maps.emplace(resource, Map{mem, ptr});
    *out = ptr;
    return true;
}

const char* kSrc = R"CUDA(
extern "C" __global__ void add_bias(float* Y, float* BI, int a, int b, int c) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= b) return;
  int row = gid / a;
  int col = gid - row * a;
  Y[row * c + col] += BI[col];
}
extern "C" __global__ void relu_fwd(float* Y, int a, int b, int c) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= b) return;
  int row = gid / a;
  int col = gid - row * a;
  int i = row * c + col;
  Y[i] = Y[i] > 0.f ? Y[i] : 0.f;
}
extern "C" __global__ void relu_bwd(float* Pre, float* Gout, float* Gin, int a, int b, int c, int d) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= b) return;
  int row = gid / a;
  int col = gid - row * a;
  float pre = Pre[row * c + col];
  float g = Gout[row * d + col];
  Gin[row * d + col] = pre > 0.f ? g : 0.f;
}
extern "C" __global__ void gate_add(float* GP, float* GI, float* GH, float* BI, float* BH, int a, int b) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= b * a) return;
  int q = gid - (gid / a) * a;
  GP[gid] = GI[gid] + GH[gid] + BI[q] + BH[q];
}
extern "C" __global__ void lstm_cell_fwd(float* GP, float* CP, float* Hout, float* Cout,
                                        int h, int B, int hasCp) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int u = blockIdx.y * blockDim.y + threadIdx.y;
  if (b >= B || u >= h) return;
  int base = b * (4 * h);
  float iv = 1.f / (1.f + expf(-GP[base + u]));
  float fv = 1.f / (1.f + expf(-GP[base + h + u]));
  float gz = tanhf(GP[base + 2 * h + u]);
  float ov = 1.f / (1.f + expf(-GP[base + 3 * h + u]));
  float pc = hasCp ? CP[b * h + u] : 0.f;
  float cc = fv * pc + iv * gz;
  Cout[b * h + u] = cc;
  Hout[b * h + u] = ov * tanhf(cc);
}
extern "C" __global__ void lstm_cell_bwd(float* GH, float* GP, float* CN, float* CP,
                                        float* DH, float* DC, float* DG,
                                        int h, int B, int hasCp) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int u = blockIdx.y * blockDim.y + threadIdx.y;
  if (b >= B || u >= h) return;
  int uh = h;
  int base = b * (4 * uh);
  float tc = tanhf(CN[b * uh + u]);
  float iv = 1.f / (1.f + expf(-GP[base + u]));
  float fv = 1.f / (1.f + expf(-GP[base + uh + u]));
  float gz = tanhf(GP[base + 2 * uh + u]);
  float ov = 1.f / (1.f + expf(-GP[base + 3 * uh + u]));
  float dhn = GH[b * uh + u] + DH[b * uh + u];
  float pc = hasCp ? CP[b * uh + u] : 0.f;
  float dct = dhn * ov * (1.f - tc * tc) + DC[b * uh + u];
  DG[base + u] = dct * gz * iv * (1.f - iv);
  DG[base + uh + u] = dct * pc * fv * (1.f - fv);
  DG[base + 2 * uh + u] = dct * iv * (1.f - gz * gz);
  DG[base + 3 * uh + u] = dhn * tc * ov * (1.f - ov);
  DC[b * uh + u] = dct * fv;
}
// One block owns one batch row and walks every time step. The next step only
// needs this row's hidden state, so the block keeps it in shared memory and
// the whole unroll is a single launch.
extern "C" __global__ void lstm_seq_fwd(
    float* H, const float* W, const float* GI, const float* BI, const float* BH,
    float* GP, float* C,
    int h, int B, int T, int hs, int ws, int gis, int gps, int cs) {
  extern __shared__ float sm[];
  int b = (int)blockIdx.x;
  if (b >= B) return;
  for (int u = (int)threadIdx.x; u < h; u += (int)blockDim.x) {
    sm[u] = 0.f;
    H[(size_t)b * hs + u] = 0.f;
  }
  __syncthreads();
  for (int t = 0; t < T; ++t) {
    float neu[8];
    int nu = 0;
    for (int u = (int)threadIdx.x; u < h; u += (int)blockDim.x) {
      const float* gi = GI + (size_t)(t * B + b) * gis;
      float a0 = gi[u] + BI[u] + BH[u];
      float a1 = gi[h + u] + BI[h + u] + BH[h + u];
      float a2 = gi[2 * h + u] + BI[2 * h + u] + BH[2 * h + u];
      float a3 = gi[3 * h + u] + BI[3 * h + u] + BH[3 * h + u];
      for (int k = 0; k < h; ++k) {
        float hv = sm[k];
        const float* wk = W + (size_t)k * ws;
        a0 += hv * wk[u];
        a1 += hv * wk[h + u];
        a2 += hv * wk[2 * h + u];
        a3 += hv * wk[3 * h + u];
      }
      float* gp = GP + (size_t)(t * B + b) * gps;
      gp[u] = a0;
      gp[h + u] = a1;
      gp[2 * h + u] = a2;
      gp[3 * h + u] = a3;
      float iv = 1.f / (1.f + expf(-a0));
      float fv = 1.f / (1.f + expf(-a1));
      float gz = tanhf(a2);
      float ov = 1.f / (1.f + expf(-a3));
      float pc = t ? C[(size_t)((t - 1) * B + b) * cs + u] : 0.f;
      float cc = fv * pc + iv * gz;
      C[(size_t)(t * B + b) * cs + u] = cc;
      float hn = ov * tanhf(cc);
      H[(size_t)((t + 1) * B + b) * hs + u] = hn;
      neu[nu++] = hn;
    }
    __syncthreads();
    nu = 0;
    for (int u = (int)threadIdx.x; u < h; u += (int)blockDim.x) sm[u] = neu[nu++];
    __syncthreads();
  }
}
// Cell backward and the recurrent input gradient (dg @ Wh) for every step.
// Weight and bias reductions stay outside so their sum order does not change.
extern "C" __global__ void lstm_seq_bwd(
    const float* GH, const float* GP, const float* CN, const float* WH, float* DG,
    int h, int B, int T, int ghs, int gps, int cs, int dgs, int whs) {
  extern __shared__ float sm[];
  float* dg = sm;
  float* dh = sm + 4 * h;
  float* dc = dh + h;
  int b = (int)blockIdx.x;
  if (b >= B) return;
  for (int u = (int)threadIdx.x; u < h; u += (int)blockDim.x) {
    dh[u] = 0.f;
    dc[u] = 0.f;
  }
  __syncthreads();
  for (int t = T - 1; t >= 0; --t) {
    for (int u = (int)threadIdx.x; u < h; u += (int)blockDim.x) {
      size_t row = (size_t)(t * B + b);
      const float* gp = GP + row * gps;
      float tc = tanhf(CN[row * cs + u]);
      float iv = 1.f / (1.f + expf(-gp[u]));
      float fv = 1.f / (1.f + expf(-gp[h + u]));
      float gz = tanhf(gp[2 * h + u]);
      float ov = 1.f / (1.f + expf(-gp[3 * h + u]));
      float dhn = GH[row * ghs + u] + dh[u];
      float pc = t ? CN[(size_t)((t - 1) * B + b) * cs + u] : 0.f;
      float dct = dhn * ov * (1.f - tc * tc) + dc[u];
      float d0 = dct * gz * iv * (1.f - iv);
      float d1 = dct * pc * fv * (1.f - fv);
      float d2 = dct * iv * (1.f - gz * gz);
      float d3 = dhn * tc * ov * (1.f - ov);
      dg[u] = d0;
      dg[h + u] = d1;
      dg[2 * h + u] = d2;
      dg[3 * h + u] = d3;
      float* grow = DG + row * dgs;
      grow[u] = d0;
      grow[h + u] = d1;
      grow[2 * h + u] = d2;
      grow[3 * h + u] = d3;
      dc[u] = dct * fv;
    }
    __syncthreads();
    for (int u = (int)threadIdx.x; u < h; u += (int)blockDim.x) {
      float s = 0.f;
      for (int q = 0; q < 4 * h; ++q) s += dg[q] * WH[(size_t)q * whs + u];
      dh[u] = s;
    }
    __syncthreads();
  }
}
extern "C" __global__ void concat2(float* Z, float* A, float* B,
                                  int n1, int n2, int rows, int zs, int as, int bs) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int cols = n1 + n2;
  if (gid >= rows * cols) return;
  int i = gid / cols;
  int j = gid - i * cols;
  float v = (j < n1) ? A[i * as + j] : B[i * bs + (j - n1)];
  Z[i * zs + j] = v;
}
extern "C" __global__ void split2(float* Z, float* A, float* B,
                                 int n1, int n2, int rows, int zs, int as, int bs) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int cols = n1 + n2;
  if (gid >= rows * cols) return;
  int i = gid / cols;
  int j = gid - i * cols;
  float v = Z[i * zs + j];
  if (j < n1) A[i * as + j] = v;
  else B[i * bs + (j - n1)] = v;
}
extern "C" __global__ void zero_and_last(float* All, float* GH, int B, int T, int h) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int n = B * T * h;
  if (gid >= n) return;
  int span = B * h;
  int t = gid / span;
  int rem = gid - t * span;
  All[gid] = (t + 1 == T) ? GH[rem] : 0.f;
}
extern "C" __global__ void mask_dyn(float* Logits, float* DS, float* Mask, int N, int B, int ls, int ms) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= B * N) return;
  int i = gid / N;
  int a = gid - i * N;
  int li = i * ls + a;
  if (Mask[i * ms + a] > 0.5f) Logits[li] += DS[(i * N + a) * 4];
  else Logits[li] = -1.0e9f;
}
extern "C" __global__ void add_to(float* Dst, float* Src, int rows, int cols, int dc, int sc) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= rows * cols) return;
  int i = gid / cols;
  int j = gid - i * cols;
  int di = i * dc + j;
  Dst[di] += Src[i * sc + j];
}
extern "C" __global__ void bias_grad_add(float* G, float* DB, int B, int o, int gs) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= o) return;
  float s = 0.f;
  for (int i = 0; i < B; ++i) s += G[i * gs + gid];
  DB[gid] += s;
}
extern "C" __global__ void slice_cols(float* Dst, float* Src, int B, int h, int off, int ds, int ss) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= B * h) return;
  int i = gid / h;
  int j = gid - i * h;
  Dst[i * ds + j] = Src[i * ss + off + j];
}
extern "C" __global__ void flatten_rows(float* Dst, float* Src, int B, int N, int ss) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= B * N) return;
  int i = gid / N;
  int a = gid - i * N;
  Dst[gid * 4] = Src[i * ss + a];
}
extern "C" __global__ void copy_mat(float* Dst, float* Src, int rows, int cols, int ds, int ss) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= rows * cols) return;
  int i = gid / cols;
  int j = gid - i * cols;
  Dst[i * ds + j] = Src[i * ss + j];
}
extern "C" __global__ void adam_step(float* W, float* DW, float* M, float* V,
                                    int n, float lr, float b1, float b2, float eps, float bc1, float bc2) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= n) return;
  float g = DW[gid];
  float mm = b1 * M[gid] + (1.f - b1) * g;
  float vv = b2 * V[gid] + (1.f - b2) * g * g;
  M[gid] = mm;
  V[gid] = vv;
  W[gid] = W[gid] - lr * (mm / bc1) / (sqrtf(vv / bc2) + eps);
}
)CUDA";

bool ensureModule() {
    if (g_mod) return true;
    nvrtcProgram prog = nullptr;
    nvrtcResult nr = g_api.nvrtcCreateProgram(&prog, kSrc, "elem.cu", 0, nullptr, nullptr);
    if (nr != NVRTC_SUCCESS) {
        std::snprintf(g_err, sizeof(g_err), "nvrtcCreateProgram failed (%d)", (int)nr);
        return false;
    }
    int major = 0, minor = 0;
    g_api.cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, g_dev);
    g_api.cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, g_dev);
    char arch[64];
    std::snprintf(arch, sizeof(arch), "--gpu-architecture=sm_%d%d", major, minor);
    const char* opts[] = {arch, "--fmad=true"};
    nr = g_api.nvrtcCompileProgram(prog, 2, opts);
    if (nr != NVRTC_SUCCESS) {
        size_t sz = 0;
        g_api.nvrtcGetProgramLogSize(prog, &sz);
        std::string log(sz, '\0');
        g_api.nvrtcGetProgramLog(prog, log.data());
        std::fprintf(stderr, "nvrtc: %s\n", log.c_str());
        std::snprintf(g_err, sizeof(g_err), "nvrtcCompileProgram failed");
        g_api.nvrtcDestroyProgram(&prog);
        return false;
    }
    size_t cubinSize = 0;
    g_api.nvrtcGetCUBINSize(prog, &cubinSize);
    std::vector<char> cubin(cubinSize);
    g_api.nvrtcGetCUBIN(prog, cubin.data());
    g_api.nvrtcDestroyProgram(&prog);
    CUresult st = g_api.cuModuleLoadData(&g_mod, cubin.data());
    if (st != CUDA_SUCCESS) {
        setCu(st, "cuModuleLoadData");
        return false;
    }
    return true;
}

CUfunction functionOf(const char* name) {
    auto it = g_fn.find(name);
    if (it != g_fn.end()) return it->second;
    CUfunction fn = nullptr;
    CUresult st = g_api.cuModuleGetFunction(&fn, g_mod, name);
    if (st != CUDA_SUCCESS) {
        setCu(st, name);
        return nullptr;
    }
    g_fn.emplace(name, fn);
    return fn;
}

}  // namespace

void crumb(const char*) {}

bool cudaBridgeGemm(ID3D12Device* device,
                    ID3D12Resource* a, size_t aOff,
                    ID3D12Resource* b, size_t bOff,
                    ID3D12Resource* c, size_t cOff,
                    char transA, char transB,
                    int M, int N, int K, int lda, int ldb, int ldc) {
    char line[200];
    std::snprintf(line, sizeof(line), "gemm %c%c %d %d %d ld %d %d %d",
                  transA, transB, M, N, K, lda, ldb, ldc);
    crumb(line);
    std::lock_guard<std::mutex> lock(g_mu);
    if (!ensureInit()) return false;
    CUdeviceptr pa = 0, pb = 0, pc = 0;
    if (!mapOf(device, a, &pa) || !mapOf(device, b, &pb) || !mapOf(device, c, &pc)) {
        crumb("gemm map fail");
        return false;
    }
    std::snprintf(line, sizeof(line), "gemm ptr %llx %llx %llx off %zu %zu %zu",
                  (unsigned long long)pa, (unsigned long long)pb, (unsigned long long)pc,
                  aOff, bOff, cOff);
    crumb(line);
    cublasOperation_t opA = (transA == 'T') ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = (transB == 'T') ? CUBLAS_OP_T : CUBLAS_OP_N;
    const float alpha = 1.f, beta = 0.f;
    const float* A = reinterpret_cast<const float*>(pa + aOff);
    const float* B = reinterpret_cast<const float*>(pb + bOff);
    float* C = reinterpret_cast<float*>(pc + cOff);
    Lane& ln = g_lanes[laneIndex()];
    cublasStatus_t bs = g_api.cublasSgemm(ln.blas, opB, opA, N, M, K, &alpha, B, ldb, A, lda, &beta, C, ldc);
    std::snprintf(line, sizeof(line), "gemm done %d", (int)bs);
    crumb(line);
    if (bs != CUBLAS_STATUS_SUCCESS) {
        std::snprintf(g_err, sizeof(g_err), "cublasSgemm failed (%d)", (int)bs);
        return false;
    }
    ln.busy = true;
    return true;
}

bool cudaBridgeKernel(ID3D12Device* device, const char* name,
                      ID3D12Resource** resources, const size_t* offsets, int nbuf,
                      const void* cst, size_t cbytes,
                      unsigned gridX, unsigned gridY) {
    char line[220];
    std::snprintf(line, sizeof(line), "kern %s %u %u nbuf %d cbytes %zu",
                  name, gridX, gridY, nbuf, cbytes);
    crumb(line);
    std::lock_guard<std::mutex> lock(g_mu);
    if (!ensureInit() || !ensureModule()) {
        crumb("kern init fail");
        return false;
    }
    CUfunction fn = functionOf(name);
    if (!fn) {
        crumb("kern no fn");
        return false;
    }
    CUdeviceptr base[8] = {};
    if (nbuf > 8) return false;
    for (int i = 0; i < nbuf; ++i) {
        std::snprintf(line, sizeof(line), "kern map %d %p off %zu", i, (void*)resources[i], offsets[i]);
        crumb(line);
        if (!resources[i] || !mapOf(device, resources[i], &base[i])) {
            crumb("kern map fail");
            return false;
        }
        base[i] += offsets[i];
    }
    crumb("kern mapped");
    int ints[16] = {};
    int nint = (int)(cbytes / sizeof(int));
    if (nint > 16) nint = 16;
    if (cst && nint > 0) std::memcpy(ints, cst, (size_t)nint * sizeof(int));

    unsigned bx = 64, by = 1;
    unsigned shmem = 0;
    if (std::strcmp(name, "lstm_cell_fwd") == 0 || std::strcmp(name, "lstm_cell_bwd") == 0) {
        bx = 8;
        by = 8;
    } else if (std::strcmp(name, "lstm_seq_fwd") == 0) {
        bx = 128;
        by = 1;
        if (nint >= 1 && ints[0] > 0) shmem = (unsigned)ints[0] * sizeof(float);
    } else if (std::strcmp(name, "lstm_seq_bwd") == 0) {
        bx = 128;
        by = 1;
        if (nint >= 1 && ints[0] > 0) shmem = (unsigned)ints[0] * 6u * sizeof(float);
    } else if (std::strcmp(name, "adam_step") == 0) {
        bx = 128;
    }
    if (gridX == 0) gridX = 1;
    if (gridY == 0) gridY = 1;

    void* args[16] = {};
    int na = 0;
    for (int i = 0; i < nbuf; ++i) args[na++] = &base[i];

    struct Adam {
        int n;
        float lr, b1, b2, eps, bc1, bc2;
    } adam{};
    if (std::strcmp(name, "adam_step") == 0) {
        if (cbytes < sizeof(adam)) return false;
        std::memcpy(&adam, cst, sizeof(adam));
        args[na++] = &adam.n;
        args[na++] = &adam.lr;
        args[na++] = &adam.b1;
        args[na++] = &adam.b2;
        args[na++] = &adam.eps;
        args[na++] = &adam.bc1;
        args[na++] = &adam.bc2;
    } else {
        for (int i = 0; i < nint; ++i) args[na++] = &ints[i];
    }
    crumb("kern launch");
    Lane& ln = g_lanes[laneIndex()];
    CUresult st = g_api.cuLaunchKernel(fn, gridX, gridY, 1, bx, by, 1, shmem, ln.stream, args, nullptr);
    if (st != CUDA_SUCCESS) {
        setCu(st, name);
        return false;
    }
    ln.busy = true;
    return true;
}

void cudaBridgeSetStream(int seat) {
    if (seat < 0) seat = 0;
    t_lane = seat % kLanes;
}

bool syncLane(int idx) {
    CUstream s = nullptr;
    HANDLE ev = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        if (!ensureInit()) return false;
        if (idx < 0 || idx >= kLanes) return false;
        if (!g_lanes[idx].busy) return true;
        s = g_lanes[idx].stream;
        ev = g_lanes[idx].done;
        ResetEvent(ev);
        CUresult st = g_api.cuLaunchHostFunc(s, onLaneDone, ev);
        if (st != CUDA_SUCCESS) {
            setCu(st, "cuLaunchHostFunc");
            return false;
        }
    }
    // OS wait, not a CUDA synchronize: another seat can launch while this one runs.
    WaitForSingleObject(ev, INFINITE);
    CUresult st = g_api.cuStreamSynchronize(s);
    std::lock_guard<std::mutex> lock(g_mu);
    if (st != CUDA_SUCCESS) {
        setCu(st, "cuStreamSynchronize");
        return false;
    }
    g_lanes[idx].busy = false;
    return true;
}

bool cudaBridgeHasWork() {
    std::lock_guard<std::mutex> lock(g_mu);
    int idx = t_lane < 0 ? 0 : t_lane;
    return g_lanes[idx].busy;
}

bool cudaBridgeSync() {
    int idx = t_lane < 0 ? 0 : t_lane;
    return syncLane(idx);
}

bool cudaBridgeSyncAll() {
    for (int i = 0; i < kLanes; ++i)
        if (!syncLane(i)) return false;
    return true;
}

bool cudaBridgeDtoH(ID3D12Device* device, ID3D12Resource* resource, size_t offset,
                    void* host, size_t bytes) {
    char line[160];
    std::snprintf(line, sizeof(line), "dtoh %zu", bytes);
    crumb(line);
    if (!cudaBridgeSync()) return false;
    std::lock_guard<std::mutex> lock(g_mu);
    if (!ensureInit()) return false;
    CUdeviceptr ptr = 0;
    if (!mapOf(device, resource, &ptr)) return false;
    CUresult st = g_api.cuMemcpyDtoH(host, ptr + offset, bytes);
    if (st != CUDA_SUCCESS) {
        setCu(st, "cuMemcpyDtoH");
        return false;
    }
    return true;
}

void cudaBridgeDrop(ID3D12Resource* resource) {
    if (!resource) return;
    if (!cudaBridgeSyncAll()) return;
    std::lock_guard<std::mutex> lock(g_mu);
    if (!g_ready) return;
    auto it = g_maps.find(resource);
    if (it == g_maps.end()) return;
    if (it->second.mem) g_api.cuDestroyExternalMemory(it->second.mem);
    g_maps.erase(it);
}

void cudaBridgeReset() {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!g_api.nvcuda) return;
    dropAll();
    g_failed = false;
}

const char* cudaBridgeError() { return g_err; }

}  // namespace nn

#else

namespace nn {
bool cudaBridgeGemm(ID3D12Device*, ID3D12Resource*, size_t, ID3D12Resource*, size_t,
                    ID3D12Resource*, size_t, char, char, int, int, int, int, int, int) { return false; }
bool cudaBridgeKernel(ID3D12Device*, const char*, ID3D12Resource**, const size_t*, int,
                      const void*, size_t, unsigned, unsigned) { return false; }
void cudaBridgeSetStream(int) {}
bool cudaBridgeHasWork() { return false; }
bool cudaBridgeSync() { return true; }
bool cudaBridgeSyncAll() { return true; }
bool cudaBridgeDtoH(ID3D12Device*, ID3D12Resource*, size_t, void*, size_t) { return false; }
void cudaBridgeDrop(ID3D12Resource*) {}
void cudaBridgeReset() {}
const char* cudaBridgeError() { return "built without cuBLAS"; }
}  // namespace nn

#endif
