#include "nn/gemm_cuda_bridge.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <d3d12.h>

#if defined(PD_HAVE_CUBLAS)
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cublas_v2.h>
#endif

#include <algorithm>
#include <cwchar>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>

namespace nn {

#if defined(PD_HAVE_CUBLAS)
namespace {

struct Api {
    HMODULE cuda = nullptr;
    HMODULE cublas = nullptr;
    decltype(&cuInit) init = nullptr;
    decltype(&cuDeviceGet) deviceGet = nullptr;
    decltype(&cuDevicePrimaryCtxRetain) primaryCtxRetain = nullptr;
    decltype(&cuCtxSetCurrent) ctxSetCurrent = nullptr;
    decltype(&cuCtxSynchronize) ctxSync = nullptr;
    decltype(&cuImportExternalMemory) importMemory = nullptr;
    decltype(&cuExternalMemoryGetMappedBuffer) mapBuffer = nullptr;
    decltype(&cuMemFree_v2) memFree = nullptr;
    decltype(&cuMemcpyDtoH_v2) copyDtoH = nullptr;
    decltype(&cuDestroyExternalMemory) destroyMemory = nullptr;
    decltype(&cuImportExternalSemaphore) importSemaphore = nullptr;
    decltype(&cuWaitExternalSemaphoresAsync) waitSemaphore = nullptr;
    decltype(&cuSignalExternalSemaphoresAsync) signalSemaphore = nullptr;
    decltype(&cuDestroyExternalSemaphore) destroySemaphore = nullptr;
    decltype(&cublasCreate_v2) blasCreate = nullptr;
    decltype(&cublasSetMathMode) setMathMode = nullptr;
    decltype(&cublasSgemm_v2) sgemm = nullptr;
    CUcontext context = nullptr;
    cublasHandle_t handle = nullptr;
    bool tried = false;
    bool ok = false;
    char error[256] = "CUDA bridge not initialized";
};

Api& api() {
    static Api a;
    return a;
}

template <class T>
bool load(HMODULE module, const char* name, T& out) {
    out = reinterpret_cast<T>(GetProcAddress(module, name));
    return out != nullptr;
}

void fail(const char* text, int code = 0) {
    Api& a = api();
    if (code)
        std::snprintf(a.error, sizeof(a.error), "%s (%d)", text, code);
    else
        std::snprintf(a.error, sizeof(a.error), "%s", text);
}

bool init() {
    Api& a = api();
    if (a.tried) return a.ok;
    a.tried = true;

    a.cuda = LoadLibraryW(L"nvcuda.dll");
    wchar_t root[512] = {};
    DWORD n = GetEnvironmentVariableW(L"CUDA_PATH", root, 512);
    wchar_t blasPath[640] = {};
    if (n > 0 && n < 512)
        ::swprintf(blasPath, 640, L"%ls\\bin\\x64\\cublas64_13.dll", root);
    a.cublas = blasPath[0] ? LoadLibraryW(blasPath) : nullptr;
    if (!a.cublas) a.cublas = LoadLibraryW(L"cublas64_13.dll");
    if (!a.cuda || !a.cublas) {
        fail("cannot load nvcuda.dll/cublas64_13.dll");
        return false;
    }

    bool symbols =
        load(a.cuda, "cuInit", a.init) &&
        load(a.cuda, "cuDeviceGet", a.deviceGet) &&
        load(a.cuda, "cuDevicePrimaryCtxRetain", a.primaryCtxRetain) &&
        load(a.cuda, "cuCtxSetCurrent", a.ctxSetCurrent) &&
        load(a.cuda, "cuCtxSynchronize", a.ctxSync) &&
        load(a.cuda, "cuImportExternalMemory", a.importMemory) &&
        load(a.cuda, "cuExternalMemoryGetMappedBuffer", a.mapBuffer) &&
        load(a.cuda, "cuMemFree_v2", a.memFree) &&
        load(a.cuda, "cuMemcpyDtoH_v2", a.copyDtoH) &&
        load(a.cuda, "cuDestroyExternalMemory", a.destroyMemory) &&
        load(a.cuda, "cuImportExternalSemaphore", a.importSemaphore) &&
        load(a.cuda, "cuWaitExternalSemaphoresAsync", a.waitSemaphore) &&
        load(a.cuda, "cuSignalExternalSemaphoresAsync", a.signalSemaphore) &&
        load(a.cuda, "cuDestroyExternalSemaphore", a.destroySemaphore) &&
        load(a.cublas, "cublasCreate_v2", a.blasCreate) &&
        load(a.cublas, "cublasSetMathMode", a.setMathMode) &&
        load(a.cublas, "cublasSgemm_v2", a.sgemm);
    if (!symbols) {
        fail("CUDA/cuBLAS symbol missing");
        return false;
    }

    CUdevice dev = 0;
    CUresult cr = a.init(0);
    if (cr == CUDA_SUCCESS) cr = a.deviceGet(&dev, 0);
    if (cr == CUDA_SUCCESS) cr = a.primaryCtxRetain(&a.context, dev);
    if (cr == CUDA_SUCCESS) cr = a.ctxSetCurrent(a.context);
    if (cr != CUDA_SUCCESS) {
        fail("CUDA context creation failed", (int)cr);
        return false;
    }
    cublasStatus_t bs = a.blasCreate(&a.handle);
    if (bs != CUBLAS_STATUS_SUCCESS) {
        fail("cublasCreate failed", (int)bs);
        return false;
    }
    // Avoid TF32 and other reduced-precision substitutions.
    bs = a.setMathMode(a.handle, CUBLAS_PEDANTIC_MATH);
    if (bs != CUBLAS_STATUS_SUCCESS) {
        fail("cublasSetMathMode failed", (int)bs);
        return false;
    }
    a.ok = true;
    std::snprintf(a.error, sizeof(a.error), "ok");
    std::fprintf(stderr, "CUDA GEMM bridge: cuBLAS initialized\n");
    return true;
}

struct Mapping {
    ID3D12Resource* resource = nullptr;
    CUexternalMemory memory = nullptr;
    CUdeviceptr pointer = 0;
};

void release(Mapping& m) {
    Api& a = api();
    if (m.pointer) a.memFree(m.pointer);
    if (m.memory) a.destroyMemory(m.memory);
    m = {};
}

bool mapResource(ID3D12Device* device, ID3D12Resource* resource, Mapping& out) {
    HANDLE shared = nullptr;
    HRESULT hr = device->CreateSharedHandle(resource, nullptr, GENERIC_ALL,
                                             nullptr, &shared);
    if (FAILED(hr)) {
        fail("CreateSharedHandle failed", (int)hr);
        return false;
    }
    D3D12_RESOURCE_DESC rd = resource->GetDesc();
    D3D12_RESOURCE_ALLOCATION_INFO allocation =
        device->GetResourceAllocationInfo(0, 1, &rd);
    CUDA_EXTERNAL_MEMORY_HANDLE_DESC md = {};
    md.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
    md.handle.win32.handle = shared;
    md.size = allocation.SizeInBytes;
    md.flags = CUDA_EXTERNAL_MEMORY_DEDICATED;
    CUresult cr = api().importMemory(&out.memory, &md);
    CloseHandle(shared);
    if (cr != CUDA_SUCCESS) {
        fail("cuImportExternalMemory failed", (int)cr);
        return false;
    }
    CUDA_EXTERNAL_MEMORY_BUFFER_DESC bd = {};
    bd.offset = 0;
    bd.size = rd.Width;
    cr = api().mapBuffer(&out.pointer, out.memory, &bd);
    if (cr != CUDA_SUCCESS) {
        fail("cuExternalMemoryGetMappedBuffer failed", (int)cr);
        release(out);
        return false;
    }
    out.resource = resource;
    return true;
}

}  // namespace

bool cudaBridgeGemm(ID3D12Device* device,
                    ID3D12CommandQueue* queue,
                    ID3D12Fence* fence, unsigned long long waitValue,
                    unsigned long long signalValue,
                    ID3D12Resource* aRes, size_t aOff,
                    ID3D12Resource* bRes, size_t bOff,
                    ID3D12Resource* cRes, size_t cOff,
                    char transA, char transB,
                    int M, int N, int K, int lda, int ldb, int ldc,
                    float* hostOut, size_t hostOutBytes) {
    static std::mutex bridgeMutex;
    std::lock_guard<std::mutex> bridgeLock(bridgeMutex);
    if (!init() || !device || !queue || !fence ||
        !aRes || !bRes || !cRes) return false;
    CUresult setCurrent = api().ctxSetCurrent(api().context);
    if (setCurrent != CUDA_SUCCESS) {
        fail("cuCtxSetCurrent failed", (int)setCurrent);
        return false;
    }
    HANDLE fenceHandle = nullptr;
    HRESULT hr = device->CreateSharedHandle(fence, nullptr, GENERIC_ALL,
                                             nullptr, &fenceHandle);
    if (FAILED(hr)) {
        fail("CreateSharedHandle(fence) failed", (int)hr);
        return false;
    }
    CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC sd = {};
    sd.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE;
    sd.handle.win32.handle = fenceHandle;
    CUexternalSemaphore semaphore = nullptr;
    CUresult cr = api().importSemaphore(&semaphore, &sd);
    CloseHandle(fenceHandle);
    if (cr != CUDA_SUCCESS) {
        fail("cuImportExternalSemaphore failed", (int)cr);
        return false;
    }
    CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS wp = {};
    wp.params.fence.value = waitValue;
    cr = api().waitSemaphore(&semaphore, &wp, 1, nullptr);
    if (cr != CUDA_SUCCESS) {
        fail("cuWaitExternalSemaphoresAsync failed", (int)cr);
        api().destroySemaphore(semaphore);
        return false;
    }
    ID3D12Resource* resources[] = {aRes, bRes, cRes};
    std::vector<Mapping> maps;
    maps.reserve(3);
    auto pointerFor = [&](ID3D12Resource* resource, size_t off,
                          CUdeviceptr& pointer) -> bool {
        auto it = std::find_if(maps.begin(), maps.end(),
                               [&](const Mapping& m) {
                                   return m.resource == resource;
                               });
        if (it == maps.end()) {
            maps.emplace_back();
            if (!mapResource(device, resource, maps.back())) return false;
            it = maps.end() - 1;
        }
        pointer = it->pointer + off;
        return true;
    };

    CUdeviceptr ap = 0, bp = 0, cp = 0;
    bool mapped = pointerFor(resources[0], aOff, ap) &&
                  pointerFor(resources[1], bOff, bp) &&
                  pointerFor(resources[2], cOff, cp);
    cublasStatus_t bs = CUBLAS_STATUS_INTERNAL_ERROR;
    if (mapped) {
        const float alpha = 1.0f, beta = 0.0f;
        cublasOperation_t opB =
            transB == 'T' ? CUBLAS_OP_T : CUBLAS_OP_N;
        cublasOperation_t opA =
            transA == 'T' ? CUBLAS_OP_T : CUBLAS_OP_N;
        // Row-major C = op(A) op(B) is column-major
        // C^T = op(B)^T op(A)^T. The row strides become leading dimensions.
        bs = api().sgemm(api().handle, opB, opA, N, M, K, &alpha,
                         reinterpret_cast<const float*>(bp), ldb,
                         reinterpret_cast<const float*>(ap), lda, &beta,
                         reinterpret_cast<float*>(cp), ldc);
        if (bs == CUBLAS_STATUS_SUCCESS) {
            if (hostOut && hostOutBytes) {
                cr = api().copyDtoH(hostOut, cp, hostOutBytes);
                if (cr != CUDA_SUCCESS) {
                    fail("cuMemcpyDtoH(output) failed", (int)cr);
                    mapped = false;
                }
            }
            CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS sp = {};
            sp.params.fence.value = signalValue;
            cr = api().signalSemaphore(&semaphore, &sp, 1, nullptr);
            // Queue the consumer wait while the CUDA signal is still pending;
            // this is the cross-API memory dependency, not merely a CPU check
            // of an already-completed fence value.
            if (cr == CUDA_SUCCESS &&
                FAILED(queue->Wait(fence, signalValue)))
                cr = CUDA_ERROR_UNKNOWN;
            if (cr == CUDA_SUCCESS) cr = api().ctxSync();
            if (cr != CUDA_SUCCESS) {
                fail("CUDA semaphore/synchronize failed", (int)cr);
                mapped = false;
            }
        } else {
            fail("cublasSgemm failed", (int)bs);
            mapped = false;
        }
    }
    for (Mapping& m : maps) release(m);
    api().destroySemaphore(semaphore);
    return mapped && bs == CUBLAS_STATUS_SUCCESS;
}

const char* cudaBridgeError() { return api().error; }

#else

bool cudaBridgeGemm(ID3D12Device*, ID3D12CommandQueue*, ID3D12Fence*,
                    unsigned long long,
                    unsigned long long, ID3D12Resource*, size_t,
                    ID3D12Resource*, size_t, ID3D12Resource*, size_t,
                    char, char, int, int, int, int, int, int,
                    float*, size_t) {
    return false;
}

const char* cudaBridgeError() {
    return "built without CUDA Toolkit/cuBLAS";
}

#endif

}  // namespace nn
