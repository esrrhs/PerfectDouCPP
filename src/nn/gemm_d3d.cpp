// Direct3D 12 backend for Windows. Mirrors the Metal stream in gemm_mps.mm:
// tiled GEMM + the same elementwise kernels, encoded onto a per-thread
// compute command list and fenced only at gpuWait().
//
// Discrete GPUs do not share Apple's unified memory, so every host buffer
// lives in a default-heap resource. Seeds are uploaded and readbacks copied
// explicitly; residency windows suballocate those heaps so a training step
// does not create one resource per tensor.
#include "nn/gemm.h"
#include "nn/gemm_cuda_bridge.h"
#include "nn/kernels.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace nn {

namespace {

struct Res {
    ID3D12Resource* p = nullptr;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
    size_t cap = 0;
    // Never transitioned. Used for the per-thread unused-UAV stand-in.
    bool sticky = false;
};

struct View {
    Res* res = nullptr;
    size_t off = 0;
    size_t len = 0;
    D3D12_GPU_VIRTUAL_ADDRESS va() const {
        return res->p->GetGPUVirtualAddress() + off;
    }
};

struct CopyOp {
    Res* dst = nullptr;
    size_t dstOff = 0;
    ID3D12Resource* src = nullptr;
    size_t srcOff = 0;
    size_t bytes = 0;
};

struct Upload {
    ID3D12Resource* p = nullptr;
    char* map = nullptr;
    size_t cap = 0;
    size_t used = 0;
};

struct UpAlloc {
    ID3D12Resource* res = nullptr;
    size_t off = 0;
    char* ptr = nullptr;
};

struct Chunk {
    Res res;
    size_t used = 0;
};

struct Slot {
    Res gpu;
    size_t bytes = 0;
    bool dirty = false;
    bool shadowValid = false;
    void** owner = nullptr;
    // Thread that inserted this slot. liveSlots is thread-local, so a drop
    // marshalled onto the GPU owner must unlink it here or kick() writes
    // through the freed slot. homeMu guards that vector against kick().
    std::vector<Slot*>* home = nullptr;
    std::mutex* homeMu = nullptr;
    std::vector<char> shadow;
};

struct Bind {
    void* host = nullptr;
    size_t len = 0;
    bool toHost = false;
    View view;
};

struct ReadOp {
    Res* src = nullptr;
    size_t srcOff = 0;
    size_t dstOff = 0;
    void* host = nullptr;
    size_t bytes = 0;
    Slot* grad = nullptr;
};

struct Pipe {
    const char* name = nullptr;
    ID3D12PipelineState* pso = nullptr;
};

ID3D12Device* g_device = nullptr;
ID3D12RootSignature* g_root = nullptr;

Pipe g_pipes[24];
int g_npipes = 0;
char g_label[320] = "GPU (D3D12)";
bool g_initTried = false;
bool g_initOk = false;
std::atomic<bool> g_lost{false};
Res g_dummy;

bool deviceLost() { return g_lost.load(std::memory_order_acquire); }

struct Crumb {
    const char* name = nullptr;
    unsigned gx = 0, gy = 0;
    int a = 0, b = 0, c = 0, d = 0;
};
constexpr int kCrumbN = 12;
Crumb g_crumb[kCrumbN];
int g_crumbCount = 0;

struct BufInfo {
    unsigned long long va = 0;
    size_t off = 0, len = 0, cap = 0;
};
struct LastCmd {
    const char* name = nullptr;
    unsigned gx = 0, gy = 0;
    int nconst = 0;
    int cst[16] = {};
    int nbuf = 0;
    BufInfo buf[8];
};
LastCmd g_last;

int reproLog() {
    static int on = -1;
    if (on < 0) on = std::getenv("PD_REPRO") ? 1 : 0;
    return on;
}

int kickEvery() {
    static int n = -1;
    if (n < 0) {
        const char* e = std::getenv("PD_KICK");
        n = e ? std::atoi(e) : 32;
        if (n < 1) n = 1;
    }
    return n;
}

void writeLastCmd() {
    if (!reproLog() || !g_last.name) return;
    CreateDirectoryA("build-win", nullptr);
    CreateDirectoryA("build-win\\repro", nullptr);
    FILE* f = std::fopen("build-win/repro/last_cmd.txt", "w");
    if (!f) return;
    std::fprintf(f, "%s grid %u %u\n", g_last.name, g_last.gx, g_last.gy);
    std::fprintf(f, "const");
    for (int i = 0; i < g_last.nconst; ++i) std::fprintf(f, " %d", g_last.cst[i]);
    std::fprintf(f, "\n");
    for (int i = 0; i < g_last.nbuf; ++i) {
        const BufInfo& b = g_last.buf[i];
        unsigned long long tail =
            b.cap > b.off + b.len ? (unsigned long long)(b.cap - (b.off + b.len)) : 0;
        std::fprintf(f, "buf %d va 0x%llx off %zu len %zu cap %zu tail %llu\n",
                     i, b.va, b.off, b.len, b.cap, tail);
    }
    std::fclose(f);
}

void remember(const char* name, unsigned gx, unsigned gy, const void* cst,
              size_t cbytes) {
    Crumb& cr = g_crumb[g_crumbCount++ % kCrumbN];
    cr.name = name;
    cr.gx = gx;
    cr.gy = gy;
    cr.a = cr.b = cr.c = cr.d = 0;
    if (cst && cbytes >= sizeof(int) * 4) {
        const int* p = static_cast<const int*>(cst);
        cr.a = p[0];
        cr.b = p[1];
        cr.c = p[2];
        cr.d = p[3];
    }
}

void preserveRepro() {
    CreateDirectoryA("build-win", nullptr);
    CreateDirectoryA("build-win\\repro", nullptr);
    CreateDirectoryA("build-win\\repro\\lost", nullptr);
    const char* names[] = {
        "last_cmd.txt", "round.txt", "xImp.bin", "seq.bin", "mask.bin",
        "dyn.bin", "extra.bin", "logits.bin", "values.bin",
        "actor_w.bin", "critic_w.bin"};
    for (const char* name : names) {
        char src[160], dst[160];
        std::snprintf(src, sizeof(src), "build-win/repro/%s", name);
        std::snprintf(dst, sizeof(dst), "build-win/repro/lost/%s", name);
        CopyFileA(src, dst, FALSE);
    }
}

void noteLost(HRESULT hr) {
    if (!g_lost.exchange(true, std::memory_order_acq_rel)) {
        std::fprintf(stderr, "D3D12 device removed: 0x%08lx\n",
                     (unsigned long)hr);
        int n = g_crumbCount < kCrumbN ? g_crumbCount : kCrumbN;
        int begin = g_crumbCount - n;
        std::fprintf(stderr, "last %d GPU commands:\n", n);
        for (int i = 0; i < n; ++i) {
            const Crumb& cr = g_crumb[(begin + i) % kCrumbN];
            std::fprintf(stderr, "  %s grid %u %u args %d %d %d %d\n",
                         cr.name ? cr.name : "-", cr.gx, cr.gy, cr.a, cr.b,
                         cr.c, cr.d);
        }
        if (g_last.name) {
            std::fprintf(stderr, "hung %s grid %u %u\n", g_last.name, g_last.gx,
                         g_last.gy);
            for (int i = 0; i < g_last.nbuf; ++i) {
                const BufInfo& b = g_last.buf[i];
                unsigned long long tail = b.cap > b.off + b.len
                                               ? (unsigned long long)(b.cap - (b.off + b.len))
                                               : 0;
                std::fprintf(stderr,
                             "  buf %d off %zu len %zu cap %zu tail %llu\n",
                             i, b.off, b.len, b.cap, tail);
            }
        }
        preserveRepro();
        std::fflush(stderr);
    }
}

// One training wave (forward, or backward, or Adam) records every copy and
// every CUDA launch, then submits the copies once. Launching each GEMM as
// its own D3D packet was ~8000 CPU fence waits per seat per update.
struct CudaOp {
    bool gemm = false;
    char name[32] = {};
    ID3D12Resource* res[8] = {};
    size_t off[8] = {};
    int nbuf = 0;
    char transA = 'N';
    char transB = 'N';
    int M = 0, N = 0, K = 0, lda = 0, ldb = 0, ldc = 0;
    unsigned gx = 1, gy = 1;
    unsigned char cst[64] = {};
    size_t cbytes = 0;
};

struct Ctx {
    ID3D12CommandQueue* queue = nullptr;
    ID3D12Fence* fence = nullptr;
    std::vector<ID3D12CommandAllocator*> allocs;
    int allocIndex = 0;
    int sinceKick = 0;
    ID3D12GraphicsCommandList* list = nullptr;
    HANDLE event = nullptr;
    UINT64 fenceValue = 0;
    bool recording = false;
    std::deque<Chunk> chunks;
    std::vector<Upload> uploads;
    std::vector<CopyOp> pending;
    std::unordered_map<uintptr_t, Bind> window;
    std::vector<Slot*> dirtyGrads;
    std::vector<Slot*> graveyard;
    std::vector<Slot*> liveSlots;
    std::mutex slotMu;
    std::vector<CudaOp> cudaOps;
    ID3D12Resource* readback = nullptr;
    char* rbMap = nullptr;
    size_t rbCap = 0;
    long long nWait = 0, nCommit = 0;
    long long stageBytes = 0, flushBytes = 0;
    double waitSec = 0, encSec = 0;
    ~Ctx();
};

Ctx& ctx() {
    // Heap, not a thread_local object. MinGW runs the thread_local destructor
    // after the TLS block is released, so ~Ctx frees its vectors through a
    // dead heap header (0xC0000374) and, before that guard, calls through
    // 0xFEEEFEEE. The context stays for the life of the process; the OS
    // reclaims it. d3dReleaseThread() is the explicit COM release path.
    static thread_local Ctx* c = new Ctx();
    return *c;
}

size_t align256(size_t n) { return (n + 255u) & ~size_t(255); }

ID3D12Resource* createBuffer(size_t bytes, D3D12_HEAP_TYPE heap,
                             D3D12_RESOURCE_FLAGS flags,
                             D3D12_RESOURCE_STATES state) {
    // A new committed resource while CUDA still owns the device has crashed
    // this driver. The stream is idle before the allocation.
    if (cudaBridgeHasWork() && !cudaBridgeSync()) {
        std::fprintf(stderr, "CUDA sync before D3D alloc failed: %s\n", cudaBridgeError());
        noteLost(E_FAIL);
        return nullptr;
    }
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = heap;
    hp.CreationNodeMask = 1;
    hp.VisibleNodeMask = 1;
    D3D12_RESOURCE_DESC d = {};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = bytes;
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    d.Flags = flags;
    ID3D12Resource* res = nullptr;
    D3D12_HEAP_FLAGS heapFlags = D3D12_HEAP_FLAG_NONE;
    if (heap == D3D12_HEAP_TYPE_DEFAULT && std::getenv("PD_CUBLAS"))
        heapFlags = D3D12_HEAP_FLAG_SHARED;
    HRESULT hr = g_device->CreateCommittedResource(
        &hp, heapFlags, &d, state, nullptr, IID_PPV_ARGS(&res));
    if (FAILED(hr)) {
        HRESULT why = g_device ? g_device->GetDeviceRemovedReason() : hr;
        noteLost(FAILED(why) ? why : hr);
        return nullptr;
    }
    return res;
}

void eraseSlotLocked(std::vector<Slot*>* live, std::mutex* mu, Slot* s) {
    if (!live || !mu) return;
    std::lock_guard<std::mutex> lk(*mu);
    live->erase(std::remove(live->begin(), live->end(), s), live->end());
}

void releaseSlot(Slot* s) {
    if (!s) return;
    if (s->owner && *s->owner == s) *s->owner = nullptr;
    std::vector<Slot*>* home = s->home;
    std::mutex* homeMu = s->homeMu;
    std::vector<Slot*>* here = &ctx().liveSlots;
    std::mutex* hereMu = &ctx().slotMu;
    s->home = nullptr;
    s->homeMu = nullptr;
    if (home == here) {
        eraseSlotLocked(home, homeMu, s);
    } else if (reinterpret_cast<uintptr_t>(homeMu) < reinterpret_cast<uintptr_t>(hereMu)) {
        eraseSlotLocked(home, homeMu, s);
        eraseSlotLocked(here, hereMu, s);
    } else {
        eraseSlotLocked(here, hereMu, s);
        eraseSlotLocked(home, homeMu, s);
    }
    if (s->gpu.p) {
        cudaBridgeDrop(s->gpu.p);
        s->gpu.p->Release();
    }
    delete s;
}

void closeRecording(Ctx& c) {
    if (c.recording && c.list) c.list->Close();
    c.recording = false;
}

void teardown(Ctx& c) {
    // Close before releasing anything the list still names. A device-lost
    // path used to clear the recording flag and then free those resources
    // underneath an open command list.
    closeRecording(c);
    if (c.event && c.fence && c.fenceValue > 0) {
        while (c.fence->GetCompletedValue() < c.fenceValue) {
            c.fence->SetEventOnCompletion(c.fenceValue, c.event);
            if (WaitForSingleObject(c.event, 5000) != WAIT_OBJECT_0) break;
        }
    }
    std::vector<Slot*> slots;
    slots.swap(c.liveSlots);
    c.graveyard.clear();
    c.dirtyGrads.clear();
    for (Slot* s : slots) {
        if (!s) continue;
        if (s->owner && *s->owner == s) *s->owner = nullptr;
        if (s->gpu.p) {
            cudaBridgeDrop(s->gpu.p);
            s->gpu.p->Release();
        }
        delete s;
    }
    for (Upload& u : c.uploads) {
        if (u.p) {
            u.p->Unmap(0, nullptr);
            u.p->Release();
        }
    }
    c.uploads.clear();
    c.pending.clear();
    for (Chunk& ch : c.chunks)
        if (ch.res.p) {
            cudaBridgeDrop(ch.res.p);
            ch.res.p->Release();
        }
    c.chunks.clear();
    if (c.readback) {
        if (c.rbMap) c.readback->Unmap(0, nullptr);
        c.readback->Release();
        c.readback = nullptr;
        c.rbMap = nullptr;
        c.rbCap = 0;
    }
    c.window.clear();
    if (c.list) {
        c.list->Release();
        c.list = nullptr;
    }
    for (ID3D12CommandAllocator* a : c.allocs)
        if (a) a->Release();
    c.allocs.clear();
    if (c.queue) {
        c.queue->Release();
        c.queue = nullptr;
    }
    if (c.fence) {
        c.fence->Release();
        c.fence = nullptr;
    }
    if (c.event) {
        CloseHandle(c.event);
        c.event = nullptr;
    }
    c.fenceValue = 0;
    c.sinceKick = 0;
    c.cudaOps.clear();
}

// Buffers decay to COMMON when ExecuteCommandLists finishes, including
// ones we explicitly transitioned. The next list must barrier from COMMON;
// a stale UAV→COPY barrier on a decayed buffer removes the device.
void noteBufferDecay() {
    Ctx& c = ctx();
    for (Chunk& ch : c.chunks) ch.res.state = D3D12_RESOURCE_STATE_COMMON;
    std::lock_guard<std::mutex> lk(c.slotMu);
    for (Slot* s : c.liveSlots)
        if (s) s->gpu.state = D3D12_RESOURCE_STATE_COMMON;
}

Ctx::~Ctx() {
    // Live weight/grad slots outlive this thread: the owning Param releases
    // them after join. Freeing them here, while another learner is still
    // submitting, removes the device. d3dReleaseThread() is the path that
    // drops them, and only after every learner has stopped.
    // A worker's TLS block can already be the heap free-fill pattern
    // (0xFEEEFEEE) by the time this destructor runs. Those pointers are not
    // COM objects; calling through them aborts process shutdown.
    auto wild = [](const void* p) {
        return reinterpret_cast<uintptr_t>(p) > 0x00007FFFFFFFFFFFULL;
    };
    if (wild(queue) || wild(fence) || wild(list)) return;
    if (event && fence && fenceValue > 0) {
        while (fence->GetCompletedValue() < fenceValue) {
            fence->SetEventOnCompletion(fenceValue, event);
            if (WaitForSingleObject(event, 5000) != WAIT_OBJECT_0) break;
        }
    }
    if (recording && list) {
        list->Close();
        recording = false;
    }
    for (Slot* s : graveyard) releaseSlot(s);
    graveyard.clear();
    for (Upload& u : uploads) {
        if (u.p) {
            u.p->Unmap(0, nullptr);
            u.p->Release();
        }
    }
    uploads.clear();
    for (Chunk& ch : chunks) {
        if (ch.res.p) {
            cudaBridgeDrop(ch.res.p);
            ch.res.p->Release();
        }
    }
    chunks.clear();
    if (readback) {
        if (rbMap) readback->Unmap(0, nullptr);
        readback->Release();
        readback = nullptr;
        rbMap = nullptr;
    }
    if (list) {
        list->Release();
        list = nullptr;
    }
    for (ID3D12CommandAllocator* a : allocs)
        if (a) a->Release();
    allocs.clear();
    if (queue) {
        queue->Release();
        queue = nullptr;
    }
    if (fence) {
        fence->Release();
        fence = nullptr;
    }
    if (event) {
        CloseHandle(event);
        event = nullptr;
    }
}

bool ensureThread() {
    Ctx& c = ctx();
    if (c.list) return true;
    if (!g_device || deviceLost()) return false;
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    if (FAILED(g_device->CreateCommandQueue(&qd, IID_PPV_ARGS(&c.queue)))) {
        noteLost(E_FAIL);
        return false;
    }
    D3D12_FENCE_FLAGS fenceFlags = std::getenv("PD_CUBLAS")
                                       ? D3D12_FENCE_FLAG_SHARED
                                       : D3D12_FENCE_FLAG_NONE;
    if (FAILED(g_device->CreateFence(0, fenceFlags,
                                     IID_PPV_ARGS(&c.fence)))) {
        noteLost(E_FAIL);
        return false;
    }
    c.fenceValue = 0;
    ID3D12CommandAllocator* alloc = nullptr;
    if (FAILED(g_device->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&alloc)))) {
        noteLost(E_FAIL);
        return false;
    }
    c.allocs.push_back(alloc);
    c.allocIndex = 0;
    if (FAILED(g_device->CreateCommandList(
            0, D3D12_COMMAND_LIST_TYPE_COMPUTE, alloc, nullptr,
            IID_PPV_ARGS(&c.list)))) {
        noteLost(E_FAIL);
        return false;
    }
    c.list->Close();
    c.event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!c.event) noteLost(E_FAIL);
    return c.event != nullptr;
}

bool ensureRecording() {
    if (!ensureThread()) return false;
    Ctx& c = ctx();
    if (c.recording) return true;
    ID3D12CommandAllocator* alloc = c.allocs[c.allocIndex];
    if (FAILED(alloc->Reset()) || FAILED(c.list->Reset(alloc, nullptr))) {
        noteLost(E_FAIL);
        return false;
    }
    c.list->SetComputeRootSignature(g_root);
    c.recording = true;
    return true;
}

// One packet in flight process-wide. Each learner has its own queue, but
// three queues executing together never let the GPU go idle and all bind
// the shared dummy UAV. Either one removes the device on this driver.
std::mutex& flightMu() {
    static std::mutex m;
    return m;
}

// Windows resets the device if one command list runs longer than the TDR
// window (about 2s). Close and wait, then keep recording. Waiting before
// the allocator is reused is required; resetting a list that is still
// in flight corrupts later dispatches. Leaving the GPU continuously busy
// also trips that timeout, so the next packet is not recorded in parallel.
void kick(bool reopen) {
    Ctx& c = ctx();
    if (!c.recording) return;
    HRESULT hr = c.list->Close();
    c.recording = false;
    if (FAILED(hr) || deviceLost() || !c.queue || !g_device) {
        if (FAILED(hr)) noteLost(hr);
        return;
    }
    std::lock_guard<std::mutex> flight(flightMu());
    ID3D12CommandList* lists[] = {c.list};
    c.queue->ExecuteCommandLists(1, lists);
    // Written before the fence wait, so a TDR that never returns still leaves
    // the packet's kernel and buffer placement on disk.
    writeLastCmd();
    // Flushed before the fence wait, so a TDR that never returns still leaves
    // the last packet's kernel on disk. Off unless PD_GPU_LOG is set.
    static int logExec = -1;
    if (logExec < 0) logExec = std::getenv("PD_GPU_LOG") ? 1 : 0;
    if (logExec) {
        const Crumb& cr = g_crumb[(g_crumbCount + kCrumbN - 1) % kCrumbN];
        std::fprintf(stderr, "[d3d] execute #%d last=%s grid %u %u args %d %d %d %d\n",
                     g_crumbCount, cr.name ? cr.name : "-", cr.gx, cr.gy, cr.a,
                     cr.b, cr.c, cr.d);
        std::fflush(stderr);
    }
    UINT64 fv = ++c.fenceValue;
    c.queue->Signal(c.fence, fv);
    // Decay is sequenced before the next list on this queue, so record
    // against COMMON even while this packet is still in flight.
    noteBufferDecay();
    // An auto-reset event can already be signaled. Wait until the fence
    // itself has reached fv, otherwise the allocator is reset too early and
    // the device hangs.
    while (c.fence->GetCompletedValue() < fv) {
        c.fence->SetEventOnCompletion(fv, c.event);
        WaitForSingleObject(c.event, INFINITE);
    }
    HRESULT removed = g_device->GetDeviceRemovedReason();
    if (FAILED(removed)) {
        noteLost(removed);
        return;
    }
    ++c.nWait;
    c.sinceKick = 0;
    ID3D12CommandAllocator* alloc = c.allocs[c.allocIndex];
    if (FAILED(alloc->Reset())) {
        noteLost(E_FAIL);
        return;
    }
    if (!reopen) return;
    if (FAILED(c.list->Reset(alloc, nullptr))) {
        noteLost(E_FAIL);
        return;
    }
    c.list->SetComputeRootSignature(g_root);
    c.recording = true;
}

void transition(Res* r, D3D12_RESOURCE_STATES to) {
    if (!r || !r->p || r->sticky || r->state == to) return;
    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = r->p;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = r->state;
    b.Transition.StateAfter = to;
    ctx().list->ResourceBarrier(1, &b);
    r->state = to;
}

UpAlloc allocUpload(size_t bytes) {
    Ctx& c = ctx();
    size_t need = align256(std::max(bytes, size_t(1)));
    for (Upload& u : c.uploads) {
        if (u.used + need <= u.cap) {
            size_t off = u.used;
            u.used += need;
            return {u.p, off, u.map + off};
        }
    }
    Upload u;
    u.cap = std::max(need, size_t(32u << 20));
    u.p = createBuffer(u.cap, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                       D3D12_RESOURCE_STATE_GENERIC_READ);
    if (!u.p) {
        noteLost(E_FAIL);
        return {};
    }
    u.p->Map(0, nullptr, (void**)&u.map);
    u.used = need;
    c.uploads.push_back(u);
    return {u.p, 0, u.map};
}

View dummyView() {
    View v;
    v.res = g_dummy.p ? &g_dummy : nullptr;
    v.off = 0;
    v.len = 256;
    return v;
}

View allocScratch(size_t bytes, const void* seed) {
    if (bytes == 0) return dummyView();
    Ctx& c = ctx();
    size_t need = align256(bytes);
    Chunk* ch = nullptr;
    for (Chunk& cand : c.chunks) {
        if (cand.used + need <= cand.res.cap) {
            ch = &cand;
            break;
        }
    }
    if (!ch) {
        c.chunks.emplace_back();
        ch = &c.chunks.back();
        ch->res.cap = std::max(need, size_t(32u << 20));
        ch->res.state = D3D12_RESOURCE_STATE_COMMON;
        ch->res.p = createBuffer(ch->res.cap, D3D12_HEAP_TYPE_DEFAULT,
                                 D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                 D3D12_RESOURCE_STATE_COMMON);
        if (!ch->res.p) {
            c.chunks.pop_back();
            return {};
        }
        ch->used = 0;
    }
    View v;
    v.res = &ch->res;
    v.off = ch->used;
    v.len = bytes;
    ch->used += need;
    if (seed) {
        UpAlloc u = allocUpload(bytes);
        if (!u.ptr) return {};
        std::memcpy(u.ptr, seed, bytes);
        c.pending.push_back({v.res, v.off, u.res, u.off, bytes});
        c.stageBytes += (long long)bytes;
    }
    return v;
}

void flushPending() {
    Ctx& c = ctx();
    if (c.pending.empty()) return;
    if (!ensureRecording()) return;
    for (CopyOp& op : c.pending) transition(op.dst, D3D12_RESOURCE_STATE_COPY_DEST);
    for (CopyOp& op : c.pending)
        c.list->CopyBufferRegion(op.dst->p, op.dstOff, op.src, op.srcOff, op.bytes);
    for (CopyOp& op : c.pending)
        transition(op.dst, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    c.pending.clear();
}

View bindResolve(const void* p) {
    Ctx& c = ctx();
    auto it = c.window.find(reinterpret_cast<uintptr_t>(p));
    if (it != c.window.end()) return it->second.view;
    View best;
    size_t bestLen = SIZE_MAX;
    for (auto& kv : c.window) {
        auto* lo = static_cast<char*>(kv.second.host);
        auto* cur = static_cast<const char*>(p);
        if (cur < lo || cur >= lo + kv.second.len) continue;
        if (kv.second.len < bestLen) {
            bestLen = kv.second.len;
            best = kv.second.view;
            best.off += size_t(cur - lo);
            best.len -= size_t(cur - lo);
        }
    }
    return best;
}

View bindFresh(const void* p, size_t bytes, const void* seed, bool toHost) {
    Ctx& c = ctx();
    auto it = c.window.find(reinterpret_cast<uintptr_t>(p));
    if (it != c.window.end()) return it->second.view;
    Bind b;
    b.host = const_cast<void*>(p);
    b.len = bytes;
    b.toHost = toHost;
    b.view = allocScratch(bytes, seed);
    c.window.emplace(reinterpret_cast<uintptr_t>(p), b);
    return b.view;
}

View kerIn(const void* p, size_t bytes) {
    View v = bindResolve(p);
    if (v.res) return v;
    // Remember the upload. A second read of the same host pointer in this
    // wave must reuse the device copy; allocating again recopied every input
    // on every GEMM (about 2.5GB per seat per update).
    return bindFresh(p, bytes, p, false);
}

View kerOut(const void* p, size_t bytes) {
    View v = bindResolve(p);
    if (v.res) return v;
    return bindFresh(p, bytes, nullptr, false);
}

View kerOutSeed(const void* p, size_t bytes) {
    View v = bindResolve(p);
    if (v.res) return v;
    return bindFresh(p, bytes, p, false);
}

View scalarRmw(const void* p, size_t bytes) {
    View v = bindResolve(p);
    if (v.res) return v;
    return bindFresh(p, bytes, p, true);
}

ID3D12PipelineState* findPso(const char* name) {
    for (int i = 0; i < g_npipes; ++i)
        if (std::strcmp(g_pipes[i].name, name) == 0) return g_pipes[i].pso;
    std::fprintf(stderr, "D3D12 missing pipeline %s\n", name);
    std::abort();
}

bool useCuda() {
    static int on = -1;
    if (on < 0) on = std::getenv("PD_CUBLAS") != nullptr ? 1 : 0;
    return on == 1;
}

// A chunk is one D3D resource. Copying into one offset while CUDA still
// writes another offset of that resource races, so any upload waits until
// the CUDA stream is idle, then the CPU waits for that copy.
bool settleD3D() {
    Ctx& c = ctx();
    if (c.pending.empty() && !c.recording) return !deviceLost();
    if (cudaBridgeHasWork()) {
        if (!cudaBridgeSync()) {
            std::fprintf(stderr, "CUDA sync before D3D copy failed: %s\n", cudaBridgeError());
            noteLost(E_FAIL);
            return false;
        }
    }
    // Keep the CUDA mapping. Destroying it on every copy leaked device VA on
    // this driver. The stream is idle, and kick() waits for the copy, so the
    // next launch sees the uploaded bytes.
    flushPending();
    if (deviceLost()) return false;
    if (c.recording) kick(false);
    return !deviceLost();
}

bool queueKernel(const char* name, const View* bufs, int nbuf, const void* cst,
                 size_t cbytes, unsigned gx, unsigned gy) {
    if (cbytes > sizeof(CudaOp::cst)) {
        noteLost(E_FAIL);
        return false;
    }
    CudaOp op;
    op.gemm = false;
    std::snprintf(op.name, sizeof(op.name), "%s", name ? name : "");
    op.nbuf = nbuf < 8 ? nbuf : 8;
    for (int i = 0; i < op.nbuf; ++i) {
        if (!bufs[i].res || !bufs[i].res->p) {
            noteLost(E_FAIL);
            return false;
        }
        op.res[i] = bufs[i].res->p;
        op.off[i] = bufs[i].off;
    }
    op.gx = std::max(gx, 1u);
    op.gy = std::max(gy, 1u);
    op.cbytes = cbytes;
    if (cst && cbytes) std::memcpy(op.cst, cst, cbytes);
    ctx().cudaOps.push_back(op);
    return true;
}

bool queueGemm(const View& a, const View& b, const View& c, char transA, char transB,
               int M, int N, int K, int lda, int ldb, int ldc) {
    if (!a.res || !a.res->p || !b.res || !b.res->p || !c.res || !c.res->p) {
        noteLost(E_FAIL);
        return false;
    }
    CudaOp op;
    op.gemm = true;
    op.res[0] = a.res->p;
    op.res[1] = b.res->p;
    op.res[2] = c.res->p;
    op.off[0] = a.off;
    op.off[1] = b.off;
    op.off[2] = c.off;
    op.transA = transA;
    op.transB = transB;
    op.M = M;
    op.N = N;
    op.K = K;
    op.lda = lda;
    op.ldb = ldb;
    op.ldc = ldc;
    ctx().cudaOps.push_back(op);
    return true;
}

// Copies land first, as one D3D packet. Every queued GEMM and elementwise
// kernel then runs back to back on the CUDA stream. The CPU waits for that
// stream once, at readback.
bool replayCuda() {
    Ctx& c = ctx();
    if (c.cudaOps.empty() && c.pending.empty() && !c.recording) return !deviceLost();
    if (cudaBridgeHasWork() && !cudaBridgeSync()) {
        std::fprintf(stderr, "CUDA sync before replay failed: %s\n", cudaBridgeError());
        noteLost(E_FAIL);
        c.cudaOps.clear();
        return false;
    }
    if (!c.pending.empty() || c.recording) {
        flushPending();
        if (deviceLost()) {
            c.cudaOps.clear();
            return false;
        }
        if (c.recording) kick(false);
        if (deviceLost()) {
            c.cudaOps.clear();
            return false;
        }
    }
    auto launchT0 = std::chrono::steady_clock::now();
    for (CudaOp& op : c.cudaOps) {
        bool ok = op.gemm
                      ? cudaBridgeGemm(g_device, op.res[0], op.off[0], op.res[1], op.off[1],
                                       op.res[2], op.off[2], op.transA, op.transB, op.M, op.N,
                                       op.K, op.lda, op.ldb, op.ldc)
                      : cudaBridgeKernel(g_device, op.name, op.res, op.off, op.nbuf, op.cst,
                                         op.cbytes, op.gx, op.gy);
        if (!ok) {
            std::fprintf(stderr, "CUDA replay %s failed: %s\n",
                         op.gemm ? "gemm" : op.name, cudaBridgeError());
            noteLost(E_FAIL);
            c.cudaOps.clear();
            return false;
        }
        ++c.nCommit;
    }
    c.encSec += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - launchT0)
                    .count();
    c.cudaOps.clear();
    return !deviceLost();
}

void dispatch(const char* name, const View* bufs, int nbuf, const void* cst,
              size_t cbytes, UINT gx, UINT gy) {
    if (deviceLost()) return;
    for (int i = 0; i < nbuf; ++i) {
        if (!bufs[i].res || !bufs[i].res->p) {
            noteLost(E_FAIL);
            return;
        }
    }
    if (useCuda()) {
        if (std::strncmp(name, "gemm_", 5) == 0) {
            std::fprintf(stderr, "internal: %s reached the D3D dispatch under CUDA\n", name);
            noteLost(E_FAIL);
            return;
        }
        queueKernel(name, bufs, nbuf, cst, cbytes, gx, gy);
        return;
    }
    auto t0 = std::chrono::steady_clock::now();
    flushPending();
    if (deviceLost()) return;
    if (!ensureRecording()) {
        noteLost(E_FAIL);
        return;
    }
    Ctx& c = ctx();
    for (int i = 0; i < nbuf; ++i)
        transition(bufs[i].res, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    // Metal orders every kernel on the command buffer. D3D12 does not: two
    // dispatches that touch the same UAV race unless a UAV barrier separates
    // them. A null barrier waits for every earlier UAV access.
    D3D12_RESOURCE_BARRIER uav = {};
    uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    c.list->ResourceBarrier(1, &uav);
    c.list->SetPipelineState(findPso(name));
    for (int i = 0; i < nbuf; ++i)
        c.list->SetComputeRootUnorderedAccessView((UINT)i, bufs[i].va());
    if (cbytes)
        c.list->SetComputeRoot32BitConstants(7, (UINT)(cbytes / 4), cst, 0);
    c.list->Dispatch(std::max(gx, 1u), std::max(gy, 1u), 1);
    remember(name, gx, gy, cst, cbytes);
    g_last.name = name;
    g_last.gx = gx;
    g_last.gy = gy;
    g_last.nconst = 0;
    if (cst && cbytes >= sizeof(int)) {
        int n = (int)std::min(cbytes / sizeof(int), sizeof(g_last.cst) / sizeof(int));
        const int* p = static_cast<const int*>(cst);
        for (int i = 0; i < n; ++i) g_last.cst[i] = p[i];
        g_last.nconst = n;
    }
    g_last.nbuf = std::min(nbuf, 8);
    for (int i = 0; i < g_last.nbuf; ++i) {
        BufInfo& b = g_last.buf[i];
        b.off = bufs[i].off;
        b.len = bufs[i].len;
        b.cap = bufs[i].res ? bufs[i].res->cap : 0;
        b.va = bufs[i].res && bufs[i].res->p
                   ? (unsigned long long)bufs[i].va()
                   : 0;
        if (bufs[i].len == 0 || !bufs[i].res) continue;
        if (bufs[i].off + bufs[i].len > bufs[i].res->cap) {
            std::fprintf(stderr,
                         "D3D buffer past resource %s buf %d off %zu len %zu cap %zu\n",
                         name, i, bufs[i].off, bufs[i].len, bufs[i].res->cap);
            std::fflush(stderr);
        }
    }
    ++c.nCommit;
    if (++c.sinceKick >= kickEvery()) kick(true);
    c.encSec += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t0)
                    .count();
}

void launch1(const char* name, const std::vector<View>& bufs, const void* cst,
             size_t cbytes, int threads) {
    UINT groups = (UINT)((std::max(threads, 0) + 63) / 64);
    dispatch(name, bufs.data(), (int)bufs.size(), cst, cbytes, groups, 1);
}

void launch2(const char* name, const std::vector<View>& bufs, const void* cst,
             size_t cbytes, int B, int H) {
    UINT gx = (UINT)((std::max(B, 0) + 7) / 8);
    UINT gy = (UINT)((std::max(H, 0) + 7) / 8);
    dispatch(name, bufs.data(), (int)bufs.size(), cst, cbytes, gx, gy);
}

View rIn(const Mat& m) { return kerIn(m.data(), size_t(m.r) * m.s * 4); }
View rOut(Mat& m) { return kerOut(m.data(), size_t(m.r) * m.s * 4); }
View rOutSeed(Mat& m) { return kerOutSeed(m.data(), size_t(m.r) * m.s * 4); }
View rScalar(const std::vector<float>& v) {
    size_t bytes = (v.size() * 4 + 15) & ~size_t(15);
    return kerIn(v.data(), bytes);
}

Slot* createSlot(void** owner, const void* host, size_t bytes) {
    if (deviceLost()) return nullptr;
    auto* s = new Slot();
    s->owner = owner;
    s->bytes = bytes;
    s->gpu.cap = align256(std::max(bytes, size_t(256)));
    s->gpu.state = D3D12_RESOURCE_STATE_COMMON;
    s->gpu.p = createBuffer(s->gpu.cap, D3D12_HEAP_TYPE_DEFAULT,
                            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                            D3D12_RESOURCE_STATE_COMMON);
    if (!s->gpu.p) {
        delete s;
        return nullptr;
    }
    s->home = &ctx().liveSlots;
    s->homeMu = &ctx().slotMu;
    {
        std::lock_guard<std::mutex> lk(ctx().slotMu);
        ctx().liveSlots.push_back(s);
    }
    s->shadow.assign(bytes, 0);
    if (bytes && host) {
        UpAlloc u = allocUpload(bytes);
        if (!u.ptr) {
            releaseSlot(s);
            return nullptr;
        }
        std::memcpy(u.ptr, host, bytes);
        ctx().pending.push_back({&s->gpu, 0, u.res, u.off, bytes});
        ctx().stageBytes += (long long)bytes;
    }
    return s;
}

// Weight and grad resources outlive the learner thread that created them.
// The next thread must track them so buffer-state decay stays accurate.
void adoptSlot(Slot* s) {
    if (!s) return;
    Ctx& c = ctx();
    auto& live = c.liveSlots;
    if (s->home == &live) {
        std::lock_guard<std::mutex> lk(c.slotMu);
        if (std::find(live.begin(), live.end(), s) == live.end()) live.push_back(s);
        return;
    }
    std::mutex* oldMu = s->homeMu;
    std::vector<Slot*>* old = s->home;
    if (reinterpret_cast<uintptr_t>(oldMu) < reinterpret_cast<uintptr_t>(&c.slotMu)) {
        eraseSlotLocked(old, oldMu, s);
        std::lock_guard<std::mutex> lk(c.slotMu);
        s->home = &live;
        s->homeMu = &c.slotMu;
        s->gpu.state = D3D12_RESOURCE_STATE_COMMON;
        if (std::find(live.begin(), live.end(), s) == live.end()) live.push_back(s);
    } else {
        std::lock_guard<std::mutex> lk(c.slotMu);
        eraseSlotLocked(old, oldMu, s);
        s->home = &live;
        s->homeMu = &c.slotMu;
        s->gpu.state = D3D12_RESOURCE_STATE_COMMON;
        if (std::find(live.begin(), live.end(), s) == live.end()) live.push_back(s);
    }
}

void markDirty(Slot* s) {
    if (!s || s->dirty) return;
    adoptSlot(s);
    s->dirty = true;
    ctx().dirtyGrads.push_back(s);
}

bool ensureReadback(size_t need) {
    Ctx& c = ctx();
    if (c.readback && c.rbCap >= need) return true;
    if (c.readback) {
        c.readback->Unmap(0, nullptr);
        c.readback->Release();
        c.readback = nullptr;
        c.rbMap = nullptr;
    }
    size_t cap = align256(std::max(need, size_t(8u << 20)));
    c.readback = createBuffer(cap, D3D12_HEAP_TYPE_READBACK,
                              D3D12_RESOURCE_FLAG_NONE,
                              D3D12_RESOURCE_STATE_COPY_DEST);
    if (!c.readback) return false;
    c.readback->Map(0, nullptr, (void**)&c.rbMap);
    c.rbCap = cap;
    return true;
}

// ---------------------------------------------------------------------------
// HLSL. One compilation unit per entry so groupshared stays per kernel.
// Layout of every cbuffer matches the host struct passed as root constants.
// ---------------------------------------------------------------------------
const char* kPrelude = R"HLSL(
float ld(RWByteAddressBuffer b, uint i) { return asfloat(b.Load(i * 4u)); }
void st(RWByteAddressBuffer b, uint i, float v) { b.Store(i * 4u, asuint(v)); }
)HLSL";

struct ShaderSrc { const char* name; const char* body; };

const ShaderSrc kShaders[] = {
{"add_bias", R"HLSL(
cbuffer CB : register(b0) { int a, b, c; };
RWByteAddressBuffer Y : register(u0);
RWByteAddressBuffer BI : register(u1);
[numthreads(64,1,1)]
void add_bias(uint3 gtid : SV_DispatchThreadID) {
    uint gid = gtid.x;
    if (gid >= (uint)b) return;
    uint i = gid / (uint)a;
    uint j = gid - i * (uint)a;
    uint yi = i * (uint)c + j;
    st(Y, yi, ld(Y, yi) + ld(BI, j));
}
)HLSL"},
{"relu_fwd", R"HLSL(
cbuffer CB : register(b0) { int a, b, c; };
RWByteAddressBuffer X : register(u0);
[numthreads(64,1,1)]
void relu_fwd(uint3 gtid : SV_DispatchThreadID) {
    uint gid = gtid.x;
    if (gid >= (uint)b) return;
    uint i = gid / (uint)a;
    uint j = gid - i * (uint)a;
    uint xi = i * (uint)c + j;
    float v = ld(X, xi);
    st(X, xi, v > 0.0f ? v : 0.0f);
}
)HLSL"},
{"relu_bwd", R"HLSL(
cbuffer CB : register(b0) { int a, b, c, d; };
RWByteAddressBuffer Pre : register(u0);
RWByteAddressBuffer Gout : register(u1);
RWByteAddressBuffer Gin : register(u2);
[numthreads(64,1,1)]
void relu_bwd(uint3 gtid : SV_DispatchThreadID) {
    uint gid = gtid.x;
    if (gid >= (uint)b) return;
    uint i = gid / (uint)a;
    uint j = gid - i * (uint)a;
    float pre = ld(Pre, i * (uint)c + j);
    float g = ld(Gout, i * (uint)d + j);
    st(Gin, i * (uint)d + j, pre > 0.0f ? g : 0.0f);
}
)HLSL"},
{"gate_add", R"HLSL(
cbuffer CB : register(b0) { int a, b; };
RWByteAddressBuffer GP : register(u0);
RWByteAddressBuffer GI : register(u1);
RWByteAddressBuffer GH : register(u2);
RWByteAddressBuffer BI : register(u3);
RWByteAddressBuffer BH : register(u4);
[numthreads(64,1,1)]
void gate_add(uint3 gtid : SV_DispatchThreadID) {
    uint gid = gtid.x;
    if (gid >= (uint)b * (uint)a) return;
    uint q = gid - (gid / (uint)a) * (uint)a;
    st(GP, gid, (ld(GI, gid) + ld(GH, gid)) + ld(BI, q) + ld(BH, q));
}
)HLSL"},
{"lstm_cell_fwd", R"HLSL(
cbuffer CB : register(b0) { int h, B, hasCp; };
RWByteAddressBuffer GP : register(u0);
RWByteAddressBuffer CP : register(u1);
RWByteAddressBuffer Hout : register(u2);
RWByteAddressBuffer Cout : register(u3);
[numthreads(8,8,1)]
void lstm_cell_fwd(uint3 gtid : SV_DispatchThreadID) {
    int b = (int)gtid.x;
    int u = (int)gtid.y;
    if (b >= B || u >= h) return;
    uint base = (uint)b * (uint)(4 * h);
    float iv = 1.0f / (1.0f + exp(-ld(GP, base + (uint)u)));
    float fv = 1.0f / (1.0f + exp(-ld(GP, base + (uint)h + (uint)u)));
    float gz = tanh(ld(GP, base + (uint)(2 * h) + (uint)u));
    float ov = 1.0f / (1.0f + exp(-ld(GP, base + (uint)(3 * h) + (uint)u)));
    float pc = hasCp != 0 ? ld(CP, (uint)b * (uint)h + (uint)u) : 0.0f;
    float cc = fv * pc + iv * gz;
    st(Cout, (uint)b * (uint)h + (uint)u, cc);
    st(Hout, (uint)b * (uint)h + (uint)u, ov * tanh(cc));
}
)HLSL"},
{"lstm_cell_bwd", R"HLSL(
cbuffer CB : register(b0) { int h, B, hasCp; };
RWByteAddressBuffer GH : register(u0);
RWByteAddressBuffer GP : register(u1);
RWByteAddressBuffer CN : register(u2);
RWByteAddressBuffer CP : register(u3);
RWByteAddressBuffer DH : register(u4);
RWByteAddressBuffer DC : register(u5);
RWByteAddressBuffer DG : register(u6);
[numthreads(8,8,1)]
void lstm_cell_bwd(uint3 gtid : SV_DispatchThreadID) {
    int b = (int)gtid.x, u = (int)gtid.y, hh = h;
    if (b >= B || u >= hh) return;
    uint uh = (uint)hh, uu = (uint)u, ub = (uint)b;
    float tc = tanh(ld(CN, ub * uh + uu));
    uint base = ub * (4u * uh);
    float iv = 1.0f / (1.0f + exp(-ld(GP, base + uu)));
    float fv = 1.0f / (1.0f + exp(-ld(GP, base + uh + uu)));
    float gz = tanh(ld(GP, base + 2u * uh + uu));
    float ov = 1.0f / (1.0f + exp(-ld(GP, base + 3u * uh + uu)));
    float dhn = ld(GH, ub * uh + uu) + ld(DH, ub * uh + uu);
    float pc = hasCp != 0 ? ld(CP, ub * uh + uu) : 0.0f;
    float dct = dhn * ov * (1.0f - tc * tc) + ld(DC, ub * uh + uu);
    st(DG, base + uu, dct * gz * iv * (1.0f - iv));
    st(DG, base + uh + uu, dct * pc * fv * (1.0f - fv));
    st(DG, base + 2u * uh + uu, dct * iv * (1.0f - gz * gz));
    st(DG, base + 3u * uh + uu, dhn * tc * ov * (1.0f - ov));
    st(DC, ub * uh + uu, dct * fv);
}
)HLSL"},
{"concat2", R"HLSL(
cbuffer CB : register(b0) { int n1, n2, rows, zs, as, bs; };
RWByteAddressBuffer Z : register(u0);
RWByteAddressBuffer A : register(u1);
RWByteAddressBuffer B : register(u2);
[numthreads(64,1,1)]
void concat2(uint3 gtid : SV_DispatchThreadID) {
    uint cols = (uint)n1 + (uint)n2;
    uint gid = gtid.x;
    if (gid >= (uint)rows * cols) return;
    uint i = gid / cols;
    uint j = gid - i * cols;
    float v = (j < (uint)n1) ? ld(A, i * (uint)as + j)
                             : ld(B, i * (uint)bs + (j - (uint)n1));
    st(Z, i * (uint)zs + j, v);
}
)HLSL"},
{"split2", R"HLSL(
cbuffer CB : register(b0) { int n1, n2, rows, zs, as, bs; };
RWByteAddressBuffer Z : register(u0);
RWByteAddressBuffer A : register(u1);
RWByteAddressBuffer B : register(u2);
[numthreads(64,1,1)]
void split2(uint3 gtid : SV_DispatchThreadID) {
    uint cols = (uint)n1 + (uint)n2;
    uint gid = gtid.x;
    if (gid >= (uint)rows * cols) return;
    uint i = gid / cols;
    uint j = gid - i * cols;
    float v = ld(Z, i * (uint)zs + j);
    if (j < (uint)n1) st(A, i * (uint)as + j, v);
    else st(B, i * (uint)bs + (j - (uint)n1), v);
}
)HLSL"},
{"zero_and_last", R"HLSL(
cbuffer CB : register(b0) { int B, T, h; };
RWByteAddressBuffer All : register(u0);
RWByteAddressBuffer GH : register(u1);
[numthreads(64,1,1)]
void zero_and_last(uint3 gtid : SV_DispatchThreadID) {
    uint n = (uint)B * (uint)T * (uint)h;
    uint gid = gtid.x;
    if (gid >= n) return;
    uint span = (uint)B * (uint)h;
    uint t = gid / span;
    uint rem = gid - t * span;
    st(All, gid, (t + 1u == (uint)T) ? ld(GH, rem) : 0.0f);
}
)HLSL"},
{"mask_dyn", R"HLSL(
cbuffer CB : register(b0) { int N, B, ls, ms; };
RWByteAddressBuffer Logits : register(u0);
RWByteAddressBuffer DS : register(u1);
RWByteAddressBuffer Mask : register(u2);
[numthreads(64,1,1)]
void mask_dyn(uint3 gtid : SV_DispatchThreadID) {
    uint gid = gtid.x;
    if (gid >= (uint)B * (uint)N) return;
    uint i = gid / (uint)N;
    uint a = gid - i * (uint)N;
    uint li = i * (uint)ls + a;
    if (ld(Mask, i * (uint)ms + a) > 0.5f)
        st(Logits, li, ld(Logits, li) + ld(DS, (i * (uint)N + a) * 4u));
    else
        st(Logits, li, -1e9f);
}
)HLSL"},
{"add_to", R"HLSL(
cbuffer CB : register(b0) { int rows, cols, dc, sc; };
RWByteAddressBuffer Dst : register(u0);
RWByteAddressBuffer Src : register(u1);
[numthreads(64,1,1)]
void add_to(uint3 gtid : SV_DispatchThreadID) {
    uint gid = gtid.x;
    if (gid >= (uint)rows * (uint)cols) return;
    uint i = gid / (uint)cols;
    uint j = gid - i * (uint)cols;
    uint di = i * (uint)dc + j;
    st(Dst, di, ld(Dst, di) + ld(Src, i * (uint)sc + j));
}
)HLSL"},
{"bias_grad_add", R"HLSL(
cbuffer CB : register(b0) { int B, o, gs; };
RWByteAddressBuffer G : register(u0);
RWByteAddressBuffer DB : register(u1);
[numthreads(64,1,1)]
void bias_grad_add(uint3 gtid : SV_DispatchThreadID) {
    uint gid = gtid.x;
    if (gid >= (uint)o) return;
    float s = 0.0f;
    for (int i = 0; i < B; ++i) s += ld(G, (uint)i * (uint)gs + gid);
    st(DB, gid, ld(DB, gid) + s);
}
)HLSL"},
{"slice_cols", R"HLSL(
cbuffer CB : register(b0) { int B, h, off, ds, ss; };
RWByteAddressBuffer Dst : register(u0);
RWByteAddressBuffer Src : register(u1);
[numthreads(64,1,1)]
void slice_cols(uint3 gtid : SV_DispatchThreadID) {
    uint gid = gtid.x;
    if (gid >= (uint)B * (uint)h) return;
    uint i = gid / (uint)h;
    uint j = gid - i * (uint)h;
    st(Dst, i * (uint)ds + j, ld(Src, i * (uint)ss + (uint)off + j));
}
)HLSL"},
{"flatten_rows", R"HLSL(
cbuffer CB : register(b0) { int B, N, ss; };
RWByteAddressBuffer Dst : register(u0);
RWByteAddressBuffer Src : register(u1);
[numthreads(64,1,1)]
void flatten_rows(uint3 gtid : SV_DispatchThreadID) {
    uint gid = gtid.x;
    if (gid >= (uint)B * (uint)N) return;
    uint i = gid / (uint)N;
    uint a = gid - i * (uint)N;
    st(Dst, gid * 4u, ld(Src, i * (uint)ss + a));
}
)HLSL"},
{"copy_mat", R"HLSL(
cbuffer CB : register(b0) { int rows, cols, ds, ss; };
RWByteAddressBuffer Dst : register(u0);
RWByteAddressBuffer Src : register(u1);
[numthreads(64,1,1)]
void copy_mat(uint3 gtid : SV_DispatchThreadID) {
    uint gid = gtid.x;
    if (gid >= (uint)rows * (uint)cols) return;
    uint i = gid / (uint)cols;
    uint j = gid - i * (uint)cols;
    st(Dst, i * (uint)ds + j, ld(Src, i * (uint)ss + j));
}
)HLSL"},
{"adam_step", R"HLSL(
cbuffer CB : register(b0) { int n; float lr, b1, b2, eps, bc1, bc2; };
RWByteAddressBuffer W : register(u0);
RWByteAddressBuffer DW : register(u1);
RWByteAddressBuffer M : register(u2);
RWByteAddressBuffer V : register(u3);
[numthreads(128,1,1)]
void adam_step(uint3 gtid : SV_DispatchThreadID) {
    uint gid = gtid.x;
    if (gid >= (uint)n) return;
    float g = ld(DW, gid);
    float mm = b1 * ld(M, gid) + (1.0f - b1) * g;
    float vv = b2 * ld(V, gid) + (1.0f - b2) * g * g;
    st(M, gid, mm);
    st(V, gid, vv);
    st(W, gid, ld(W, gid) - lr * (mm / bc1) / (sqrt(vv / bc2) + eps));
}
)HLSL"},
{"gemm_tiled", R"HLSL(
// One thread per output. A shared-memory tile plus GroupMemoryBarrier hangs
// on this driver when the last row tile is only partly filled: threads
// outside the tile skip the barrier. These GEMMs are the M<192, N<512 set.
cbuffer CB : register(b0) { int M, N, K, lda, ldb, ldc, tA, tB, epi; };
RWByteAddressBuffer A : register(u0);
RWByteAddressBuffer B : register(u1);
RWByteAddressBuffer C : register(u2);
RWByteAddressBuffer Bias : register(u3);
[numthreads(16,16,1)]
void gemm_tiled(uint3 gt : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    int row = int(gid.y * 16u + gt.y);
    int col = int(gid.x * 16u + gt.x);
    if (row >= M || col >= N) return;
    precise float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        float av = (tA == 0) ? ld(A, (uint)row * (uint)lda + (uint)k)
                             : ld(A, (uint)k * (uint)lda + (uint)row);
        float bv = (tB == 0) ? ld(B, (uint)k * (uint)ldb + (uint)col)
                             : ld(B, (uint)col * (uint)ldb + (uint)k);
        acc += av * bv;
    }
    if (epi == 1) {
        acc += ld(Bias, (uint)col);
        acc = acc > 0.0f ? acc : 0.0f;
    }
    st(C, (uint)row * (uint)ldc + (uint)col, acc);
}
)HLSL"},
{"gemm_block", R"HLSL(
groupshared float As[32][17];
groupshared float Bs[16][33];
cbuffer CB : register(b0) { int M, N, K, lda, ldb, ldc, tA, tB, epi; };
RWByteAddressBuffer A : register(u0);
RWByteAddressBuffer B : register(u1);
RWByteAddressBuffer C : register(u2);
RWByteAddressBuffer Bias : register(u3);
[numthreads(8,8,1)]
void gemm_block(uint3 gt : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    precise float acc[4][4];
    [unroll] for (int a = 0; a < 4; ++a)
        [unroll] for (int b = 0; b < 4; ++b) acc[a][b] = 0.0f;
    uint i0 = gid.y * 32u;
    uint j0 = gid.x * 32u;
    uint tr = gt.y, tc = gt.x;
    uint tid = tr * 8u + tc;
    for (int k0 = 0; k0 < K; k0 += 16) {
        [unroll]
        for (uint q = 0u; q < 8u; ++q) {
            uint idx = tid + 64u * q;
            uint r = idx / 16u, c = idx % 16u;
            uint ar = i0 + r, ak = (uint)k0 + c;
            float v = 0.0f;
            if (ar < (uint)M && ak < (uint)K)
                v = (tA == 0) ? ld(A, ar * (uint)lda + ak)
                              : ld(A, ak * (uint)lda + ar);
            As[r][c] = v;
        }
        [unroll]
        for (uint q = 0u; q < 8u; ++q) {
            uint idx = tid + 64u * q;
            uint r = idx / 32u, c = idx % 32u;
            uint bk = (uint)k0 + r, bj = j0 + c;
            float v = 0.0f;
            if (bk < (uint)K && bj < (uint)N)
                v = (tB == 0) ? ld(B, bk * (uint)ldb + bj)
                              : ld(B, bj * (uint)ldb + bk);
            Bs[r][c] = v;
        }
        GroupMemoryBarrierWithGroupSync();
        [unroll]
        for (int z = 0; z < 16; ++z) {
            float av[4], bv[4];
            [unroll] for (int a = 0; a < 4; ++a) av[a] = As[tr * 4u + (uint)a][z];
            [unroll] for (int b = 0; b < 4; ++b) bv[b] = Bs[z][tc * 4u + (uint)b];
            [unroll] for (int a = 0; a < 4; ++a)
                [unroll] for (int b = 0; b < 4; ++b) acc[a][b] += av[a] * bv[b];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    [unroll]
    for (int a = 0; a < 4; ++a) {
        uint row = i0 + tr * 4u + (uint)a;
        if (row >= (uint)M) continue;
        uint col = j0 + tc * 4u;
        uint base = row * (uint)ldc + col;
        if (epi == 0) {
            [unroll] for (int b = 0; b < 4; ++b)
                if (col + (uint)b < (uint)N) st(C, base + (uint)b, acc[a][b]);
        } else {
            [unroll] for (int b = 0; b < 4; ++b)
                if (col + (uint)b < (uint)N) {
                    float v = acc[a][b] + ld(Bias, col + (uint)b);
                    st(C, base + (uint)b, v > 0.0f ? v : 0.0f);
                }
        }
    }
}
)HLSL"},
{"gemm_blockn", R"HLSL(
groupshared float As[16][17];
groupshared float Bs[16][33];
cbuffer CB : register(b0) { int M, N, K, lda, ldb, ldc, tA, tB, epi; };
RWByteAddressBuffer A : register(u0);
RWByteAddressBuffer B : register(u1);
RWByteAddressBuffer C : register(u2);
RWByteAddressBuffer Bias : register(u3);
[numthreads(8,8,1)]
void gemm_blockn(uint3 gt : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    precise float acc[2][4];
    [unroll] for (int a = 0; a < 2; ++a)
        [unroll] for (int b = 0; b < 4; ++b) acc[a][b] = 0.0f;
    uint i0 = gid.y * 16u;
    uint j0 = gid.x * 32u;
    uint tr = gt.y, tc = gt.x;
    uint tid = tr * 8u + tc;
    for (int k0 = 0; k0 < K; k0 += 16) {
        [unroll]
        for (uint q = 0u; q < 4u; ++q) {
            uint idx = tid + 64u * q;
            uint r = idx / 16u, c = idx % 16u;
            uint ar = i0 + r, ak = (uint)k0 + c;
            float v = 0.0f;
            if (ar < (uint)M && ak < (uint)K)
                v = (tA == 0) ? ld(A, ar * (uint)lda + ak)
                              : ld(A, ak * (uint)lda + ar);
            As[r][c] = v;
        }
        [unroll]
        for (uint q = 0u; q < 8u; ++q) {
            uint idx = tid + 64u * q;
            uint r = idx / 32u, c = idx % 32u;
            uint bk = (uint)k0 + r, bj = j0 + c;
            float v = 0.0f;
            if (bk < (uint)K && bj < (uint)N)
                v = (tB == 0) ? ld(B, bk * (uint)ldb + bj)
                              : ld(B, bj * (uint)ldb + bk);
            Bs[r][c] = v;
        }
        GroupMemoryBarrierWithGroupSync();
        [unroll]
        for (int z = 0; z < 16; ++z) {
            float av[2], bv[4];
            [unroll] for (int a = 0; a < 2; ++a) av[a] = As[tr * 2u + (uint)a][z];
            [unroll] for (int b = 0; b < 4; ++b) bv[b] = Bs[z][tc * 4u + (uint)b];
            [unroll] for (int a = 0; a < 2; ++a)
                [unroll] for (int b = 0; b < 4; ++b) acc[a][b] += av[a] * bv[b];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    [unroll]
    for (int a = 0; a < 2; ++a) {
        uint row = i0 + tr * 2u + (uint)a;
        if (row >= (uint)M) continue;
        uint col = j0 + tc * 4u;
        uint base = row * (uint)ldc + col;
        if (epi == 0) {
            [unroll] for (int b = 0; b < 4; ++b)
                if (col + (uint)b < (uint)N) st(C, base + (uint)b, acc[a][b]);
        } else {
            [unroll] for (int b = 0; b < 4; ++b)
                if (col + (uint)b < (uint)N) {
                    float v = acc[a][b] + ld(Bias, col + (uint)b);
                    st(C, base + (uint)b, v > 0.0f ? v : 0.0f);
                }
        }
    }
}
)HLSL"},
};

bool compileShaders() {
    UINT flags = D3DCOMPILE_OPTIMIZATION_LEVEL3 | D3DCOMPILE_IEEE_STRICTNESS;
    for (const ShaderSrc& s : kShaders) {
        std::string src = std::string(kPrelude) + s.body;
        ID3DBlob* blob = nullptr;
        ID3DBlob* err = nullptr;
        HRESULT hr = D3DCompile(src.data(), src.size(), s.name, nullptr, nullptr,
                                s.name, "cs_5_0", flags, 0, &blob, &err);
        if (FAILED(hr)) {
            std::fprintf(stderr, "D3DCompile %s failed:\n%s\n", s.name,
                         err ? (const char*)err->GetBufferPointer() : "(no log)");
            if (err) err->Release();
            if (blob) blob->Release();
            return false;
        }
        if (err) err->Release();
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
        desc.pRootSignature = g_root;
        desc.CS.pShaderBytecode = blob->GetBufferPointer();
        desc.CS.BytecodeLength = blob->GetBufferSize();
        ID3D12PipelineState* pso = nullptr;
        hr = g_device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso));
        blob->Release();
        if (FAILED(hr)) {
            std::fprintf(stderr, "D3D12 PSO %s failed: 0x%08lx\n", s.name,
                         (unsigned long)hr);
            return false;
        }
        g_pipes[g_npipes++] = {s.name, pso};
    }
    return true;
}

bool createRoot() {
    D3D12_ROOT_PARAMETER params[8] = {};
    for (UINT i = 0; i < 7; ++i) {
        params[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        params[i].Descriptor.ShaderRegister = i;
        params[i].Descriptor.RegisterSpace = 0;
        params[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }
    params[7].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[7].Constants.ShaderRegister = 0;
    params[7].Constants.RegisterSpace = 0;
    params[7].Constants.Num32BitValues = 16;
    params[7].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC desc = {};
    desc.NumParameters = 8;
    desc.pParameters = params;
    desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
    ID3DBlob* blob = nullptr;
    ID3DBlob* err = nullptr;
    HRESULT hr = D3D12SerializeRootSignature(
        &desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err);
    if (FAILED(hr)) {
        std::fprintf(stderr, "D3D12SerializeRootSignature: %s\n",
                     err ? (const char*)err->GetBufferPointer() : "?");
        if (err) err->Release();
        return false;
    }
    hr = g_device->CreateRootSignature(0, blob->GetBufferPointer(),
                                       blob->GetBufferSize(),
                                       IID_PPV_ARGS(&g_root));
    blob->Release();
    if (err) err->Release();
    return SUCCEEDED(hr);
}

}  // namespace

bool d3dInit() {
    if (g_initTried) return g_initOk;
    g_initTried = true;
    IDXGIFactory1* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return false;
    IDXGIAdapter1* best = nullptr;
    SIZE_T bestMem = 0;
    bool have = false;
    for (UINT i = 0;; ++i) {
        IDXGIAdapter1* ad = nullptr;
        if (factory->EnumAdapters1(i, &ad) == DXGI_ERROR_NOT_FOUND) break;
        DXGI_ADAPTER_DESC1 desc = {};
        ad->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
            ad->Release();
            continue;
        }
        if (!have || desc.DedicatedVideoMemory > bestMem) {
            if (best) best->Release();
            best = ad;
            bestMem = desc.DedicatedVideoMemory;
            have = true;
        } else {
            ad->Release();
        }
    }
    if (!best) {
        factory->Release();
        return false;
    }
    HRESULT hr = D3D12CreateDevice(best, D3D_FEATURE_LEVEL_11_0,
                                   IID_PPV_ARGS(&g_device));
    if (FAILED(hr)) {
        best->Release();
        factory->Release();
        return false;
    }
    DXGI_ADAPTER_DESC1 desc = {};
    best->GetDesc1(&desc);
    char name[160] = {};
    WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, name, (int)sizeof(name),
                        nullptr, nullptr);
    std::snprintf(g_label, sizeof(g_label), "GPU (D3D12: %s)", name);
    best->Release();
    factory->Release();
    g_dummy.cap = 256;
    g_dummy.state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    g_dummy.sticky = true;
    g_dummy.p = createBuffer(256, D3D12_HEAP_TYPE_DEFAULT,
                             D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                             D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (!g_dummy.p || !createRoot() || !compileShaders()) {
        g_initOk = false;
        return false;
    }
    g_initOk = true;
    return true;
}

bool d3dAvailable() { return d3dInit(); }
const char* d3dLabel() { return g_label; }

bool d3dDeviceOk() { return !g_initOk || !deviceLost(); }

void d3dReleaseThread() { teardown(ctx()); }

bool d3dRecreate() {
    cudaBridgeReset();
    for (int i = 0; i < g_npipes; ++i) {
        if (g_pipes[i].pso) g_pipes[i].pso->Release();
        g_pipes[i].pso = nullptr;
        g_pipes[i].name = nullptr;
    }
    g_npipes = 0;
    if (g_root) {
        g_root->Release();
        g_root = nullptr;
    }
    if (g_dummy.p) {
        g_dummy.p->Release();
        g_dummy.p = nullptr;
    }
    if (g_device) {
        g_device->Release();
        g_device = nullptr;
    }
    g_initTried = false;
    g_initOk = false;
    g_lost.store(false, std::memory_order_release);
    return d3dInit();
}

void d3dPrintStats(const char* tag) {
    Ctx& c = ctx();
    std::fprintf(stderr,
                 "[d3d %s] CB=%lld enc=%lld waitMs=%.1f encMs=%.1f "
                 "stageMB=%.1f flushMB=%.1f\n",
                 tag, c.nWait, c.nCommit, c.waitSec * 1e3, c.encSec * 1e3,
                 c.stageBytes / 1e6, c.flushBytes / 1e6);
    c.nWait = c.nCommit = 0;
    c.stageBytes = c.flushBytes = 0;
    c.waitSec = c.encSec = 0;
}

void d3dMarkHost(const void* p) {
    Ctx& c = ctx();
    auto it = c.window.find(reinterpret_cast<uintptr_t>(p));
    if (it != c.window.end()) it->second.toHost = true;
}

void d3dStageInput(const void* p, size_t bytes) {
    Ctx& c = ctx();
    if (c.window.find(reinterpret_cast<uintptr_t>(p)) != c.window.end()) return;
    bindFresh(p, bytes, p, false);
}

void cudaReadback(bool keepWindow) {
    Ctx& c = ctx();
    bool hostRead = false;
    for (auto& kv : c.window)
        if (kv.second.toHost) hostRead = true;
    // Dirty gradients stay on the device until d3dFlushGrad. Copying them on
    // every minibatch fence was more than a gigabyte per seat.
    if (c.cudaOps.empty() && !c.recording && c.pending.empty() && !hostRead &&
        !cudaBridgeHasWork())
        return;
    auto t0 = std::chrono::steady_clock::now();
    if (!replayCuda() || !cudaBridgeSync()) {
        if (cudaBridgeHasWork() || !c.cudaOps.empty()) {
            std::fprintf(stderr, "CUDA readback sync failed: %s\n", cudaBridgeError());
            noteLost(E_FAIL);
        }
        c.cudaOps.clear();
        c.pending.clear();
        c.window.clear();
        c.dirtyGrads.clear();
        return;
    }
    auto copyOut = [&](ID3D12Resource* res, size_t off, void* host, size_t bytes) {
        if (!res || !host || bytes == 0) return true;
        if (!cudaBridgeDtoH(g_device, res, off, host, bytes)) {
            std::fprintf(stderr, "CUDA DtoH failed: %s\n", cudaBridgeError());
            noteLost(E_FAIL);
            return false;
        }
        c.flushBytes += (long long)bytes;
        return true;
    };
    for (auto& kv : c.window) {
        Bind& b = kv.second;
        if (!b.toHost) continue;
        if (!b.view.res || !copyOut(b.view.res->p, b.view.off, b.host, b.len)) {
            c.pending.clear();
            c.window.clear();
            c.dirtyGrads.clear();
            return;
        }
    }
    if (!keepWindow) {
        c.window.clear();
        for (Chunk& ch : c.chunks) ch.used = 0;
    }
    for (Upload& u : c.uploads) u.used = 0;
    for (Slot* s : c.graveyard) releaseSlot(s);
    c.graveyard.clear();
    ++c.nWait;
    c.waitSec += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void d3dWait(bool keepWindow) {
    Ctx& c = ctx();
    if (deviceLost()) {
        closeRecording(c);
        c.pending.clear();
        c.window.clear();
        c.dirtyGrads.clear();
        return;
    }
    if (useCuda()) {
        cudaReadback(keepWindow);
        return;
    }
    if (!c.recording && c.pending.empty() && c.dirtyGrads.empty()) return;
    auto t0 = std::chrono::steady_clock::now();
    flushPending();
    if (!ensureRecording() && !c.recording) return;

    struct Job {
        Res* src;
        size_t srcOff;
        size_t dstOff;
        void* host;
        size_t bytes;
        Slot* grad;
    };
    std::vector<Job> jobs;
    size_t need = 0;
    auto pushJob = [&](Res* src, size_t srcOff, void* host, size_t bytes, Slot* g) {
        if (!src || !host || bytes == 0) return;
        need = align256(need) + align256(bytes);
        jobs.push_back({src, srcOff, 0, host, bytes, g});
    };
    for (auto& kv : c.window) {
        Bind& b = kv.second;
        if (b.toHost) pushJob(b.view.res, b.view.off, b.host, b.len, nullptr);
    }
    for (Slot* s : c.dirtyGrads)
        pushJob(&s->gpu, 0, s->shadow.data(), s->bytes, s);
    if (!jobs.empty()) {
        if (!ensureReadback(std::max(need, size_t(256)))) return;
        size_t cursor = 0;
        for (Job& j : jobs) {
            cursor = align256(cursor);
            j.dstOff = cursor;
            cursor += j.bytes;
        }
        for (Job& j : jobs) transition(j.src, D3D12_RESOURCE_STATE_COPY_SOURCE);
        for (Job& j : jobs)
            c.list->CopyBufferRegion(c.readback, j.dstOff, j.src->p, j.srcOff, j.bytes);
        for (Job& j : jobs) transition(j.src, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
    if (c.recording) kick(false);
    if (deviceLost() || !g_device) {
        c.pending.clear();
        c.window.clear();
        c.dirtyGrads.clear();
        for (Chunk& ch : c.chunks) ch.used = 0;
        for (Upload& u : c.uploads) u.used = 0;
        return;
    }
    if (c.fence && c.fenceValue != 0 &&
        c.fence->GetCompletedValue() < c.fenceValue) {
        c.fence->SetEventOnCompletion(c.fenceValue, c.event);
        WaitForSingleObject(c.event, INFINITE);
    }
    HRESULT removed = g_device->GetDeviceRemovedReason();
    if (FAILED(removed)) {
        noteLost(removed);
        c.pending.clear();
        c.window.clear();
        c.dirtyGrads.clear();
        for (Chunk& ch : c.chunks) ch.used = 0;
        for (Upload& u : c.uploads) u.used = 0;
        return;
    }
    for (Job& j : jobs) {
        std::memcpy(j.host, c.rbMap + j.dstOff, j.bytes);
        c.flushBytes += (long long)j.bytes;
        if (j.grad) {
            j.grad->shadowValid = true;
            j.grad->dirty = false;
        }
    }
    c.dirtyGrads.clear();
    if (!keepWindow) {
        c.window.clear();
        for (Chunk& ch : c.chunks) ch.used = 0;
    }
    for (Upload& u : c.uploads) u.used = 0;
    for (Slot* s : c.graveyard) releaseSlot(s);
    c.graveyard.clear();
    c.waitSec += std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - t0)
                     .count();
}

void d3dDropCache(void** slotp) {
    if (!slotp || !*slotp) return;
    Slot* s = static_cast<Slot*>(*slotp);
    *slotp = nullptr;
    adoptSlot(s);
    Ctx& c = ctx();
    auto& d = c.dirtyGrads;
    d.erase(std::remove(d.begin(), d.end(), s), d.end());
    if (c.recording || !c.pending.empty()) c.graveyard.push_back(s);
    else releaseSlot(s);
}

void* d3dWeightCache(void** slot, const void* host, size_t bytes) {
    if (!slot || deviceLost()) return nullptr;
    if (*slot == nullptr) {
        *slot = createSlot(slot, host, bytes);
        if (!*slot) return nullptr;
    } else {
        adoptSlot(static_cast<Slot*>(*slot));
    }
    return *slot;
}

void* d3dGradCache(void** slot, const void* host, size_t bytes) {
    if (!slot || deviceLost()) return nullptr;
    if (*slot == nullptr) {
        *slot = createSlot(slot, host, bytes);
        if (!*slot) return nullptr;
    } else {
        adoptSlot(static_cast<Slot*>(*slot));
    }
    markDirty(static_cast<Slot*>(*slot));
    return *slot;
}

void d3dFlushGrad(void* slot, void* host, size_t bytes) {
    if (!slot || !host || bytes == 0 || deviceLost()) return;
    Slot* s = static_cast<Slot*>(slot);
    adoptSlot(s);
    if (useCuda()) {
        if (!ctx().cudaOps.empty() || !ctx().pending.empty() || ctx().recording ||
            cudaBridgeHasWork())
            d3dWait(true);
        if (deviceLost()) return;
        if (s->dirty || !s->shadowValid) {
            if (!s->gpu.p ||
                !cudaBridgeDtoH(g_device, s->gpu.p, 0, s->shadow.data(), s->bytes)) {
                std::fprintf(stderr, "CUDA grad DtoH failed: %s\n", cudaBridgeError());
                noteLost(E_FAIL);
                return;
            }
            s->shadowValid = true;
            s->dirty = false;
            ctx().flushBytes += (long long)s->bytes;
        }
        size_t n = std::min(bytes, s->shadow.size());
        std::memcpy(host, s->shadow.data(), n);
        return;
    }
    if (s->dirty || !s->shadowValid) d3dWait(true);
    if (deviceLost() || !s->shadowValid) return;
    size_t n = std::min(bytes, s->shadow.size());
    std::memcpy(host, s->shadow.data(), n);
    ctx().flushBytes += (long long)n;
}

struct GParam {
    int M, N, K, lda, ldb, ldc, tA, tB, epi;
};

void d3dCustomGemm(const GemmOp& g, const void* biasV, size_t biasBytes, int epi) {
    const float* bias = static_cast<const float*>(biasV);
    size_t ar = (g.transA == 'N') ? g.M : g.K;
    size_t br = (g.transB == 'N') ? g.K : g.N;
    View rA = kerIn(g.A, ar * g.lda * 4);
    View rB;
    if (g.wSlot) {
        Slot* s = static_cast<Slot*>(d3dWeightCache(g.wSlot, g.B, br * g.ldb * 4));
        if (!s) return;
        rB.res = &s->gpu;
        rB.off = 0;
        rB.len = s->bytes;
    } else {
        rB = kerIn(g.B, br * g.ldb * 4);
    }
    View rC = bindFresh(g.C, size_t(g.M) * g.ldc * 4, nullptr, false);
    View rBias = bias ? kerIn(bias, biasBytes) : rC;
    View bufs[4] = {rA, rB, rC, rBias};
    GParam p{g.M, g.N, g.K, g.lda, g.ldb, g.ldc,
             g.transA == 'T' ? 1 : 0, g.transB == 'T' ? 1 : 0, epi};
    if (useCuda()) {
        if (!queueGemm(rA, rB, rC, g.transA, g.transB, g.M, g.N, g.K, g.lda, g.ldb, g.ldc))
            return;
        if (epi == 1) {
            struct BiasP { int a, b, c; } bp{g.N, g.M * g.N, g.ldc};
            View add[2] = {rC, rBias};
            dispatch("add_bias", add, 2, &bp, sizeof(bp),
                     (UINT)((g.M * g.N + 63) / 64), 1);
            struct ReluP { int a, b, c; } rp{g.N, g.M * g.N, g.ldc};
            dispatch("relu_fwd", &rC, 1, &rp, sizeof(rp),
                     (UINT)((g.M * g.N + 63) / 64), 1);
        }
        return;
    }
    int variant = 2;
    if (std::getenv("PD_GEMM_OLD")) variant = 2;
    else if (const char* f = std::getenv("PD_GEMM_FORCE")) variant = std::atoi(f);
    else if (g.M >= 192) variant = 0;
    else if (g.N >= 512) variant = 1;
    // gemm_db (force=3) matches gemm_block's FMA order; the double-buffered
    // variant is not ported.
    if (variant == 3) variant = 0;
    if (variant < 0 || variant > 2) variant = 0;
    // gemm_block's K-loop barrier hangs this driver when K is shorter than the
    // 16-wide tile and the dispatch is the last command in the packet. The
    // value head backward is exactly that shape (K=1). The barrier-free kernel
    // covers it; one product has the same rounding either way.
    if (variant != 2 && g.K < 16) variant = 2;
    const char* name = variant == 0 ? "gemm_block" : variant == 1 ? "gemm_blockn"
                                                                  : "gemm_tiled";
    UINT gx, gy;
    if (variant == 0) {
        gx = (UINT)((g.N + 31) / 32);
        gy = (UINT)((g.M + 31) / 32);
    } else if (variant == 1) {
        gx = (UINT)((g.N + 31) / 32);
        gy = (UINT)((g.M + 15) / 16);
    } else {
        gx = (UINT)((g.N + 15) / 16);
        gy = (UINT)((g.M + 15) / 16);
    }
    dispatch(name, bufs, 4, &p, sizeof(p), gx, gy);
    auto floats = [](int rows, int stride, int cols) -> size_t {
        if (rows <= 0 || cols <= 0 || stride <= 0) return 0;
        return (size_t)(rows - 1) * (size_t)stride + (size_t)cols;
    };
    size_t need[4] = {
        (g.transA == 'T' ? floats(g.K, g.lda, g.M) : floats(g.M, g.lda, g.K)) * 4,
        (g.transB == 'T' ? floats(g.N, g.ldb, g.K) : floats(g.K, g.ldb, g.N)) * 4,
        floats(g.M, g.ldc, g.N) * 4,
        epi ? (size_t)g.N * 4 : 0,
    };
    for (int i = 0; i < 4; ++i) {
        if (i == 3 && epi == 0) continue;
        if (bufs[i].len == 0) continue;
        if (need[i] > bufs[i].len) {
            std::fprintf(stderr,
                         "D3D gemm OOB %s buf %d need %zu len %zu M %d N %d K %d\n",
                         name, i, need[i], bufs[i].len, g.M, g.N, g.K);
            std::fflush(stderr);
        }
    }
}

void d3dCommitGemms(int count, const GemmOp* ops, long long) {
    for (int e = 0; e < count; ++e) d3dCustomGemm(ops[e], nullptr, 0, 0);
}

void d3dAddBias(Mat& y, const std::vector<float>& b) {
    View yv = rOut(y);
    View bv = rScalar(b);
    struct P { int a, b, c; } q{y.c, y.r * y.c, y.s};
    launch1("add_bias", {yv, bv}, &q, sizeof(q), y.r * y.c);
}
void d3dRelu(Mat& x) {
    struct P { int a, b, c; } q{x.c, x.r * x.c, x.s};
    launch1("relu_fwd", {rOutSeed(x)}, &q, sizeof(q), x.r * x.c);
}
void d3dReluBwd(const Mat& pre, const Mat& gout, Mat& gin) {
    struct P { int a, b, c, d; } p{pre.c, pre.r * pre.c, pre.s, gout.s};
    launch1("relu_bwd", {rIn(pre), rIn(gout), rOut(gin)}, &p, sizeof(p),
            pre.r * pre.c);
}
void d3dGateAdd(Mat& gp, const Mat& gi, const Mat& gh,
                const std::vector<float>& bi, const std::vector<float>& bh) {
    struct P { int H, B; } p{gp.c, gp.r};
    launch1("gate_add",
            {rOut(gp), rIn(gi), rIn(gh), rScalar(bi), rScalar(bh)}, &p,
            sizeof(p), gp.r * gp.c);
}
void d3dLstmCellFwd(const Mat& gp, const float* cp, Mat& hOut, Mat& cOut,
                    int hidden) {
    struct P { int h, B, hasCp; } p{hidden, gp.r, cp ? 1 : 0};
    View cpv = cp ? kerIn(cp, size_t(gp.r) * hidden * 4) : dummyView();
    launch2("lstm_cell_fwd", {rIn(gp), cpv, rOut(hOut), rOut(cOut)}, &p,
            sizeof(p), gp.r, hidden);
}
void d3dLstmCellBwd(const Mat& gh, const Mat& gp, const Mat& cNow,
                    const float* cp, Mat& dh, Mat& dc, Mat& dg, int hidden) {
    struct P { int h, B, hasCp; } p{hidden, gp.r, cp ? 1 : 0};
    View cpv = cp ? kerIn(cp, size_t(gp.r) * hidden * 4) : dummyView();
    launch2("lstm_cell_bwd",
            {rIn(gh), rIn(gp), rIn(cNow), cpv, rIn(dh), rIn(dc), rOut(dg)},
            &p, sizeof(p), gp.r, hidden);
}
void d3dConcat(Mat& z, const Mat& a, int n1, const Mat& b, int n2) {
    struct P { int n1, n2, B, zs, as, bs; } p{n1, n2, z.r, z.s, a.s, b.s};
    launch1("concat2", {rOut(z), rIn(a), rIn(b)}, &p, sizeof(p), z.r * (n1 + n2));
}
void d3dSplit(const Mat& z, int n1, Mat& a, Mat& b, int n2) {
    struct P { int n1, n2, B, zs, as, bs; } p{n1, n2, z.r, z.s, a.s, b.s};
    launch1("split2", {rIn(z), rOut(a), rOut(b)}, &p, sizeof(p), z.r * (n1 + n2));
}
void d3dZeroAndLast(Mat& all, const Mat& gh, int B, int T, int h) {
    struct P { int B, T, h; } p{B, T, h};
    launch1("zero_and_last", {rOut(all), rIn(gh)}, &p, sizeof(p), B * T * h);
}
void d3dMaskDyn(Mat& logits, const Mat& ds, const Mat& mask, int N) {
    struct P { int N, B, ls, ms; } p{N, logits.r, logits.s, mask.s};
    launch1("mask_dyn", {rOut(logits), rIn(ds), rIn(mask)}, &p, sizeof(p),
            logits.r * N);
}
void d3dAddTo(float* dst, void** gSlot, int dstCols, const Mat& src) {
    struct P { int rows, cols, dc, sc; } p{src.r, src.c, dstCols, src.s};
    size_t rounded = (size_t(src.r) * dstCols * 4 + 15) & ~size_t(15);
    Slot* s = static_cast<Slot*>(d3dGradCache(gSlot, dst, rounded));
    if (!s) return;
    View rd;
    rd.res = &s->gpu;
    rd.off = 0;
    rd.len = s->bytes;
    launch1("add_to", {rd, rIn(src)}, &p, sizeof(p), src.r * src.c);
}
void d3dBiasGradAdd(const Mat& g, float* db, void** dbSlot) {
    struct P { int B, o, gs; } p{g.r, g.c, g.s};
    size_t rounded = (size_t(g.c) * 4 + 15) & ~size_t(15);
    Slot* s = static_cast<Slot*>(d3dGradCache(dbSlot, db, rounded));
    if (!s) return;
    View rd;
    rd.res = &s->gpu;
    rd.off = 0;
    rd.len = s->bytes;
    launch1("bias_grad_add", {rIn(g), rd}, &p, sizeof(p), g.c);
}
void d3dSlice(Mat& dst, const Mat& src, int off, int h) {
    struct P { int B, h, off, ds, ss; } p{src.r, h, off, dst.s, src.s};
    launch1("slice_cols", {rOut(dst), rIn(src)}, &p, sizeof(p), src.r * h);
}
void d3dFlatten(Mat& dst, const Mat& src, int N) {
    struct P { int B, N, ss; } p{src.r, N, src.s};
    launch1("flatten_rows", {rOut(dst), rIn(src)}, &p, sizeof(p), src.r * N);
}
void d3dCopyMat(Mat& dst, const Mat& src) {
    struct P { int rows, cols, ds, ss; } p{src.r, src.c, dst.s, src.s};
    launch1("copy_mat", {rOut(dst), rIn(src)}, &p, sizeof(p), src.r * src.c);
}
void d3dAdam(std::vector<float>& w, std::vector<float>& dw,
             std::vector<float>& m, std::vector<float>& v, int n, float lr,
             float beta1, float beta2, float eps, float bc1, float bc2) {
    struct P {
        int n;
        float lr, b1, b2, eps, bc1, bc2;
    } p{n, lr, beta1, beta2, eps, bc1, bc2};
    size_t bytes = (size_t(n) + 3 & ~size_t(3)) * 4;
    View bufs[4] = {scalarRmw(w.data(), bytes), scalarRmw(dw.data(), bytes),
                    scalarRmw(m.data(), bytes), scalarRmw(v.data(), bytes)};
    UINT groups = (UINT)((n + 127) / 128);
    dispatch("adam_step", bufs, 4, &p, sizeof(p), groups, 1);
}

}  // namespace nn
