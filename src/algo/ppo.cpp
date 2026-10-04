#include "algo/ppo.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <mutex>
#include <numeric>
#include <thread>
#include <unordered_map>

namespace algo {

void computeGAE(std::vector<Transition>& tr, float gamma, float lambda) {
    // group indices by game (chronological within each game)
    std::unordered_map<int, std::vector<size_t>> groups;
    for (size_t i = 0; i < tr.size(); ++i) groups[tr[i].gameId].push_back(i);

    for (auto& [id, is] : groups) {
        (void)id;
        float gae = 0.0f;
        float nextValue = 0.0f;  // bootstrap with 0 at terminal
        for (int k = static_cast<int>(is.size()) - 1; k >= 0; --k) {
            Transition& t = tr[is[k]];
            float delta = t.reward + gamma * nextValue - t.value;
            gae = delta + gamma * lambda * gae;
            t.adv = gae;
            t.ret = gae + t.value;
            nextValue = t.value;
        }
    }
}

namespace {

void maskedSoftmax(const float* logits, int n, float* probs) {
    float mx = -1e30f;
    for (int a = 0; a < n; ++a) mx = std::max(mx, logits[a]);
    float sum = 0.0f;
    for (int a = 0; a < n; ++a) {
        probs[a] = logits[a] < -1e8f ? 0.0f : std::exp(logits[a] - mx);
        sum += probs[a];
    }
    float inv = 1.0f / sum;
    for (int a = 0; a < n; ++a) probs[a] *= inv;
}

}  // namespace

bool ppoUpdate(nn::Actor& actor, nn::Critic& critic,
               std::vector<Transition>& tr, const PPOConfig& cfg,
               nn::Adam& actorOpt, nn::Adam& criticOpt, nn::Rng64& rng,
               PPOStats& stats) {
    int N = static_cast<int>(tr.size());
    if (N == 0) return true;

    computeGAE(tr, cfg.gamma, cfg.lambda);

    double mean = 0.0, var = 0.0;
    for (const Transition& t : tr) mean += t.adv;
    mean /= N;
    for (const Transition& t : tr) var += (t.adv - float(mean)) * (t.adv - float(mean));
    float std = float(std::sqrt(var / N + 1e-8));
    for (Transition& t : tr) t.adv = (t.adv - float(mean)) / std;

    double sumRet = 0;
    for (const Transition& t : tr) sumRet += t.ret;
    stats.meanRet = sumRet / N;
    stats.meanAdv = mean;

    std::vector<int> order(N);
    std::iota(order.begin(), order.end(), 0);

    // One float copy of the whole stream, reused for every epoch. About
    // 150 MB per seat, so the prepare thread only gathers a minibatch.
    std::vector<float> xAll, seqAll, eAll;
    cacheTransitionFeatures(tr, xAll, seqAll, eAll);
    const int xStride = nn::padStride(ddz::kImpSize);
    const int eStride = nn::padStride(ddz::kExtraSize);
    struct PreparedBatch {
        std::vector<Transition*> mb;
        Batch data;
    };
    constexpr size_t kQueueDepth = 2;
    std::array<PreparedBatch, kQueueDepth> queue;
    std::atomic<size_t> produced{0};
    std::atomic<size_t> consumed{0};
    std::atomic<bool> stopProducer{false};
    const int batchesPerEpoch =
        (N + std::max(cfg.minibatch, 1) - 1) / std::max(cfg.minibatch, 1);
    const int totalBatches = cfg.epochs * batchesPerEpoch;

    // This producer owns the shuffle RNG and writes one SPSC ring slot at a
    // time. The learner keeps ownership of its D3D context and only consumes
    // fully materialized host batches.
    std::thread prepareThread([&] {
        // Host only. A second command queue on this thread cannot be used
        // by the learner, and dropping a device cache here would free it
        // off the thread that created it.
        nn::gemmSetThreadGpu(0);
        for (int ep = 0; ep < cfg.epochs && !stopProducer.load(); ++ep) {
            for (int i = N - 1; i > 0; --i) {
                int j = int(rng.nextU64() % uint64_t(i + 1));
                std::swap(order[i], order[j]);
            }
            for (int lo = 0; lo < N && !stopProducer.load(); lo += cfg.minibatch) {
                size_t tail = produced.load(std::memory_order_relaxed);
                while (tail - consumed.load(std::memory_order_acquire) >=
                       kQueueDepth) {
                    if (stopProducer.load(std::memory_order_acquire)) return;
                    std::this_thread::yield();
                }
                int hi = std::min(lo + cfg.minibatch, N);
                PreparedBatch& out = queue[tail % kQueueDepth];
                out.mb.resize(hi - lo);
                for (int i = lo; i < hi; ++i)
                    out.mb[i - lo] = &tr[order[i]];
                buildBatchFromCache(
                    out.mb, tr.data(), xAll.data(), xStride, seqAll.data(),
                    eAll.data(), eStride, out.data.xImp, out.data.seq,
                    out.data.mask, out.data.dynFeat, out.data.extra);
                produced.store(tail + 1, std::memory_order_release);
            }
        }
    });

    nn::Mat dLogits, dValue;

    // The D3D residency window is one per device context. Two seats in
    // that window at once reuse each other's scratch and remove the device.
    // One seat runs its minibatches through Adam before the next seat starts.
    static std::mutex oneSeat;
    std::lock_guard<std::mutex> seatLock(oneSeat);
    nn::gpuInvoke([&] {
        actor.zeroGrad();
        critic.zeroGrad();
    });

    double pgLossSum = 0, vLossSum = 0, entSum = 0;
    int mbCount = 0;
    bool deviceOk = true;

    for (int batchNo = 0; batchNo < totalBatches; ++batchNo) {
            size_t head = consumed.load(std::memory_order_relaxed);
            while (produced.load(std::memory_order_acquire) == head) {
                if (stopProducer.load(std::memory_order_acquire)) break;
                std::this_thread::yield();
            }
            if (produced.load(std::memory_order_acquire) == head) break;
            PreparedBatch& in = queue[head % kQueueDepth];
            std::vector<Transition*>& mb = in.mb;
            Batch& batch = in.data;
            int B = static_cast<int>(mb.size());

            // Forward and backward run on the one thread that owns the
            // compute queue. The host loss below stays on this learner.
            nn::Mat* logitRows = nullptr;
            nn::Mat* valueRows = nullptr;
            nn::gpuInvoke([&] {
                logitRows = &actor.forward(batch.xImp, batch.seq, batch.mask,
                                           batch.dynFeat);
                valueRows = &critic.forward(batch.xImp, batch.seq, batch.extra);
                nn::gpuMarkHost(logitRows->data());
                nn::gpuMarkHost(valueRows->data());
                nn::gpuWaitEx(true);  // flush logits & values; keep fwd binds
            });
            if (!nn::gpuDeviceOk()) {
                deviceOk = false;
                break;
            }

            nn::Mat& logits = *logitRows;
            nn::Mat& values = *valueRows;

            // ---- loss gradients on host ----
            dLogits.resize(B, nn::kNumActions);
            std::fill(dLogits.d.begin(), dLogits.d.end(), 0.0f);
            for (int i = 0; i < B; ++i) {
                float probs[nn::kNumActions];
                maskedSoftmax(logits.row(i), nn::kNumActions, probs);
                int act = mb[i]->action;
                float newLogp = std::log(std::max(probs[act], 1e-12f));
                float ratio = std::exp(newLogp - mb[i]->logp);
                float s = mb[i]->adv;
                float rc = std::clamp(ratio, 1.0f - cfg.clip, 1.0f + cfg.clip);
                float l1 = ratio * s, l2 = rc * s;
                float surrogate = std::min(l1, l2);
                pgLossSum += surrogate;

                float entropy = 0.0f;
                for (int a = 0; a < nn::kNumActions; ++a)
                    if (probs[a] > 1e-12f) entropy -= probs[a] * std::log(probs[a]);
                entSum += entropy;

                float g;
                if (l1 <= l2)
                    g = ratio * s;
                else
                    g = 0.0f;
                float scale = 1.0f / float(N);
                float pa = probs[act];
                float sumW = g * scale +
                             cfg.entCoef * scale * (1.0f - entropy);
                for (int a = 0; a < nn::kNumActions; ++a) {
                    if (probs[a] <= 0.0f) continue;
                    float W = (a == act ? -g / std::max(pa, 1e-8f) * scale
                                       : 0.0f) +
                              cfg.entCoef * scale * (std::log(probs[a]) + 1.0f);
                    dLogits.row(i)[a] = probs[a] * (W - sumW);
                }
            }

            dValue.resize(B, 1);
            for (int i = 0; i < B; ++i) {
                float err = values.row(i)[0] - mb[i]->ret;
                vLossSum += 0.5 * err * err;
                dValue.row(i)[0] = cfg.vfCoef * err / float(N);
            }

            nn::gpuInvoke([&] {
                actor.backward(dLogits);
                critic.backward(dValue);
                // Fence before this iteration's host buffers (dLogits, inputs)
                // go out of scope and the next minibatch overwrites them.
                nn::gpuWait();
            });
            if (!nn::gpuDeviceOk()) {
                deviceOk = false;
                break;
            }
            ++mbCount;
            // Release only after the final fence: the producer may now reuse
            // this slot's host buffers for a later minibatch.
            consumed.store(head + 1, std::memory_order_release);
    }
    stopProducer.store(true, std::memory_order_release);
    prepareThread.join();
    if (!deviceOk) return false;

    nn::gpuInvoke([&] {
        actorOpt.applyGradNorm(actor.params(), cfg.maxGradNorm);
        criticOpt.applyGradNorm(critic.params(), cfg.maxGradNorm);
        if (!nn::gpuDeviceOk()) return;
        // Weights changed on the device; drop their cached device copies (the
        // transposed wt Mats drop theirs in prepareInference via resize).
        for (nn::Param* p : actor.params()) nn::gpuDropCache(&p->devW);
        for (nn::Param* p : critic.params()) nn::gpuDropCache(&p->devW);
        nn::gpuPrintStats("ppo-update");
    });
    if (!nn::gpuDeviceOk()) return false;
    // Still holding the seat lock: rebuilding the transposed weights drops
    // device caches. Doing that after the lock lets the next seat record
    // against a buffer this thread is freeing.
    actor.prepareInference();
    critic.prepareInference();

    double samples = double(N) * cfg.epochs;
    stats.pgLoss = -pgLossSum / samples;
    stats.vLoss = vLossSum / samples;
    stats.entropy = entSum / samples;
    stats.meanAbsOldLogp = 0;
    (void)mbCount;
    return true;
}

}  // namespace algo
