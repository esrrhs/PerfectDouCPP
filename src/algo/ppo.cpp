#include "algo/ppo.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <condition_variable>
#include <mutex>
#include <numeric>
#include <thread>
#include <unordered_map>

namespace algo {

void assignEpisodeReturns(std::vector<Transition>& tr, float gamma,
                          float lambda) {
    std::unordered_map<int, std::vector<size_t>> groups;
    for (size_t i = 0; i < tr.size(); ++i) groups[tr[i].gameId].push_back(i);

    const float gaeFactor = gamma * lambda;
    for (auto& [id, is] : groups) {
        (void)id;
        if (is.empty()) continue;
        float gae = 0.0f;
        // Backward pass through the trajectory to compute GAE advantages
        for (int step = static_cast<int>(is.size()) - 1; step >= 0; --step) {
            size_t k = is[step];
            float nextValue = 0.0f;
            if (!tr[k].terminal && step + 1 < static_cast<int>(is.size())) {
                nextValue = tr[is[step + 1]].value;
            }
            float delta = tr[k].reward + gamma * nextValue - tr[k].value;
            gae = delta + (tr[k].terminal ? 0.0f : gaeFactor * gae);
            tr[k].adv = gae;
            tr[k].ret = tr[k].adv + tr[k].value;
        }
    }
}

void maskedSoftmax(const float* logits, int n, float* probs) {
    float mx = -1e30f;
    for (int a = 0; a < n; ++a) {
        if (logits[a] > mx) mx = logits[a];
    }
    float sum = 0.0f;
    for (int a = 0; a < n; ++a) {
        probs[a] = (logits[a] < -1e8f || !std::isfinite(logits[a])) ? 0.0f : std::exp(logits[a] - mx);
        sum += probs[a];
    }
    if (sum > 0.0f && std::isfinite(sum)) {
        float inv = 1.0f / sum;
        for (int a = 0; a < n; ++a) probs[a] *= inv;
    } else {
        float uni = 1.0f / float(n);
        for (int a = 0; a < n; ++a) probs[a] = uni;
    }
}

namespace {

}  // namespace

namespace {

bool reproDump() {
    static int on = -1;
    if (on < 0) on = std::getenv("PD_REPRO") ? 1 : 0;
    return on != 0;
}

void writeMat(const char* path, const nn::Mat& m) {
    FILE* f = std::fopen(path, "wb");
    if (!f) return;
    int hdr[3] = {m.r, m.c, m.s};
    std::fwrite(hdr, sizeof(int), 3, f);
    if (m.r > 0 && m.s > 0 && m.data())
        std::fwrite(m.data(), sizeof(float), size_t(m.r) * m.s, f);
    std::fclose(f);
}

void dumpWeights(const nn::Actor& actor, const nn::Critic& critic) {
    std::filesystem::create_directories("build-win/repro");
    auto dumpParams = [](const char* path, const std::vector<nn::Param*>& ps) {
        FILE* f = std::fopen(path, "wb");
        if (!f) return;
        int n = (int)ps.size();
        std::fwrite(&n, sizeof(int), 1, f);
        for (const nn::Param* p : ps) {
            int meta[3] = {p->rows, p->cols, (int)p->w.size()};
            std::fwrite(meta, sizeof(int), 3, f);
            if (!p->w.empty())
                std::fwrite(p->w.data(), sizeof(float), p->w.size(), f);
        }
        std::fclose(f);
    };
    dumpParams("build-win/repro/actor_w.bin",
               const_cast<nn::Actor&>(actor).params());
    dumpParams("build-win/repro/critic_w.bin",
               const_cast<nn::Critic&>(critic).params());
}

void dumpRound(int batchNo, const std::vector<Transition*>& mb, const Batch& batch) {
    std::filesystem::create_directories("build-win/repro");
    FILE* meta = std::fopen("build-win/repro/round.txt", "w");
    if (meta) {
        std::fprintf(meta, "batch %d B %d\n", batchNo, (int)mb.size());
        for (size_t i = 0; i < mb.size(); ++i) {
            const Transition* t = mb[i];
            std::fprintf(meta, "%zu action %d logp %.8g adv %.8g ret %.8g reward %.8g value %.8g\n",
                         i, t->action, t->logp, t->adv, t->ret, t->reward, t->value);
        }
        std::fclose(meta);
    }
    writeMat("build-win/repro/xImp.bin", batch.xImp);
    writeMat("build-win/repro/seq.bin", batch.seq);
    writeMat("build-win/repro/action.bin", batch.actionFeat);
    writeMat("build-win/repro/action_sample.bin", batch.actionSample);
    writeMat("build-win/repro/action_id.bin", batch.actionId);
    writeMat("build-win/repro/extra.bin", batch.extra);
    std::remove("build-win/repro/logits.bin");
    std::remove("build-win/repro/values.bin");
}

}  // namespace

bool ppoUpdate(nn::Actor& actor, nn::Critic& critic,
               std::vector<Transition>& tr, const PPOConfig& cfg,
               nn::Adam& actorOpt, nn::Adam& criticOpt, nn::Rng64& rng,
               PPOStats& stats) {
    int N = static_cast<int>(tr.size());
    if (N == 0) return true;

    assignEpisodeReturns(tr, cfg.gamma, cfg.lambda);

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
    const int xStride = nn::padStride(ddz::kNodeSize);
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
    // yield() is Sleep(0) on this runtime and showed up as about a fifth of
    // process CPU. Wait on the slot instead.
    std::mutex slotMu;
    std::condition_variable slotCv;
    const int batchesPerEpoch =
        (N + std::max(cfg.minibatch, 1) - 1) / std::max(cfg.minibatch, 1);
    const int totalBatches = cfg.epochs * batchesPerEpoch;

    const bool singleThread = std::getenv("PD_SINGLE_THREAD") != nullptr;
    int prodEp = 0;
    int prodLo = 0;
    bool needShuffle = true;
    auto produceOne = [&]() -> bool {
        while (prodEp < cfg.epochs && prodLo >= N) {
            ++prodEp;
            needShuffle = true;
            prodLo = 0;
        }
        if (prodEp >= cfg.epochs) return false;
        if (needShuffle) {
            for (int i = N - 1; i > 0; --i) {
                int j = int(rng.nextU64() % uint64_t(i + 1));
                std::swap(order[i], order[j]);
            }
            needShuffle = false;
            prodLo = 0;
        }
        int hi = std::min(prodLo + cfg.minibatch, N);
        size_t tail = produced.load(std::memory_order_relaxed);
        PreparedBatch& out = queue[tail % kQueueDepth];
        out.mb.resize(hi - prodLo);
        for (int i = prodLo; i < hi; ++i)
            out.mb[i - prodLo] = &tr[order[i]];
        buildBatchFromCache(
            out.mb, tr.data(), xAll.data(), xStride, seqAll.data(),
            eAll.data(), eStride, out.data.xImp, out.data.seq,
            out.data.actionFeat, out.data.actionSample, out.data.actionId,
            out.data.actionOffset, out.data.extra);
        produced.store(tail + 1, std::memory_order_release);
        prodLo = hi;
        return true;
    };

    // This producer owns the shuffle RNG and writes one SPSC ring slot at a
    // time. The learner keeps ownership of its D3D context and only consumes
    // fully materialized host batches. Single-thread mode fills the next
    // batch on this same thread instead.
    std::thread prepareThread;
    if (!singleThread) prepareThread = std::thread([&] {
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
                size_t tail = 0;
                {
                    std::unique_lock<std::mutex> lk(slotMu);
                    slotCv.wait(lk, [&] {
                        if (stopProducer.load(std::memory_order_acquire)) return true;
                        tail = produced.load(std::memory_order_relaxed);
                        return tail - consumed.load(std::memory_order_acquire) <
                               kQueueDepth;
                    });
                }
                if (stopProducer.load(std::memory_order_acquire)) return;
                int hi = std::min(lo + cfg.minibatch, N);
                PreparedBatch& out = queue[tail % kQueueDepth];
                out.mb.resize(hi - lo);
                for (int i = lo; i < hi; ++i)
                    out.mb[i - lo] = &tr[order[i]];
                buildBatchFromCache(
                    out.mb, tr.data(), xAll.data(), xStride, seqAll.data(),
                    eAll.data(), eStride, out.data.xImp, out.data.seq,
                    out.data.actionFeat, out.data.actionSample,
                    out.data.actionId, out.data.actionOffset, out.data.extra);
                produced.store(tail + 1, std::memory_order_release);
                slotCv.notify_one();
            }
        }
    });

    nn::Mat dLogits, dValue;

    // CUDA: this learner records and launches on its own thread. The three
    // seats share one D3D queue (a queue each removes the device) and each
    // has its own CUDA stream. The pure D3D path stays on the owner thread.
    const bool ownGpu = std::getenv("PD_CUBLAS") != nullptr;
    auto onGpu = [&](const std::function<void()>& fn) {
        if (ownGpu) fn();
        else nn::gpuInvoke(fn);
    };
    static std::atomic<int> seatNext{0};
    static thread_local int seat = -1;
    if (seat < 0) seat = seatNext.fetch_add(1, std::memory_order_relaxed);
    nn::gpuBindSeat(seat);
    onGpu([&] {
        actor.zeroGrad();
        critic.zeroGrad();
    });

    double pgLossSum = 0, vLossSum = 0, entSum = 0;
    double klSum = 0, clipCountSum = 0;
    double epochKLSum = 0;
    double lastEpochKL = 0;
    int epochSamples = 0;
    int mbCount = 0;
    int totalSamplesProcessed = 0;
    bool deviceOk = true;
    bool earlyStopped = false;
    if (reproDump()) dumpWeights(actor, critic);

    for (int batchNo = 0; batchNo < totalBatches; ++batchNo) {
            size_t head = consumed.load(std::memory_order_relaxed);
            if (singleThread) {
                if (produced.load(std::memory_order_acquire) == head &&
                    !produceOne())
                    break;
            } else {
                std::unique_lock<std::mutex> lk(slotMu);
                slotCv.wait(lk, [&] {
                    return stopProducer.load(std::memory_order_acquire) ||
                           produced.load(std::memory_order_acquire) != head;
                });
                if (produced.load(std::memory_order_acquire) == head) break;
            }
            PreparedBatch& in = queue[head % kQueueDepth];
            std::vector<Transition*>& mb = in.mb;
            Batch& batch = in.data;
            int B = static_cast<int>(mb.size());
            if (reproDump()) dumpRound(batchNo, mb, batch);

            // Forward and backward run on the one thread that owns the
            // compute queue. The host loss below stays on this learner.
            nn::Mat* logitRows = nullptr;
            nn::Mat* valueRows = nullptr;
            onGpu([&] {
                logitRows =
                    &actor.forward(batch.xImp, batch.seq, batch.actionFeat,
                                   batch.actionSample, batch.actionId,
                                   batch.actionOffset);
                valueRows = &critic.forward(batch.xImp, batch.seq, batch.extra);
                nn::gpuMarkHost(logitRows->data());
                nn::gpuMarkHost(valueRows->data());
                if (ownGpu) nn::gpuSubmit();
                else nn::gpuWaitEx(true);
            });
            if (ownGpu) {
                nn::gpuSync();
                nn::gpuWaitEx(true);
            }
            if (!nn::gpuDeviceOk()) {
                deviceOk = false;
                break;
            }

            nn::Mat& logits = *logitRows;
            nn::Mat& values = *valueRows;
            if (reproDump()) {
                writeMat("build-win/repro/logits.bin", logits);
                writeMat("build-win/repro/values.bin", values);
            }

            // ---- loss gradients on host ----
            dLogits.resize(B, nn::kNumActions);
            std::fill(dLogits.d.begin(), dLogits.d.end(), 0.0f);
            float mbKLSum = 0.0f;
            float mbClipSum = 0.0f;
            for (int i = 0; i < B; ++i) {
                float probs[nn::kNumActions];
                maskedSoftmax(logits.row(i), nn::kNumActions, probs);
                int act = mb[i]->action;
                float newLogp = std::log(std::max(probs[act], 1e-12f));
                float logRatio = newLogp - mb[i]->logp;
                float ratio = std::exp(logRatio);
                // Approx KL divergence: (ratio - 1) - log(ratio) (k3 approximation, non-negative)
                float approxKL = (ratio - 1.0f) - logRatio;
                if (!std::isfinite(approxKL) || approxKL < 0.0f) approxKL = 0.0f;
                mbKLSum += approxKL;

                float s = mb[i]->adv;
                float rc = std::clamp(ratio, 1.0f - cfg.clip, 1.0f + cfg.clip);
                if ((s > 0.0f && ratio > 1.0f + cfg.clip) ||
                    (s < 0.0f && ratio < 1.0f - cfg.clip)) {
                    mbClipSum += 1.0f;
                }
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
                // Each PPO minibatch is an optimizer batch (paper: 1024
                // samples total), rather than a tile accumulated into one
                // giant full-rollout gradient.
                float scale = 1.0f / float(B);
                float pa = probs[act];
                // sumW is E[W]. The taken-action weight is -g/p, so its
                // expectation is -g, not +g. The positive baseline kept a
                // full-sized step after the chosen action was already certain.
                float sumW = -g * scale +
                             cfg.entCoef * scale * (1.0f - entropy);
                for (int a = 0; a < nn::kNumActions; ++a) {
                    if (probs[a] <= 0.0f) continue;
                    float W = (a == act ? -g / std::max(pa, 1e-8f) * scale
                                       : 0.0f) +
                              cfg.entCoef * scale * (std::log(probs[a]) + 1.0f);
                    dLogits.row(i)[a] = probs[a] * (W - sumW);
                }
            }
            klSum += mbKLSum;
            clipCountSum += mbClipSum;
            totalSamplesProcessed += B;

            dValue.resize(B, 1);
            for (int i = 0; i < B; ++i) {
                float vPred = values.row(i)[0];
                float vTarget = mb[i]->ret;
                float err = vPred - vTarget;
                if (cfg.clipVf) {
                    // PPO Value Clipping: L_vf = max((v - target)^2, (v_clipped - target)^2)
                    // If the clipped surrogate is larger (l2 > l1), v is outside the [-vfClip, vfClip] interval,
                    // so d(v_clipped)/d(v) = 0, meaning the gradient with respect to v is 0.
                    float vOld = mb[i]->value;
                    float vClipped = vOld + std::clamp(vPred - vOld, -cfg.vfClip, cfg.vfClip);
                    float errClipped = vClipped - vTarget;
                    float l1 = err * err;
                    float l2 = errClipped * errClipped;
                    if (l2 > l1) {
                        vLossSum += 0.5 * l2;
                        dValue.row(i)[0] = 0.0f;
                    } else {
                        vLossSum += 0.5 * l1;
                        dValue.row(i)[0] = cfg.vfCoef * err / float(B);
                    }
                } else {
                    vLossSum += 0.5 * err * err;
                    dValue.row(i)[0] = cfg.vfCoef * err / float(B);
                }
            }

            onGpu([&] {
                actor.backward(dLogits);
                critic.backward(dValue);
                if (ownGpu) nn::gpuSubmit();
                else nn::gpuWait();
            });
            // CUDA copies are already in this seat's scratch, so the producer
            // may refill the host batch while the stream finishes. The D3D
            // path waited inside onGpu.
            consumed.store(head + 1, std::memory_order_release);
            slotCv.notify_one();
            if (ownGpu) {
                nn::gpuSync();
                nn::gpuWait();
            }
            if (!nn::gpuDeviceOk()) {
                deviceOk = false;
                break;
            }

            // PPO performs an Adam update for each shuffled minibatch.  The
            // previous implementation accumulated every minibatch and epoch
            // into one step, making the published batch size only a tiling
            // parameter and leaving old-logp clipping mostly ineffective.
            onGpu([&] {
                actorOpt.applyGradNorm(actor.params(), cfg.maxGradNorm);
                criticOpt.applyGradNorm(critic.params(), cfg.criticMaxGradNorm);
                if (!nn::gpuDeviceOk()) return;
                for (nn::Param* p : actor.params())
                    nn::gpuStaleCache(&p->devW);
                for (nn::Param* p : critic.params())
                    nn::gpuStaleCache(&p->devW);
            });
            if (!nn::gpuDeviceOk()) {
                deviceOk = false;
                break;
            }
            actor.prepareInference();
            critic.prepareInference();
            onGpu([&] {
                actor.zeroGrad();
                critic.zeroGrad();
            });
            ++mbCount;

            // Early stopping check at the end of each epoch
            epochKLSum += mbKLSum;
            epochSamples += B;
            if ((batchNo + 1) % batchesPerEpoch == 0) {
                lastEpochKL = (epochSamples > 0) ? (epochKLSum / epochSamples) : 0.0;
                epochKLSum = 0.0;
                epochSamples = 0;
                if (cfg.targetKL > 0.0f && lastEpochKL > 1.5 * cfg.targetKL) {
                    earlyStopped = true;
                    break;
                }
            }
    }
    stopProducer.store(true, std::memory_order_release);
    slotCv.notify_all();
    if (prepareThread.joinable()) prepareThread.join();
    if (!deviceOk) return false;

    onGpu([&] { nn::gpuPrintStats("ppo-update"); });
    if (!nn::gpuDeviceOk()) return false;

    double samples = std::max(1, totalSamplesProcessed);
    stats.pgLoss = -pgLossSum / samples;
    stats.vLoss = vLossSum / samples;
    stats.entropy = entSum / samples;
    stats.approxKL = klSum / samples;
    stats.lastEpochKL = (lastEpochKL > 0.0) ? lastEpochKL : stats.approxKL;
    stats.clipFraction = clipCountSum / samples;
    stats.epochsCompleted = (mbCount + batchesPerEpoch - 1) / std::max(1, batchesPerEpoch);
    (void)earlyStopped;

    for (nn::Param* p : actor.params()) {
        for (float w : p->w) {
            if (!std::isfinite(w)) {
                std::fprintf(stderr, "FATAL: Actor parameter contains NaN/Inf after ppoUpdate!\n");
                return false;
            }
        }
    }
    for (nn::Param* p : critic.params()) {
        for (float w : p->w) {
            if (!std::isfinite(w)) {
                std::fprintf(stderr, "FATAL: Critic parameter contains NaN/Inf after ppoUpdate!\n");
                return false;
            }
        }
    }
    if (!std::isfinite(stats.approxKL) || !std::isfinite(stats.pgLoss) || !std::isfinite(stats.vLoss)) {
        std::fprintf(stderr, "FATAL: PPOStats contains NaN/Inf (approxKL=%f, pgLoss=%f, vLoss=%f)!\n",
                     stats.approxKL, stats.pgLoss, stats.vLoss);
        return false;
    }
    return true;
}

}  // namespace algo
