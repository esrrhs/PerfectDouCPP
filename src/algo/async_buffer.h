#pragma once

#include "algo/rollout.h"
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <vector>

namespace algo {

struct RolloutChunk {
    std::array<std::vector<Transition>, 3> streams;
    RolloutStats stats;
};

// Thread-safe Async Bounded Pipeline Buffer for decoupling CPU rollout from GPU training.
// Rollout workers continuously push completed episode chunks without blocking.
// Learner pops complete fresh chunks, ensuring every trajectory is trained once without stale spinning.
class AsyncReplayBuffer {
public:
    explicit AsyncReplayBuffer(size_t maxChunks = 4)
        : maxCapacity(std::max<size_t>(maxChunks, 1)) {}

    void push(RolloutChunk&& chunk, std::atomic<bool>& stopFlag) {
        std::unique_lock<std::mutex> lock(mu);
        cvNotFull.wait(lock, [&] {
            return stopFlag.load() || queue.size() < maxCapacity;
        });
        if (stopFlag.load()) return;
        for (int s = 0; s < 3; ++s) {
            totalPushedTransitions[s] += chunk.streams[s].size();
        }
        queue.push_back(std::move(chunk));
        cvNotEmpty.notify_one();
    }

    bool pop(RolloutChunk& chunk, std::atomic<bool>& stopFlag) {
        std::unique_lock<std::mutex> lock(mu);
        cvNotEmpty.wait(lock, [&] {
            return stopFlag.load() || !queue.empty();
        });
        if (queue.empty()) return false;
        chunk = std::move(queue.front());
        queue.pop_front();
        cvNotFull.notify_one();
        return true;
    }

    bool waitMinChunks(size_t minCount, std::atomic<bool>& stopFlag) {
        std::unique_lock<std::mutex> lock(mu);
        cvNotEmpty.wait(lock, [&] {
            return stopFlag.load() || queue.size() >= minCount;
        });
        return !stopFlag.load();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mu);
        return queue.size();
    }

    long long getTotalPushed(int seat) const {
        std::lock_guard<std::mutex> lock(mu);
        return (seat >= 0 && seat < 3) ? totalPushedTransitions[seat] : 0;
    }

    void notifyAll() {
        std::lock_guard<std::mutex> lock(mu);
        cvNotEmpty.notify_all();
        cvNotFull.notify_all();
    }

private:
    size_t maxCapacity;
    mutable std::mutex mu;
    std::condition_variable cvNotEmpty;
    std::condition_variable cvNotFull;
    std::deque<RolloutChunk> queue;
    std::array<long long, 3> totalPushedTransitions{0, 0, 0};
};

}  // namespace algo
