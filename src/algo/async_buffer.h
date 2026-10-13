#pragma once

#include "algo/rollout.h"
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <vector>

namespace algo {

// Thread-safe Async Ring Replay Buffer for decoupling CPU rollout from GPU training.
// Workers continuously push completed episode trajectories without blocking.
// Learner samples continuous chunks from the ring buffer.
class AsyncReplayBuffer {
public:
    explicit AsyncReplayBuffer(size_t capacityTransitions = 32768)
        : capacity(std::max<size_t>(capacityTransitions, 2048)) {
        for (int s = 0; s < 3; ++s) {
            storage[s].resize(capacity);
        }
    }

    // Push transitions for a specific seat
    void push(int seat, const std::vector<Transition>& trs) {
        if (trs.empty() || seat < 0 || seat >= 3) return;
        std::unique_lock<std::mutex> lock(mu);
        for (const auto& t : trs) {
            storage[seat][head[seat]] = t;
            head[seat] = (head[seat] + 1) % capacity;
            if (count[seat] < capacity) {
                ++count[seat];
            }
        }
        totalPushed[seat] += trs.size();
        cv.notify_all();
    }

    // Wait until buffer has at least minCount transitions for the given seat
    bool waitMinSamples(int seat, size_t minCount, std::atomic<bool>& stopFlag) {
        std::unique_lock<std::mutex> lock(mu);
        cv.wait(lock, [&] {
            return stopFlag.load() || count[seat] >= minCount;
        });
        return !stopFlag.load();
    }

    // Get current available count in buffer for seat
    size_t size(int seat) const {
        std::lock_guard<std::mutex> lock(mu);
        return count[seat];
    }

    // Sample the most recent N transitions (with episode GAE boundary safety)
    std::vector<Transition> sampleRecent(int seat, size_t n) {
        std::vector<Transition> res;
        std::lock_guard<std::mutex> lock(mu);
        size_t available = count[seat];
        if (available == 0) return res;
        size_t take = std::min(n, available);
        res.reserve(take);

        // Read backwards from current head
        size_t startIdx = (head[seat] + capacity - take) % capacity;
        for (size_t i = 0; i < take; ++i) {
            size_t idx = (startIdx + i) % capacity;
            res.push_back(storage[seat][idx]);
        }
        return res;
    }

    void notifyAll() {
        std::lock_guard<std::mutex> lock(mu);
        cv.notify_all();
    }

private:
    size_t capacity;
    mutable std::mutex mu;
    std::condition_variable cv;
    std::array<std::vector<Transition>, 3> storage;
    std::array<size_t, 3> head{0, 0, 0};
    std::array<size_t, 3> count{0, 0, 0};
    std::array<size_t, 3> totalPushed{0, 0, 0};
};

}  // namespace algo
