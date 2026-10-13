#include "algo/ppo.h"

#include <cmath>
#include <vector>

namespace algo {

void assignEpisodeReturns(std::vector<Transition>& tr, float gamma,
                          float lambda) {
    int N = static_cast<int>(tr.size());
    if (N == 0) return;
    int maxGameId = 0;
    for (const auto& t : tr) {
        if (t.gameId > maxGameId) maxGameId = t.gameId;
    }
    // Linked list per gameId: head[gameId] points to latest transition,
    // nextTr[i] points to previous transition in that game.
    // Zero dynamic allocations beyond two flat vectors.
    std::vector<int> head(maxGameId + 1, -1);
    std::vector<int> nextTr(N, -1);
    for (int i = 0; i < N; ++i) {
        int gid = tr[i].gameId;
        if (gid >= 0) {
            nextTr[i] = head[gid];
            head[gid] = i;
        }
    }

    const float gaeFactor = gamma * lambda;
    for (int g = 0; g <= maxGameId; ++g) {
        int curr = head[g];
        if (curr < 0) continue;
        float gae = 0.0f;
        float nextValue = 0.0f;
        bool isFirstInBackward = true;
        while (curr >= 0) {
            float vNext = isFirstInBackward ? 0.0f : nextValue;
            float delta = tr[curr].reward + gamma * vNext - tr[curr].value;
            gae = delta + (tr[curr].terminal ? 0.0f : gaeFactor * gae);
            tr[curr].adv = gae;
            tr[curr].ret = tr[curr].adv + tr[curr].value;
            nextValue = tr[curr].value;
            isFirstInBackward = false;
            curr = nextTr[curr];
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

}  // namespace algo
