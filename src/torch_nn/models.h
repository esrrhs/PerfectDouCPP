#pragma once

#include <torch/torch.h>
#include <vector>
#include <string>

namespace torch_nn {

constexpr int64_t kImpInput = 1676;
constexpr int64_t kExtraInput = 362;
constexpr int64_t kLstmSteps = 10;
constexpr int64_t kLstmIn = 540;
constexpr int64_t kNumActions = 2048;
constexpr int64_t kActionInput = 186;

struct ModelConfig {
    int64_t hidden = 256;
    int64_t lstmHidden = 128;
};

// Actor (Imperfect Information Policy Network)
// LSTM(540 -> lstmHidden) over 10 steps
// node = concat(xImp, lstm_h_last) -> [B, 1676 + lstmHidden]
// joint = concat(node[actionSample], actionFeat) -> [L, 1676 + lstmHidden + 186]
// MLP: Linear(joint_dim -> hidden) -> ReLU -> Linear(hidden -> hidden) -> ReLU
//      -> Linear(hidden -> hidden) -> ReLU -> Linear(hidden -> hidden*2) -> ReLU
//      -> Linear(hidden*2 -> 1)
// scatter scores [L, 1] to logits [B, 2048] initialized to -1e9
class ActorImpl : public torch::nn::Module {
public:
    ActorImpl() : ActorImpl(ModelConfig{}) {}
    ActorImpl(const ModelConfig& cfg);

    // Forward pass
    // seq: [B, 10, 540]
    // xImp: [B, 1676]
    // actionFeat: [L, 186]
    // actionSample: [L] (indices in 0..B-1)
    // actionId: [L] (action indices in 0..2047)
    // Returns logits: [B, 2048]
    torch::Tensor forward(const torch::Tensor& xImp,
                          const torch::Tensor& seq,
                          const torch::Tensor& actionFeat,
                          const torch::Tensor& actionSample,
                          const torch::Tensor& actionId) const;

    void save(const std::string& path) const;
    void load(const std::string& path);

    ModelConfig config;
    mutable torch::nn::LSTM lstm{nullptr};
    mutable torch::nn::Linear l1{nullptr}, l2{nullptr}, l3{nullptr}, l4{nullptr}, head{nullptr};
};
TORCH_MODULE(Actor);

// Critic (Perfect Information Value Network)
// LSTM(540 -> lstmHidden) over 10 steps
// node = concat(xImp, lstm_h_last) -> [B, 1676 + lstmHidden]
// all = concat(node, extra) -> [B, 1676 + lstmHidden + 362]
// MLP: Linear(all_dim -> hidden) -> ReLU -> Linear(hidden -> hidden) -> ReLU
//      -> Linear(hidden -> hidden) -> ReLU -> Linear(hidden -> hidden) -> ReLU
//      -> Linear(hidden -> 1)
// Returns value: [B, 1]
class CriticImpl : public torch::nn::Module {
public:
    CriticImpl() : CriticImpl(ModelConfig{}) {}
    CriticImpl(const ModelConfig& cfg);

    torch::Tensor forward(const torch::Tensor& xImp,
                          const torch::Tensor& seq,
                          const torch::Tensor& extra) const;

    void save(const std::string& path) const;
    void load(const std::string& path);

    ModelConfig config;
    mutable torch::nn::LSTM lstm{nullptr};
    mutable torch::nn::Linear c1{nullptr}, c2{nullptr}, c3{nullptr}, c4{nullptr}, out{nullptr};
};
TORCH_MODULE(Critic);

}  // namespace torch_nn
