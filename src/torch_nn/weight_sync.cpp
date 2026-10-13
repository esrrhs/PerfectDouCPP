#include "torch_nn/weight_sync.h"
#include <cstring>
#include <stdexcept>

namespace torch_nn {

namespace {

void copyTensorToParam(const torch::Tensor& t, nn::Param& p) {
    torch::Tensor cpuT = t.detach().to(torch::kCPU).contiguous();
    size_t count = cpuT.numel();
    if (count > p.w.size()) {
        p.w.resize(nn::padStride(count));
    }
    std::memcpy(p.w.data(), cpuT.data_ptr<float>(), count * sizeof(float));
    nn::gpuDropCache(&p.devW);
}

void copyParamToTensor(const nn::Param& p, torch::Tensor& t) {
    torch::NoGradGuard no_grad;
    size_t count = t.numel();
    std::memcpy(t.data_ptr<float>(), p.w.data(), count * sizeof(float));
}

// Convert PyTorch LSTM weight [4*H, In] (gates order: i, f, g, o)
// to legacy Lstm Wi [4*H, In] (gates order: i, f, g, o)
// In PyTorch:
//   weight_ih_l0: [4*H, In]
//   weight_hh_l0: [4*H, H]
//   bias_ih_l0: [4*H]
//   bias_hh_l0: [4*H]
// Legacy Lstm:
//   Wi: [4*H, In], bi: [1, 4*H]
//   Wh: [4*H, H],  bh: [1, 4*H]
}  // namespace

void syncTorchToLegacyActor(const Actor& src, nn::Actor& dst) {
    torch::NoGradGuard no_grad;

    // Linear layers
    // Note: PyTorch Linear weight is [Out, In]. Legacy Linear W is [Out, In].
    copyTensorToParam(src->l1->weight, dst.l1.W);
    copyTensorToParam(src->l1->bias, dst.l1.b);

    copyTensorToParam(src->l2->weight, dst.l2.W);
    copyTensorToParam(src->l2->bias, dst.l2.b);

    copyTensorToParam(src->l3->weight, dst.l3.W);
    copyTensorToParam(src->l3->bias, dst.l3.b);

    copyTensorToParam(src->l4->weight, dst.l4.W);
    copyTensorToParam(src->l4->bias, dst.l4.b);

    copyTensorToParam(src->head->weight, dst.head.W);
    copyTensorToParam(src->head->bias, dst.head.b);

    // LSTM
    auto named_params = src->lstm->named_parameters();
    copyTensorToParam(named_params["weight_ih_l0"], dst.lstm.Wi);
    copyTensorToParam(named_params["bias_ih_l0"], dst.lstm.bi);
    copyTensorToParam(named_params["weight_hh_l0"], dst.lstm.Wh);
    copyTensorToParam(named_params["bias_hh_l0"], dst.lstm.bh);

    dst.prepareInference();
}

void syncTorchToLegacyCritic(const Critic& src, nn::Critic& dst) {
    torch::NoGradGuard no_grad;

    copyTensorToParam(src->c1->weight, dst.c1.W);
    copyTensorToParam(src->c1->bias, dst.c1.b);

    copyTensorToParam(src->c2->weight, dst.c2.W);
    copyTensorToParam(src->c2->bias, dst.c2.b);

    copyTensorToParam(src->c3->weight, dst.c3.W);
    copyTensorToParam(src->c3->bias, dst.c3.b);

    copyTensorToParam(src->c4->weight, dst.c4.W);
    copyTensorToParam(src->c4->bias, dst.c4.b);

    copyTensorToParam(src->out->weight, dst.out.W);
    copyTensorToParam(src->out->bias, dst.out.b);

    auto named_params = src->lstm->named_parameters();
    copyTensorToParam(named_params["weight_ih_l0"], dst.lstm.Wi);
    copyTensorToParam(named_params["bias_ih_l0"], dst.lstm.bi);
    copyTensorToParam(named_params["weight_hh_l0"], dst.lstm.Wh);
    copyTensorToParam(named_params["bias_hh_l0"], dst.lstm.bh);

    dst.prepareInference();
}

void syncLegacyToTorchActor(const nn::Actor& src, Actor& dst) {
    torch::NoGradGuard no_grad;

    copyParamToTensor(src.l1.W, dst->l1->weight);
    copyParamToTensor(src.l1.b, dst->l1->bias);

    copyParamToTensor(src.l2.W, dst->l2->weight);
    copyParamToTensor(src.l2.b, dst->l2->bias);

    copyParamToTensor(src.l3.W, dst->l3->weight);
    copyParamToTensor(src.l3.b, dst->l3->bias);

    copyParamToTensor(src.l4.W, dst->l4->weight);
    copyParamToTensor(src.l4.b, dst->l4->bias);

    copyParamToTensor(src.head.W, dst->head->weight);
    copyParamToTensor(src.head.b, dst->head->bias);

    auto named_params = dst->lstm->named_parameters();
    copyParamToTensor(src.lstm.Wi, named_params["weight_ih_l0"]);
    copyParamToTensor(src.lstm.bi, named_params["bias_ih_l0"]);
    copyParamToTensor(src.lstm.Wh, named_params["weight_hh_l0"]);
    copyParamToTensor(src.lstm.bh, named_params["bias_hh_l0"]);
}

void syncLegacyToTorchCritic(const nn::Critic& src, Critic& dst) {
    torch::NoGradGuard no_grad;

    copyParamToTensor(src.c1.W, dst->c1->weight);
    copyParamToTensor(src.c1.b, dst->c1->bias);

    copyParamToTensor(src.c2.W, dst->c2->weight);
    copyParamToTensor(src.c2.b, dst->c2->bias);

    copyParamToTensor(src.c3.W, dst->c3->weight);
    copyParamToTensor(src.c3.b, dst->c3->bias);

    copyParamToTensor(src.c4.W, dst->c4->weight);
    copyParamToTensor(src.c4.b, dst->c4->bias);

    copyParamToTensor(src.out.W, dst->out->weight);
    copyParamToTensor(src.out.b, dst->out->bias);

    auto named_params = dst->lstm->named_parameters();
    copyParamToTensor(src.lstm.Wi, named_params["weight_ih_l0"]);
    copyParamToTensor(src.lstm.bi, named_params["bias_ih_l0"]);
    copyParamToTensor(src.lstm.Wh, named_params["weight_hh_l0"]);
    copyParamToTensor(src.lstm.bh, named_params["bias_hh_l0"]);
}

}  // namespace torch_nn
