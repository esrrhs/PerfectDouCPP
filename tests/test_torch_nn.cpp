#include <gtest/gtest.h>
#include "torch_nn/models.h"
#include "torch_nn/ppo_learner.h"
#include "torch_nn/tensor_bridge.h"

TEST(TorchNNTest, ActorForwardBackward) {
    torch_nn::ModelConfig cfg;
    cfg.hidden = 64;
    cfg.lstmHidden = 32;

    torch_nn::Actor actor(cfg);

    int64_t B = 4;
    int64_t L = 16;

    torch::Tensor xImp = torch::randn({B, torch_nn::kImpInput});
    torch::Tensor seq = torch::randn({B, torch_nn::kLstmSteps, torch_nn::kLstmIn});
    torch::Tensor actionFeat = torch::randn({L, torch_nn::kActionInput});
    torch::Tensor actionSample = torch::randint(0, B, {L}, torch::kLong);
    torch::Tensor actionId = torch::randint(0, torch_nn::kNumActions, {L}, torch::kLong);

    torch::Tensor logits = actor->forward(xImp, seq, actionFeat, actionSample, actionId);

    ASSERT_EQ(logits.size(0), B);
    ASSERT_EQ(logits.size(1), torch_nn::kNumActions);

    torch::Tensor loss = logits.sum();
    loss.backward();

    ASSERT_TRUE(actor->l1->weight.grad().defined());
}

TEST(TorchNNTest, CriticForwardBackward) {
    torch_nn::ModelConfig cfg;
    cfg.hidden = 64;
    cfg.lstmHidden = 32;

    torch_nn::Critic critic(cfg);

    int64_t B = 4;
    torch::Tensor xImp = torch::randn({B, torch_nn::kImpInput});
    torch::Tensor seq = torch::randn({B, torch_nn::kLstmSteps, torch_nn::kLstmIn});
    torch::Tensor extra = torch::randn({B, torch_nn::kExtraInput});

    torch::Tensor val = critic->forward(xImp, seq, extra);

    ASSERT_EQ(val.size(0), B);
    ASSERT_EQ(val.size(1), 1);

    torch::Tensor loss = val.sum();
    loss.backward();

    ASSERT_TRUE(critic->c1->weight.grad().defined());
}

TEST(TorchNNTest, PPOLearnerStep) {
    torch_nn::ModelConfig cfg;
    cfg.hidden = 64;
    cfg.lstmHidden = 32;

    torch_nn::Actor actor(cfg);
    torch_nn::Critic critic(cfg);

    torch_nn::PPOConfig ppoCfg;
    ppoCfg.epochs = 2;
    ppoCfg.minibatch = 4;

    torch_nn::PPOLearner learner(actor, critic, ppoCfg);

    int64_t B = 8;
    int64_t L = 24;

    torch_nn::BatchTensors bt;
    bt.xImp = torch::randn({B, torch_nn::kImpInput});
    bt.seq = torch::randn({B, torch_nn::kLstmSteps, torch_nn::kLstmIn});
    bt.extra = torch::randn({B, torch_nn::kExtraInput});
    bt.actions = torch::randint(0, torch_nn::kNumActions, {B}, torch::kLong);
    bt.oldLogp = torch::randn({B});
    bt.adv = torch::randn({B});
    bt.ret = torch::randn({B});
    bt.oldValue = torch::randn({B});

    bt.actionFeat = torch::randn({L, torch_nn::kActionInput});
    bt.actionSample = torch::randint(0, B, {L}, torch::kLong);
    bt.actionId = torch::randint(0, torch_nn::kNumActions, {L}, torch::kLong);

    auto stats = learner.update(bt);
    ASSERT_GT(stats.epochsCompleted, 0);
}
