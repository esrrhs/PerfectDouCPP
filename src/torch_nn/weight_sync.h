#pragma once

#include "torch_nn/models.h"
#include "nn/net.h"
#include <vector>

namespace torch_nn {

// Weights synchronization: copy parameters from a LibTorch Actor to an nn::Actor
void syncTorchToLegacyActor(const Actor& src, nn::Actor& dst);

// Weights synchronization: copy parameters from a LibTorch Critic to an nn::Critic
void syncTorchToLegacyCritic(const Critic& src, nn::Critic& dst);

// Weights synchronization: copy parameters from an nn::Actor to a LibTorch Actor
void syncLegacyToTorchActor(const nn::Actor& src, Actor& dst);

// Weights synchronization: copy parameters from an nn::Critic to a LibTorch Critic
void syncLegacyToTorchCritic(const nn::Critic& src, Critic& dst);

}  // namespace torch_nn
