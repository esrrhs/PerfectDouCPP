# PerfectDouCPP

[![CI](https://github.com/esrrhs/PerfectDouCPP/actions/workflows/ci.yml/badge.svg)](https://github.com/esrrhs/PerfectDouCPP/actions/workflows/ci.yml)
[![Language](https://img.shields.io/badge/language-C%2B%2B17-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B17)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)](https://github.com/esrrhs/PerfectDouCPP)

[**English**](README.md) | [**中文说明**](README_zh.md)

**PerfectDouCPP** is a high-performance, production-grade, end-to-end self-play reinforcement learning and evaluation system for DouDizhu (Fighting the Landlord), written natively in **pure C++17** and powered by **LibTorch**.

---

### Lineage & Academic Heritage

This project draws deep inspiration and foundational game rules from two milestone research works in DouDizhu reinforcement learning:

- **[DouZero](https://github.com/kwai/DouZero)** (*ICML 2021, "Mastering DouDizhu with Self-Play Deep Reinforcement Learning"*): Laid the cornerstone for modern DouDizhu AI. We incorporate its comprehensive Tencent DouDizhu 15-combination rule parser, beating logic, and state-card tensor representations. In addition, its official `DouZero-ADP` model serves as the primary external benchmark baseline inside our evaluation pipeline.
- **[PerfectDou](https://github.com/Vincent-Guan/PerfectDou)** (*NeurIPS 2022, "Dominating DouDizhu with Perfect Information Distillation"*): Pioneered the **Perfect Information Distillation** paradigm in imperfect information games. We adopt its central methodology: leveraging an omniscient Critic (with global hand state) to guide imperfect-information Actors via advantage estimation, combined with a 621-dimensional abstract action space and dense Oracle difference rewards.

Building upon these pioneering academic works, **PerfectDouCPP** delivers a complete, industrial-grade engineering implementation:
- Native neural network architectures and operators using the LibTorch C++ API (zero Python overhead in the core loop).
- Ultra-fast card engine and DP/DFS memoized minimum-step Oracle.
- Multi-threaded CPU self-play rollout sampling coupled with synchronous GPU/MPS PPO training.
- Stratified league training with strict Peasant team policy consistency.
- Robust numerical safety guardrails (Masked Softmax anti-NaN fallback, dynamic KL divergence early stopping, and decoupled gradient clipping).
- Embedded asynchronous DouZero TCP evaluation loop with duplicate matching and automatic rolling snapshot pruning.
- 100% lossless state serialization and seamless checkpoint resumption.

---

## Architecture & Training Pipeline

The entire self-play reinforcement learning training loop operates as a tightly integrated, synchronous pipeline:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   1. Card Engine & Feature Engineering                 │
│  Tencent Rules (15 Types) ──► 621 Action Decoding ──► Oracle Min-Steps │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│             2. Multi-threaded Self-Play Rollout Collection             │
│  3 Independent Seats (Landlord / Landlord Down / Landlord Up)          │
│  Peasant Consistency (both peasants bound to same policy snapshot)     │
│  Stratified League: Current Latest + Rolling Pool + Sparse Archive     │
│  Dense Step-level Oracle Difference Shaping Reward (ΔAdv)              │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│              3. Terminal Settlement & Advantage Estimation             │
│  Terminal ADP Zero-Sum Reward (Bomb Doubling) ──► GAE (γ=1.0, λ=0.95)  │
│  Batch Advantage Normalization                                         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│             4. Perfect Information Distillation & PPO Learn            │
│  Omniscient Critic Distillation ──► Clipped Policy Loss L_CLIP         │
│  Decoupled Adam Optimizers ──► Masked Softmax (Anti-NaN) Guard         │
│  Adaptive KL Divergence Early Stopping                                 │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│             5. Policy Synchronization & Checkpoint Resumption          │
│  GPU Weights Synced back to CPU Workers ──► Periodic Full Checkpoint   │
│  (Actor/Critic Weights + Adam m/v/t + Update Steps + League Snapshots) │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│          6. Embedded Async DouZero Evaluation & Monitoring             │
│  Zero-overhead Model Deep Copy ──► Background Thread Duplicate Match   │
│  Auto-rolling Old Snapshot Pruning ──► plot_eval.py Curve Rendering    │
└────────────────────────────────────────────────────────────────────────┘
```

### 1. Card Engine & Feature Engineering
- **Rules and Valid Actions**: Full implementation of Tencent DouDizhu 15 valid card patterns (solo, pair, trio with/without attachments, straight, consecutive pairs, airplane, four with two, bombs, rocket). Maps valid combinations into a 621-dimensional abstract action space and rapidly decodes optimal card instances.
- **Oracle Min-Steps Solver**: Built-in dynamic programming and depth-first search (DFS) with thread-local memoization, evaluating the minimum number of steps to clear any hand in sub-milliseconds.
- **Dual Observation Representations**:
  - **Imperfect Information Observation (for Actor)**: Historical move sequences (10 steps × 540-dim fed into LSTM) + current player's hand and public bottom cards (1676-dim).
  - **Perfect Information Observation (for Critic)**: Global omniscient card matrices for all three players plus hand min-steps (362-dim).

### 2. Multi-threaded Self-Play Rollout
- **Three Independent Positions**: Dedicated models for Landlord, Landlord Down, and Landlord Up.
- **Peasant Team Consistency**: In rollout sampling, the Landlord independently samples an opponent policy, whereas both Peasants are strictly bound to the exact same policy snapshot to eliminate contradictory team behaviors.
- **Stratified League Training**: Mitigates policy cyclicality by sampling opponents across:
  - Current latest policy;
  - Rolling snapshot pool (recent iterations);
  - Geometrically spaced long-term archive (historical milestones).
- **Dense Oracle Rewards**: Follows PerfectDou Section 4.4 by computing step-level omniscient minimum step difference updates $\Delta Adv = \Delta (n_L - \min(n_D, n_U))$ to guide early learning.

### 3. Credit Assignment & GAE
- **ADP Zero-Sum Terminal Settlement**: Calculates exact ADP scores based on win/loss and bomb count: $\pm 2 \times 2^{\text{bomb}}$ for Landlord and $\pm 1 \times 2^{\text{bomb}}$ for Peasants.
- **Generalized Advantage Estimation (GAE)**: Uses $\gamma=1.0, \lambda=0.95$ for non-discounted episodic card games.
- **Advantage Normalization**: Batch-level mean-variance normalization stabilizes gradient steps.

### 4. Perfect Information Distillation & Synchronous PPO
- **Value Distillation (PTIE)**: The Critic receives global game information and provides low-variance, unbiased baseline value predictions to guide imperfect-information Actors.
- **Synchronous On-Policy Loop**: Eliminates policy staleness inherent to asynchronous off-policy buffers. Data sampled from CPU threads is batch-optimized on GPU/MPS before pushing updated weights back.
- **Adaptive KL Early Stopping**: Monitors empirical KL divergence each epoch, triggering early truncation when exceeding $1.5 \times \text{targetKL}$.
- **Numerical Safeguards**:
  - **Masked Softmax Fallback**: Uniform distribution fallback on near-zero/infinite action logits to prevent NaN propagation.
  - **Robust KL Clamping**: Boundary protection against divide-by-zero and log-zero.
  - **Sample Cleansing**: Automatic filtering of non-finite float samples.
  - **Decoupled Gradient Clipping**: $0.5$ norm clipping for Actor, $5.0$ for Critic to handle large bomb multipliers.

### 5. Checkpointing & Resumption
- Atomically persists complete checkpoint bundles (`meta.txt`, weights, Adam first/second moments $m/v/t$, league pool state).
- Seamless `--resume <DIR>` resumes training with deterministic deck sequences without repeating or skipping samples.

### 6. Embedded Asynchronous DouZero Evaluation
- Deep-copies models every $N$ updates (default 200) into an independent background thread.
- Performs duplicate match evaluations (swapping seats on the same deck) against a DouZero TCP inference server.
- Automatically prunes oldest snapshots (`--eval-max-snapshots`) to prevent disk exhaustion.
- Results are logged to CSV for live plotting via `tools/plot_eval.py`.

---

## Build & Test

Built with standard CMake (C++17) and LibTorch. Works out-of-the-box on Linux, macOS, and Windows.

### Prerequisites
- C++17 compatible compiler (`g++ >= 9`, `clang++ >= 11`, or MSVC 2019+)
- CMake >= 3.16
- [LibTorch (PyTorch C++ Library)](https://pytorch.org/get-started/locally/)
- Python 3.8+ (for DouZero benchmark server and plotting tools)

### Compilation

```bash
# Configure & compile (Release mode)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Run the Google Test suite (30 core tests)
ctest --test-dir build --output-on-failure
```

The test suite thoroughly covers rule validity, card combinations, 621-dim action decoding, Oracle DP memoization, neural network forward/backward passes, GAE/PPO numerical stability, checkpoint serialization/deserialization, anti-NaN fallbacks, and DouZero socket protocols.

---

## Training Guide

### 1. Start DouZero Baseline Server (Optional but Recommended)
To benchmark your policy against official `DouZero-ADP` during training, launch the lightweight TCP service (runs on CPU, without consuming training GPU memory):

```bash
# Option A (Recommended wrapper script, auto-detects dependencies & forces CPU isolation):
python tools/run_douzero_serve.py

# Option B (Manual):
export PYTHONPATH=third_party/DouZero   # Windows: set PYTHONPATH=third_party/DouZero
python tools/douzero_serve.py --port 18765 --ckpt-dir baselines/douzero_ADP
```

### 2. Launch Training

```bash
# Standard self-play training (GPU acceleration with regular DouZero evaluation)
./build/perfectdou_train --backend cuda --updates 100000 --games 768 --threads 16 --out ckpt \
    --eval-every 200 --eval-decks 1000 --eval-port 18765 --eval-save-dir eval_snapshots

# Apple Silicon (macOS Metal Performance Shaders)
./build/perfectdou_train --backend mps --updates 50000 --games 512 --threads 10 --out ckpt

# Fast local validation / CPU debug
./build/perfectdou_train --backend cpu --updates 300 --games 128 --threads 8 --out ckpt_debug
```

> **Hardware Backend (`--backend`)**:
> - `cuda`: NVIDIA GPU (Linux / Windows) for PPO updates, CPU threads for rollouts.
> - `mps`: Apple Silicon Metal GPU acceleration.
> - `cpu`: Standard multi-core CPU execution.

### 3. One-Click Background Automation Scripts

Production automation scripts are provided in `scripts/`:

```bash
# Linux:
# Auto-starts DouZero server + plot watcher + auto-detects resumption from meta.txt
bash scripts/run_all_linux.sh

# Monitor live training log
tail -f train.log

# Gracefully stop all background services and processes
bash scripts/stop_all_linux.sh
```

```bat
:: Windows:
scripts\run_all_detached.bat   :: Launch detached background processes
scripts\stop_all.bat            :: Stop all services
```

### 4. Log Metrics & Interpretation

Each update produces structured metrics:

```
upd  500 | wall 3.8s learn 0.42s games 768 mb 2048 ep 2,2,2/2 | WP 0.532 ADP   0.15 (pure WP 0.548 ADP   0.21 [550 g]) bomb/g 0.35 moves/g 51.2 | lr 3.00e-04 ent 0.852/0.710/0.708 vL    1.15/   0.32/   0.34 kl 0.0018/0.0015/0.0016 cf 0.04/0.03/0.03 | train_n 14820/11610/11590
[eval u500] done games 2000 | WP 0.565 ADP 0.380 | landlord WP 0.580 ADP 0.450 | peasant WP 0.550 ADP 0.310
```

- **WP / ADP**: Landlord win rate and average difference points against mixed league opponents.
- **pure WP / pure ADP**: Win rate and ADP strictly between the latest policy models.
- **ep**: Actual PPO epochs executed per position (early truncations indicate KL early stopping).
- **kl / cf**: Approximate KL divergence and PPO clip fraction.
- **[eval uX]**: Benchmark results against DouZero (total win rate, landlord win rate, peasant win rate).

### 5. Plotting Evaluation Curves & A/B Testing

Generate dynamic or static win-rate curves directly from logged CSV:

```bash
# Render static plot
python tools/plot_eval.py --csv ckpt/eval_vs_douzero.csv --out ckpt/eval_curve.png

# Live watch mode (auto re-renders every 30 seconds)
python tools/plot_eval.py --csv ckpt/eval_vs_douzero.csv --out ckpt/eval_curve.png --watch 30

# A/B Comparison between two experiments
python tools/plot_eval.py \
    --csv exp_baseline/eval_vs_douzero.csv exp_lr_1e4/eval_vs_douzero.csv \
    --labels "Baseline (3e-4)" "Low LR (1e-4)" \
    --out ab_comparison.png
```

---

## Practical Training Tips vs DouZero

1. **Two-Stage Oracle Reward Curriculum**:
   - *Phase 1 (Warmup & Mid-stage, updates 0 ~ 50,000)*: Enable Oracle rewards (`--oracle-reward 1 --oracle-coef 0.2`). Step-level $\Delta Adv$ dense rewards help agents quickly grasp card combination hierarchies, suppression, and efficient hand clearing.
   - *Phase 2 (Late-stage mastery)*: Lower the coefficient (`--oracle-coef 0.05` or `0`). Because Oracle does not account for bomb doubling ($2^{\text{bombs}}$) or peasant sacrifice cooperation, pure ADP terminal rewards optimize for explosive point maximization against DouZero.
2. **Compute Scale Perspective**:
   - Official DouZero ADP weights were trained using Deep Monte Carlo across hundreds of CPU/GPU cores over hundreds of millions of game steps.
   - PerfectDou achieves higher theoretical ceiling through omniscient Critic distillation and 10-step LSTM history tracking, but requires sufficient rollout steps (tens of thousands of updates) for strategy consolidation.
3. **Duplicate Match Protocol**:
   - Always evaluate using duplicate matches (both sides play identical decks from both Landlord and Peasant perspectives) to eliminate deal variance.
4. **Resumption Golden Rule**:
   - Keep identical arguments as original training, and simply append `--resume <DIR>`.
   - `meta.txt` tracks the exact update index; seeds derive deterministically (`seed + upd * 7919ULL`), guaranteeing seamless deck continuity.

---

## Command Line Arguments Reference

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--backend <B>` | `cpu` | **Hardware backend (`cuda` / `mps` / `cpu`)** |
| `--out <DIR>` | `ckpt_long` | **Output directory for weights, optimizer moments, and logs** |
| `--updates N` | 100000 | Total PPO update steps |
| `--games N` | 768 | Games collected per update cycle (recommended 512–1024) |
| `--threads N` | CPU cores | Number of concurrent self-play rollout threads |
| `--seed <S>` | 1 | Global PRNG seed for deterministic reproduction |
| `--hidden H` | 256 | MLP hidden dimension size |
| `--lstm-hidden H`| 128 | LSTM hidden state sequence dimension |
| `--epochs N` | 2 | PPO optimization epochs per update batch |
| `--mb N` | 2048 | PPO mini-batch size |
| `--lr N` | 3e-4 | Initial learning rate (Adam optimizer) |
| `--ent N` | 0.03 | Policy entropy regularization coefficient |
| `--clip N` | 0.2 | PPO policy ratio clipping threshold $\epsilon$ |
| `--target-kl N` | 0.03 | Adaptive KL divergence threshold for early stopping |
| `--gamma / --lambda` | 1.0 / 0.95 | GAE discount factor and variance trade-off parameter |
| **`--oracle-reward`** | **1** | **Enable dense Oracle step-difference rewards** |
| **`--oracle-coef`** | **0.2** | **Oracle difference reward scaling coefficient** |
| `--pool-size N` | 16 | League rolling snapshot pool size |
| `--archive-size N`| 16 | League geometric sparse long-term archive size |
| `--pool-every K` | 20 | Update frequency for adding policy to league pool |
| `--historical-prob`| 0.2 | Probability of sampling league opponents during rollout |
| `--snapshot-every K`| 50 | Frequency of persisting full training checkpoints |
| `--resume DIR` | empty | Resume training from existing checkpoint directory |
| **`--eval-every N`** | **200** | **DouZero evaluation frequency (0 to disable)** |
| **`--eval-decks D`** | **1000** | **Decks per evaluation cycle (1000 duplicate decks = 2000 games)** |
| **`--eval-port P`** | **18765**| **DouZero TCP inference server port** |
| **`--eval-max-snapshots N`**| **0** | **Max eval checkpoints kept on disk (0 for unlimited; prevents disk overflow)** |

---

## Directory Structure

```
scripts/       One-click background launch, stop, and monitoring scripts (Linux & Windows)
src/ddz/       Card rules, 15 combination validations, 621-dim action space, Oracle DFS solver
src/torch_nn/  LibTorch neural networks (LSTM, Actor, Critic), PPO learner, tensor serialization
src/algo/      Self-play rollout collector, GAE estimator, league pool manager, DouZero client
src/           Main training binary train.cpp, standalone evaluation eval_douzero.cpp
tools/         DouZero TCP server douzero_serve.py, evaluation plotter plot_eval.py, benchmarks
tests/         Google Test suite (game rules, rollout, forward/backward, PPO, NaN protection)
```
