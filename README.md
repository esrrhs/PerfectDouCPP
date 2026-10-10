# PerfectDouCPP

[![CI](https://github.com/esrrhs/PerfectDouCPP/actions/workflows/ci.yml/badge.svg)](https://github.com/esrrhs/PerfectDouCPP/actions/workflows/ci.yml)
[![Language](https://img.shields.io/badge/language-C%2B%2B17-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B17)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)](https://github.com/esrrhs/PerfectDouCPP)

**PerfectDouCPP** 是一套高性能、工业级的纯 C++17 斗地主强化学习端到端自对弈训练与评测系统。

### 算法渊源与致敬
本项目在理论思想和规则体系上深度借鉴了斗地主强化学习领域的两项标杆工作：
- **[DouZero](https://github.com/kwai/DouZero)**（ICML 2021, *Mastering DouDizhu with Self-Play Deep Reinforcement Learning*）：奠定了现代斗地主 AI 的基石。我们借鉴了其完备的腾讯斗地主 15 种牌型解析/压牌逻辑与特征张量化思想，并在系统内直接将其官方 DouZero-ADP 模型作为衡量我方策略水平的核心对战基线；
- **[PerfectDou](https://github.com/Vincent-Guan/PerfectDou)**（NeurIPS 2022, *Dominating DouDizhu with Perfect Information Distillation*）：开创性地提出了非完全信息博弈下的**完美信息蒸馏（Perfect Information Distillation）**范式。我们借鉴了其全知 Critic（全知手牌价值估计）指导不完全信息 Actor 的核心思路，以及 621 维抽象动作空间的设计。

在上述前沿学术研究的启发下，本项目重点进行了**全栈工业级工程落地与算法现代化重构**：抛弃了重型分布式集群与复杂外部依赖，纯 C++17 原生从零构建了超高速牌局引擎、多层正反向神经网络算子、双农民严格同策的分层联赛对手池（League Training）、工业级防 NaN 熔断与动态 KL 自适应早停机制，并直接内置了对 DouZero 的异步对战评测闭环。单机多线程即可实现极高吞吐与稳定收敛。

---

## 训练架构与从头到尾的流程

整个强化学习自对弈训练流程为一个高内聚、流水线化的闭环：

```
┌────────────────────────────────────────────────────────────────────────┐
│                        1. 牌局引擎与特征工程                             │
│   腾讯规则 (15种牌型)  ──►  621动作空间解码  ──►  最小步数 Oracle 记忆化缓存 │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   2. 多线程自对弈数据采集 (Rollout)                      │
│  三个座位独立决策 (地主/下家/上家) ◄── 阵营协同保障 (双农民绑定同策略快照)     │
│  对手池混合采样: 当前最新策略 + 滚动快照池 + 几何稀疏归档池 + 规则智能体基线     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   3. 终局收益结算与优势估计 (GAE)                         │
│  终局 ADP 奖励 (炸弹翻倍结算) ──► GAE (γ=1.0, λ=0.95) ──► 全局优势标准化  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 4. 完美信息蒸馏与并行 PPO 优化 (Learn)                  │
│  Critic 全知价值蒸馏 ──► 策略裁剪 L_CLIP ──► 余弦退火 (lr & 熵) ──► Adam   │
│  数值防爆熔断: Masked Softmax 防 NaN ──► 动态 KL 散度自适应早停         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 5. 策略热更新、模型落盘与断点续接                        │
│  原子更新当前策略 ──► 周期性落盘完整 Checkpoint (权重 + Adam 动量)       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│            6. 内置后台异步 DouZero 评测与趋势监控 (Eval Loop)            │
│  零开销深拷贝模型 ──► 独立后台线程对弈 (正反手 200 局) ──► 写入 CSV/快照  │
│  自动滚动淘汰旧快照 (硬防爆盘) ──► plot_eval.py 实时渲染胜率/ADP 曲线    │
└────────────────────────────────────────────────────────────────────────┘
```

### 1. 牌局引擎与特征工程
- **规则与合法动作**：严格实现腾讯斗地主 15 种合法牌型（单张、对子、三带、顺子、连对、飞机、炸弹、王炸等），动作空间映射为 621 维抽象动作，并通过算法快速解码最优实际出牌。
- **最小出牌步数 Oracle**：内置动态规划与记忆化深度优先搜索（DFS）的求解器，工作线程配备私有缓存，能够毫秒级评估任意手牌清牌所需的最少手数。
- **双重特征视角**：
  - **不完美信息观察（用于 Actor）**：各家打牌历史序列矩阵（送入 LSTM 捕捉时序动态）+ 手牌与公开底牌特征矩阵。
  - **完美信息观察（用于 Critic）**：包含所有玩家手牌的全知特征，赋予评论家全局评估能力。

### 2. 多线程自对弈采集（Rollout）
- **独立三座位决策**：地主（Landlord）、地主下家农民（Landlord Down）、地主上家农民（Landlord Up）拥有独立的神经网络。
- **阵营合作一致性（Team Consistency）**：在自对弈采样中，地主独立抽样对手，而两个农民座位严格绑定相同的历史策略快照，防止因不同版本农民混搭导致策略自相矛盾。
- **分层联赛机制（League Training）**：为防止自对弈陷入单一策略自转（Policy Cyclicality），对手由多层池子按概率混合组成：
  - 当前最新策略快照；
  - 滚动历史快照池（Rolling Pool，追踪近期迭代）；
  - 几何稀疏长期归档池（Long-term Archive，保留最早和各关键阶段的锚点基线）；
  - 经典启发式规则智能体（RuleAgent，维持基本人类牌理底线）。

### 3. 终局收益结算与优势估计（Credit Assignment）
- **ADP 终局奖励结算**：对局结束时，根据输赢及全局炸弹数量结算真实差分分值：地主为 $\pm 2 \times 2^{\text{bomb}}$，农民为 $\pm 1 \times 2^{\text{bomb}}$。
- **广义优势估计（GAE）**：默认采用 $\gamma=1.0, \lambda=0.95$ 计算广义优势，有效降低长程轨迹多步决策的方差，辅以微量残局清牌引导 Shaping。
- **优势标准化（Advantage Normalization）**：对整批轨迹的优势值做均值与方差归一化，稳定不同局况下的梯度更新尺度。

### 4. 完美信息蒸馏与并行 PPO 优化（Learn）
- **完美信息价值蒸馏（PTIE）**：Critic 接收包含三家手牌的全知特征，评估出的无偏状态价值通过优势函数指导仅能看到不完全信息的 Actor，完成隐式蒸馏。
- **多座位并行更新**：三个座位的训练批次完全解耦，可在多线程下并行计算反向传播与梯度更新。
- **自适应早停与余弦退火**：每个 Epoch 评估真实均值 KL 散度，超出 $1.5 \times \text{targetKL}$ 立即提前终止，防止策略突变崩溃；学习率与策略熵均沿余弦曲线平滑退火。
- **工业级防爆与数值熔断**：
  - **Masked Softmax 防 NaN**：当所有合法动作分数为极小值或溢出时，自动兜底均匀分布；
  - **KL 散度边界保护**：防除零、防对数无穷大与非法概率裁剪；
  - **训练样本清洗**：自动跳过包含非有限浮点数的脏样本；
  - **原子动量存取**：Adam 优化器一阶/二阶动量安全序列化。

### 5. 策略热更新与模型落盘
- 训练完成后，最新权重无锁推进至自对弈池；
- 按照 `--snapshot-every` 定期向输出目录保存完整检查点（包含 3 个座位的 Actor 权重、Critic 权重以及对应的 Adam 动量文件），支持随时通过 `--resume` 续训。

### 6. 内置后台 DouZero 评测与胜率趋势监控
- **原生内嵌对战评测**：每达一定步数（推荐 200~500 轮），训练进程自动将当前三个 Actor 深拷贝，扔进独立后台线程，并通过跨平台 Socket 直连开源 DouZero 服务展开 200 局（100 副牌正反手互换）对抗评测。
- **快照存档与自动防爆盘**：评测触发时自动将模型备份到快照目录；提供 `--eval-max-snapshots` 参数自动滚动淘汰最旧快照，彻底杜绝超长周期训练爆满磁盘。
- **可视化趋势呈现**：评测结果自动追加到 CSV，可使用轻量绘图工具 `plot_eval.py` 生成包含平滑拟合线的 WP（胜率）与 ADP 趋势图。

---

## 构建与测试

项目使用 CMake 构建，内置 Google Test 单元测试套件（34 项高覆盖度测试），跨平台原生支持 Linux (GCC/Clang)、macOS (Apple Clang/Accelerate) 与 Windows (MSVC/CUDA)。

```bash
# 配置并编译
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 运行完整 Google Test 单元测试套件（34 项全部通过）
ctest --test-dir build --output-on-failure
```

测试套件覆盖出牌规则合法性、15 种牌型解析、621 维动作解码、Oracle 状态机、多层神经网络手写数值与解析梯度核验（96 项）、PPO 优化稳定性、防 NaN 熔断机制、并发多车道无死锁压力测试以及 DouZero 对弈快照回溯。

---

## 训练指南

### 1. 启动对弈基准服务端（可选但推荐）
如需在训练过程中定期与官方 DouZero-ADP 进行基准对抗评测，先启动基于 TCP 的 DouZero 推理服务（纯 CPU 即可运行，不占用训练 GPU）：

```bash
# 需准备 third_party/DouZero 与 baselines/douzero_ADP/*.ckpt
export PYTHONPATH=third_party/DouZero   # Windows: set PYTHONPATH=...
python tools/douzero_serve.py --port 18765 --ckpt-dir baselines/douzero_ADP
```

### 2. 启动从头到尾训练

```bash
# 标准自对弈训练（开启内置后台定期评测与快照归档）
./build/perfectdou_train --updates 100000 --games 256 --threads 10 --out ckpt \
    --eval-every 500 --eval-decks 100 --eval-port 18765 --eval-save-dir eval_snapshots

# 快速验证 / 低配机器调试
./build/perfectdou_train --updates 300 --games 128 --threads 8 \
    --hidden 128 --epochs 2 --out ckpt
```

> **硬件加速后端说明**：
> - **Apple Silicon (macOS)**：默认走 Accelerate / AMX 硬件加速矩阵乘，LSTM 走 vForce；
> - **Windows / Linux**：`--backend auto` 优先检测并使用 CUDA（cuBLAS GEMM），无 GPU 时自动回退至高优化 CPU 向量化路径；
> - 自对弈采样在 x86-64 上固定走 AVX2/FMA 指令集，兼顾高吞吐与多核并发。

### 3. 训练日志与监控指标

训练过程中每轮输出结构化监控日志：

```
upd   500 | q 1/1 left 0 end 1 | wall 4.2s rollout 0.5s learn 3.7s games 256 mb 1024 ep 4,4,4/4 threads 10 | WP 0.532 ADP   0.15 (pure WP 0.548 ADP   0.21 [180 g]) bomb/g 0.32 moves/g 53.4 | lr 2.95e-04 ent 1.082/0.892/0.890 vL    1.21/   0.38/   0.39 kl 0.0021/0.0018/0.0019 cf 0.05/0.04/0.04 | ret     0.1/   -0.0/   -0.0 | n 4820/3610/3590
[eval u500] done games 200 | WP 0.545 ADP 0.230 | landlord WP 0.560 ADP 0.310 | peasant WP 0.530 ADP 0.150
```

- **WP / ADP**：混合对手对局下的地主胜率与平均分差；
- **pure WP / pure ADP**：仅统计最新模型之间纯自对弈的胜率与均分（剔除规则与历史对手干扰）；
- **ep**：三个座位实际执行的 PPO Epoch 轮数（若提前结束说明触发了 KL 早停保护）；
- **kl / cf**：近似 KL 散度与 PPO 策略裁剪触发比例；
- **[eval uX]**：后台 DouZero 评测的对局统计，包含总胜率、地主胜率与农民胜率。

### 4. 胜率趋势图实时渲染与 A/B 对照

使用轻量绘图脚本随时根据评测追加的 CSV 渲染胜率与 ADP 变化曲线：

```bash
# 单实验渲染静态图片
python tools/plot_eval.py --csv ckpt/eval_vs_douzero.csv --out ckpt/eval_curve.png

# 实时监视模式（每 30 秒自动重新绘图刷新）
python tools/plot_eval.py --csv ckpt/eval_vs_douzero.csv --out ckpt/eval_curve.png --watch 30
```

#### A/B 对照实验使用指引
系统天然支持完整的 A/B 对照实验目录隔离。通过 `--out <DIR>` 即可将所有实验产物（训练 Checkpoint、评测 CSV、评测模型快照）整体隔离在各自独立的目录下：

```bash
# 实验 A（基准实验）
./build/perfectdou_train --out exp_baseline --lr 3e-4 --eval-every 500

# 实验 B（新参数对照）
./build/perfectdou_train --out exp_lr_1e4 --lr 1e-4 --eval-every 500
```

每个实验会自动生成独立的产物：
- `exp_baseline/`：最终模型、`eval_vs_douzero.csv`、`eval_snapshots/`（包含各步数快照）；
- `exp_lr_1e4/`：最终模型、`eval_vs_douzero.csv`、`eval_snapshots/`。

若需要将两个或多个实验画在同一张图上进行直观对比：
```bash
# 一键生成 A/B 对比趋势图（支持任意多个实验）
python tools/plot_eval.py \
    --csv exp_baseline/eval_vs_douzero.csv exp_lr_1e4/eval_vs_douzero.csv \
    --labels "Baseline (3e-4)" "Low LR (1e-4)" \
    --out ab_comparison.png
```

---

## 常用训练参数参考

| 参数 | 默认值 | 作用与建议 |
| :--- | :--- | :--- |
| `--updates N` | 200 | 训练总更新步数 |
| `--games N` | 256 | 每次更新自对弈收集的牌局数（推荐 256～1024） |
| `--threads N` | CPU核数 | 自对弈并发采样与多座位 PPO 优化的并行线程数 |
| `--hidden H` | 256 | 神经网络隐层神经元宽度（128 可显著提升吞吐） |
| `--lstm-hidden H`| 128 | LSTM 隐层时序特征维度 |
| `--epochs N` | 4 | 每次更新每批数据的 PPO 迭代轮数 |
| `--mb N` | 1024 | PPO Mini-batch 大小 |
| `--lr N` | 3e-4 | 初始学习率（默认余弦退火，`--no-lr-decay` 保持固定） |
| `--ent N` | 0.03 | 策略熵正则系数（默认余弦退火，`--no-ent-decay` 保持固定） |
| `--clip N` | 0.2 | PPO 策略概率比率截断阈值 |
| `--target-kl N` | 0.03 | 自适应 KL 散度早停阈值（超过自动截断 Epoch，0 为禁用） |
| `--gamma / --lambda` | 1.0 / 0.95 | GAE 折扣因子与方差权衡参数 |
| `--pool-size N` | 16 | 联赛滚动快照池大小 |
| `--archive-size N`| 16 | 联赛长期几何稀疏归档池大小 |
| `--pool-every K` | 20 | 每 K 轮向联赛历史池归档一次最新模型 |
| `--historical-prob`| 0.2 | 自对弈中抽样历史模型的概率 |
| `--rule-prob` | 0.1 | 自对弈中抽样经典规则智能体的概率 |
| `--snapshot-every K`| 20 | 检查点落盘间隔（保存完整权重及优化器动量） |
| `--resume DIR` | 空 | 从指定目录恢复模型权重及 Adam 动量续训 |
| **`--eval-every N`** | **0** | **内置 DouZero 评测间隔（推荐 200～500，0 为关闭）** |
| **`--eval`** | - | **快捷开启内置评测（等价于 `--eval-every 500`）** |
| **`--eval-decks D`** | **100** | **每次评测副数（默认 100 副正反手互换即 200 局）** |
| **`--eval-port P`** | **18765**| **DouZero TCP 服务端口** |
| **`--eval-max-snapshots N`**| **0** | **磁盘最大保留快照数（0 为不限；设置后自动滚动淘汰旧快照硬防爆盘）** |

> **大规模训练（如 25 亿帧样本）磁盘规划建议**：
> - 25 亿帧样本折合约 160,000 次 Update（按每次 256 局折算）；
> - 每次快照（3 个 Actor）占用约 **12.8 MB**；
> - 推荐设置 `--eval-every 500`：整个训练产生约 320 个高密度评测点，磁盘快照总大小仅约 **4 GB**，且评测在后台每隔 15~30 分钟进行一次（耗时 ~20 秒），对主训练吞吐几乎零影响；
> - 若磁盘空间极度紧张，可直接指定 `--eval-max-snapshots 100`，训练将始终只保留最新的 100 份模型。

---

## 目录结构

```
src/ddz/      出牌规则、15 种牌型判定/生成、621 维抽象动作空间、Oracle 记忆化求解器、特征工程
src/nn/       基础矩阵算子、GEMM 加速、Linear/ReLU/LSTM、Actor/Critic 神经网络、Adam 优化器
src/algo/     自对弈 Rollout 采集器、GAE 优势估计、PPO 策略更新、内置 DouZero 对战评测驱动与快照
src/          训练主入口 train.cpp、独立评测入口 eval_douzero.cpp
tools/        DouZero TCP 服务端 douzero_serve.py、胜率曲线绘制 plot_eval.py
tests/        Google Test 单元测试套件（规则、自对弈、反向传播梯度、PPO 算法、防爆熔断、Mock 评测）
```

---

## 参考文献与基准

1. **DouZero**: Daochen Zha, et al. *"DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement Learning"*, ICML 2021. [[Paper]](https://arxiv.org/abs/2106.06135) [[GitHub]](https://github.com/kwai/DouZero)
2. **PerfectDou**: Yang Guan, et al. *"PerfectDou: Dominating DouDizhu with Perfect Information Distillation"*, NeurIPS 2022. [[Paper]](https://arxiv.org/abs/2203.16406) [[GitHub]](https://github.com/Vincent-Guan/PerfectDou)
