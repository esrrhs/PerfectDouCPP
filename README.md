# PerfectDouCPP

[![CI](https://github.com/esrrhs/PerfectDouCPP/actions/workflows/ci.yml/badge.svg)](https://github.com/esrrhs/PerfectDouCPP/actions/workflows/ci.yml)
[![Language](https://img.shields.io/badge/language-C%2B%2B17-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B17)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)](https://github.com/esrrhs/PerfectDouCPP)

**PerfectDouCPP** 是一套高性能、工业级的纯 C++17 斗地主强化学习端到端自对弈训练与评测系统。

### 算法渊源与致敬
本项目在理论思想和规则体系上深度借鉴了斗地主强化学习领域的两项标杆工作：
- **[DouZero](https://github.com/kwai/DouZero)**（ICML 2021, *Mastering DouDizhu with Self-Play Deep Reinforcement Learning*）：奠定了现代斗地主 AI 的基石。我们借鉴了其完备的腾讯斗地主 15 种牌型解析/压牌逻辑与特征张量化思想，并在系统内直接将其官方 DouZero-ADP 模型作为衡量我方策略水平的核心对战基线；
- **[PerfectDou](https://github.com/Vincent-Guan/PerfectDou)**（NeurIPS 2022, *Dominating DouDizhu with Perfect Information Distillation*）：开创性地提出了非完全信息博弈下的**完美信息蒸馏（Perfect Information Distillation）**范式。我们借鉴了其全知 Critic（全知手牌价值估计）指导不完全信息 Actor 的核心思路，以及 621 维抽象动作空间与 Oracle 差分奖励的设计。

在上述前沿学术研究的启发下，本项目重点进行了**全栈工业级工程落地与现代化重构**：基于 LibTorch C++ API 构建原生神经网络，设计了超高速牌局引擎、多线程 CPU 并发 Rollout 采样、GPU 同步 PPO 优化、双农民严格同策的分层联赛对手池（League Training）、工业级防 NaN 熔断与动态 KL 自适应早停机制，并直接内置了对 DouZero 的异步对战评测闭环与完整断点续接。单机多线程即可实现极高吞吐与稳定收敛。

---

## 训练架构与流水线流程

整个强化学习自对弈训练流程为一个高内聚、流水线化的同步闭环：

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
│  分层联赛对手池混合采样: 当前最新策略 + 滚动快照池 + 几何稀疏归档池           │
│  论文 Sec 4.4 Oracle 密集差分奖励 (ΔAdv) 步级塑形                       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   3. 终局收益结算与优势估计 (GAE)                         │
│  终局 ADP 零和奖励 (炸弹翻倍结算) ──► GAE (γ=1.0, λ=0.95) ──► 优势标准化   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 4. 完美信息蒸馏与同步 PPO 优化 (Learn)                  │
│  Critic 全知价值蒸馏 ──► 策略比率截断 L_CLIP ──► 动量解耦双 Adam 优化器   │
│  数值防爆熔断: Masked Softmax 防 NaN ──► 动态 KL 散度自适应早停         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 5. 策略同步、完整检查点落盘与断点续接                    │
│  GPU 训练权重同步回 CPU 采样实例 ──► 周期性落盘完整 Checkpoint            │
│  (Actor/Critic 权重 + Adam 动量 m/v/t + 更新步数 + 历史对手池完整快照)  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│            6. 内置后台异步 DouZero 评测与趋势监控 (Eval Loop)            │
│  零开销深拷贝模型 ──► 独立后台线程对弈 (正反手镜像对调) ──► 写入 CSV/快照  │
│  自动滚动淘汰旧快照 (硬防爆盘) ──► plot_eval.py 实时渲染胜率/ADP 曲线    │
└────────────────────────────────────────────────────────────────────────┘
```

### 1. 牌局引擎与特征工程
- **规则与合法动作**：严格实现腾讯斗地主 15 种合法牌型（单张、对子、三带、顺子、连对、飞机、四带二、炸弹、王炸等），动作空间映射为 621 维抽象动作，并通过算法快速解码最优实际出牌。
- **最小出牌步数 Oracle**：内置动态规划与记忆化深度优先搜索（DFS）求解器，工作线程配备私有缓存，能够毫秒级评估任意手牌清牌所需的最少手数。
- **双重特征视角**：
  - **不完美信息观察（用于 Actor）**：各家打牌历史序列矩阵（送入 LSTM 捕捉时序动态，10 步 × 540 维）+ 手牌与公开底牌特征矩阵（1676 维）。
  - **完美信息观察（用于 Critic）**：包含所有玩家手牌的全知特征矩阵与手牌最小步数（362 维），赋予评论家全局评估能力。

### 2. 多线程自对弈采集（Rollout）
- **独立三座位决策**：地主（Landlord）、地主下家农民（Landlord Down）、地主上家农民（Landlord Up）拥有独立的神经网络。
- **阵营合作一致性（Team Consistency）**：在自对弈采样中，地主独立抽样对手，而两个农民座位严格绑定相同的历史策略快照，防止因不同版本农民混搭导致策略自相矛盾。
- **分层联赛机制（League Training）**：为防止自对弈陷入单一策略自转（Policy Cyclicality），对手按概率混合采样：
  - 当前最新策略快照；
  - 滚动历史快照池（Rolling Pool，追踪近期迭代）；
  - 几何稀疏长期归档池（Long-term Archive，保留最早和各关键阶段的锚点基线）。
- **Oracle 密集差分奖励（Dense Oracle Reward）**：严格遵循 PerfectDou 论文 Section 4.4，在对局每一步计算全知最小手数差值变化 $\Delta Adv = \Delta (n_L - \min(n_D, n_U))$ 作为即时塑形奖励，引导智能体快速习得合规且高效的牌理。

### 3. 终局收益结算与优势估计（Credit Assignment）
- **ADP 终局奖励结算**：对局结束时，根据输赢及全局炸弹数量结算真实差分分值：地主为 $\pm 2 \times 2^{\text{bomb}}$，农民为 $\pm 1 \times 2^{\text{bomb}}$，严格满足零和博弈。
- **广义优势估计（GAE）**：默认采用 $\gamma=1.0, \lambda=0.95$ 计算广义优势，有效降低长程轨迹多步决策的方差。
- **优势标准化（Advantage Normalization）**：对整批轨迹的优势值做均值与方差归一化，稳定不同局况下的梯度更新尺度。

### 4. 完美信息蒸馏与同步 PPO 优化（Learn）
- **完美信息价值蒸馏（PTIE）**：Critic 接收包含三家手牌的全知特征，评估出的无偏状态价值通过优势函数指导仅能看到不完全信息的 Actor，完成隐式蒸馏。
- **同步无偏优化**：摒弃存在严重策略陈旧度的异步双缓冲队列，采用纯同步 On-policy 循环：Rollout 收集当前策略最新轨迹 $\to$ GPU 批量优化更新 $\to$ 权重同步更新 CPU 模型，学习稳定性大幅跃升。
- **自适应早停**：每个 Epoch 评估真实均值 KL 散度，超出 $1.5 \times \text{targetKL}$ 立即提前终止，防止策略突变崩溃。
- **数值安全熔断**：
  - **Masked Softmax 防 NaN**：当所有合法动作分数为极小值或溢出时，自动兜底均匀分布；
  - **KL 散度边界保护**：防除零、防对数无穷大与非法概率裁剪；
  - **训练样本清洗**：自动跳过包含非有限浮点数的脏样本；
  - **梯度裁剪解耦**：Actor 采用 $0.5$ 范数截断，Critic 针对大炸弹分差采用 $5.0$ 范数截断，防止价值梯度爆炸。

### 5. 策略同步、检查点落盘与断点续接
- 训练完成后，GPU/MPS 优化后的最新权重直接同步到 CPU 实例供后续采集使用；
- 按照 `--snapshot-every` 定期向输出目录保存完整检查点（包含 3 个座位的 Actor 权重、Critic 权重、对应的 Adam 动量文件 $m/v/t$、当前 Update 步数以及完整的联赛历史对手池快照及元数据）；
- 支持随时通过 `--resume <DIR>` 100% 无损恢复状态继续训练。

### 6. 内置后台 DouZero 评测与胜率趋势监控
- **原生内嵌对战评测**：每达一定步数（默认 200 轮），训练进程自动将当前三个 Actor 深拷贝，送入独立后台线程，并通过 Socket 直连 DouZero 推理服务展开镜像对局（换边互换对弈）。
- **快照存档与自动防爆盘**：评测触发时自动将模型备份到快照目录；提供 `--eval-max-snapshots` 参数自动滚动淘汰最旧快照，彻底杜绝超长周期训练爆满磁盘。
- **可视化趋势呈现**：评测结果自动追加到 CSV，可使用轻量绘图工具 `plot_eval.py` 生成包含平滑拟合线的 WP（胜率）与 ADP 趋势图。

---

## 构建与测试

项目使用 CMake 构建，基于 LibTorch 原生开发，内置 Google Test 单元测试套件（30 项核心测试），跨平台支持 Linux、macOS 与 Windows。

```bash
# 配置并编译 (Release 模式)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 运行完整 Google Test 单元测试套件（30 项全部通过）
ctest --test-dir build --output-on-failure
```

测试套件覆盖出牌规则合法性、15 种牌型解析、621 维动作解码、Oracle 记忆化 DP 状态机、多层神经网络 LibTorch 前后向计算、GAE/PPO 算法数值稳定性、断点序列化与完整恢复、防 NaN 熔断机制以及 DouZero 对弈协议。

---

## 训练指南

### 1. 启动对弈基准服务端（可选但推荐）
如需在训练过程中定期与官方 DouZero-ADP 进行基准对抗评测，先启动基于 TCP 的 DouZero 推理服务（纯 CPU 即可运行，不占用训练 GPU）：

```bash
# 准备 third_party/DouZero 与 baselines/douzero_ADP/*.ckpt
export PYTHONPATH=third_party/DouZero   # Windows: set PYTHONPATH=...
python tools/douzero_serve.py --port 18765 --ckpt-dir baselines/douzero_ADP
```

### 2. 启动从头到尾训练

```bash
# 标准自对弈训练（开启 GPU 加速与后台定期评测）
./build/perfectdou_train --backend cuda --updates 100000 --games 768 --threads 16 --out ckpt \
    --eval-every 200 --eval-decks 1000 --eval-port 18765 --eval-save-dir eval_snapshots

# Apple Silicon (macOS) 运行
./build/perfectdou_train --backend mps --updates 50000 --games 512 --threads 10 --out ckpt

# 快速验证 / CPU 调试
./build/perfectdou_train --backend cpu --updates 300 --games 128 --threads 8 --out ckpt_debug
```

> **硬件加速后端说明**（`--backend` 必须显式指定，可选 `cuda` / `mps` / `cpu`）：
> - **NVIDIA GPU (Linux / Windows)**：使用 `--backend cuda`，LibTorch 将在 GPU 上执行 PPO 反向传播与优化；CPU 多线程负责高吞吐 Rollout 采样；
> - **Apple Silicon (macOS)**：使用 `--backend mps` 调用 Metal GPU 加速；
> - **纯 CPU 平台**：使用 `--backend cpu`。

### 3. 训练日志与监控指标

训练过程中每轮输出结构化监控日志：

```
upd  500 | wall 3.8s learn 0.42s games 768 mb 2048 ep 2,2,2/2 | WP 0.532 ADP   0.15 (pure WP 0.548 ADP   0.21 [550 g]) bomb/g 0.35 moves/g 51.2 | lr 3.00e-04 ent 0.852/0.710/0.708 vL    1.15/   0.32/   0.34 kl 0.0018/0.0015/0.0016 cf 0.04/0.03/0.03 | train_n 14820/11610/11590
[eval u500] done games 2000 | WP 0.565 ADP 0.380 | landlord WP 0.580 ADP 0.450 | peasant WP 0.550 ADP 0.310
```

- **WP / ADP**：混合对手对局下的地主胜率与平均分差；
- **pure WP / pure ADP**：仅统计最新模型之间纯自对弈的胜率与均分（剔除历史对手干扰）；
- **ep**：三个座位实际执行的 PPO Epoch 轮数（若小于设定值说明触发了 KL 早停保护）；
- **kl / cf**：近似 KL 散度与 PPO 策略裁剪触发比例；
- **[eval uX]**：后台 DouZero 评测的对局统计，包含总胜率、地主胜率与农民胜率。

### 4. 胜率趋势图实时渲染与 A/B 对照

使用绘图脚本随时根据评测追加的 CSV 渲染胜率与 ADP 变化曲线：

```bash
# 单实验渲染静态图片
python tools/plot_eval.py --csv ckpt/eval_vs_douzero.csv --out ckpt/eval_curve.png

# 实时监视模式（每 30 秒自动重新绘图刷新）
python tools/plot_eval.py --csv ckpt/eval_vs_douzero.csv --out ckpt/eval_curve.png --watch 30
```

#### A/B 对照实验使用指引
通过 `--out <DIR>` 将实验产物隔离在各自独立目录下：
```bash
# 实验 A（基准实验）
./build/perfectdou_train --out exp_baseline --lr 3e-4 --eval-every 200

# 实验 B（新参数对照）
./build/perfectdou_train --out exp_lr_1e4 --lr 1e-4 --eval-every 200

# 一键生成 A/B 对比趋势图
python tools/plot_eval.py \
    --csv exp_baseline/eval_vs_douzero.csv exp_lr_1e4/eval_vs_douzero.csv \
    --labels "Baseline (3e-4)" "Low LR (1e-4)" \
    --out ab_comparison.png
```

---

## 训练 Tips：对战 DouZero 胜率与调优指南

以下为实战中压制 DouZero、提升胜率与稳定收敛的核心经验总结：

### 1. Oracle 差分奖励的两阶段使用（密集引导 vs 纯 ADP 零和目标）
* **原理**：
  * **第一阶段（启动与中前期，前 20,000 ~ 50,000 轮）**：开启 Oracle 奖励（`--oracle-reward 1 --oracle-coef 0.2`）。Oracle 的 $\text{minSteps}$ 步数差分能在对局每一步提供明确的牌理梯度，避免模型在稀疏奖励下盲目出牌，使地主和农民以极快速度学会管牌、组顺子和清牌。
  * **第二阶段（后期冲刺极致胜率）**：由于 Oracle 仅考虑最小出牌步数，不感知炸弹翻倍（$2^{\text{bombCount}}$），且不完全契合农民“顶大牌牺牲自己掩护队友”的高级配合，在训练后期建议调小 Oracle 系数（如 `--oracle-coef 0.05`）或纯使用 ADP 奖励。这能让策略专注于炸弹博弈和团队最大净收益，对战 DouZero 杀伤力达到顶峰。

### 2. 算力规模与训练时长认知
* **DouZero 的底蕴**：DouZero 官方 ADP 权重是在数百核 CPU/GPU 集群上通过 Deep Monte Carlo（DMC）训练了数亿游戏步长（数百天等效单卡时间）凝练而成，残局算力极高。
* **PerfectDou 的破局点**：PerfectDou 拥有全知 Critic 价值蒸馏、621 维抽象动作空间及 10 步 LSTM 历史感知，在特征深度上完全超越 DouZero。但强化学习需要足够的采样体量（通常建议训练数万个 updates，积累数千万局对局经验）。在训练初期（如前几千步），对弈胜率可能存在一定波动，达到充分迭代后胜率将出现稳定的压制性上升。

### 3. 评测协议与双盲镜像换边
* 评测模块默认启用**换边对局（Duplicate Match）**：同一副牌上下半场阵容对调（上半场我方地主 vs DouZero 农民，下半场 DouZero 地主 vs 我方农民），彻底消除发牌运气成分；
* 评测推断统一采用确定性贪心（Greedy Argmax），完全对标 DouZero 官方评测标准。

### 4. 保持固定的探索性与平稳学习率
* 斗地主自对弈博弈空间极大，取消硬编码的余弦退火、保持适度的策略熵系数（默认 `0.03`）能防止策略过早收敛到局部僵化的打法；配合分层历史联赛池（20% 概率抽查历史模型），有效避免策略循环相克。

---

## 常用训练参数参考

| 参数 | 默认值 | 作用与建议 |
| :--- | :--- | :--- |
| `--backend <B>` | `cpu` | **硬件加速后端（可选 `cuda` / `mps` / `cpu`）** |
| `--updates N` | 100000 | 训练总更新步数 |
| `--games N` | 768 | 每次更新多线程收集的对局数（推荐 512～1024） |
| `--threads N` | CPU核数 | 自对弈并发采样线程数 |
| `--hidden H` | 256 | MLP 隐层神经元宽度（标准 PerfectDou 规模） |
| `--lstm-hidden H`| 128 | LSTM 隐层时序特征维度 |
| `--epochs N` | 2 | 每次更新每批数据的 PPO 迭代轮数 |
| `--mb N` | 2048 | PPO Mini-batch 大小 |
| `--lr N` | 3e-4 | 初始学习率（Adam 优化器） |
| `--ent N` | 0.03 | 策略熵正则系数（维持稳定探索） |
| `--clip N` | 0.2 | PPO 策略概率比率截断阈值 $\epsilon$ |
| `--target-kl N` | 0.03 | 自适应 KL 散度早停阈值（超出后截断当前 Epoch 迭代） |
| `--gamma / --lambda` | 1.0 / 0.95 | GAE 折扣因子与方差权衡参数（斗地主为无折扣单局博弈） |
| **`--oracle-reward`** | **1** | **开启论文 Sec 4.4 Oracle 步数差分密集奖励** |
| **`--oracle-coef`** | **0.2** | **Oracle 差分奖励缩放系数（后期精调可调至 0.05 或 0）** |
| `--pool-size N` | 16 | 联赛滚动快照池大小 |
| `--archive-size N`| 16 | 联赛长期几何稀疏归档池大小 |
| `--pool-every K` | 20 | 每 K 轮向联赛历史池归档一次最新模型 |
| `--historical-prob`| 0.2 | 自对弈中抽样历史模型的概率 |
| `--snapshot-every K`| 50 | 检查点落盘间隔（保存完整权重、优化器动量与对手池） |
| `--resume DIR` | 空 | 从指定目录恢复模型权重、优化器动量及历史对手池续训 |
| **`--eval-every N`** | **200** | **内置 DouZero 评测间隔（推荐 200～500，0 为关闭）** |
| **`--eval-decks D`** | **1000** | **每次评测副数（1000 副正反手互换即 2000 局）** |
| **`--eval-port P`** | **18765**| **DouZero TCP 服务端口** |
| **`--eval-max-snapshots N`**| **0** | **磁盘最大保留快照数（0 为不限；设置后自动滚动淘汰旧快照硬防爆盘）** |

---

## 目录结构

```
src/ddz/       出牌规则、15 种牌型判定/生成、621 维抽象动作空间、Oracle 记忆化求解器、特征工程
src/torch_nn/  基于 LibTorch 的神经网络（LSTM、Actor、Critic）、PPO 学习器、张量桥接与参数持久化
src/algo/      自对弈 Rollout 采集器、GAE 优势估计、分层对手池管理、内置 DouZero 对战评测驱动
src/           训练主入口 train.cpp、独立评测入口 eval_douzero.cpp
tools/         DouZero TCP 服务端 douzero_serve.py、胜率曲线绘制 plot_eval.py、评测牌堆转换 convert_eval_data.py
tests/         Google Test 单元测试套件（规则、自对弈、前后向计算、PPO 算法、防爆熔断、Mock 评测）
```

---

## 参考文献与基准

1. **DouZero**: Daochen Zha, et al. *"DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement Learning"*, ICML 2021. [[Paper]](https://arxiv.org/abs/2106.06135) [[GitHub]](https://github.com/kwai/DouZero)
2. **PerfectDou**: Yang Guan, et al. *"PerfectDou: Dominating DouDizhu with Perfect Information Distillation"*, NeurIPS 2022. [[Paper]](https://arxiv.org/abs/2203.16406) [[GitHub]](https://github.com/Vincent-Guan/PerfectDou)
