# PerfectDouCPP

[![CI](https://github.com/esrrhs/PerfectDouCPP/actions/workflows/ci.yml/badge.svg)](https://github.com/esrrhs/PerfectDouCPP/actions/workflows/ci.yml)
[![Language](https://img.shields.io/badge/language-C%2B%2B17-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B17)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)](https://github.com/esrrhs/PerfectDouCPP)

斗地主 AI **PerfectDou**（NeurIPS 2022, *Dominating DouDizhu with Perfect
Information Distillation*）的纯 C++17 工业级强化学习完整实现：零第三方依赖（仅
pthread），包含超高速牌局引擎、特征工程、神经网络反向传播、PPO 自对弈与分层联赛（League Training）。

## 与论文的对应关系与工业级增强

| 论文 / 算法组件 | 本实现细节与工业级增强 |
| --- | --- |
| 牌局规则（附录 B，腾讯规则） | `src/ddz/moves.cpp`（15 种牌型，移植 DouZero 的生成/识别/压牌逻辑） |
| 621 抽象动作 + 解码（附录 E.2） | `src/ddz/action_space.{h,cpp}`，Algo 2 kicker 打分 |
| 不完美信息特征 24 个牌矩阵 + 手牌张数/炸弹 one-hot | `src/ddz/features.cpp`（包含公共底牌变化追踪） |
| 完美信息特征 +2 手牌 +2 步数（critic） | 同上 |
| 动作特征为实际打出的牌（含带牌）+6 | `legalOptions()` 动态计算 |
| 最小出牌步数 oracle（附录 E.1） | `src/ddz/oracle.cpp`（DP + 记忆化 DFS，线程私有缓存，线程安全） |
| LSTM(5×540，即每步拼接 3 次出牌) + 对每个合法动作共享 MLP[256,256,256,512,1] | `src/nn/net.cpp`（支持 CPU AMX/AVX2；Windows 训练 GPU 为 CUDA cuBLAS + D3D12 其余核） |
| 完美信息价值网络 MLP[256×4] | `src/nn/net.cpp`（PTIE：完美 critic 通过优势蒸馏给不完美 actor） |
| 终局 ADP 收益目标与微量 Shaping | 终局收益为地主 $\pm 2 \times 2^{\text{bomb}}$，农民 $\pm 1 \times 2^{\text{bomb}}$；默认启用 GAE（$\lambda=0.95$）降低多步决策方差（可通过 `--lambda 1.0` 退化为纯蒙特卡洛 ADP），辅以微量残局 Shaping（`--shaping-cap 0.05`，引导清牌且不扰动胜负格局） |
| PPO 工业级策略裁剪与价值估计 | `src/algo/ppo.cpp`，实现策略裁剪 $L_{\text{CLIP}}$ 与优势值标准化（Advantage Normalization）；价值网络支持价值裁剪 $L_{\text{VF}}$（默认禁用 `--no-clip-vf`，避免绝对截断抑制炸弹翻倍大样本回归；可通过 `--clip-vf` 开启，阈值 `--vf-clip 32.0`） |
| 动态 KL 散度与自适应早停 | $\text{approxKL} = (r - 1) - \ln r$，每个 Epoch 独立评估均值，超过 $1.5 \times \text{targetKL}$ 触发早停，防止策略崩溃 |
| 分层联赛训练池（League Training） | `src/algo/rollout.cpp`，混合最近快照（Rolling Pool）、几何稀疏长期归档（Long-term Archive，锚定最早基线）与规则智能体（Heuristic RuleAgent），阵营级采样（地主独立抽样，两农民座位严格同步绑定同策略快照），保证农民合作一致性 |
| 三个座位独立模型、批量自对弈 | `src/algo/rollout.cpp`（多线程无锁快照推进，原子安全） |

论文的叫牌阶段与分布式集群（880 CPU + 8 GPU、25 亿帧）未实现：本项目只训练
出牌阶段（landlord 固定 20 张并先手，与 DouZero/PerfectDou 评估协议一致），
单机多线程即可跑出工业级吞吐与稳定收敛。

官方仓库没有公开训练代码，只提供评测代码、二进制特征编码器和最终 ONNX。
本实现对可验证部分同时对照论文和官方 ONNX；PPO epoch/clip/max-grad-norm 等论文
未披露的值采用常见 PPO2 默认值。

## 构建与测试

项目使用 CMake 构建，内置 Google Test 单元测试套件（34 项高覆盖度测试），跨平台支持 Linux (GCC/Clang)、macOS (Apple Clang/Accelerate) 与 Windows (MSVC/CUDA)。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure   # 运行全部 34 项 Google Test 单元测试
```

测试套件涵盖：
- **`RulesTest`**：出牌规则合法性、15 种牌型解析生成、621 维抽象动作空间、最小步数 Oracle 记忆化、随机自对弈状态机及特征编码；
- **`GradTest`**：反向传播数值与解析梯度校验（包含多层 MLP 与 LSTM 96 项核验）；
- **`AlgoTest`**：GAE/优势值归一化、PPO 策略/价值裁剪、自适应 KL 早停、Adam 动量原子序列化；
- **健壮性与防爆测试**：Masked Softmax 防 NaN 熔断防护、KL 散度防除零与无穷、训练样本非法值过滤跳过、PPO 并发多车道无死锁压力测试；
- **评测系统单测**：Actor 线程安全深拷贝、Mock DouZero 评测通信、CSV 写入与磁盘快照回溯验证。

## 训练

```bash
# macOS 默认走 Accelerate/AMX。
# Windows：--backend auto 优先 CUDA（cuBLAS GEMM + D3D12 其余核），否则 CPU。
# --backend cuda / --backend cpu 可强制切换（已移除会挂驱动的纯 D3D12 gpu 路径）
./build/perfectdou_train --updates 1000 --games 256 --threads 10 --out ckpt

# 开启内置定期 DouZero 评测与模型快照落盘（推荐间隔 200~500 步）
./build/perfectdou_train --updates 100000 --games 256 --threads 10 --out ckpt \
    --eval-every 500 --eval-decks 100 --eval-port 18765 --eval-save-dir eval_snapshots

# 快速/低配机器
./build/perfectdou_train --updates 300 --games 128 --threads 8 \
    --hidden 128 --epochs 2 --out ckpt
```

主要参数：

```
--games N           每次更新自对弈牌局数（默认 256）
--updates N         更新轮数
--threads N         自对弈线程数（三个座位的 PPO 更新也会并行）
--hidden H          MLP 宽度（论文 256；128 可显著提速）
--lstm-hidden H     LSTM 隐层（论文 128）
--epochs N          每批数据 PPO epoch 数（论文未披露；默认采用 PPO2 常见值 4）
--mb N              minibatch（论文 batch size 1024，8 卡时每卡 128；单机默认 1024）
--buffer N          rollout 队列深度（默认 1，对齐论文最大模型延迟 1）
--lr/--ent           初始学习率（默认 3e-4）和熵系数（默认 0.03），均默认按余弦曲线退火；`--no-ent-decay` 可固定熵系数
--no-lr-decay       禁用学习率余弦退火（保持固定学习率）
--gamma N           GAE 折扣因子 gamma（默认 1.0）
--lambda N          GAE 权衡参数 lambda（默认 0.95 降低多步决策方差；1.0 为纯蒙特卡洛终局 ADP）
--clip N            PPO clip（默认 0.2）
--target-kl N       动态 KL 散度早停阈值（默认 0.03，0 禁用）
--clip-vf           启用 PPO 价值函数裁剪损失（默认禁用 clipVf，避免炸弹高回报饱和）
--no-clip-vf        禁用 PPO 价值函数裁剪损失（默认）
--vf-clip N         PPO 价值函数裁剪阈值（默认 32.0，与斗地主 ADP 炸弹翻倍尺度相匹配）
--shaping-cap N     残局微量奖励塑造上限（默认 0.05，平局打破与出牌紧凑度引导）
--pool-size N       联赛最近滚动快照池大小（默认 16）
--archive-size N    联赛长期几何稀疏归档池大小（默认 16）
--pool-every K      每 K 轮向联赛历史池增加一次当前模型快照（默认 20）
--historical-prob P       自对弈中抽样历史对手的概率（默认 0.2）
--rule-prob P       自对弈中抽样经典规则智能体的概率（默认 0.1）
--snapshot-every K  每 K 轮落盘，最后一轮必存
--start-update U    指定起始更新轮数（默认 1，若从带有 meta.txt 的断点 resume 则自动续接）
--resume DIR        从 actor/critic 权重与 Adam 动量继续（支持自动续接进度与学习率）
--backend auto|cuda|cpu   GEMM 后端（Windows：cuda = cuBLAS GEMM + D3D12 其余核）

[内置对 DouZero 定期评测参数]
--eval-every N      每 N 轮更新触发一次与 DouZero 的对弈评测（默认 0 不开启；推荐 200~500）
--eval              快捷开启内置评测（默认每 500 轮评测一次）
--eval-decks D      每次评测副数（默认 100，正反手对换即 200 局）
--eval-host H       DouZero 服务 IP 地址（默认 127.0.0.1）
--eval-port P       DouZero 服务端口（默认 18765）
--eval-csv PATH     评测结果追加写入的 CSV 路径（默认 eval_vs_douzero.csv）
--eval-save-dir DIR 评测时自动归档模型权重的目录（默认 eval_snapshots，存放 u{step} 快照）
--eval-max-snapshots N 磁盘上保留的最大快照文件夹数量（默认 0 保存全部；>0 时自动滚动清理最老快照硬防爆盘）
--eval-sync         使用同步阻塞模式评测（默认异步后台线程评测，不阻塞主训练）
```

后端说明：

- Apple Silicon：默认 PPO 与自对弈都走 CPU。大矩阵乘法用 Accelerate（AMX），LSTM 的
  sigmoid/tanh 用 vForce。参考实测（M1 Pro，hidden=256 / 256 局 / epochs=4）：约
  **5 秒/轮**（采样 0.5 秒 + 学习 4.6 秒）。
- Windows：仅支持 `cuda` 与 `cpu`。`--backend auto` 在检测到 CUDA Toolkit / cuBLAS
  时启用 GPU，否则回退 CPU。手写 D3D12 `gemm_block` 路径已移除（部分 NVIDIA 驱动上
  会 TDR）。`--backend cuda` 要求编译时找到 CUDA headers，且运行时能加载
  `cublas`/`nvrtc`；其余 elementwise / LSTM 核仍走 D3D12。GTX 10xx（Pascal）请用
  **CUDA Toolkit 12.x**（13+ 已移除 sm_61）；Turing 及以上可用 12 或 13。x86-64
  自对弈采样固定走 AVX2/FMA CPU 路径，只有 PPO 学习上 GPU。
- 数值为 FP32。`tests/test_gemm` 校验 CPU/GPU GEMM，`tests/test_grad` 校验整网梯度。
- 没有 CUDA 时使用 CPU GEMM（Apple 为 Accelerate，ARM 为 NEON，其余为标量）。

输出文件：`ckpt/actor{0,1,2}.bin`、`ckpt/critic{0,1,2}.bin`、`ckpt/opt_actor{0,1,2}.bin`、`ckpt/opt_critic{0,1,2}.bin`、`ckpt/meta.txt`
（座位 0=landlord，1=landlord_down，2=landlord_up），供后续推理程序加载或断点续训。

每轮打印：
- **牌局统计**：WP（地主胜率）、ADP（均差分）、pure WP/ADP（两阵营纯最新自对弈指标）、炸弹率（bomb/g）、平均步数（moves/g）；
- **优化监控**：lr（当前学习率）、ent（三座位策略熵）、vL（三座位价值网络损失）；
- **稳定指标**：ep（三座位实际完成的 Epoch 数，显示早停状态）、kl（末轮近似 KL 散度）、cf（PPO Clip 触发比例）；
- **收益规模**：ret（三座位平均累积回报）、n（三座位样本数量）。

## 对 DouZero-ADP 周期性评测与胜率趋势图

### 1. 启动 DouZero 评测服务端
评测需要先启动官方开源 DouZero 模型的对弈服务（基于 TCP 通信，CPU 推理，默认端口 18765）：
```bash
# 准备 third_party/DouZero 与 baselines/douzero_ADP/*.ckpt
export PYTHONPATH=third_party/DouZero   # Windows: set PYTHONPATH=...
python tools/douzero_serve.py --port 18765 --ckpt-dir baselines/douzero_ADP
```

### 2. 训练内置评测与模型快照落盘（推荐）
在训练命令中直接加上 `--eval-every`（或快捷开关 `--eval`），训练进程会在达到指定步数时：
1. **自动克隆与落盘**：对当前 Actor 模型执行线程安全深拷贝，并立即保存权重到 `eval_snapshots/u<step>/actor{0,1,2}.bin` 与初始 `meta.txt`，供随时复盘回溯；
2. **异步后台对战**：在独立后台线程与 DouZero 服务进行每副牌正反手互换对打（默认 100 副牌共 200 局）；
3. **记录与回写**：评测结束后自动将最终 WP、ADP、分角色胜率等追加至 `eval_vs_douzero.csv`，并回写快照目录下的 `meta.txt`。

```bash
./build/perfectdou_train --updates 100000 --games 256 --threads 10 \
    --eval-every 500 --eval-decks 100 --eval-port 18765 \
    --eval-save-dir eval_snapshots --eval-csv eval_vs_douzero.csv
```

> **大规模训练（如论文 25 亿帧）磁盘容量与间隔建议**：
> - 25 亿帧折合约 160,000 次 Update（按每次 256 局算）。单次快照（3 个 Actor）约 **12.8 MB**；
> - 推荐设置 `--eval-every 500`（或 200~500）：整个训练周期产生约 320 个高解析度数据点，磁盘总快照仅占用 **~4 GB**，后台对弈每 15~30 分钟打一次（耗时 ~20 秒），对主训练吞吐 **0 影响**；
> - 若磁盘空间紧张，可加 `--eval-max-snapshots 100`（仅保留最新 100 份快照，硬防爆盘）。

### 3. 渲染胜率/ADP 趋势曲线
使用 `tools/plot_eval.py` 读取评测 CSV，自动绘制地主/农民双阵营与整体胜率及 ADP 趋势曲线（含平滑拟合线）：
```bash
# 生成静态趋势图
python tools/plot_eval.py --csv eval_vs_douzero.csv --out eval_curve.png

# 实时监视模式（每 30 秒自动刷新图片）
python tools/plot_eval.py --csv eval_vs_douzero.csv --out eval_curve.png --watch 30
```

> 提示：若不希望内嵌在训练中，仍可使用外部独立轮询脚本 `tools/eval_vs_douzero_loop.py` 或独立对弈程序 `build/perfectdou_eval`。

## 目录结构

```
src/ddz/      牌、牌型识别/生成、621 动作空间、oracle、对局、特征
src/nn/       矩阵运算、Linear/ReLU/LSTM、演员/评论家、Adam、模型存取
src/algo/     批量自对弈 rollout、GAE、PPO 更新、DouZero 对弈评测与快照
src/          训练入口 train.cpp、独立评测入口 eval_douzero.cpp
tools/        DouZero 服务端、评测曲线绘制工具 plot_eval.py、外部评测轮询脚本
tests/        Google Test 单元测试（规则/自对弈/反向传播数值与解析梯度/PPO 算法与稳定性/防爆熔断）
```

## 实现说明

- 所有网络为手写反向传播；`tests/test_grad` 对包括 LSTM 在内的演员/评论家
  做数值与解析梯度校验（96 项）。
- 手牌/历史等 0/1 特征以 uint8 紧凑存储；前向使用转置权重 + 向量化 GEMM。
- oracle 结果在 rollout 工作线程内记忆化，DP 表一次性预计算，线程安全。
- DouZero 生成器会产生少量其检测器判为 WRONG 的退化组合（不符合腾讯规则），
  映射抽象动作时直接剔除。
- 完整包含跨平台 Google Test 单元测试，在 GitHub Actions CI（Linux / macOS / Windows）实现自动化回归守护。
