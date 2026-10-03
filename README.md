# PerfectDouCPP

斗地主 AI **PerfectDou**（NeurIPS 2022, *Dominating DouDizhu with Perfect
Information Distillation*）的纯 C++17 完整训练实现：零第三方依赖（仅
pthread），包含牌局引擎、特征工程、神经网络与 PPO 自对弈训练。

## 与论文的对应关系

| 论文组件 | 本实现 |
| --- | --- |
| 牌局规则（附录 B，腾讯规则） | `src/ddz/moves.cpp`（15 种牌型，移植 DouZero 的生成/识别/压牌逻辑） |
| 621 抽象动作 + 解码（附录 E.2） | `src/ddz/action_space.{h,cpp}`，Algo 2 kicker 打分 |
| 不完美信息特征 23×12×15+6（表 1/5） | `src/ddz/features.cpp` |
| 完美信息特征 +2 手牌 +2 步数（critic） | 同上 |
| 动作特征 12×15+7（表 6） | `legalOptions()` 动态计算 |
| 最小出牌步数 oracle（附录 E.1） | `src/ddz/oracle.cpp`（DP + 记忆化 DFS） |
| LSTM(15 步历史) + MLP[256,256,256,512] 演员 | `src/nn/net.cpp` |
| 完美信息价值网络 MLP[256×4] | `src/nn/net.cpp`（PTIE：完美 critic 通过优势蒸馏给不完美 actor） |
| PPO+GAE（γ=1, λ=0.95, ent=0.1, lr=3e-4, Adam） | `src/algo/ppo.cpp` |
| oracle 距离奖励 r=±ΔAdv·l（l=50）+ 终端 ADP | `src/algo/rollout.cpp` |
| 三个座位独立模型、批量自对弈 | `src/algo/rollout.cpp`（多线程批量推进） |

论文的叫牌阶段与分布式集群（880 CPU + 8 GPU、25 亿帧）未实现：本项目只训练
出牌阶段（landlord 固定 20 张并先手，与 DouZero/PerfectDou 评估协议一致），
单机多线程即可运行。

## 构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build          # 规则测试 + 神经网络数值梯度检查
```

## 训练

```bash
# macOS 默认走 Accelerate/AMX（比当前 Metal GEMM 更快）。
# Windows 默认走本机显卡（Direct3D 12）。采样在 CPU，PPO 学习在 GPU。
# --backend cpu / --backend gpu 可强制切换
./build/perfectdou_train --updates 1000 --games 256 --threads 10 --out ckpt

# 快速/低配机器
./build/perfectdou_train --updates 300 --games 128 --threads 8 \
    --hidden 128 --epochs 2 --out ckpt

# 论文的最终微调阶段：去掉 oracle 塑形，只保留 ADP
./build/perfectdou_train --resume ckpt --reward-scale 0 --out ckpt_adp
```

主要参数：

```
--games N           每次更新自对弈牌局数（默认 256）
--updates N         更新轮数
--threads N         自对弈线程数（三个座位的 PPO 更新也会并行）
--hidden H          MLP 宽度（论文 256；128 可显著提速）
--lstm-hidden H     LSTM 隐层（论文 128）
--epochs N          每批数据 PPO epoch 数（论文默认 4）
--mb N              minibatch（1024 总批 / GPU128；单机默认 256）
--lr/--clip/--ent/--gae/--gamma   PPO 超参，默认论文值
--reward-scale L    oracle 塑形系数 l（默认 50，0 表示纯 ADP）
--snapshot-every K  每 K 轮落盘，最后一轮必存
--resume DIR        从 actor{0,1,2}.bin / critic{0,1,2}.bin 继续
--backend auto|gpu|cpu   GEMM 后端（Apple 默认 cpu：Accelerate/AMX；Windows 默认 gpu：D3D12）
```

后端说明：

- Apple Silicon：默认 PPO 与自对弈都走 CPU。大矩阵乘法用 Accelerate（AMX），LSTM 的
  sigmoid/tanh 用 vForce。`--backend gpu` 使用 Metal GEMM。参考实测（M1 Pro，
  hidden=256 / 256 局 / epochs=4）：约 **5 秒/轮**（采样 0.5 秒 + 学习 4.6 秒）。
  同配置 Metal 学习阶段大约慢一倍。
- Windows：默认使用 Direct3D 12 计算着色器，和 Metal 同一套分块 GEMM、LSTM、Adam。
  多显卡时选择专用显存最大的一块（笔记本上的独显会优先于核显）。`--backend cpu`
  退回标量 CPU GEMM。自对弈采样仍固定走 CPU，只有 PPO 学习上 GPU。
- 数值为 FP32。`tests/test_gemm` 校验 CPU/GPU GEMM，`tests/test_grad` 校验整网梯度。
- 没有 GPU 时使用 CPU GEMM（Apple 为 Accelerate，ARM 为 NEON，其余为标量）。

输出文件：`ckpt/actor{0,1,2}.bin`、`ckpt/critic{0,1,2}.bin`
（座位 0=landlord，1=landlord_down，2=landlord_up），供后续推理程序加载。

每轮打印 WP（地主胜率）、ADP（均差分）、炸弹率、平均出牌数、三个座位的
策略熵 / 价值损失 / 平均回报。

## 目录结构

```
src/ddz/      牌、牌型识别/生成、621 动作空间、oracle、对局、特征
src/nn/       矩阵运算、Linear/ReLU/LSTM、演员/评论家、Adam、模型存取
src/algo/     批量自对弈 rollout、GAE、PPO 更新
src/          训练入口 train.cpp
tests/        规则单测（随机自对弈校验）与数值梯度检查
```

## 实现说明

- 所有网络为手写反向传播；`tests/test_grad` 对包括 LSTM 在内的演员/评论家
  做数值梯度校验（96 项）。
- 手牌/历史等 0/1 特征以 uint8 紧凑存储；前向使用转置权重 + 向量化 GEMM。
- oracle 结果在 rollout 工作线程内记忆化，DP 表一次性预计算，线程安全。
- DouZero 生成器会产生少量其检测器判为 WRONG 的退化组合（不符合腾讯规则），
  映射抽象动作时直接剔除。
