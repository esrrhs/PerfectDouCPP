# PerfectDouCPP

斗地主 AI **PerfectDou**（NeurIPS 2022, *Dominating DouDizhu with Perfect
Information Distillation*）的纯 C++17 完整训练实现：零第三方依赖（仅
pthread），包含牌局引擎、特征工程、神经网络与 PPO 自对弈训练。

## 与论文的对应关系

| 论文组件 | 本实现 |
| --- | --- |
| 牌局规则（附录 B，腾讯规则） | `src/ddz/moves.cpp`（15 种牌型，移植 DouZero 的生成/识别/压牌逻辑） |
| 621 抽象动作 + 解码（附录 E.2） | `src/ddz/action_space.{h,cpp}`，Algo 2 kicker 打分 |
| 不完美信息特征 24 个牌矩阵 + 手牌张数/炸弹 one-hot | `src/ddz/features.cpp` |
| 完美信息特征 +2 手牌 +2 步数（critic） | 同上 |
| 动作特征为实际打出的牌（含带牌）+6 | `legalOptions()` 动态计算 |
| 最小出牌步数 oracle（附录 E.1） | `src/ddz/oracle.cpp`（DP + 记忆化 DFS） |
| LSTM(5×540，即每步拼接 3 次出牌) + 对每个合法动作共享 MLP[256,256,256,512,1] | `src/nn/net.cpp` |
| 完美信息价值网络 MLP[256×4] | `src/nn/net.cpp`（PTIE：完美 critic 通过优势蒸馏给不完美 actor） |
| PPO，每一步的目标都是该座位的终局 ADP | `src/algo/ppo.cpp` |
| 终局 ADP（地主 ±2×2^炸弹，农民 ±1×2^炸弹），无中间奖励 | `src/algo/rollout.cpp` |
| 三个座位独立模型、批量自对弈 | `src/algo/rollout.cpp`（多线程批量推进） |

论文的叫牌阶段与分布式集群（880 CPU + 8 GPU、25 亿帧）未实现：本项目只训练
出牌阶段（landlord 固定 20 张并先手，与 DouZero/PerfectDou 评估协议一致），
单机多线程即可运行。

官方仓库没有公开训练代码，只提供评测代码、二进制特征编码器和最终 ONNX。
本实现对可验证部分同时对照论文和官方 ONNX；PPO epoch/clip/max-grad-norm 等论文
未披露的值采用常见 PPO2 默认值。

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
# --backend cpu / --backend gpu / --backend cuda 可强制切换
./build/perfectdou_train --updates 1000 --games 256 --threads 10 --out ckpt

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
--lr/--ent           学习率（默认 3e-4）和熵系数（默认 0.1）
--clip N            PPO clip（默认 0.2）
--snapshot-every K  每 K 轮落盘，最后一轮必存
--resume DIR        从 actor{0,1,2}.bin / critic{0,1,2}.bin 继续
--backend auto|gpu|cuda|cpu   GEMM 后端（Windows 的 cuda 用 cuBLAS GEMM + D3D12 其余核）
```

后端说明：

- Apple Silicon：默认 PPO 与自对弈都走 CPU。大矩阵乘法用 Accelerate（AMX），LSTM 的
  sigmoid/tanh 用 vForce。`--backend gpu` 使用 Metal GEMM。参考实测（M1 Pro，
  hidden=256 / 256 局 / epochs=4）：约 **5 秒/轮**（采样 0.5 秒 + 学习 4.6 秒）。
  同配置 Metal 学习阶段大约慢一倍。
- Windows：默认使用 Direct3D 12 计算着色器，和 Metal 同一套分块 GEMM、LSTM、Adam。
  多显卡时选择专用显存最大的一块（笔记本上的独显会优先于核显）。`--backend cpu`
  退回 CPU GEMM。x86-64 Windows 使用运行时检测的 AVX2/FMA 推理内核；自对弈采样
  固定走该 CPU 路径，只有 PPO 学习上 GPU。
- 安装 CUDA Toolkit 后可用 `--backend cuda`，以 cuBLAS 替换 PPO 的 GEMM，其余核
  仍使用 D3D12。当前 CUDA 输出经上传堆回到 D3D12，主要用于稳定性诊断，速度会比
  纯 D3D12 慢。
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
