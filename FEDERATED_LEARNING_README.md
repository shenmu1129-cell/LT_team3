# 联邦学习 + 主动推理 - 车联网入侵检测系统

## 项目概述

本项目在现有的Qwen3-VL自动驾驶攻击检测系统基础上，新增了基于主动推理(Active Inference)的联邦学习架构，实现端云协同的大模型训练与推理。

### 核心特性

✅ **向后兼容**: 完全保留原有单机训练功能  
✅ **Logits双向传递**: 联邦蒸馏范式，保护模型隐私  
✅ **主动推理驱动**: 自由能机制动态评估客户端可信度  
✅ **端云对等**: 客户端和服务器均运行完整Qwen3-VL模型  
✅ **最小侵入**: 模块化设计，无缝集成现有代码  

## 项目结构

```
project_root/
├── test_local_train_mini_qwen3vl_debug.py  # 原有单机训练脚本(保持不变)
├── run_federated_qwenvl.py                 # 联邦学习入口脚本(新增)
├── example_config.json                      # 示例配置文件(新增)
├── FEDERATED_LEARNING_README.md            # 本文档(新增)
│
├── federation/                              # 联邦学习模块(新增)
│   ├── __init__.py
│   ├── client.py                            # 联邦客户端
│   ├── server.py                            # 联邦服务器
│   ├── active_inference.py                  # 主动推理与自由能计算
│   ├── comm.py                              # 通信与序列化
│   ├── config.py                            # 配置管理
│   ├── logger.py                            # 日志记录
│   ├── utils.py                             # 工具函数
│   └── README.md                            # 详细文档
│
├── .kiro/specs/federated-active-inference/  # 设计文档(新增)
│   ├── requirements.md                      # 需求文档
│   ├── design.md                            # 设计文档
│   └── tasks.md                             # 任务列表
│
├── checkpoints_federated/                   # 联邦模型检查点(运行时生成)
└── logs_federated/                          # 联邦训练日志(运行时生成)
```

## 快速开始

### 1. 原有单机训练（不受影响）

```bash
# 单机训练仍可正常运行
python test_local_train_mini_qwen3vl_debug.py

# 快速测试
python test_local_train_mini_qwen3vl_debug.py test
```

### 2. 联邦学习训练

```bash
# 基础运行：3客户端，10回合
python run_federated_qwenvl.py --num_clients 3 --num_rounds 10

# 使用配置文件
python run_federated_qwenvl.py --config example_config.json

# 查看所有选项
python run_federated_qwenvl.py --help
```

### 3. 自定义配置示例

```bash
# 使用CE方案的自由能，5个客户端，20回合
python run_federated_qwenvl.py \
    --num_clients 5 \
    --num_rounds 20 \
    --free_energy_mode ce_entropy \
    --alpha 0.6 \
    --beta 0.4 \
    --tau 0.5 \
    --temperature 4.0

# 启用服务器端更新
python run_federated_qwenvl.py \
    --num_clients 3 \
    --num_rounds 10 \
    --enable_server_update \
    --server_lr 1e-5
```

## 核心概念

### 主动推理与自由能

**自由能(Free Energy)** 是主动推理理论中的核心概念，用于衡量客户端预测的质量：

- **F_i 越小** → 客户端越可信 → 获得更大权重 w_i
- **F_i 越大** → 客户端不可信 → 获得较小权重 w_i

#### 方案1: KL散度 + 熵 (推荐)

```
F_i = KL(q_i || p_s) + λ * H(q_i)
```

- 衡量客户端预测与服务器先验的偏离程度
- 适用于有服务器先验知识的场景

#### 方案2: 交叉熵 + 熵

```
F_i = CE(y_true, q_i) + γ * H(q_i)
```

- 衡量客户端预测对真实标签的解释能力
- 适用于有真实标签的场景

### 权重计算

```
w_i = softmax(-F_i / τ)
```

- 自由能越小的客户端获得越大的权重
- τ 控制权重分布的平滑度

### 联邦蒸馏

客户端使用蒸馏损失更新本地模型：

```
Loss = α * CE(y_true, local_pred) + β * KL(local_pred || global_pred)
```

- α: 硬标签（真实标签）的权重
- β: 软标签（全局logits）的权重

## 联邦回合流程

```
Round N:
  1. 客户端本地推理 → 提取logits_i
  2. 上传logits到服务器
  3. 服务器计算自由能F_i和权重w_i
  4. 服务器聚合: global_logits = Σ(w_i * logits_i)
  5. 服务器自我更新(可选)
  6. 下发global_logits给所有客户端
  7. 客户端使用蒸馏损失更新本地模型
  8. 记录指标: F_i, w_i, 通信量, 性能指标
```

## 关键参数说明

### 必需参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_clients` | 3 | 客户端数量（模拟车端设备） |
| `--num_rounds` | 10 | 联邦训练回合数 |

### 蒸馏参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--alpha` | 0.5 | 硬标签权重 |
| `--beta` | 0.5 | 软标签权重 |
| `--temperature` | 3.0 | 蒸馏温度T（建议3.0-5.0） |

### 主动推理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--free_energy_mode` | kl_entropy | 自由能计算模式 |
| `--tau` | 1.0 | 权重计算温度（建议0.5-2.0） |
| `--lambda_entropy` | 0.1 | KL方案的熵权重 |
| `--gamma_entropy` | 0.1 | CE方案的熵权重 |

### 其他重要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch_size` | 1 | 批次大小（Qwen3-VL显存大，建议1） |
| `--local_epochs` | 1 | 每轮本地训练epoch数 |
| `--enable_server_update` | False | 是否启用服务器端更新 |

## 输出和日志

### 训练日志

```
logs_federated/
├── federated_training_20241216_143022.log  # 详细日志
├── metrics_20241216_143022.json            # 指标JSON
└── training_curves.png                      # 训练曲线
```

### 模型检查点

```
checkpoints_federated/
├── config.json                  # 配置文件
├── best_server_model.pth        # 最佳服务器模型
├── best_client_0_model.pth      # 最佳客户端模型
├── best_client_1_model.pth
├── best_client_2_model.pth
└── checkpoint_round_5.pth       # 定期检查点
```

### 每轮输出示例

```
================================================================================
Round 1 Started - 3 clients
================================================================================
阶段1: 客户端本地推理
  Client 0: logits shape torch.Size([1, 1])
  Client 1: logits shape torch.Size([1, 1])
  Client 2: logits shape torch.Size([1, 1])

阶段3: 服务器计算自由能和权重
  Client 0: F_i = 0.8234
  Client 1: F_i = 0.9102
  Client 2: F_i = 0.7891
  Client 0: w_i = 0.3421
  Client 1: w_i = 0.3012
  Client 2: w_i = 0.3567

Server Aggregation:
  Free Energies: ['0.8234', '0.9102', '0.7891']
  Weights: ['0.3421', '0.3012', '0.3567']
  Global Logits Stats:
    mean: -0.1234
    std: 0.4567
    max: 0.8901
    min: -1.2345

阶段7: 客户端本地更新
Client 0: F=0.8234, w=0.3421, loss=0.6543, acc=0.7200, f1=0.6890
Client 1: F=0.9102, w=0.3012, loss=0.7012, acc=0.6800, f1=0.6543
Client 2: F=0.7891, w=0.3567, loss=0.6234, acc=0.7500, f1=0.7123

Communication Stats:
  Round Sent: 1.23 MB
  Round Received: 1.11 MB
  Round Total: 2.34 MB
  Cumulative Total: 2.34 MB

Round 1 Completed
  Average Metrics:
    avg_accuracy: 0.7167
    avg_f1_score: 0.6852
    avg_free_energy: 0.8409
    avg_weight: 0.3333
================================================================================
```

## 设计文档

详细的设计文档位于 `.kiro/specs/federated-active-inference/`:

- **requirements.md**: 12个详细需求，包含用户故事和验收标准
- **design.md**: 完整的架构设计、组件接口、数据模型、正确性属性
- **tasks.md**: 实现任务列表，包含12个主要任务组

## 技术亮点

### 1. 最小侵入集成

- 原有代码 `test_local_train_mini_qwen3vl_debug.py` 完全不受影响
- 通过 `--federated` 开关提示用户使用联邦入口
- 复用现有的数据加载、攻击模拟、模型定义

### 2. 主动推理机制

- 两种自由能计算方案可切换
- 动态权重分配，自适应客户端贡献
- 数值稳定性处理（epsilon项、softmax技巧）

### 3. 联邦蒸馏范式

- 通过logits双向传递共享知识
- 避免直接共享模型参数，保护隐私
- 减少通信开销（相比参数聚合）

### 4. 完整的日志系统

- 每轮详细记录F_i、w_i、通信量、性能指标
- 自动生成训练曲线图
- JSON格式保存所有指标，便于后续分析

### 5. 模块化设计

- 清晰的模块划分（client、server、active_inference等）
- 完整的类型标注和docstring
- 易于扩展和维护

## 性能优化建议

1. **显存优化**: 
   - 使用 `batch_size=1`
   - 考虑使用梯度累积

2. **通信优化**:
   - Logits通信量远小于参数通信
   - 可考虑压缩或量化logits

3. **训练效率**:
   - `local_epochs=1` 避免客户端过拟合
   - 适当的客户端数量（3-5个）

4. **超参数调优**:
   - 蒸馏温度T: 3.0-5.0
   - 权重温度τ: 0.5-2.0
   - α和β的平衡

## 故障排查

### 问题1: 显存不足

```bash
# 解决方案1: 减小batch_size
python run_federated_qwenvl.py --batch_size 1

# 解决方案2: 使用CPU
python run_federated_qwenvl.py --device cpu
```

### 问题2: Logits形状不匹配

检查所有客户端是否使用相同的 `logits_mode`。

### 问题3: 自由能出现NaN

- 检查数据是否有异常值
- 增大温度参数T
- 框架会自动使用均匀权重并记录警告

### 问题4: 训练不收敛

- 调整α和β的比例
- 增大蒸馏温度T
- 检查学习率是否合适

## 实验建议

### 基线实验

```bash
# 实验1: 基础联邦学习（KL方案）
python run_federated_qwenvl.py \
    --num_clients 3 \
    --num_rounds 10 \
    --free_energy_mode kl_entropy

# 实验2: CE方案对比
python run_federated_qwenvl.py \
    --num_clients 3 \
    --num_rounds 10 \
    --free_energy_mode ce_entropy

# 实验3: 不同客户端数量
python run_federated_qwenvl.py --num_clients 5 --num_rounds 10
python run_federated_qwenvl.py --num_clients 7 --num_rounds 10
```

### 消融实验

```bash
# 实验4: 不同温度参数τ
python run_federated_qwenvl.py --tau 0.5
python run_federated_qwenvl.py --tau 1.0
python run_federated_qwenvl.py --tau 2.0

# 实验5: 不同蒸馏权重
python run_federated_qwenvl.py --alpha 0.7 --beta 0.3
python run_federated_qwenvl.py --alpha 0.5 --beta 0.5
python run_federated_qwenvl.py --alpha 0.3 --beta 0.7

# 实验6: 服务器更新的影响
python run_federated_qwenvl.py --enable_server_update
```

## 下一步工作

- [ ] 支持多进程并行模式
- [ ] 实现Non-IID数据分区
- [ ] 添加差分隐私机制
- [ ] 支持异步联邦学习
- [ ] 实现模型压缩和量化
- [ ] 添加更多自由能计算方案

## 参考文献

1. Active Inference: Friston, K. (2010). The free-energy principle.
2. Federated Learning: McMahan, B., et al. (2017). Communication-efficient learning of deep networks from decentralized data.
3. Federated Distillation: Lin, T., et al. (2020). Ensemble distillation for robust model fusion in federated learning.

## 联系方式

如有问题或建议，请联系项目维护者。

---

**注意**: 本框架为研究原型，生产环境使用前请进行充分测试。
