# 联邦学习框架 - 基于主动推理的端云协同架构

## 概述

本框架实现了车联网环境下的端云协同联邦学习系统，用于Qwen3-VL大模型的分布式训练和推理。核心特性包括：

- **Logits双向传递**: 采用联邦蒸馏范式，通过软标签共享知识而非直接共享模型参数
- **主动推理机制**: 使用自由能(Free Energy)动态评估客户端可信度并分配贡献权重
- **端云对等架构**: 客户端和服务器均运行完整的Qwen3-VL模型
- **最小侵入集成**: 保留原有单机训练功能，通过模块化设计实现无缝切换

## 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                    Federated Learning System                 │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌──────────────┐      ┌────────────┐│
│  │  Client 1    │      │  Client 2    │      │  Client N  ││
│  │ Qwen3-VL     │      │ Qwen3-VL     │      │ Qwen3-VL   ││
│  └──────┬───────┘      └──────┬───────┘      └──────┬─────┘│
│         │ logits_i            │ logits_i            │       │
│         ▼                     ▼                     ▼       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Federated Server (Cloud)                │  │
│  │  1. Collect logits                                   │  │
│  │  2. Compute Free Energy: F_i                         │  │
│  │  3. Calculate Weights: w_i = softmax(-F_i/τ)        │  │
│  │  4. Aggregate: global_logits = Σ(w_i * logits_i)    │  │
│  └──────────────────────┬───────────────────────────────┘  │
│         global_logits   │ global_logits   global_logits    │
│         ▼               ▼                 ▼                 │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐       │
│  │  Client 1    │ │  Client 2    │ │  Client N    │       │
│  │ Local Update │ │ Local Update │ │ Local Update │       │
│  └──────────────┘ └──────────────┘ └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

## 模块说明

### 核心模块

- **client.py**: `FederatedClient` - 客户端类，负责本地训练和推理
- **server.py**: `FederatedServer` - 服务器类，负责聚合和协调
- **active_inference.py**: 主动推理模块，实现自由能计算和权重分配
- **config.py**: `FederatedConfig` - 配置管理
- **comm.py**: `CommunicationManager` - 通信和序列化
- **logger.py**: `FederatedLogger` - 日志记录和可视化
- **utils.py**: 工具函数集合

## 快速开始

### 1. 基础运行

```bash
# 3个客户端，10个联邦回合
python run_federated_qwenvl.py --num_clients 3 --num_rounds 10
```

### 2. 自定义配置

```bash
# 使用CE方案的自由能，调整蒸馏权重
python run_federated_qwenvl.py \
    --num_clients 5 \
    --num_rounds 20 \
    --free_energy_mode ce_entropy \
    --alpha 0.6 \
    --beta 0.4 \
    --tau 0.5
```

### 3. 启用服务器更新

```bash
# 服务器端也进行模型更新
python run_federated_qwenvl.py \
    --num_clients 3 \
    --num_rounds 10 \
    --enable_server_update \
    --server_lr 1e-5
```

### 4. 使用配置文件

```bash
# 从JSON文件加载配置
python run_federated_qwenvl.py --config my_config.json
```

## 配置参数说明

### 联邦学习参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_clients` | 3 | 客户端数量 |
| `--num_rounds` | 10 | 联邦训练回合数 |
| `--local_epochs` | 1 | 每轮本地训练epoch数 |

### 蒸馏损失参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--alpha` | 0.5 | 交叉熵权重 |
| `--beta` | 0.5 | KL散度权重 |
| `--temperature` | 3.0 | 蒸馏温度T |

### 主动推理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--free_energy_mode` | kl_entropy | 自由能计算模式 (kl_entropy/ce_entropy) |
| `--lambda_entropy` | 0.1 | KL方案的熵权重λ |
| `--gamma_entropy` | 0.1 | CE方案的熵权重γ |
| `--tau` | 1.0 | 权重计算温度τ |

### 其他参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--logits_mode` | cls | Logits提取模式 (cls/last_token/mean_pool) |
| `--enable_server_update` | False | 启用服务器端模型更新 |
| `--batch_size` | 1 | 批次大小 |
| `--lr` | 1e-4 | 学习率 |

## 自由能计算方案

### 方案1: KL散度 + 熵 (推荐)

```
F_i = KL(q_i || p_s) + λ * H(q_i)
```

- `q_i`: 客户端预测分布
- `p_s`: 服务器先验分布
- `λ`: 熵权重，控制不确定性惩罚

**适用场景**: 有服务器先验知识，强调与全局一致性

### 方案2: 交叉熵 + 熵

```
F_i = CE(y_true, q_i) + γ * H(q_i)
```

- `y_true`: 真实标签
- `γ`: 熵权重

**适用场景**: 有真实标签，强调预测准确性

## 权重计算

```
w_i = softmax(-F_i / τ)
```

- 自由能越小的客户端获得越大的权重
- `τ` 控制权重分布的平滑度：
  - `τ` 较小：权重更集中在低自由能客户端
  - `τ` 较大：权重分布更均匀

## 联邦回合流程

每个联邦回合包含以下阶段：

1. **客户端本地推理**: 各客户端用本地数据进行前向推理，提取logits
2. **上传logits**: 客户端将logits上传到服务器
3. **计算自由能和权重**: 服务器计算每个客户端的F_i和w_i
4. **聚合logits**: 服务器计算加权聚合 `global_logits = Σ(w_i * logits_i)`
5. **服务器更新** (可选): 服务器用聚合logits更新自身模型
6. **下发global_logits**: 服务器将全局logits广播给所有客户端
7. **客户端本地更新**: 客户端使用蒸馏损失更新本地模型
8. **记录指标**: 记录F_i, w_i, 通信量, 性能指标等

## 日志和输出

### 日志文件

训练过程中会生成以下文件：

```
logs_federated/
├── federated_training_YYYYMMDD_HHMMSS.log  # 详细日志
├── metrics_YYYYMMDD_HHMMSS.json            # 指标JSON
└── training_curves.png                      # 训练曲线图
```

### 模型检查点

```
checkpoints_federated/
├── config.json                              # 配置文件
├── best_server_model.pth                    # 最佳服务器模型
├── best_client_0_model.pth                  # 最佳客户端0模型
├── best_client_1_model.pth                  # 最佳客户端1模型
└── checkpoint_round_5.pth                   # 定期检查点
```

### 每轮输出示例

```
Round 1 Started - 3 clients
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
阶段7: 客户端本地更新
  Client 0: F=0.8234, w=0.3421, loss=0.6543, acc=0.7200, f1=0.6890
  Client 1: F=0.9102, w=0.3012, loss=0.7012, acc=0.6800, f1=0.6543
  Client 2: F=0.7891, w=0.3567, loss=0.6234, acc=0.7500, f1=0.7123
Communication Stats:
  Round Total: 2.34 MB
Round 1 Completed
```

## 与原有代码的集成

### 保持向后兼容

原有的单机训练脚本 `test_local_train_mini_qwen3vl_debug.py` 仍可正常运行：

```bash
# 单机训练（原有功能）
python test_local_train_mini_qwen3vl_debug.py

# 快速测试
python test_local_train_mini_qwen3vl_debug.py test
```

### 联邦学习开关

如果尝试在原脚本中启用联邦模式，会提示使用联邦入口：

```bash
python test_local_train_mini_qwen3vl_debug.py --federated
# 输出: 请使用 run_federated_qwenvl.py 启动联邦训练
```

## 高级用法

### 自定义数据分区

```python
from federation.utils import partition_data_iid, partition_data_non_iid

# IID分区（均匀随机）
client_indices = partition_data_iid(dataset_size, num_clients)

# Non-IID分区（按标签分片）
client_indices = partition_data_non_iid(dataset_size, labels, num_clients)
```

### 自定义自由能计算

```python
from federation.active_inference import compute_free_energy

# 使用KL方案
free_energy = compute_free_energy(
    client_logits=logits,
    server_prior_logits=prior_logits,
    mode='kl_entropy',
    temperature=3.0,
    lambda_entropy=0.1
)

# 使用CE方案
free_energy = compute_free_energy(
    client_logits=logits,
    labels=labels,
    mode='ce_entropy',
    temperature=3.0,
    gamma_entropy=0.1
)
```

### 加载和保存模型

```python
# 保存服务器模型
server.save_model('server_model.pth')

# 加载服务器模型
server.load_model('server_model.pth')

# 保存客户端模型
client_state = client.get_model_state()
torch.save(client_state, 'client_model.pth')

# 加载客户端模型
client_state = torch.load('client_model.pth')
client.set_model_state(client_state)
```

## 性能优化建议

1. **批次大小**: Qwen3-VL显存占用较大，建议 `batch_size=1`
2. **客户端数量**: 建议3-5个客户端，过多会增加通信开销
3. **本地epoch数**: 建议 `local_epochs=1`，避免客户端过拟合
4. **温度参数**: 
   - 蒸馏温度T建议3.0-5.0
   - 权重温度τ建议0.5-2.0
5. **自由能方案**: 
   - 有先验知识时使用KL方案
   - 有真实标签时使用CE方案

## 故障排查

### 显存不足

```bash
# 减小batch_size
python run_federated_qwenvl.py --batch_size 1

# 或使用CPU
python run_federated_qwenvl.py --device cpu
```

### Logits形状不匹配

检查 `logits_mode` 配置，确保所有客户端使用相同的提取模式。

### 自由能异常值

如果出现NaN或Inf，框架会自动使用均匀权重并记录警告。检查：
- 数据是否有异常
- 温度参数是否过小

## 引用

如果使用本框架，请引用：

```bibtex
@software{federated_active_inference_2024,
  title={Federated Learning Framework with Active Inference for Autonomous Driving},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
```

## 许可证

[Your License]

## 联系方式

如有问题或建议，请联系：[your-email@example.com]
