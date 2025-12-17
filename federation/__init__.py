"""
联邦学习模块 - 基于主动推理的端云协同架构

该模块实现了车联网环境下的联邦学习框架，支持：
- 基于logits双向传递的联邦蒸馏
- 主动推理驱动的自由能计算和权重分配
- 端云协同的Qwen3-VL大模型训练
"""

from .client import FederatedClient
from .server import FederatedServer
from .config import FederatedConfig
from .active_inference import (
    free_energy_kl_entropy,
    free_energy_ce_entropy,
    compute_client_weights
)
from .comm import CommunicationManager
from .logger import FederatedLogger, RoundMetrics, ClientMetrics

__version__ = "1.0.0"

__all__ = [
    "FederatedClient",
    "FederatedServer",
    "FederatedConfig",
    "free_energy_kl_entropy",
    "free_energy_ce_entropy",
    "compute_client_weights",
    "CommunicationManager",
    "FederatedLogger",
    "RoundMetrics",
    "ClientMetrics",
]
