"""
联邦学习配置模块

提供FederatedConfig数据类，管理所有联邦学习相关的超参数和配置。
"""

from dataclasses import dataclass, field
from typing import Optional
import json


@dataclass
class FederatedConfig:
    """联邦学习配置"""
    
    # ========== 联邦学习基础参数 ==========
    num_clients: int = 3
    num_rounds: int = 10
    local_epochs: int = 1
    
    # ========== 模型参数 ==========
    model_path: str = "/home/sutongtong/wwt/model/Qwen3-VL-2B-Instruct"
    pointcloud_dim: int = 1024
    qwen_hidden_dim: int = 3072
    
    # ========== 蒸馏损失参数 ==========
    alpha: float = 0.5  # 交叉熵权重
    beta: float = 0.5   # KL散度权重
    temperature: float = 3.0  # 蒸馏温度T
    
    # ========== 主动推理参数 ==========
    free_energy_mode: str = "kl_entropy"  # "kl_entropy" 或 "ce_entropy"
    lambda_entropy: float = 0.1  # KL方案的熵权重λ
    gamma_entropy: float = 0.1   # CE方案的熵权重γ
    tau: float = 1.0  # 权重计算温度τ
    
    # ========== Logits提取参数 ==========
    logits_mode: str = "cls"  # "cls", "last_token", "mean_pool"
    
    # ========== 服务器更新参数 ==========
    enable_server_update: bool = False
    server_lr: float = 1e-5
    
    # ========== 数据参数 ==========
    dataroot: str = "/home/sutongtong/LanTu_team3/dataset/nuScenes/train"
    version: str = "v1.0-trainval"
    batch_size: int = 1
    max_batches: int = 0  # 每轮每客户端最大训练batch数，0表示跑完整个epoch
    num_workers: int = 2
    attack_ratio: float = 0.3
    num_points: int = 2048
    
    # ========== 训练参数 ==========
    lr: float = 1e-4
    weight_decay: float = 0.01
    
    # ========== 设备参数 ==========
    device: str = "cuda"
    
    # ========== 日志参数 ==========
    log_dir: str = "./logs_federated"
    save_dir: str = "./checkpoints_federated"
    verbose: bool = True
    
    # ========== 部署模式 ==========
    mode: str = "single_process"  # "single_process" 或 "multi_process"
    
    def validate(self) -> None:
        """验证配置参数的合法性"""
        assert self.num_clients > 0, "num_clients必须大于0"
        assert self.num_rounds > 0, "num_rounds必须大于0"
        assert self.local_epochs > 0, "local_epochs必须大于0"
        
        assert 0 < self.alpha <= 1, "alpha必须在(0, 1]范围内"
        assert 0 < self.beta <= 1, "beta必须在(0, 1]范围内"
        assert self.temperature > 0, "temperature必须大于0"
        
        assert self.free_energy_mode in ["kl_entropy", "ce_entropy"], \
            "free_energy_mode必须是'kl_entropy'或'ce_entropy'"
        assert self.lambda_entropy >= 0, "lambda_entropy必须非负"
        assert self.gamma_entropy >= 0, "gamma_entropy必须非负"
        assert self.tau > 0, "tau必须大于0"
        
        assert self.logits_mode in ["cls", "last_token", "mean_pool"], \
            "logits_mode必须是'cls', 'last_token'或'mean_pool'"
        
        assert self.batch_size > 0, "batch_size必须大于0"
        assert self.lr > 0, "lr必须大于0"
        
        assert self.mode in ["single_process", "multi_process"], \
            "mode必须是'single_process'或'multi_process'"
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
    
    def save(self, filepath: str) -> None:
        """保存配置到JSON文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=4, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: str) -> 'FederatedConfig':
        """从JSON文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def __str__(self) -> str:
        """字符串表示"""
        lines = ["FederatedConfig:"]
        for key, value in self.to_dict().items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)
