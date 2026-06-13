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
    model_type: str = "qwen3vl"  # "qwen3vl", "ovis", 或 "internvl"
    pointcloud_dim: int = 1024
    qwen_hidden_dim: Optional[int] = None  # None表示自动检测
    
    # ========== 蒸馏损失参数 ==========
    alpha: float = 0.5  # 交叉熵权重
    beta: float = 0.5   # KL散度权重
    temperature: float = 3.0  # 蒸馏温度T
    
    # ========== LLM生成训练参数 ==========
    train_generation: bool = False  # 是否训练LLM生成任务
    generation_alpha: float = 1.0   # 分类/蒸馏损失权重
    generation_beta: float = 0.3    # 检测生成损失权重
    generation_gamma: float = 0.3   # 防御生成损失权重
    
    # ========== 主动推理参数 ==========
    free_energy_mode: str = "kl_entropy"  # "kl_entropy" 或 "ce_entropy"
    lambda_entropy: float = 0.1  # KL方案的熵权重λ
    gamma_entropy: float = 0.1   # CE方案的熵权重γ
    beta_divergence: float = 0.2 # CE方案的全局一致性权重β
    tau: float = 1.0  # 权重计算温度τ
    
    # ========== 聚合策略参数 ==========
    aggregation_method: str = "active_inference"  # "active_inference", "fedavg", "fedprox"
    fedprox_mu: float = 0.01  # FedProx正则化项权重μ
    
    # ========== Logits提取参数 ==========
    logits_mode: str = "cls"  # "cls", "last_token", "mean_pool"
    
    # ========== 服务器更新参数 ==========
    enable_server_update: bool = False
    server_lr: float = 1e-5
    
    # ========== 数据参数 ==========
    dataroot: str = "/home/sutongtong/LanTu_team3/dataset/nuScenes/train"
    version: str = "v1.0-trainval"
    batch_size: int = 3
    partition_mode: str = "iid"  # "iid", "non-iid-dirichlet", "non-iid-shard"
    dirichlet_alpha: float = 1.0 # 狄利克雷分布参数α，越小异构程度越高
    max_batches: int = 0  # 每轮每客户端最大训练batch数，0表示跑完整个epoch
    server_max_batches: int = 0  # 服务器公共数据集最大batch数，0表示跑完整个数据集
    num_workers: int = 2
    attack_ratio: float = 0.3  # 全局基础比例（如果启用随机则作为参考）
    num_clean_clients: int = 0 # 干净客户端数量 (attack_ratio 固定为 0)
    malicious_client_ratio: float = 0.0  # 恶意客户端比例 (0.0-1.0)
    num_points: int = 2048
    
    # ========== 攻击数据集参数 ==========
    use_attack_dataset: bool = True   # 是否使用真实攻击生成数据集
    use_synthetic_data: bool = False  # 是否使用合成数据（无需NuScenes）
    num_synthetic_samples: int = 1000 # 合成数据样本数
    num_classes: int = 5              # 分类类别数 (1正常 + 4攻击)
    
    # ========== 训练参数 ==========
    lr: float = 1e-5
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
        assert 0 <= self.malicious_client_ratio <= 1.0, "malicious_client_ratio必须在[0, 1]范围内"
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
        
        assert self.aggregation_method in ["active_inference", "fedavg", "fedprox"], \
            "aggregation_method必须是'active_inference', 'fedavg'或'fedprox'"
        assert self.fedprox_mu >= 0, "fedprox_mu必须非负"
        
        assert self.partition_mode in ["iid", "non-iid-dirichlet", "non-iid-shard"], \
            "partition_mode必须是'iid', 'non-iid-dirichlet'或'non-iid-shard'"
        assert self.dirichlet_alpha > 0, "dirichlet_alpha必须大于0"
        
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
