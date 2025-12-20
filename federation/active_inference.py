"""
主动推理模块

实现基于自由能(Free Energy)的主动推理机制，用于评估客户端可信度并计算贡献权重。

支持两种自由能计算方案：
1. KL散度 + 熵: F_i = KL(q_i || p_s) + λ * H(q_i)
2. 交叉熵 + 熵: F_i = CE(y_true, q_i) + γ * H(q_i)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List


def safe_softmax(logits: torch.Tensor, temperature: float = 1.0, epsilon: float = 1e-8) -> torch.Tensor:
    """
    数值稳定的softmax
    
    Args:
        logits: 输入logits [B, C]
        temperature: 温度参数
        epsilon: 数值稳定性常数
        
    Returns:
        torch.Tensor: softmax概率分布 [B, C]
    """
    logits = logits / temperature
    
    # 处理二分类单输出情况 [B, 1] -> [B, 2]
    if logits.shape[-1] == 1:
        prob1 = torch.sigmoid(logits)
        prob0 = 1.0 - prob1
        return torch.cat([prob0, prob1], dim=-1)
        
    # 减去最大值防止溢出
    logits = logits - torch.max(logits, dim=-1, keepdim=True)[0]
    exp_logits = torch.exp(logits)
    return exp_logits / (exp_logits.sum(dim=-1, keepdim=True) + epsilon)


def safe_entropy(probs: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    数值稳定的熵计算
    H(p) = -Σ p * log(p)
    
    Args:
        probs: 概率分布 [B, C]
        epsilon: 数值稳定性常数
        
    Returns:
        torch.Tensor: 熵值 [B]
    """
    probs = torch.clamp(probs, min=epsilon, max=1.0)
    return -torch.sum(probs * torch.log(probs + epsilon), dim=-1)


def safe_kl_divergence(q: torch.Tensor, p: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    数值稳定的KL散度计算
    KL(q || p) = Σ q * log(q / p)
    
    Args:
        q: 预测分布 [B, C]
        p: 先验分布 [B, C]
        epsilon: 数值稳定性常数
        
    Returns:
        torch.Tensor: KL散度 [B]
    """
    q = torch.clamp(q, min=epsilon, max=1.0)
    p = torch.clamp(p, min=epsilon, max=1.0)
    return torch.sum(q * torch.log((q + epsilon) / (p + epsilon)), dim=-1)


def free_energy_kl_entropy(
    client_logits: torch.Tensor,
    server_prior_logits: torch.Tensor,
    temperature: float = 1.0,
    lambda_entropy: float = 0.1,
    epsilon: float = 1e-8
) -> float:
    """
    方案1: 基于KL散度和熵的自由能
    F_i = KL(q_i || p_s) + λ * H(q_i)
    
    该方案衡量客户端预测与服务器先验的偏离程度，以及预测的不确定性。
    F_i越小表示客户端预测与先验一致且确定性高，因此更可信。
    
    Args:
        client_logits: 客户端预测logits [B, C]
        server_prior_logits: 服务器先验logits [B, C]
        temperature: 温度参数T，用于软化分布
        lambda_entropy: 熵权重λ，控制不确定性惩罚
        epsilon: 数值稳定性常数
        
    Returns:
        float: 自由能值（标量）
    """
    # 转换为概率分布
    q_i = safe_softmax(client_logits, temperature, epsilon)  # 客户端预测分布
    p_s = safe_softmax(server_prior_logits, temperature, epsilon)  # 服务器先验分布
    
    # 计算KL散度
    kl_div = safe_kl_divergence(q_i, p_s, epsilon)  # [B]
    
    # 计算熵
    entropy = safe_entropy(q_i, epsilon)  # [B]
    
    # 自由能 = KL散度 + λ * 熵
    free_energy = kl_div + lambda_entropy * entropy  # [B]
    
    # 返回批次平均值
    return free_energy.mean().item()


def free_energy_ce_entropy(
    client_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    gamma_entropy: float = 0.1,
    epsilon: float = 1e-8
) -> float:
    """
    方案2: 基于交叉熵和熵的自由能
    F_i = CE(y_true, q_i) + γ * H(q_i)
    
    该方案衡量客户端预测对真实标签的解释能力和不确定性。
    F_i越小表示客户端能准确预测且确定性高，因此更可信。
    
    Args:
        client_logits: 客户端预测logits [B, C]
        labels: 真实标签 [B]
        temperature: 温度参数T，用于软化分布
        gamma_entropy: 熵权重γ，控制不确定性惩罚
        epsilon: 数值稳定性常数
        
    Returns:
        float: 自由能值（标量）
    """
    # 转换为概率分布
    q_i = safe_softmax(client_logits, temperature, epsilon)  # [B, C]
    
    # 计算交叉熵损失
    if client_logits.shape[-1] == 1:
        # 二分类单输出情况
        ce_loss = F.binary_cross_entropy_with_logits(
            client_logits / temperature,
            labels.float().view(-1, 1),
            reduction='none'
        ).squeeze(-1)
    else:
        # 多分类情况
        ce_loss = F.cross_entropy(
            client_logits / temperature,
            labels,
            reduction='none'
        )  # [B]
    
    # 计算熵
    entropy = safe_entropy(q_i, epsilon)  # [B]
    
    # 自由能 = 交叉熵 + γ * 熵
    free_energy = ce_loss + gamma_entropy * entropy  # [B]
    
    # 返回批次平均值
    return free_energy.mean().item()


def compute_client_weights(
    free_energies: List[float],
    tau: float = 1.0,
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    基于自由能计算归一化的客户端权重
    w_i = softmax(-F_i / τ)
    
    自由能越小的客户端获得越大的权重。
    温度参数τ控制权重分布的平滑度：
    - τ较小：权重分布更集中在低自由能客户端
    - τ较大：权重分布更均匀
    
    Args:
        free_energies: 各客户端的自由能列表 [F_1, F_2, ..., F_N]
        tau: 温度参数τ，控制权重分布的平滑度
        epsilon: 数值稳定性常数
        
    Returns:
        np.ndarray: 归一化权重数组 [w_1, w_2, ..., w_N]，和为1
    """
    if not free_energies:
        return np.array([])
    
    # 转换为numpy数组
    F = np.array(free_energies, dtype=np.float64)
    
    # 处理异常值（NaN或Inf）
    if np.any(np.isnan(F)) or np.any(np.isinf(F)):
        print(f"Warning: 检测到异常自由能值，使用均匀权重")
        return np.ones(len(F)) / len(F)
    
    # 计算 -F_i / τ
    scaled_neg_F = -F / tau
    
    # 减去最大值以提高数值稳定性
    scaled_neg_F = scaled_neg_F - np.max(scaled_neg_F)
    
    # 计算softmax
    exp_values = np.exp(scaled_neg_F)
    weights = exp_values / (np.sum(exp_values) + epsilon)
    
    # 确保权重和为1（由于浮点误差可能略有偏差）
    weights = weights / np.sum(weights)
    
    return weights


def compute_free_energy(
    client_logits: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    server_prior_logits: Optional[torch.Tensor] = None,
    mode: str = "kl_entropy",
    temperature: float = 1.0,
    lambda_entropy: float = 0.1,
    gamma_entropy: float = 0.1
) -> float:
    """
    统一的自由能计算接口
    
    Args:
        client_logits: 客户端预测logits [B, C]
        labels: 真实标签 [B]，CE方案需要
        server_prior_logits: 服务器先验logits [B, C]，KL方案需要
        mode: 计算模式，"kl_entropy" 或 "ce_entropy"
        temperature: 温度参数
        lambda_entropy: KL方案的熵权重
        gamma_entropy: CE方案的熵权重
        
    Returns:
        float: 自由能值
    """
    if mode == "kl_entropy":
        if server_prior_logits is None:
            raise ValueError("KL方案需要提供server_prior_logits")
        return free_energy_kl_entropy(
            client_logits,
            server_prior_logits,
            temperature,
            lambda_entropy
        )
    elif mode == "ce_entropy":
        if labels is None:
            raise ValueError("CE方案需要提供labels")
        return free_energy_ce_entropy(
            client_logits,
            labels,
            temperature,
            gamma_entropy
        )
    else:
        raise ValueError(f"未知的自由能计算模式: {mode}")


# ========== 测试和验证函数 ==========

def validate_weight_properties(weights: np.ndarray, tolerance: float = 1e-6) -> bool:
    """
    验证权重是否满足正确性属性
    
    Property 3: 权重归一化
    - 所有权重非负
    - 权重和为1
    
    Args:
        weights: 权重数组
        tolerance: 数值容差
        
    Returns:
        bool: 是否满足属性
    """
    # 检查非负性
    if not np.all(weights >= 0):
        return False
    
    # 检查和为1
    if not np.abs(np.sum(weights) - 1.0) < tolerance:
        return False
    
    return True


def validate_monotonicity(
    free_energies: List[float],
    weights: np.ndarray,
    tolerance: float = 1e-6
) -> bool:
    """
    验证自由能单调性属性
    
    Property 4: 自由能单调性
    - 如果F_i < F_j，则w_i > w_j
    
    Args:
        free_energies: 自由能列表
        weights: 权重数组
        tolerance: 数值容差
        
    Returns:
        bool: 是否满足属性
    """
    for i in range(len(free_energies)):
        for j in range(i + 1, len(free_energies)):
            if abs(free_energies[i] - free_energies[j]) < tolerance:
                # 自由能相近，跳过
                continue
            
            if free_energies[i] < free_energies[j]:
                if weights[i] <= weights[j]:
                    return False
            else:
                if weights[i] >= weights[j]:
                    return False
    
    return True
