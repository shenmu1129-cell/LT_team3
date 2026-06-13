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
from typing import Optional, List, Dict


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
    server_prior_logits: Optional[torch.Tensor] = None,
    beta_divergence: float = 0.2,
    epsilon: float = 1e-8
) -> float:
    """
    方案2: 基于交叉熵和熵的自由能 (增强版: 增加全局一致性约束)
    F_i = CE(y_true, q_i) + γ * H(q_i) + β * KL(q_i || p_s)
    
    该方案衡量客户端预测对真实标签的解释能力、预测的不确定性，
    以及相对于全局先验的一致性。增加KL项可以防止恶意客户端通过过度的“虚假自信”降低自由能。
    
    Args:
        client_logits: 客户端预测logits [B, C]
        labels: 真实标签 [B]
        temperature: 温度参数T，用于软化分布
        gamma_entropy: 熵权重γ
        server_prior_logits: 可选的服务器先验logits [B, C]
        beta_divergence: 全局一致性权重β
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
    
    # 2. 计算预测分布的熵
    entropy = safe_entropy(q_i, epsilon)  # [B]
    
    # [优化] 针对AD场景的逻辑修正：
    # 恶意的投毒攻击往往表现为极低极低的不确定性（过分确信某个错误类别）
    # 而干净的场景（复杂路况）往往带有自然的合理不确定性（熵稍高）。
    # 如果熵过低（< 0.1），我们将其视为“可疑的过度拟合”，不再提供奖励，甚至给予微小惩罚。
    entropy_penalty = torch.where(entropy < 0.1, 1.0 - entropy, entropy)
    
    # 3. 计算相对于先验的散度 (全局一致性惩罚)
    if server_prior_logits is not None:
        p_s = safe_softmax(server_prior_logits, temperature, epsilon)
        kl_div = safe_kl_divergence(q_i, p_s, epsilon)
    else:
        kl_div = torch.zeros_like(entropy)
    
    # 自由能 = 交叉熵 + γ * 优化的熵项 + β * KL散度
    free_energy = ce_loss + gamma_entropy * entropy_penalty + beta_divergence * kl_div  # [B]
    
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
    
    优化点：增加数值平滑，防止在训练初期因自由能差异过大导致权重极化。
    """
    if not free_energies:
        return np.array([])
    
    # 转换为numpy数组
    F = np.array(free_energies, dtype=np.float64)
    
    # [优化] 如果是训练初期或自由能差异过大，进行对比度调整
    # 限制自由能的动态范围，防止单个客户端权重过低
    f_min = np.min(F)
    f_max = np.max(F)
    if f_max - f_min > 5.0:
        F = np.clip(F, f_min, f_min + 5.0)
    
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
    gamma_entropy: float = 0.1,
    beta_divergence: float = 0.2
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
        beta_divergence: 全局一致性权重 (CE方案专属)
        
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
            gamma_entropy,
            server_prior_logits=server_prior_logits,
            beta_divergence=beta_divergence
        )
    else:
        raise ValueError(f"未知的自由能计算模式: {mode}")


def compute_free_energy_components(
    client_logits: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    server_prior_logits: Optional[torch.Tensor] = None,
    mode: str = "kl_entropy",
    temperature: float = 1.0,
    lambda_entropy: float = 0.1,
    gamma_entropy: float = 0.1,
    beta_divergence: float = 0.2,
    epsilon: float = 1e-8
) -> Dict[str, float]:
    """
    返回自由能分解项，供恶意客户端区分、ROC和失败模式分析使用。
    """
    q_i = safe_softmax(client_logits, temperature, epsilon)
    entropy = safe_entropy(q_i, epsilon)

    if server_prior_logits is not None:
        p_s = safe_softmax(server_prior_logits, temperature, epsilon)
        kl_div = safe_kl_divergence(q_i, p_s, epsilon)
    else:
        kl_div = torch.zeros_like(entropy)

    if labels is not None:
        if client_logits.shape[-1] == 1:
            ce_loss = F.binary_cross_entropy_with_logits(
                client_logits / temperature,
                labels.float().view(-1, 1),
                reduction='none'
            ).squeeze(-1)
        else:
            ce_loss = F.cross_entropy(
                client_logits / temperature,
                labels,
                reduction='none'
            )
    else:
        ce_loss = torch.zeros_like(entropy)

    entropy_penalty = torch.where(entropy < 0.1, 1.0 - entropy, entropy)
    if mode == "kl_entropy":
        free_energy = kl_div + lambda_entropy * entropy
    elif mode == "ce_entropy":
        free_energy = ce_loss + gamma_entropy * entropy_penalty + beta_divergence * kl_div
    else:
        raise ValueError(f"未知的自由能计算模式: {mode}")

    return {
        'free_energy': float(free_energy.mean().item()),
        'kl_divergence': float(kl_div.mean().item()),
        'entropy': float(entropy.mean().item()),
        'entropy_penalty': float(entropy_penalty.mean().item()),
        'cross_entropy': float(ce_loss.mean().item()),
        'mode': mode,
    }


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
