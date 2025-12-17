"""
工具函数模块

提供联邦学习中的辅助功能，包括logits提取、对齐和指标计算。
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional


def extract_logits(
    model_output: torch.Tensor,
    mode: str = "cls"
) -> torch.Tensor:
    """
    从模型输出中提取logits
    
    Args:
        model_output: 模型输出，可能是 [B, 1] (分类头) 或 [B, L, V] (token级)
        mode: 提取模式
            - "cls": 直接使用分类头输出
            - "last_token": 提取最后一个token的logits
            - "mean_pool": 对所有token做平均池化
            
    Returns:
        torch.Tensor: 提取的logits
    """
    if mode == "cls":
        # 分类头输出，直接返回
        return model_output
    
    elif mode == "last_token":
        # 提取最后一个token
        if len(model_output.shape) == 3:  # [B, L, V]
            return model_output[:, -1, :]  # [B, V]
        else:
            return model_output
    
    elif mode == "mean_pool":
        # 平均池化
        if len(model_output.shape) == 3:  # [B, L, V]
            return model_output.mean(dim=1)  # [B, V]
        else:
            return model_output
    
    else:
        raise ValueError(f"未知的logits提取模式: {mode}")


def align_logits(
    logits_list: List[torch.Tensor],
    target_shape: Optional[Tuple[int, ...]] = None
) -> List[torch.Tensor]:
    """
    对齐不同客户端的logits形状
    
    Args:
        logits_list: 客户端logits列表
        target_shape: 目标形状，如果为None则使用第一个logits的形状
        
    Returns:
        List[torch.Tensor]: 对齐后的logits列表
    """
    if not logits_list:
        return []
    
    if target_shape is None:
        target_shape = logits_list[0].shape
    
    aligned_logits = []
    for logits in logits_list:
        if logits.shape == target_shape:
            aligned_logits.append(logits)
        else:
            # 尝试对齐
            if len(logits.shape) == 2 and len(target_shape) == 2:
                # [B, C] 形状
                if logits.shape[1] < target_shape[1]:
                    # Padding
                    pad_size = target_shape[1] - logits.shape[1]
                    padded = torch.nn.functional.pad(logits, (0, pad_size), value=0)
                    aligned_logits.append(padded)
                elif logits.shape[1] > target_shape[1]:
                    # 截断
                    aligned_logits.append(logits[:, :target_shape[1]])
                else:
                    aligned_logits.append(logits)
            else:
                # 形状差异太大，使用原始logits并警告
                print(f"Warning: 无法对齐logits形状 {logits.shape} 到 {target_shape}")
                aligned_logits.append(logits)
    
    return aligned_logits


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """
    计算分类指标
    
    Args:
        predictions: 预测标签 [N]
        labels: 真实标签 [N]
        
    Returns:
        dict: 包含accuracy, precision, recall, f1_score的字典
    """
    # 确保是numpy数组
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # 计算混淆矩阵元素
    tp = np.sum((predictions == 1) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    
    # 计算指标
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1_score),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }


def logits_statistics(logits: torch.Tensor) -> Dict[str, float]:
    """
    计算logits的统计信息
    
    Args:
        logits: logits张量
        
    Returns:
        dict: 包含均值、方差、最大值、最小值的字典
    """
    return {
        'mean': float(logits.mean().item()),
        'std': float(logits.std().item()),
        'max': float(logits.max().item()),
        'min': float(logits.min().item()),
        'shape': list(logits.shape)
    }


def partition_data_iid(
    dataset_size: int,
    num_clients: int
) -> List[List[int]]:
    """
    IID数据分区：将数据均匀随机分配给各客户端
    
    Args:
        dataset_size: 数据集大小
        num_clients: 客户端数量
        
    Returns:
        List[List[int]]: 每个客户端的样本索引列表
    """
    # 随机打乱索引
    indices = np.random.permutation(dataset_size).tolist()
    
    # 均匀分配
    samples_per_client = dataset_size // num_clients
    client_indices = []
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        if i == num_clients - 1:
            # 最后一个客户端获取剩余所有数据
            end_idx = dataset_size
        else:
            end_idx = (i + 1) * samples_per_client
        
        client_indices.append(indices[start_idx:end_idx])
    
    return client_indices


def partition_data_non_iid(
    dataset_size: int,
    labels: np.ndarray,
    num_clients: int,
    num_shards: int = 200
) -> List[List[int]]:
    """
    Non-IID数据分区：按标签分片，每个客户端获得少数几个分片
    
    Args:
        dataset_size: 数据集大小
        labels: 数据标签 [N]
        num_clients: 客户端数量
        num_shards: 分片数量
        
    Returns:
        List[List[int]]: 每个客户端的样本索引列表
    """
    # 按标签排序
    sorted_indices = np.argsort(labels).tolist()
    
    # 分成num_shards个分片
    shard_size = dataset_size // num_shards
    shards = []
    for i in range(num_shards):
        start_idx = i * shard_size
        if i == num_shards - 1:
            end_idx = dataset_size
        else:
            end_idx = (i + 1) * shard_size
        shards.append(sorted_indices[start_idx:end_idx])
    
    # 随机分配分片给客户端
    shards_per_client = num_shards // num_clients
    np.random.shuffle(shards)
    
    client_indices = []
    for i in range(num_clients):
        start_shard = i * shards_per_client
        if i == num_clients - 1:
            end_shard = num_shards
        else:
            end_shard = (i + 1) * shards_per_client
        
        client_data = []
        for shard_idx in range(start_shard, end_shard):
            client_data.extend(shards[shard_idx])
        
        client_indices.append(client_data)
    
    return client_indices


def aggregate_metrics(
    client_metrics_list: List[Dict[str, float]]
) -> Dict[str, float]:
    """
    聚合多个客户端的指标
    
    Args:
        client_metrics_list: 客户端指标列表
        
    Returns:
        dict: 平均指标
    """
    if not client_metrics_list:
        return {}
    
    # 收集所有指标的键
    keys = client_metrics_list[0].keys()
    
    # 计算平均值
    aggregated = {}
    for key in keys:
        values = [m[key] for m in client_metrics_list if key in m]
        if values:
            aggregated[f'avg_{key}'] = float(np.mean(values))
            aggregated[f'std_{key}'] = float(np.std(values))
    
    return aggregated


def create_uniform_prior(batch_size: int, num_classes: int, device: str = 'cpu') -> torch.Tensor:
    """
    创建均匀先验logits（用于第一轮或无先验时）
    
    Args:
        batch_size: 批次大小
        num_classes: 类别数
        device: 设备
        
    Returns:
        torch.Tensor: 均匀先验logits [B, C]
    """
    # 均匀分布对应的logits为全0（softmax后每个类别概率相等）
    return torch.zeros(batch_size, num_classes, device=device)


def save_checkpoint(
    model,
    optimizer,
    round_id: int,
    metrics: Dict[str, float],
    filepath: str
) -> None:
    """
    保存模型检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        round_id: 当前回合
        metrics: 指标字典
        filepath: 保存路径
    """
    checkpoint = {
        'round_id': round_id,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath: str, model, optimizer=None) -> Dict[str, any]:
    """
    加载模型检查点
    
    Args:
        filepath: 检查点路径
        model: 模型
        optimizer: 优化器（可选）
        
    Returns:
        dict: 包含round_id和metrics的字典
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and checkpoint['optimizer_state_dict']:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return {
        'round_id': checkpoint['round_id'],
        'metrics': checkpoint['metrics']
    }
