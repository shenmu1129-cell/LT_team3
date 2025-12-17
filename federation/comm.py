"""
通信管理模块

处理联邦学习中的logits序列化、传输和通信量统计。
"""

import torch
import numpy as np
from typing import Dict, Any
import io


class CommunicationManager:
    """通信管理器"""
    
    def __init__(self):
        self.total_bytes_sent = 0
        self.total_bytes_received = 0
        self.round_bytes_sent = 0
        self.round_bytes_received = 0
    
    def serialize_logits(self, logits: torch.Tensor) -> bytes:
        """
        序列化logits为字节流
        
        Args:
            logits: torch.Tensor, 需要序列化的logits
            
        Returns:
            bytes: 序列化后的字节流
        """
        buffer = io.BytesIO()
        torch.save(logits.cpu(), buffer)
        data = buffer.getvalue()
        
        # 统计发送字节数
        num_bytes = len(data)
        self.total_bytes_sent += num_bytes
        self.round_bytes_sent += num_bytes
        
        return data
    
    def deserialize_logits(self, data: bytes) -> torch.Tensor:
        """
        反序列化字节流为logits
        
        Args:
            data: bytes, 序列化的字节流
            
        Returns:
            torch.Tensor: 反序列化后的logits
        """
        # 统计接收字节数
        num_bytes = len(data)
        self.total_bytes_received += num_bytes
        self.round_bytes_received += num_bytes
        
        buffer = io.BytesIO(data)
        logits = torch.load(buffer)
        
        return logits
    
    def get_communication_stats(self) -> Dict[str, float]:
        """
        获取通信统计信息
        
        Returns:
            dict: 包含通信量统计的字典（单位：MB）
        """
        return {
            'total_sent_mb': self.total_bytes_sent / (1024 * 1024),
            'total_received_mb': self.total_bytes_received / (1024 * 1024),
            'round_sent_mb': self.round_bytes_sent / (1024 * 1024),
            'round_received_mb': self.round_bytes_received / (1024 * 1024),
            'total_mb': (self.total_bytes_sent + self.total_bytes_received) / (1024 * 1024),
            'round_mb': (self.round_bytes_sent + self.round_bytes_received) / (1024 * 1024),
        }
    
    def reset_round_stats(self) -> None:
        """重置当前回合的统计信息"""
        self.round_bytes_sent = 0
        self.round_bytes_received = 0
    
    def get_tensor_size_mb(self, tensor: torch.Tensor) -> float:
        """
        计算tensor的大小（MB）
        
        Args:
            tensor: torch.Tensor
            
        Returns:
            float: tensor大小（MB）
        """
        num_bytes = tensor.element_size() * tensor.nelement()
        return num_bytes / (1024 * 1024)


class ClientLogitsPackage:
    """客户端上传的logits包"""
    
    def __init__(
        self,
        client_id: int,
        logits: torch.Tensor,
        sample_ids: list,
        batch_size: int,
        timestamp: float
    ):
        self.client_id = client_id
        self.logits = logits
        self.sample_ids = sample_ids
        self.batch_size = batch_size
        self.timestamp = timestamp
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'client_id': self.client_id,
            'logits_shape': list(self.logits.shape),
            'batch_size': self.batch_size,
            'timestamp': self.timestamp,
            'num_samples': len(self.sample_ids)
        }


class GlobalLogitsPackage:
    """服务器下发的全局logits包"""
    
    def __init__(
        self,
        global_logits: torch.Tensor,
        weights: np.ndarray,
        free_energies: list,
        round_id: int,
        timestamp: float
    ):
        self.global_logits = global_logits
        self.weights = weights
        self.free_energies = free_energies
        self.round_id = round_id
        self.timestamp = timestamp
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'round_id': self.round_id,
            'global_logits_shape': list(self.global_logits.shape),
            'weights': self.weights.tolist(),
            'free_energies': self.free_energies,
            'timestamp': self.timestamp
        }
