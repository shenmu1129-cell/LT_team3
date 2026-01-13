"""
联邦服务器模块

实现FederatedServer类，负责收集客户端logits、计算权重、聚合并下发全局知识。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import time

from .config import FederatedConfig
from .active_inference import compute_client_weights
from .utils import logits_statistics, align_logits
from .comm import GlobalLogitsPackage


class FederatedServer:
    """联邦学习服务器"""
    
    def __init__(
        self,
        model: nn.Module,
        device: str,
        config: FederatedConfig,
        server_data_loader: Optional[DataLoader] = None
    ):
        """
        Args:
            model: 服务器端Qwen3VL模型
            device: 计算设备
            config: 联邦学习配置
            server_data_loader: 服务器自有数据加载器（可选）
        """
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.server_data_loader = server_data_loader
        
        # 如果启用服务器更新，初始化优化器
        if config.enable_server_update:
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.SGD(
                trainable_params,
                lr=config.server_lr,
                momentum=0.9,
                weight_decay=config.weight_decay
            )
        else:
            self.optimizer = None
        
        # 历史记录
        self.aggregation_history = []
        self.weight_history = []
        
        # 先验logits（用于第一轮或KL方案）
        self.prior_logits = None
    
    def collect_client_logits(
        self,
        client_logits_dict: Dict[int, torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """
        收集所有客户端的logits
        
        Args:
            client_logits_dict: {client_id: logits}映射
            
        Returns:
            logits_list: 按客户端ID排序的logits列表
            client_ids: 客户端ID列表
        """
        # 按客户端ID排序
        sorted_items = sorted(client_logits_dict.items(), key=lambda x: x[0])
        client_ids = [cid for cid, _ in sorted_items]
        logits_list = [logits.to(self.device) for _, logits in sorted_items]
        
        # 对齐logits形状
        logits_list = align_logits(logits_list)
        
        return logits_list, client_ids
    
    def compute_client_weights(
        self,
        free_energies: List[float]
    ) -> np.ndarray:
        """
        根据选定的聚合策略计算客户端权重
        
        Args:
            free_energies: 各客户端的自由能列表
            
        Returns:
            weights: 归一化的权重数组
        """
        method = self.config.aggregation_method
        num_clients = len(free_energies)
        
        if method == "active_inference":
            # 基于自由能的主动推理权重
            weights = compute_client_weights(
                free_energies,
                tau=self.config.tau
            )
        elif method in ["fedavg", "fedprox"]:
            # FedAvg和FedProx在聚合阶段通常使用简单平均(或按样本量加权)
            # 在没有实时获取样本量的情况下，使用 uniform 权重
            weights = np.ones(num_clients) / num_clients
        else:
            print(f"Warning: 未知的聚合方式 {method}, 退回到主动推理")
            weights = compute_client_weights(
                free_energies,
                tau=self.config.tau
            )
        
        self.weight_history.append(weights.copy())
        return weights
    
    def aggregate_logits(
        self,
        client_logits: List[torch.Tensor],
        weights: np.ndarray
    ) -> torch.Tensor:
        """
        加权聚合客户端logits
        global_logits = Σ(w_i * logits_i)
        
        Args:
            client_logits: 客户端logits列表
            weights: 权重数组
            
        Returns:
            global_logits: 聚合后的全局logits
        """
        if not client_logits:
            raise ValueError("客户端logits列表为空")
        
        # 确保权重和为1
        weights = weights / np.sum(weights)
        
        # 加权求和
        global_logits = torch.zeros_like(client_logits[0])
        for i, logits in enumerate(client_logits):
            global_logits += weights[i] * logits
        
        # 记录聚合历史
        stats = logits_statistics(global_logits)
        self.aggregation_history.append(stats)
        
        # 更新先验logits
        self.prior_logits = global_logits.clone().detach()
        
        return global_logits
    
    def broadcast_global_logits(
        self,
        global_logits: torch.Tensor,
        weights: np.ndarray,
        free_energies: List[float],
        round_id: int
    ) -> GlobalLogitsPackage:
        """
        广播全局logits给所有客户端
        
        Args:
            global_logits: 聚合的全局logits
            weights: 客户端权重
            free_energies: 客户端自由能
            round_id: 当前回合ID
            
        Returns:
            GlobalLogitsPackage: 全局logits包
        """
        return GlobalLogitsPackage(
            global_logits=global_logits.cpu(),  # 转到CPU以便传输
            weights=weights,
            free_energies=free_energies,
            round_id=round_id,
            timestamp=time.time()
        )
    
    def server_update(
        self,
        global_logits: torch.Tensor,
        batch: Optional[Dict[str, Any]] = None,
        num_epochs: int = 1
    ) -> Dict[str, float]:
        """
        服务器端模型自我更新（可选）
        使用聚合的全局logits作为软标签进行蒸馏训练
        
        Args:
            global_logits: 聚合的全局logits
            batch: 用于更新的数据批次。如果为None，则从server_data_loader中获取。
            num_epochs: 训练轮数
            
        Returns:
            dict: 包含loss的字典
        """
        if not self.config.enable_server_update:
            return {'loss': 0.0}
        
        if self.server_data_loader is None and batch is None:
            return {'loss': 0.0}
        
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        # 如果提供了特定的 batch，则只在该 batch 上更新
        if batch is not None:
            batches = [batch]
        else:
            # 否则使用整个 data_loader (注意：这通常要求 global_logits 与 loader 数据对应)
            batches = self.server_data_loader

        for epoch in range(num_epochs):
            for batch_idx, current_batch in enumerate(batches):
                # 检查是否达到最大batch限制
                if batch is None and hasattr(self.config, 'server_max_batches') and self.config.server_max_batches > 0:
                    if batch_idx >= self.config.server_max_batches:
                        break
                
                images = current_batch['images']
                pointclouds = current_batch['pointclouds'].to(self.device)
                
                self.optimizer.zero_grad()
                
                # 服务器前向传播
                server_logits = self.model(images, pointclouds, mode='train')
                
                # 使用全局logits作为软标签
                # 注意：如果 batches 包含多个 batch，这里的 global_logits 必须与当前 batch 对应
                # 在当前联邦回合逻辑中，global_logits 仅对应一个 batch
                global_logits_batch = global_logits.to(self.device)
                
                # 检查 batch size 是否匹配
                if server_logits.size(0) != global_logits_batch.size(0):
                    # 如果不匹配，可能是因为 global_logits 只对应部分数据，或者 loader 到了最后一个不完整的 batch
                    # 这里简单跳过或截断
                    min_size = min(server_logits.size(0), global_logits_batch.size(0))
                    server_logits = server_logits[:min_size]
                    global_logits_batch = global_logits_batch[:min_size]
                
                # KL散度损失
                T = self.config.temperature
                
                # 检查是否为二分类单输出情况 [B, 1]
                if server_logits.shape[-1] == 1:
                    # 转换为 [B, 2] 概率分布以计算 KL 散度
                    server_prob1 = torch.sigmoid(server_logits / T)
                    server_prob0 = 1.0 - server_prob1
                    server_soft = torch.log(torch.cat([server_prob0, server_prob1], dim=-1) + 1e-8)
                    
                    global_prob1 = torch.sigmoid(global_logits_batch / T)
                    global_prob0 = 1.0 - global_prob1
                    global_soft = torch.cat([global_prob0, global_prob1], dim=-1)
                else:
                    # 多分类情况
                    server_soft = F.log_softmax(server_logits / T, dim=-1)
                    global_soft = F.softmax(global_logits_batch / T, dim=-1)
                
                loss = F.kl_div(
                    server_soft,
                    global_soft,
                    reduction='batchmean'
                ) * (T * T)
                
                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {'loss': avg_loss}
    
    def get_prior_logits(
        self,
        batch_size: int,
        num_classes: int
    ) -> torch.Tensor:
        """
        获取先验logits（用于自由能计算）
        
        Args:
            batch_size: 批次大小
            num_classes: 类别数
            
        Returns:
            torch.Tensor: 先验logits
        """
        if self.prior_logits is not None:
            # 使用上一轮的聚合logits作为先验
            return self.prior_logits.to(self.device)
        else:
            # 第一轮：使用均匀先验
            return torch.zeros(batch_size, num_classes, device=self.device)
    
    def get_aggregation_stats(self) -> Dict[str, Any]:
        """获取聚合统计信息"""
        if not self.aggregation_history:
            return {}
        
        latest_stats = self.aggregation_history[-1]
        return {
            'latest': latest_stats,
            'history_length': len(self.aggregation_history)
        }
    
    def get_weight_stats(self) -> Dict[str, Any]:
        """获取权重统计信息"""
        if not self.weight_history:
            return {}
        
        latest_weights = self.weight_history[-1]
        return {
            'latest_weights': latest_weights.tolist(),
            'mean': float(np.mean(latest_weights)),
            'std': float(np.std(latest_weights)),
            'max': float(np.max(latest_weights)),
            'min': float(np.min(latest_weights))
        }
    
    def evaluate(self) -> Dict[str, float]:
        """
        在服务器公共数据上评估模型
        
        Returns:
            dict: 评估指标
        """
        if self.server_data_loader is None:
            return {'accuracy': 0.0, 'f1_score': 0.0}
            
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.server_data_loader):
                # 检查是否达到最大batch限制
                if hasattr(self.config, 'server_max_batches') and self.config.server_max_batches > 0:
                    if batch_idx >= self.config.server_max_batches:
                        break
                
                images = batch['images']
                pointclouds = batch['pointclouds'].to(self.device)
                labels = batch['labels']
                
                # 前向传播
                logits = self.model(images, pointclouds, mode='train')
                
                # 预测
                predictions = torch.sigmoid(logits.squeeze(-1))
                predicted_labels = (predictions > 0.5).long()
                
                all_predictions.extend(predicted_labels.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # 计算指标
        from .utils import compute_metrics
        metrics = compute_metrics(
            np.array(all_predictions),
            np.array(all_labels)
        )
        
        return metrics

    def save_model(self, filepath: str) -> None:
        """保存服务器模型"""
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'prior_logits': self.prior_logits,
            'aggregation_history': self.aggregation_history,
            'weight_history': self.weight_history
        }
        torch.save(state, filepath)
    
    def load_model(self, filepath: str) -> None:
        """加载服务器模型"""
        state = torch.load(filepath)
        self.model.load_state_dict(state['model_state_dict'])
        
        if self.optimizer and state['optimizer_state_dict']:
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
        
        self.prior_logits = state.get('prior_logits')
        self.aggregation_history = state.get('aggregation_history', [])
        self.weight_history = state.get('weight_history', [])
    
    def __repr__(self) -> str:
        return f"FederatedServer(device={self.device}, enable_update={self.config.enable_server_update})"
