"""
联邦客户端模块

实现FederatedClient类，封装客户端的本地训练、推理和与服务器的交互逻辑。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

from .config import FederatedConfig
from .active_inference import compute_free_energy
from .utils import extract_logits, compute_metrics
from .comm import ClientLogitsPackage


class FederatedClient:
    """联邦学习客户端"""
    
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        data_loader: DataLoader,
        device: str,
        config: FederatedConfig
    ):
        """
        Args:
            client_id: 客户端唯一标识
            model: Qwen3VL防御系统模型实例
            data_loader: 本地数据加载器
            device: 计算设备
            config: 联邦学习配置
        """
        self.client_id = client_id
        self.model = model.to(device)
        self.data_loader = data_loader
        self.device = device
        self.config = config
        
        # 初始化优化器
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        
        # 损失函数
        self.ce_criterion = nn.BCEWithLogitsLoss()
        
        # 统计信息
        self.local_loss_history = []
        self.free_energy_history = []
        self.weight_history = []
    
    def local_forward(
        self,
        batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        本地前向推理，提取logits
        
        Args:
            batch: 包含images, pointclouds, labels的批次数据
            
        Returns:
            logits: [B, num_classes] 预测logits
            labels: [B] 真实标签
            sample_ids: 样本ID列表
        """
        self.model.eval()
        
        images = batch['images']
        pointclouds = batch['pointclouds'].to(self.device)
        labels = batch['labels'].to(self.device)
        sample_ids = batch.get('sample_tokens', [])
        
        with torch.no_grad():
            # 使用train模式获取logits（分类头输出）
            logits = self.model(images, pointclouds, mode='train')  # [B, 1]
            
            # 根据配置提取logits
            logits = extract_logits(logits, mode=self.config.logits_mode)
        
        return logits, labels, sample_ids
    
    def compute_free_energy(
        self,
        logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        server_prior_logits: Optional[torch.Tensor] = None
    ) -> float:
        """
        计算自由能
        
        Args:
            logits: 本地预测logits
            labels: 真实标签(可选)
            server_prior_logits: 服务器先验logits(可选)
            
        Returns:
            free_energy: 自由能值
        """
        free_energy = compute_free_energy(
            client_logits=logits,
            labels=labels,
            server_prior_logits=server_prior_logits,
            mode=self.config.free_energy_mode,
            temperature=self.config.temperature,
            lambda_entropy=self.config.lambda_entropy,
            gamma_entropy=self.config.gamma_entropy
        )
        
        self.free_energy_history.append(free_energy)
        return free_energy
    
    def compute_distillation_loss(
        self,
        local_logits: torch.Tensor,
        global_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算蒸馏损失
        Loss = α * CE(y_true, local_pred) + β * KL(local_pred || global_pred)
        
        Args:
            local_logits: 本地预测logits [B, C]
            global_logits: 全局logits [B, C]
            labels: 真实标签 [B]
            
        Returns:
            torch.Tensor: 蒸馏损失
        """
        alpha = self.config.alpha
        beta = self.config.beta
        T = self.config.temperature
        
        # 交叉熵损失（硬标签）
        ce_loss = self.ce_criterion(local_logits.squeeze(-1), labels.float())
        
        # KL散度损失（软标签）
        # 使用温度T软化分布
        if local_logits.shape[-1] == 1:
            # 二分类单输出情况：转换为 [B, 2] 概率分布
            local_prob1 = torch.sigmoid(local_logits / T)
            local_prob0 = 1.0 - local_prob1
            local_soft = torch.log(torch.cat([local_prob0, local_prob1], dim=-1) + 1e-8)
            
            global_prob1 = torch.sigmoid(global_logits / T)
            global_prob0 = 1.0 - global_prob1
            global_soft = torch.cat([global_prob0, global_prob1], dim=-1)
        else:
            # 多分类情况
            local_soft = F.log_softmax(local_logits / T, dim=-1)
            global_soft = F.softmax(global_logits / T, dim=-1)
        
        kl_loss = F.kl_div(
            local_soft,
            global_soft,
            reduction='batchmean'
        ) * (T * T)  # 温度缩放
        
        # 总损失
        total_loss = alpha * ce_loss + beta * kl_loss
        
        return total_loss
    
    def local_update_with_distillation(
        self,
        global_logits: torch.Tensor,
        num_epochs: Optional[int] = None
    ) -> Dict[str, float]:
        """
        使用蒸馏损失更新本地模型
        
        Args:
            global_logits: 服务器聚合的全局logits
            num_epochs: 本地训练轮数，如果为None则使用config中的值
            
        Returns:
            dict: 包含loss和指标的字典
        """
        if num_epochs is None:
            num_epochs = self.config.local_epochs
        
        self.model.train()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        num_batches = 0
        
        for epoch in range(num_epochs):
            for batch_idx, batch in enumerate(self.data_loader):
                # 检查是否达到最大batch限制
                if hasattr(self.config, 'max_batches') and self.config.max_batches > 0:
                    if batch_idx >= self.config.max_batches:
                        break
                
                images = batch['images']
                pointclouds = batch['pointclouds'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                self.optimizer.zero_grad()
                
                # 前向传播
                local_logits = self.model(images, pointclouds, mode='train')  # [B, 1]
                
                # 确保global_logits在正确的设备上
                global_logits_batch = global_logits.to(self.device)
                
                # 计算蒸馏损失
                loss = self.compute_distillation_loss(
                    local_logits,
                    global_logits_batch,
                    labels
                )
                
                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # 统计
                total_loss += loss.item()
                num_batches += 1
                
                # 收集预测和标签用于计算指标
                with torch.no_grad():
                    predictions = torch.sigmoid(local_logits.squeeze(-1))
                    predicted_labels = (predictions > 0.5).long()
                    all_predictions.extend(predicted_labels.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
        
        print(f"  [Client {self.client_id}] 本地更新完成: 运行了 {num_batches} 个 batches")
        
        # 计算平均损失
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.local_loss_history.append(avg_loss)
        
        # 计算指标
        metrics = compute_metrics(
            np.array(all_predictions),
            np.array(all_labels)
        )
        metrics['loss'] = avg_loss
        
        return metrics
    
    def upload_logits(
        self,
        logits: torch.Tensor,
        sample_ids: List[str],
        timestamp: float
    ) -> ClientLogitsPackage:
        """
        封装logits为上传包
        
        Args:
            logits: 本地logits
            sample_ids: 样本ID列表
            timestamp: 时间戳
            
        Returns:
            ClientLogitsPackage: 上传包
        """
        return ClientLogitsPackage(
            client_id=self.client_id,
            logits=logits.cpu(),  # 转到CPU以便序列化
            sample_ids=sample_ids,
            batch_size=logits.size(0),
            timestamp=timestamp
        )
    
    def receive_global_logits(
        self,
        global_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        接收服务器下发的全局logits
        
        Args:
            global_logits: 全局logits
            
        Returns:
            torch.Tensor: 全局logits（确保在正确设备上）
        """
        return global_logits.to(self.device)
    
    def evaluate(self) -> Dict[str, float]:
        """
        在本地数据上评估模型
        
        Returns:
            dict: 评估指标
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.data_loader:
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
        metrics = compute_metrics(
            np.array(all_predictions),
            np.array(all_labels)
        )
        
        return metrics
    
    def get_model_state(self) -> Dict[str, Any]:
        """获取模型状态"""
        return {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
    
    def set_model_state(self, state: Dict[str, Any]) -> None:
        """设置模型状态"""
        if 'model_state_dict' in state:
            self.model.load_state_dict(state['model_state_dict'])
        if 'optimizer_state_dict' in state:
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
    
    def __repr__(self) -> str:
        return f"FederatedClient(id={self.client_id}, device={self.device})"
