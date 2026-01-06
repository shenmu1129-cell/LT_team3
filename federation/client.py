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
        计算5分类蒸馏损失
        Loss = α * CE(y_true, local_pred) + β * KL(local_pred || global_pred)
        
        Args:
            local_logits: 本地预测logits [B, 5]
            global_logits: 全局logits [B, 5]
            labels: 真实标签 [B]，值为0-4
            
        Returns:
            torch.Tensor: 蒸馏损失
        """
        alpha = self.config.alpha
        beta = self.config.beta
        T = self.config.temperature
        
        # 交叉熵损失（硬标签）- 5分类
        ce_loss = F.cross_entropy(local_logits, labels)
        
        # KL散度损失（软标签）- 使用温度T软化分布
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
    
    def compute_proximal_loss(self, global_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算FedProx的近端项 (Proximal Term)
        Loss_prox = (mu / 2) * Σ || θ - θ_global ||^2
        
        Args:
            global_params: 全局模型参数字典
            
        Returns:
            torch.Tensor: 近端损失
        """
        prox_loss = 0.0
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in global_params:
                prox_loss += torch.norm(param - global_params[name].to(self.device))**2
        return (self.config.fedprox_mu / 2) * prox_loss
    
    def local_update_with_distillation(
        self,
        global_logits: torch.Tensor,
        target_batch: Optional[Dict[str, Any]] = None,
        num_epochs: Optional[int] = None
    ) -> Dict[str, float]:
        """
        使用蒸馏损失更新本地模型
        
        Args:
            global_logits: 服务器聚合的全局logits
            target_batch: 指定的训练批次。如果为None，则使用本地数据加载器。
            num_epochs: 本地训练轮数，如果为None则使用config中的值
            
        Returns:
            dict: 包含loss和指标的字典
        """
        if num_epochs is None:
            num_epochs = self.config.local_epochs
        
        self.model.train()
        
        # 如果是FedProx，保存当前模型权重作为参考(global_params)
        global_params = None
        if self.config.aggregation_method == "fedprox":
            global_params = {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []  # 保存预测概率用于计算AUC
        num_batches = 0
        
        # 决定训练批次
        if target_batch is not None:
            batches = [target_batch]
        else:
            batches = self.data_loader
        
        for epoch in range(num_epochs):
            for batch_idx, batch in enumerate(batches):
                # 检查是否达到最大batch限制
                if target_batch is None and hasattr(self.config, 'max_batches') and self.config.max_batches > 0:
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
                
                # 检查 batch size 是否匹配
                if local_logits.size(0) != global_logits_batch.size(0):
                    min_size = min(local_logits.size(0), global_logits_batch.size(0))
                    local_logits = local_logits[:min_size]
                    global_logits_batch = global_logits_batch[:min_size]
                    labels = labels[:min_size]
                
                # 计算蒸馏损失
                loss = self.compute_distillation_loss(
                    local_logits,
                    global_logits_batch,
                    labels
                )
                
                # 如果是FedProx，增加近端项
                if global_params is not None:
                    prox_loss = self.compute_proximal_loss(global_params)
                    loss += prox_loss
                
                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # 统计
                total_loss += loss.item()
                num_batches += 1
                
                # 收集预测和标签用于计算指标（5分类）
                with torch.no_grad():
                    # 5分类: local_logits形状为 [B, 5]
                    probs = torch.softmax(local_logits, dim=-1)  # [B, 5]
                    predicted_labels = torch.argmax(local_logits, dim=-1)  # [B] 预测类别0-4
                    attack_probs = 1 - probs[:, 0]  # 攻击概率 = 1 - P(normal)
                    
                    all_predictions.extend(predicted_labels.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probabilities.extend(attack_probs.cpu().numpy())  # 攻击概率用于AUC
        
        if self.config.verbose:
            print(f"  [Client {self.client_id}] 本地更新完成: 运行了 {num_batches} 个 batches")
        
        # 计算平均损失
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.local_loss_history.append(avg_loss)
        
        # 计算指标（5分类 + 二分类视角）
        metrics = compute_metrics(
            np.array(all_predictions),
            np.array(all_labels),
            np.array(all_probabilities) if all_probabilities else None,
            num_classes=5
        )
        
        return {
            'loss': avg_loss,
            'accuracy': metrics['accuracy'],
            'macro_f1': metrics.get('macro_f1', 0.0),
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'fpr': metrics.get('fpr', 0.0),
            'fnr': metrics.get('fnr', 0.0),
            'specificity': metrics.get('specificity', 0.0),
            'auc_roc': metrics.get('auc_roc', 0.5),
            'tp': metrics.get('tp', 0),
            'tn': metrics.get('tn', 0),
            'fp': metrics.get('fp', 0),
            'fn': metrics.get('fn', 0),
            # 5分类特有指标
            'per_class_f1': metrics.get('per_class_f1', [0.0]*5)
        }
    
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
    
    def detect_attacks(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        使用LLM检测攻击并生成分析结果
        
        Args:
            batch: 包含images, pointclouds的批次数据
            
        Returns:
            List[Dict]: 每个样本的攻击检测结果，包含:
                - is_attack: 是否检测到攻击
                - attack_type: 攻击类型
                - confidence: 置信度
                - risk_level: 风险等级
                - analysis: 详细分析
        """
        self.model.eval()
        
        images = batch['images']
        pointclouds = batch['pointclouds'].to(self.device)
        
        # 使用detect模式调用LLM进行攻击检测
        with torch.no_grad():
            detection_results = self.model(images, pointclouds, mode='detect')
        
        return detection_results
    
    def generate_defense_strategies(
        self, 
        batch: Dict[str, Any],
        attack_types: List[str] = None
    ) -> List[str]:
        """
        使用LLM生成防御策略
        
        Args:
            batch: 包含images, pointclouds的批次数据
            attack_types: 检测到的攻击类型列表
            
        Returns:
            List[str]: 每个样本的防御策略文本
        """
        self.model.eval()
        
        images = batch['images']
        pointclouds = batch['pointclouds'].to(self.device)
        
        defense_strategies = []
        batch_size = len(images) if isinstance(images, list) else images.size(0)
        
        for i in range(batch_size):
            attack_type = attack_types[i] if attack_types else None
            
            # 使用defend模式调用LLM生成防御策略
            with torch.no_grad():
                # 单样本处理
                if isinstance(images, list):
                    single_image = [images[i]]
                else:
                    single_image = images[i:i+1]
                single_pc = pointclouds[i:i+1]
                
                strategy = self.model(
                    single_image, 
                    single_pc, 
                    mode='defend',
                    attack_type=attack_type
                )
                defense_strategies.extend(strategy)
        
        return defense_strategies
    
    def full_inference(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        完整推理流程：检测攻击 + 生成防御策略
        
        这是完整的端到端推理，输出包括：
        1. 分类器的logits和预测结果（5分类）
        2. LLM的攻击检测分析（JSON格式）
        3. LLM的防御策略建议（文本格式）
        
        Args:
            batch: 包含images, pointclouds, labels的批次数据
            
        Returns:
            Dict: 完整推理结果
        """
        self.model.eval()
        
        images = batch['images']
        pointclouds = batch['pointclouds'].to(self.device)
        labels = batch.get('labels')
        
        results = {
            'sample_tokens': batch.get('sample_tokens', []),
            'ground_truth_labels': labels.numpy().tolist() if labels is not None else None,
        }
        
        # 1. 获取分类logits和预测（5分类）
        with torch.no_grad():
            logits = self.model(images, pointclouds, mode='train')  # [B, 5]
            probs = torch.softmax(logits, dim=-1)  # [B, 5]
            predicted_labels = torch.argmax(logits, dim=-1)  # [B] 0-4
            confidence = probs.max(dim=-1)[0]  # 最高类别的置信度
            
            results['logits'] = logits.cpu().numpy().tolist()
            results['predictions'] = predicted_labels.cpu().numpy().tolist()
            results['confidence'] = confidence.cpu().numpy().tolist()
            results['class_probs'] = probs.cpu().numpy().tolist()
        
        # 2. LLM攻击检测
        detection_results = self.detect_attacks(batch)
        results['detection_results'] = detection_results
        
        # 3. 对检测到攻击的样本生成防御策略
        # 5分类：0=normal, 1-4=攻击
        CLASS_NAMES = ['normal', 'adversarial_patch', 'sensor_spoofing', 'physical_attack', 'data_poisoning']
        
        attack_types = []
        for pred in results['predictions']:
            if pred > 0:  # 非normal
                attack_types.append(CLASS_NAMES[pred] if pred < len(CLASS_NAMES) else 'unknown')
            else:
                attack_types.append(None)
        
        # 只对检测到攻击的样本生成防御策略
        if any(at is not None for at in attack_types):
            defense_strategies = self.generate_defense_strategies(batch, attack_types)
            results['defense_strategies'] = defense_strategies
        else:
            results['defense_strategies'] = ['无需防御 - 未检测到攻击'] * len(attack_types)
        
        return results
    
    def local_update_with_generation_training(
        self,
        global_logits: torch.Tensor,
        target_batch: Optional[Dict[str, Any]] = None,
        num_epochs: Optional[int] = None,
        alpha: float = 1.0,  # 分类损失权重
        beta: float = 0.3,   # 检测生成损失权重  
        gamma: float = 0.3   # 防御生成损失权重
    ) -> Dict[str, float]:
        """
        使用联合损失更新本地模型，同时训练分类和LLM生成任务
        
        总损失 = α * L_蒸馏 + β * L_检测生成 + γ * L_防御生成
        
        其中 L_蒸馏 = α' * CE(y_true, local) + β' * KL(local || global)
        
        Args:
            global_logits: 服务器聚合的全局logits
            target_batch: 指定的训练批次
            num_epochs: 本地训练轮数
            alpha: 分类/蒸馏损失权重
            beta: 检测生成损失权重
            gamma: 防御生成损失权重
            
        Returns:
            dict: 包含各项损失和指标的字典
        """
        if num_epochs is None:
            num_epochs = self.config.local_epochs
        
        self.model.train()
        
        # 如果是FedProx，保存当前模型权重作为参考(global_params)
        global_params = None
        if self.config.aggregation_method == "fedprox":
            global_params = {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}
        
        total_distill_loss = 0.0
        total_detect_loss = 0.0
        total_defend_loss = 0.0
        all_predictions = []
        all_labels = []
        num_batches = 0
        
        # 决定训练批次
        if target_batch is not None:
            batches = [target_batch]
        else:
            batches = self.data_loader
        
        for epoch in range(num_epochs):
            for batch_idx, batch in enumerate(batches):
                if target_batch is None and hasattr(self.config, 'max_batches') and self.config.max_batches > 0:
                    if batch_idx >= self.config.max_batches:
                        break
                
                images = batch['images']
                pointclouds = batch['pointclouds'].to(self.device)
                labels = batch['labels'].to(self.device)
                attack_types = batch.get('attack_types', None)
                
                self.optimizer.zero_grad()
                
                # 检查模型是否支持生成训练模式
                if hasattr(self.model, 'forward') and 'train_generation' in str(type(self.model)):
                    # 使用增强模型的联合训练
                    outputs = self.model(
                        images, pointclouds,
                        mode='train_generation',
                        labels=labels,
                        attack_types_gt=attack_types
                    )
                    local_logits = outputs['cls_logits']
                    detect_loss = outputs['detect_loss']
                    defend_loss = outputs['defend_loss']
                else:
                    # 普通模型只训练分类
                    local_logits = self.model(images, pointclouds, mode='train')
                    detect_loss = torch.tensor(0.0, device=self.device)
                    defend_loss = torch.tensor(0.0, device=self.device)
                
                # 确保global_logits在正确的设备上
                global_logits_batch = global_logits.to(self.device)
                
                # 检查batch size是否匹配
                if local_logits.size(0) != global_logits_batch.size(0):
                    min_size = min(local_logits.size(0), global_logits_batch.size(0))
                    local_logits = local_logits[:min_size]
                    global_logits_batch = global_logits_batch[:min_size]
                    labels = labels[:min_size]
                
                # 计算蒸馏损失
                distill_loss = self.compute_distillation_loss(
                    local_logits,
                    global_logits_batch,
                    labels
                )
                
                # 总损失
                if torch.is_tensor(detect_loss) and torch.is_tensor(defend_loss):
                    total_loss = (alpha * distill_loss + 
                                 beta * detect_loss + 
                                 gamma * defend_loss)
                else:
                    total_loss = distill_loss
                
                # 如果是FedProx，增加近端项
                if global_params is not None:
                    prox_loss = self.compute_proximal_loss(global_params)
                    total_loss += prox_loss
                
                # 反向传播
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # 统计
                total_distill_loss += distill_loss.item()
                if torch.is_tensor(detect_loss):
                    total_detect_loss += detect_loss.item()
                if torch.is_tensor(defend_loss):
                    total_defend_loss += defend_loss.item()
                num_batches += 1
                
                # 收集预测和标签
                with torch.no_grad():
                    predictions = torch.sigmoid(local_logits.squeeze(-1))
                    predicted_labels = (predictions > 0.5).long()
                    all_predictions.extend(predicted_labels.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
        
        if self.config.verbose:
            print(f"  [Client {self.client_id}] 联合训练完成: {num_batches} batches")
            print(f"    蒸馏损失: {total_distill_loss/num_batches:.4f}")
            print(f"    检测生成损失: {total_detect_loss/num_batches:.4f}")
            print(f"    防御生成损失: {total_defend_loss/num_batches:.4f}")
        
        # 计算指标
        metrics = compute_metrics(
            np.array(all_predictions),
            np.array(all_labels)
        )
        
        return {
            'distill_loss': total_distill_loss / num_batches if num_batches > 0 else 0.0,
            'detect_loss': total_detect_loss / num_batches if num_batches > 0 else 0.0,
            'defend_loss': total_defend_loss / num_batches if num_batches > 0 else 0.0,
            'total_loss': (total_distill_loss + total_detect_loss + total_defend_loss) / num_batches if num_batches > 0 else 0.0,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score']
        }
    
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
