#!/usr/bin/env python3
"""
联邦学习入口脚本

实现完整的联邦训练闭环：
1. 客户端本地推理 → 提取logits
2. 上传logits到服务器
3. 服务器计算自由能和权重
4. 服务器聚合logits
5. 下发全局logits给客户端
6. 客户端使用蒸馏损失更新本地模型
7. 循环进行多轮训练
"""

import os
import sys
import argparse
import time
import json
import torch
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader, Subset, Dataset

# 导入联邦学习模块
from federation.client import FederatedClient
from federation.server import FederatedServer
from federation.config import FederatedConfig
from federation.comm import CommunicationManager
from federation.logger import FederatedLogger
from federation.utils import partition_data_iid, partition_data_non_iid, aggregate_metrics

# 导入模型和数据集
from test_local_train_mini_qwen3vl_fixed import (
    Qwen3VLDefenseSystem,
    NuScenesMiniDataset,
    custom_collate_fn
)

# 导入攻击数据集
try:
    from attacks import (
        NuScenesAttackDataset,
        SyntheticAttackDataset,
        attack_collate_fn,
        create_attack_dataset
    )
    ATTACK_DATASET_AVAILABLE = True
except ImportError:
    print("警告: 攻击数据集模块未找到，将使用原始数据集")
    ATTACK_DATASET_AVAILABLE = False

try:
    from attacks.attack_generator import attack_type_to_label
except ImportError:
    def attack_type_to_label(attack_type):
        mapping = {
            'normal': 0, 'adversarial_patch': 1, 'sensor_spoofing': 2,
            'physical_attack': 3, 'data_poisoning': 4
        }
        return mapping.get(attack_type, 0)


class ClientLocalAttackDataset(Dataset):
    """
    客户端本地攻击数据集包装器
    为每个客户端分配独立的数据索引，并在本地生成随机攻击。
    """
    def __init__(self, base_dataset, indices, attack_ratio, config):
        self.base_dataset = base_dataset
        self.indices = indices
        self.attack_ratio = attack_ratio
        self.config = config
        
        # 记录样本总数
        self.num_samples = len(indices)
        
        # 预计算本地攻击分配 (保证每个epoch的一致性)
        # 使用第一个索引作为种子，使每个客户端的随机性不同但可复现
        seed = indices[0] if (indices is not None and len(indices) > 0) else 42
        rng = np.random.RandomState(seed)
        
        self.local_is_attack = rng.rand(self.num_samples) < self.attack_ratio
        
        # 预先分配攻击类型
        self.attack_types_list = ['adversarial_patch', 'sensor_spoofing', 'physical_attack', 'data_poisoning']
        self.local_attack_types = []
        for i in range(self.num_samples):
            if self.local_is_attack[i]:
                self.local_attack_types.append(rng.choice(self.attack_types_list))
            else:
                self.local_attack_types.append('normal')
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # 获取基础数据（此时基础数据集应该是干净的）
        real_idx = self.indices[idx]
        data = self.base_dataset[real_idx]
        
        # 应用本地攻击逻辑
        is_attack = self.local_is_attack[idx]
        attack_type = self.local_attack_types[idx]
        
        if is_attack:
            # 1. 如果基础数据集有攻击生成器，应用真实攻击
            if hasattr(self.base_dataset, 'attack_generator'):
                image = data['images']
                # 处理点云 (可能是 Tensor 或 numpy)
                if torch.is_tensor(data['pointclouds']):
                    # 如果是 CUDA Tensor 需要转到 CPU
                    points = data['pointclouds'].detach().cpu().numpy()
                else:
                    points = data['pointclouds']
                
                # 应用攻击生成器
                attacked_image, attacked_points, attack_info = self.base_dataset.attack_generator.apply_attack(
                    image, points, attack_type
                )
                
                # 确保点云维度对齐 (攻击可能增加了或减少了点数)
                num_points = self.config.num_points
                if attacked_points.shape[0] > num_points:
                    # 如果点多了，进行随机采样
                    sampled_indices = np.random.choice(attacked_points.shape[0], num_points, replace=False)
                    attacked_points = attacked_points[sampled_indices]
                elif attacked_points.shape[0] < num_points:
                    # 如果点少了，补齐到指定点数
                    pad_size = num_points - attacked_points.shape[0]
                    padding = np.zeros((pad_size, 3))
                    attacked_points = np.vstack([attacked_points, padding])
                
                # 更新返回数据内容
                data['images'] = attacked_image
                data['pointclouds'] = torch.from_numpy(attacked_points).float()
                data['attack_info'] = attack_info
            
            # 2. 更新标签和类型 (始终更新，即使是 mini 版本也会生效)
            label = attack_type_to_label(attack_type)
            data['labels'] = torch.tensor(label, dtype=torch.long)
            data['attack_types'] = attack_type
        else:
            # 确保是正常标签
            data['labels'] = torch.tensor(0, dtype=torch.long)
            data['attack_types'] = 'normal'
            
        return data

    def get_statistics(self):
        """获取本地统计信息"""
        stats = {}
        for t in self.local_attack_types:
            stats[t] = stats.get(t, 0) + 1
        return stats


def create_model(config: FederatedConfig):
    """创建Qwen3VL防御系统模型"""
    
    if config.train_generation:
        # 使用增强模型（支持训练LLM生成任务）
        try:
            from models.enhanced_model import Qwen3VLDefenseSystemEnhanced
            model = Qwen3VLDefenseSystemEnhanced(
                pointcloud_dim=config.pointcloud_dim,
                qwen_hidden_dim=None,  # 自动检测
                model_name=config.model_path,
                train_generation=True
            )
            print("使用增强模型（支持LLM生成训练）")
        except ImportError:
            print("警告：无法导入增强模型，使用标准模型")
            model = Qwen3VLDefenseSystem(
                pointcloud_dim=config.pointcloud_dim,
                qwen_hidden_dim=None,  # 自动检测
                model_name=config.model_path
            )
    else:
        # 使用标准模型（只训练分类）
        model = Qwen3VLDefenseSystem(
            pointcloud_dim=config.pointcloud_dim,
            qwen_hidden_dim=None,  # 自动检测
            model_name=config.model_path
        )
    
    return model


def setup_clients(
    config: FederatedConfig,
    train_dataset,
    device: str,
    logger: FederatedLogger
) -> list:
    """
    初始化联邦客户端
    
    Args:
        config: 联邦配置
        train_dataset: 训练数据集
        device: 计算设备
        logger: 日志记录器
        
    Returns:
        clients: FederatedClient列表
    """
    logger.log("="*60)
    logger.log("初始化联邦客户端")
    logger.log("="*60)
    
    # 数据分区
    num_samples = len(train_dataset)
    client_indices = partition_data_iid(num_samples, config.num_clients)
    
    # 确定哪些客户端是干净的
    # 随机选择 num_clean_clients 个客户端
    clean_client_ids = np.random.choice(config.num_clients, config.num_clean_clients, replace=False)
    
    clients = []
    for i in range(config.num_clients):
        logger.log(f"\n初始化客户端 {i}...")
        
        # 确定该客户端的攻击比例
        if i in clean_client_ids:
            local_attack_ratio = 0.0
            logger.log(f"  - 客户端 {i} 被设定为干净客户端 (attack_ratio = 0.0)")
        else:
            # 随机生成攻击比例 (0.1 到 0.6 之间随机)
            local_attack_ratio = np.random.uniform(0.1, 0.6)
            logger.log(f"  - 客户端 {i} 随机生成的攻击比例: {local_attack_ratio:.1%}")
            
        # 创建客户端专属数据集
        client_dataset = ClientLocalAttackDataset(
            base_dataset=train_dataset,
            indices=client_indices[i],
            attack_ratio=local_attack_ratio,
            config=config
        )
        
        # 记录本地统计信息
        local_stats = client_dataset.get_statistics()
        logger.log(f"  - 数据量: {len(client_indices[i])} 样本, 攻击分布: {local_stats}")
        
        client_loader = DataLoader(
            client_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=True
        )
        
        logger.log(f"  - 数据量: {len(client_indices[i])} 样本")
        
        # 创建客户端模型（每个客户端有独立的模型实例）
        client_model = create_model(config)
        
        # 创建联邦客户端
        client = FederatedClient(
            client_id=i,
            model=client_model,
            data_loader=client_loader,
            device=device,
            config=config
        )
        
        clients.append(client)
        logger.log(f"  - 客户端 {i} 初始化完成")
    
    return clients


def setup_server(
    config: FederatedConfig,
    train_dataset,
    device: str,
    logger: FederatedLogger
) -> FederatedServer:
    """
    初始化联邦服务器
    
    Args:
        config: 联邦配置
        train_dataset: 训练数据集（用于可选的服务器端更新）
        device: 计算设备
        logger: 日志记录器
        
    Returns:
        server: FederatedServer实例
    """
    logger.log("\n" + "="*60)
    logger.log("初始化联邦服务器")
    logger.log("="*60)
    
    # 创建服务器模型
    server_model = create_model(config)
    
    # 可选：服务器端数据加载器
    server_loader = None
    if config.enable_server_update:
        # 使用部分数据作为服务器公共数据
        num_server_samples = min(len(train_dataset) // 10, 100)
        server_indices = np.random.choice(len(train_dataset), num_server_samples, replace=False)
        server_dataset = Subset(train_dataset, server_indices.tolist())
        server_loader = DataLoader(
            server_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=True
        )
        logger.log(f"  - 服务器公共数据: {num_server_samples} 样本")
    
    # 创建服务器
    server = FederatedServer(
        model=server_model,
        device=device,
        config=config,
        server_data_loader=server_loader
    )
    
    logger.log("  - 服务器初始化完成")
    
    return server


def run_federated_round(
    round_id: int,
    clients: list,
    server: FederatedServer,
    config: FederatedConfig,
    comm_manager: CommunicationManager,
    logger: FederatedLogger
) -> dict:
    """
    执行一个联邦训练回合
    
    完整流程:
    1. 客户端本地前向推理 → 提取logits
    2. 计算自由能
    3. 上传logits到服务器
    4. 服务器计算权重并聚合
    5. 服务器自我更新（可选）
    6. 下发全局logits
    7. 客户端蒸馏更新
    
    Args:
        round_id: 回合ID
        clients: 客户端列表
        server: 服务器
        config: 配置
        comm_manager: 通信管理器
        logger: 日志记录器
        
    Returns:
        dict: 回合指标
    """
    logger.log_round_start(round_id, config.num_rounds)
    comm_manager.reset_round_stats()
    
    round_metrics = {
        'round_id': round_id,
        'client_metrics': [],
        'free_energies': [],
        'weights': [],
    }
    
    # ========== 阶段1: 客户端本地推理 + 计算自由能 ==========
    logger.log("\n[阶段1] 客户端本地推理与自由能计算")
    
    client_logits_dict = {}
    client_labels_dict = {}
    client_batches = {}  # 保存每个客户端的batch，用于后续蒸馏
    
    for client in clients:
        # 获取一个batch进行推理
        batch = next(iter(client.data_loader))
        client_batches[client.client_id] = batch
        
        # 本地前向推理
        logits, labels, sample_ids = client.local_forward(batch)
        client_logits_dict[client.client_id] = logits
        client_labels_dict[client.client_id] = labels
        
        # 获取服务器先验logits
        prior_logits = server.get_prior_logits(
            batch_size=logits.size(0),
            num_classes=logits.size(-1) if len(logits.shape) > 1 else 1
        )
        
        # 计算自由能
        if config.free_energy_mode == "kl_entropy":
            free_energy = client.compute_free_energy(
                logits=logits,
                server_prior_logits=prior_logits
            )
        else:  # ce_entropy
            free_energy = client.compute_free_energy(
                logits=logits,
                labels=labels
            )
        #print(f"    Free Energies: {list(free_energies.values())}")
        
        round_metrics['free_energies'].append(free_energy)
        
        logger.log(f"  客户端 {client.client_id}: logits shape={logits.shape}, F={free_energy:.4f}")
    
    # ========== 阶段2: 上传logits到服务器 ==========
    logger.log("\n[阶段2] 上传logits到服务器")
    
    for client_id, logits in client_logits_dict.items():
        # 序列化（统计通信量）
        serialized = comm_manager.serialize_logits(logits)
        logger.log(f"  客户端 {client_id} 上传: {len(serialized)/1024:.2f} KB")
    
    # ========== 阶段3: 服务器计算权重 ==========
    logger.log("\n[阶段3] 服务器计算客户端权重")
    
    weights = server.compute_client_weights(round_metrics['free_energies'])
    round_metrics['weights'] = weights.tolist()
    #print(f"    Weights: {list(weights.values())}")
    
    for i, (fe, w) in enumerate(zip(round_metrics['free_energies'], weights)):
        logger.log(f"  客户端 {i}: F={fe:.4f}, weight={w:.4f}")
    
    # ========== 阶段4: 服务器聚合logits ==========
    logger.log("\n[阶段4] 服务器聚合logits")
    
    client_logits_list, client_ids = server.collect_client_logits(client_logits_dict)
    global_logits = server.aggregate_logits(client_logits_list, weights)
    
    stats = server.get_aggregation_stats()
    logger.log_server_aggregation(weights, round_metrics['free_energies'], stats.get('latest', {}))
    
    # ========== 阶段5: 服务器自我更新（可选） ==========
    if config.enable_server_update:
        logger.log("\n[阶段5] 服务器自我更新")
        # 使用聚合的全局logits进行自我蒸馏
        # 需要一个代表性的batch
        representative_batch = client_batches[0]
        server_update_result = server.server_update(
            global_logits=global_logits,
            batch=representative_batch,
            num_epochs=1
        )
        logger.log(f"  服务器更新损失: {server_update_result['loss']:.4f}")
    
    # ========== 阶段6: 下发全局logits ==========
    logger.log("\n[阶段6] 下发全局logits给客户端")
    
    global_package = server.broadcast_global_logits(
        global_logits=global_logits,
        weights=weights,
        free_energies=round_metrics['free_energies'],
        round_id=round_id
    )
    
    # 序列化（统计通信量）
    serialized_global = comm_manager.serialize_logits(global_logits)
    logger.log(f"  全局logits大小: {len(serialized_global)/1024:.2f} KB")
    
    # ========== 阶段7: 客户端蒸馏更新 ==========
    logger.log("\n[阶段7] 客户端蒸馏更新")
    
    for client in clients:
        # 接收全局logits
        global_logits_local = client.receive_global_logits(global_logits)
        
        # 根据配置选择训练方式
        if config.train_generation:
            # 联合训练：分类 + LLM生成
            update_result = client.local_update_with_generation_training(
                global_logits=global_logits_local,
                target_batch=client_batches[client.client_id],
                num_epochs=config.local_epochs,
                alpha=config.generation_alpha,
                beta=config.generation_beta,
                gamma=config.generation_gamma
            )
            
            round_metrics['client_metrics'].append({
                'client_id': client.client_id,
                'free_energy': round_metrics['free_energies'][client.client_id],
                'weight': weights[client.client_id],
                'loss': update_result['total_loss'],
                'distill_loss': update_result['distill_loss'],
                'detect_loss': update_result['detect_loss'],
                'defend_loss': update_result['defend_loss'],
                'total_loss': update_result['total_loss'],
                'accuracy': update_result['accuracy'],
                'precision': update_result['precision'],
                'recall': update_result['recall'],
                'f1_score': update_result['f1_score'],
                'fpr': update_result.get('fpr', 0.0),
                'fnr': update_result.get('fnr', 0.0),
                'specificity': update_result.get('specificity', 0.0),
                'auc_roc': update_result.get('auc_roc', 0.5),
                'auc_pr': update_result.get('auc_pr', 0.0),
                'tp': update_result.get('tp', 0),
                'tn': update_result.get('tn', 0),
                'fp': update_result.get('fp', 0),
                'fn': update_result.get('fn', 0)
            })
            
            logger.log(f"  客户端 {client.client_id}: "
                      f"蒸馏={update_result['distill_loss']:.4f}, "
                      f"检测={update_result['detect_loss']:.4f}, "
                      f"防御={update_result['defend_loss']:.4f}, "
                      f"Acc={update_result['accuracy']:.4f}")
        else:
            # 仅分类训练
            update_result = client.local_update_with_distillation(
                global_logits=global_logits_local,
                target_batch=client_batches[client.client_id],
                num_epochs=config.local_epochs
            )
            
            round_metrics['client_metrics'].append({
                'client_id': client.client_id,
                'free_energy': round_metrics['free_energies'][client.client_id],
                'weight': weights[client.client_id],
                'loss': update_result['loss'],
                'accuracy': update_result['accuracy'],
                'precision': update_result['precision'],
                'recall': update_result['recall'],
                'f1_score': update_result['f1_score'],
                'fpr': update_result.get('fpr', 0.0),
                'fnr': update_result.get('fnr', 0.0),
                'specificity': update_result.get('specificity', 0.0),
                'auc_roc': update_result.get('auc_roc', 0.5),
                'tp': update_result.get('tp', 0),
                'tn': update_result.get('tn', 0),
                'fp': update_result.get('fp', 0),
                'fn': update_result.get('fn', 0),
                # 5分类专有指标
                'macro_f1': update_result.get('macro_f1', 0.0),
                'per_class_f1': update_result.get('per_class_f1', [0.0]*5)
            })
            
            logger.log_client_metrics(
                client_id=client.client_id,
                free_energy=round_metrics['free_energies'][client.client_id],
                weight=weights[client.client_id],
                loss=update_result['loss'],
                accuracy=update_result['accuracy'],
                f1_score=update_result.get('f1_score', 0.0),
                num_samples=len(client_batches[client.client_id]['labels'])
            )
    
    # ========== 统计通信量 ==========
    comm_stats = comm_manager.get_communication_stats()
    logger.log_communication_stats(comm_stats)
    round_metrics['communication'] = comm_stats
    
    # ========== 阶段8: LLM攻击检测与防御策略生成（可选） ==========
    if config.verbose and round_id % 3 == 0:  # 每3轮展示一次LLM输出
        logger.log("\n[阶段8] LLM攻击检测与防御策略生成 (示例)")
        
        # 类别名称映射
        CLASS_NAMES = ['normal', 'adversarial_patch', 'sensor_spoofing', 'physical_attack', 'data_poisoning']
        
        # 选择第一个客户端进行演示
        demo_client = clients[0]
        demo_batch = client_batches[0]
        
        # 显示当前batch的攻击类型分布
        if 'attack_types' in demo_batch:
            attack_types_in_batch = demo_batch['attack_types']
            labels_in_batch = demo_batch['labels'].tolist()
            logger.log(f"\n  === 当前Batch攻击情况 ===")
            for i, (at, lb) in enumerate(zip(attack_types_in_batch, labels_in_batch)):
                label_name = CLASS_NAMES[lb] if lb < len(CLASS_NAMES) else f'unknown_{lb}'
                logger.log(f"    样本{i}: 标签={lb} ({label_name})")
        
        try:
            # 完整推理：分类 + 攻击检测 + 防御策略
            inference_results = demo_client.full_inference(demo_batch)
            
            logger.log(f"\n  === 样本推理结果 ===")
            
            for i, det_result in enumerate(inference_results['detection_results']):
                # 获取真实标签
                true_label = demo_batch['labels'][i].item()
                true_attack_type = demo_batch['attack_types'][i] if 'attack_types' in demo_batch else 'unknown'
                
                # 获取预测结果
                pred_label = inference_results['predictions'][i]
                pred_class_name = CLASS_NAMES[pred_label] if pred_label < len(CLASS_NAMES) else f'unknown_{pred_label}'
                
                logger.log(f"\n  样本 {i+1}:")
                logger.log(f"    【真实】: 类别={true_label} ({true_attack_type})")
                logger.log(f"    【预测】: 类别={pred_label} ({pred_class_name})")
                logger.log(f"    【置信度】: {inference_results['confidence'][i]:.4f}")
                logger.log(f"    【正确】: {'✓' if pred_label == true_label else '✗'}")
                
                # LLM检测结果
                logger.log(f"    LLM检测结果:")
                logger.log(f"      - 是否攻击: {det_result.get('is_attack', 'N/A')}")
                logger.log(f"      - 攻击类型: {det_result.get('attack_type', 'N/A')}")
                logger.log(f"      - 风险等级: {det_result.get('risk_level', 'N/A')}")
                logger.log(f"      - 置信度: {det_result.get('confidence', 'N/A')}")
                
                # 分析内容（截取前200字符）
                analysis = det_result.get('analysis', '')
                if analysis:
                    logger.log(f"      - 分析: {analysis[:200]}...")
                
                # 防御策略（截取前300字符）
                defense = inference_results['defense_strategies'][i]
                if defense and defense != '无需防御 - 未检测到攻击':
                    logger.log(f"    防御策略: {defense[:300]}...")
                else:
                    logger.log(f"    防御策略: {defense}")
                    
        except Exception as e:
            logger.log(f"  LLM推理出错: {str(e)}")
            logger.log(f"  (这可能是因为模型未完全加载或配置问题)")
    
    # ========== 回合总结 ==========
    avg_metrics = aggregate_metrics(round_metrics['client_metrics'])
    logger.log_round_end(round_id, avg_metrics)
    round_metrics['avg_metrics'] = avg_metrics
    
    return round_metrics


def run_federated_training(config: FederatedConfig):
    """
    执行完整的联邦训练
    
    Args:
        config: 联邦学习配置
    """
    # 设置设备
    device = config.device if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建日志目录
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.save_dir, exist_ok=True)
    
    # 初始化日志记录器
    logger = FederatedLogger(config.log_dir, config.verbose)
    logger.log("="*60)
    logger.log("联邦学习 + 主动推理 训练系统")
    logger.log("="*60)
    logger.log(f"\n配置:\n{config}")
    
    # 初始化通信管理器
    comm_manager = CommunicationManager()
    
    # 加载数据集
    logger.log("\n加载数据集 (统一初始化为干净数据)...")
    
    # 选择数据集类型
    if config.use_attack_dataset and ATTACK_DATASET_AVAILABLE:
        logger.log("使用增强攻击数据集 (真实攻击生成器)")
        
        # 初始全局数据集设为0，由各客户端本地按规则应用随机攻击
        attack_config = {
            'attack_ratio': 0.0,
            'attack_weights': {
                'adversarial_patch': 0.25,
                'sensor_spoofing': 0.25,
                'physical_attack': 0.25,
                'data_poisoning': 0.25
            }
        }
        
        train_dataset = create_attack_dataset(
            dataroot=config.dataroot,
            version=config.version,
            split='train',
            attack_ratio=0.0,
            num_points=config.num_points,
            use_synthetic=config.use_synthetic_data,
            num_synthetic_samples=config.num_synthetic_samples,
            attack_config=attack_config
        )
        collate_fn = attack_collate_fn
    else:
        logger.log("使用原始数据集（简单攻击标记 - 初始设置为干净）")
        train_dataset = NuScenesMiniDataset(
            dataroot=config.dataroot,
            version=config.version,
            split='train',
            attack_ratio=0.0,
            num_points=config.num_points,
            use_cache=True
        )
        collate_fn = custom_collate_fn
    
    train_dataset.get_statistics()
    
    # 初始化客户端
    clients = setup_clients(config, train_dataset, device, logger)
    
    # 初始化服务器
    server = setup_server(config, train_dataset, device, logger)
    
    # 保存配置
    config.save(os.path.join(config.save_dir, 'config.json'))
    
    # ========== 联邦训练循环 ==========
    logger.log("\n" + "="*60)
    logger.log("开始联邦训练")
    logger.log("="*60)
    
    all_round_metrics = []
    best_avg_f1 = 0.0
    
    for round_id in range(config.num_rounds):
        # 执行一轮联邦训练
        round_metrics = run_federated_round(
            round_id=round_id,
            clients=clients,
            server=server,
            config=config,
            comm_manager=comm_manager,
            logger=logger
        )
        
        all_round_metrics.append(round_metrics)
        logger.add_round_metrics(round_metrics)
        
        # 检查是否是最佳模型
        avg_f1 = round_metrics['avg_metrics'].get('avg_f1_score', 0)
        if avg_f1 > best_avg_f1:
            best_avg_f1 = avg_f1
            # 保存最佳服务器模型
            server.save_model(os.path.join(config.save_dir, 'best_server_model.pth'))
            # 保存最佳客户端模型
            for client in clients:
                torch.save(
                    client.get_model_state(),
                    os.path.join(config.save_dir, f'best_client_{client.client_id}_model.pth')
                )
            logger.log(f"  ✓ 保存最佳模型 (Avg F1: {best_avg_f1:.4f})")
        
        # 定期保存检查点
        if (round_id + 1) % 5 == 0:
            checkpoint_path = os.path.join(config.save_dir, f'checkpoint_round_{round_id+1}.pth')
            server.save_model(checkpoint_path)
            logger.log(f"  ✓ 保存检查点: {checkpoint_path}")
    
    # ========== 训练完成 ==========
    logger.log("\n" + "="*60)
    logger.log("联邦训练完成!")
    logger.log("="*60)
    logger.log(f"最佳平均F1分数: {best_avg_f1:.4f}")
    
    # 保存所有指标
    logger.save_metrics()
    
    # 绘制训练曲线
    logger.plot_training_curves()
    
    # 最终通信统计
    final_comm_stats = comm_manager.get_communication_stats()
    logger.log(f"\n总通信量: {final_comm_stats['total_mb']:.2f} MB")
    
    return all_round_metrics


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='联邦学习训练脚本')
    
    # 基础参数
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--num_clients', type=int, default=3, help='客户端数量')
    parser.add_argument('--num_rounds', type=int, default=10, help='联邦训练轮数')
    parser.add_argument('--local_epochs', type=int, default=1, help='本地训练轮数')
    
    # 主动推理参数
    parser.add_argument('--free_energy_mode', type=str, default='ce_entropy',
                        choices=['kl_entropy', 'ce_entropy'], help='自由能计算模式')
    parser.add_argument('--tau', type=float, default=1.0, help='权重计算温度')
    parser.add_argument('--lambda_entropy', type=float, default=0.1, help='KL方案熵权重')
    parser.add_argument('--gamma_entropy', type=float, default=0.1, help='CE方案熵权重')
    
    # 聚合策略参数
    parser.add_argument('--aggregation_method', type=str, default='active_inference',
                        choices=['active_inference', 'fedavg', 'fedprox'], 
                        help='聚合方式选择: 主动推理(默认), FedAvg, FedProx')
    parser.add_argument('--fedprox_mu', type=float, default=0.01, help='FedProx正则化项权重mu')
    
    # 蒸馏参数
    parser.add_argument('--alpha', type=float, default=0.5, help='交叉熵权重')
    parser.add_argument('--beta', type=float, default=0.5, help='KL散度权重')
    parser.add_argument('--temperature', type=float, default=3.0, help='蒸馏温度')
    
    # LLM生成训练参数
    parser.add_argument('--train_generation', action='store_true', 
                        help='启用LLM生成任务训练（检测+防御）')
    parser.add_argument('--generation_alpha', type=float, default=1.0, 
                        help='分类/蒸馏损失权重')
    parser.add_argument('--generation_beta', type=float, default=0.3, 
                        help='检测生成损失权重')
    parser.add_argument('--generation_gamma', type=float, default=0.3, 
                        help='防御生成损失权重')
    
    # 服务器参数
    parser.add_argument('--enable_server_update', action='store_true', help='启用服务器更新')
    parser.add_argument('--server_lr', type=float, default=1e-5, help='服务器学习率')
    
    # 数据参数
    parser.add_argument('--dataroot', type=str, 
                        default=r'/root/autodl-tmp/zrj/data/nusences',
                        help='nuScenes数据集路径')
    parser.add_argument('--batch_size', type=int, default=1, help='批次大小')
    parser.add_argument('--num_clean_clients', type=int, default=0, help='干净客户端的数量(attack_ratio为0)')
    
    # 攻击数据集参数
    parser.add_argument('--use_attack_dataset', action='store_true', default=True,
                        help='使用真实攻击生成数据集')
    parser.add_argument('--no_attack_dataset', action='store_true',
                        help='不使用攻击生成数据集（使用原始数据集）')
    parser.add_argument('--use_synthetic_data', action='store_true',
                        help='使用合成数据（无需NuScenes数据集）')
    parser.add_argument('--num_synthetic_samples', type=int, default=1000,
                        help='合成数据样本数量')
    
    # 模型参数
    parser.add_argument('--model_path', type=str,
                        default=r'Qwen/Qwen3-VL-2B-Instruct',
                        help='Qwen3-VL模型路径')
    
    # 设备和日志
    parser.add_argument('--device', type=str, default='cuda', help='计算设备')
    parser.add_argument('--log_dir', type=str, default='./logs_federated', help='日志目录')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_federated', help='保存目录')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    if args.config:
        config = FederatedConfig.load(args.config)
        print(f"从文件加载配置: {args.config}")
    else:
        # 处理攻击数据集参数
        use_attack_dataset = args.use_attack_dataset and not args.no_attack_dataset
        
        # 从命令行参数创建配置
        config = FederatedConfig(
            num_clients=args.num_clients,
            num_rounds=args.num_rounds,
            local_epochs=args.local_epochs,
            free_energy_mode=args.free_energy_mode,
            tau=args.tau,
            lambda_entropy=args.lambda_entropy,
            gamma_entropy=args.gamma_entropy,
            aggregation_method=args.aggregation_method,
            fedprox_mu=args.fedprox_mu,
            alpha=args.alpha,
            beta=args.beta,
            temperature=args.temperature,
            train_generation=args.train_generation,
            generation_alpha=args.generation_alpha,
            generation_beta=args.generation_beta,
            generation_gamma=args.generation_gamma,
            enable_server_update=args.enable_server_update,
            server_lr=args.server_lr,
            dataroot=args.dataroot,
            batch_size=args.batch_size,
            num_clean_clients=args.num_clean_clients,
            use_attack_dataset=use_attack_dataset,
            use_synthetic_data=args.use_synthetic_data,
            num_synthetic_samples=args.num_synthetic_samples,
            model_path=args.model_path,
            device=args.device,
            log_dir=args.log_dir,
            save_dir=args.save_dir,
            verbose=args.verbose
        )
    
    # 验证配置
    config.validate()
    
    # 运行联邦训练
    run_federated_training(config)


if __name__ == "__main__":
    main()
