"""
联邦学习训练入口脚本

基于主动推理的端云协同联邦学习框架

使用示例:
    # 基础运行（3客户端，10回合）
    python run_federated_qwenvl.py --num_clients 3 --num_rounds 10
    
    # 自定义配置
    python run_federated_qwenvl.py --num_clients 5 --num_rounds 20 \\
        --free_energy_mode ce_entropy --tau 0.5 --alpha 0.6 --beta 0.4
    
    # 启用服务器更新
    python run_federated_qwenvl.py --num_clients 3 --num_rounds 10 \\
        --enable_server_update
"""

import os
import sys
import argparse
import json
import time

# 禁用Tokenizer并行警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import List, Dict, Any
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np

# 导入原有模块
from test_local_train_mini_qwen3vl_debug import (
    Qwen3VLDefenseSystem,
    NuScenesMiniDataset,
    custom_collate_fn
)

# 导入联邦学习模块
from federation import (
    FederatedClient,
    FederatedServer,
    FederatedConfig,
    FederatedLogger,
    RoundMetrics,
    ClientMetrics,
    CommunicationManager
)
from federation.utils import partition_data_iid, partition_data_non_iid


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='联邦学习训练 - 基于主动推理的Qwen3-VL攻击检测'
    )
    
    # 联邦学习基础参数
    parser.add_argument('--num_clients', type=int, default=3,
                        help='客户端数量')
    parser.add_argument('--num_rounds', type=int, default=10,
                        help='联邦训练回合数')
    parser.add_argument('--local_epochs', type=int, default=1,
                        help='每轮本地训练epoch数')
    
    # 模型参数
    parser.add_argument('--model_path', type=str,
                        default='/home/sutongtong/wwt/model/Qwen3-VL-2B-Instruct',
                        help='Qwen3-VL模型路径')
    parser.add_argument('--pointcloud_dim', type=int, default=1024,
                        help='点云特征维度')
    parser.add_argument('--qwen_hidden_dim', type=int, default=2048,
                        help='Qwen3-VL隐藏层维度')
    
    # 蒸馏损失参数
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='交叉熵权重')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='KL散度权重')
    parser.add_argument('--temperature', type=float, default=3.0,
                        help='蒸馏温度T')
    
    # 主动推理参数
    parser.add_argument('--free_energy_mode', type=str, default='kl_entropy',
                        choices=['kl_entropy', 'ce_entropy'],
                        help='自由能计算模式')
    parser.add_argument('--lambda_entropy', type=float, default=0.1,
                        help='KL方案的熵权重λ')
    parser.add_argument('--gamma_entropy', type=float, default=0.1,
                        help='CE方案的熵权重γ')
    parser.add_argument('--tau', type=float, default=1.0,
                        help='权重计算温度τ')
    
    # Logits提取参数
    parser.add_argument('--logits_mode', type=str, default='cls',
                        choices=['cls', 'last_token', 'mean_pool'],
                        help='Logits提取模式')
    
    # 服务器更新参数
    parser.add_argument('--enable_server_update', action='store_true',
                        help='启用服务器端模型更新')
    parser.add_argument('--server_lr', type=float, default=1e-5,
                        help='服务器学习率')
    
    # 数据参数
    parser.add_argument('--dataroot', type=str,
                        default='/home/sutongtong/LanTu_team3/dataset/nuScenes/train',
                        help='nuScenes数据集路径')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='nuScenes版本')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='批次大小')
    parser.add_argument('--max_batches', type=int, default=0,
                        help='每轮每客户端最大训练batch数，0表示跑完整个epoch')
    parser.add_argument('--server_max_batches', type=int, default=0,
                        help='服务器公共数据集最大batch数，0表示跑完整个数据集')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='数据加载线程数')
    parser.add_argument('--attack_ratio', type=float, default=0.3,
                        help='攻击样本比例')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='点云采样点数')
    parser.add_argument('--data_partition', type=str, default='iid',
                        choices=['iid', 'non_iid'],
                        help='数据分区策略')
    
    # 训练参数
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='权重衰减')
    
    # 设备参数
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备')
    
    # 日志参数
    parser.add_argument('--log_dir', type=str, default='./logs_federated',
                        help='日志保存目录')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_federated',
                        help='模型保存目录')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='打印详细日志')
    
    # 部署模式
    parser.add_argument('--mode', type=str, default='single_process',
                        choices=['single_process', 'multi_process'],
                        help='部署模式')
    
    # 配置文件
    parser.add_argument('--config', type=str, default=None,
                        help='从JSON文件加载配置')
    
    args = parser.parse_args()
    return args


def create_config_from_args(args) -> FederatedConfig:
    """从命令行参数创建配置"""
    if args.config:
        # 从文件加载配置
        config = FederatedConfig.load(args.config)
    else:
        # 从命令行参数创建配置
        config = FederatedConfig(
            num_clients=args.num_clients,
            num_rounds=args.num_rounds,
            local_epochs=args.local_epochs,
            model_path=args.model_path,
            pointcloud_dim=args.pointcloud_dim,
            qwen_hidden_dim=args.qwen_hidden_dim,
            alpha=args.alpha,
            beta=args.beta,
            temperature=args.temperature,
            free_energy_mode=args.free_energy_mode,
            lambda_entropy=args.lambda_entropy,
            gamma_entropy=args.gamma_entropy,
            tau=args.tau,
            logits_mode=args.logits_mode,
            enable_server_update=args.enable_server_update,
            server_lr=args.server_lr,
            dataroot=args.dataroot,
            version=args.version,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
            server_max_batches=args.server_max_batches,
            num_workers=args.num_workers,
            attack_ratio=args.attack_ratio,
            num_points=args.num_points,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=args.device,
            log_dir=args.log_dir,
            save_dir=args.save_dir,
            verbose=args.verbose,
            mode=args.mode
        )
    
    # 验证配置
    config.validate()
    
    return config


def init_server(config: FederatedConfig) -> FederatedServer:
    """初始化联邦服务器"""
    print("\n初始化联邦服务器...")
    
    # 创建服务器模型
    server_model = Qwen3VLDefenseSystem(
        pointcloud_dim=config.pointcloud_dim,
        qwen_hidden_dim=config.qwen_hidden_dim,
        model_name=config.model_path
    )
    
    # 加载公共数据集（用于评估和服务器更新）
    print("加载服务器公共数据集 (val split)...")
    try:
        common_dataset = NuScenesMiniDataset(
            dataroot=config.dataroot,
            version=config.version,
            split='val',
            attack_ratio=config.attack_ratio,
            num_points=config.num_points,
            use_cache=True
        )
        
        common_loader = DataLoader(
            common_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=True
        )
        print(f"✓ 公共数据集加载成功: {len(common_dataset)} 样本")
    except Exception as e:
        print(f"⚠️ 公共数据集加载失败: {e}，将不使用服务器数据")
        common_loader = None
    
    # 创建服务器
    server = FederatedServer(
        model=server_model,
        device=config.device,
        config=config,
        server_data_loader=common_loader
    )
    
    print(f"✓ 服务器初始化完成: {server}")
    
    return server


def init_clients(config: FederatedConfig) -> List[FederatedClient]:
    """初始化联邦客户端"""
    print(f"\n初始化 {config.num_clients} 个联邦客户端...")
    
    # 加载完整数据集
    full_dataset = NuScenesMiniDataset(
        dataroot=config.dataroot,
        version=config.version,
        split='train',
        attack_ratio=config.attack_ratio,
        num_points=config.num_points,
        use_cache=True
    )
    
    print(f"完整数据集大小: {len(full_dataset)}")
    
    # 数据分区
    if hasattr(full_dataset, 'attack_labels'):
        labels = full_dataset.attack_labels
    else:
        labels = np.zeros(len(full_dataset))
    
    if config.num_clients == 1:
        # 单客户端：使用全部数据
        client_indices_list = [list(range(len(full_dataset)))]
    else:
        # 多客户端：根据策略分区
        if hasattr(config, 'data_partition') and config.data_partition == 'non_iid':
            client_indices_list = partition_data_non_iid(
                len(full_dataset),
                labels,
                config.num_clients
            )
        else:
            client_indices_list = partition_data_iid(
                len(full_dataset),
                config.num_clients
            )
    
    # 创建客户端
    clients = []
    for client_id in range(config.num_clients):
        # 创建客户端数据子集
        client_indices = client_indices_list[client_id]
        client_dataset = Subset(full_dataset, client_indices)
        
        # 创建数据加载器
        client_loader = DataLoader(
            client_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=True
        )
        
        # 创建客户端模型
        client_model = Qwen3VLDefenseSystem(
            pointcloud_dim=config.pointcloud_dim,
            qwen_hidden_dim=config.qwen_hidden_dim,
            model_name=config.model_path
        )
        
        # 创建客户端
        client = FederatedClient(
            client_id=client_id,
            model=client_model,
            data_loader=client_loader,
            device=config.device,
            config=config
        )
        
        clients.append(client)
        
        print(f"✓ Client {client_id}: {len(client_indices)} 样本")
    
    print(f"✓ 所有客户端初始化完成")
    
    return clients



def run_federated_round(
    round_id: int,
    clients: List[FederatedClient],
    server: FederatedServer,
    comm_manager: CommunicationManager,
    logger: FederatedLogger,
    config: FederatedConfig
) -> RoundMetrics:
    """
    执行一个联邦训练回合
    
    Args:
        round_id: 当前回合ID
        clients: 客户端列表
        server: 服务器
        comm_manager: 通信管理器
        logger: 日志记录器
        config: 配置
        
    Returns:
        RoundMetrics: 本轮指标
    """
    logger.log_round_start(round_id, len(clients))
    comm_manager.reset_round_stats()
    
    # ========== 阶段1: 客户端本地推理 ==========
    logger.log("阶段1: 客户端本地推理")
    
    # 如果启用了服务器更新，使用服务器公共数据的一个 batch 进行对齐推理
    inference_batch = None
    if config.enable_server_update and server.server_data_loader is not None:
        if not hasattr(server, '_data_iter') or server._data_iter is None:
            server._data_iter = iter(server.server_data_loader)
        try:
            inference_batch = next(server._data_iter)
        except StopIteration:
            server._data_iter = iter(server.server_data_loader)
            inference_batch = next(server._data_iter)
    
    client_logits_dict = {}
    client_labels_dict = {}
    
    for client in clients:
        # 如果有统一的 inference_batch，则使用它；否则使用客户端自己的数据
        batch = inference_batch if inference_batch is not None else next(iter(client.data_loader))
        
        # 本地前向推理
        logits, labels, sample_ids = client.local_forward(batch)
        
        client_logits_dict[client.client_id] = logits
        client_labels_dict[client.client_id] = labels
        
        logger.log(f"  Client {client.client_id}: logits shape {logits.shape}")
    
    # ========== 阶段2: 上传logits到服务器 ==========
    logger.log("阶段2: 上传logits到服务器")
    
    # 模拟序列化和传输（统计通信量）
    for client_id, logits in client_logits_dict.items():
        _ = comm_manager.serialize_logits(logits)
    
    # ========== 阶段3: 服务器计算自由能和权重 ==========
    logger.log("阶段3: 服务器计算自由能和权重")
    
    # 收集logits
    client_logits_list, client_ids = server.collect_client_logits(client_logits_dict)
    
    # 获取先验logits（用于KL方案）
    if config.free_energy_mode == 'kl_entropy':
        batch_size = client_logits_list[0].size(0)
        num_classes = client_logits_list[0].size(1)
        prior_logits = server.get_prior_logits(batch_size, num_classes)
    else:
        prior_logits = None
    
    # 计算每个客户端的自由能
    free_energies = []
    for i, client_id in enumerate(client_ids):
        client = clients[client_id]
        logits = client_logits_list[i]
        labels = client_labels_dict[client_id]
        
        free_energy = client.compute_free_energy(
            logits=logits,
            labels=labels,
            server_prior_logits=prior_logits
        )
        free_energies.append(free_energy)
        
        logger.log(f"  Client {client_id}: F_i = {free_energy:.4f}")
    
    # 计算权重
    weights = server.compute_client_weights(free_energies)
    
    for i, client_id in enumerate(client_ids):
        logger.log(f"  Client {client_id}: w_i = {weights[i]:.4f}")
    
    # ========== 阶段4: 服务器聚合logits ==========
    logger.log("阶段4: 服务器聚合logits")
    
    global_logits = server.aggregate_logits(client_logits_list, weights)
    
    # 记录聚合统计
    global_logits_stats = {
        'mean': float(global_logits.mean().item()),
        'std': float(global_logits.std().item()),
        'max': float(global_logits.max().item()),
        'min': float(global_logits.min().item())
    }
    
    logger.log_server_aggregation(free_energies, weights, global_logits_stats)
    
    # ========== 阶段5: 服务器自我更新（可选） ==========
    if config.enable_server_update:
        logger.log("阶段5: 服务器自我更新")
        server_metrics = server.server_update(global_logits, batch=inference_batch)
        logger.log(f"  Server loss: {server_metrics['loss']:.4f}")
    
    # ========== 阶段6: 下发global_logits ==========
    logger.log("阶段6: 下发global_logits")
    
    # 模拟序列化和传输
    _ = comm_manager.serialize_logits(global_logits)
    
    # ========== 阶段7: 客户端本地更新 ==========
    logger.log("阶段7: 客户端本地更新")
    
    round_metrics = RoundMetrics(round_id=round_id)
    
    for client in clients:
        # 接收全局logits
        global_logits_local = client.receive_global_logits(global_logits)
        
        # 本地更新 (如果使用了统一的 inference_batch，则在该 batch 上蒸馏)
        update_metrics = client.local_update_with_distillation(
            global_logits_local, 
            target_batch=inference_batch
        )
        logger.log(f"  Client {client.client_id}: 本地更新完成 (max_batches={config.max_batches})")
        
        # 记录客户端指标
        client_metrics = ClientMetrics(
            client_id=client.client_id,
            free_energy=free_energies[client.client_id],
            weight=float(weights[client.client_id]),
            local_loss=update_metrics['loss'],
            accuracy=update_metrics['accuracy'],
            precision=update_metrics['precision'],
            recall=update_metrics['recall'],
            f1_score=update_metrics['f1_score'],
            num_samples=len(client.data_loader.dataset)
        )
        
        round_metrics.client_metrics[client.client_id] = client_metrics
        logger.log_client_metrics(client_metrics)
    
    # ========== 阶段8: 记录通信量和指标 ==========
    comm_stats = comm_manager.get_communication_stats()
    round_metrics.communication_mb = comm_stats['round_mb']
    
    logger.log_communication_stats(comm_stats)
    
    # ========== 阶段9: 全局评估 (使用公共数据集) ==========
    if server.server_data_loader is not None:
        logger.log("阶段9: 全局评估 (使用公共数据集)")
        # 使用服务器模型在公共数据集上评估
        global_metrics = server.evaluate()
        logger.log(f"  Global Metrics: Acc={global_metrics['accuracy']:.4f}, F1={global_metrics['f1_score']:.4f}")
        
        # 将全局指标存入 round_metrics 以便记录
        round_metrics.server_metrics.update({
            'global_accuracy': global_metrics['accuracy'],
            'global_f1': global_metrics['f1_score']
        })
    
    # 计算平均指标
    avg_metrics = {
        'avg_accuracy': np.mean([cm.accuracy for cm in round_metrics.client_metrics.values()]),
        'avg_f1_score': np.mean([cm.f1_score for cm in round_metrics.client_metrics.values()]),
        'avg_free_energy': np.mean(free_energies),
        'avg_weight': np.mean(weights)
    }
    
    if 'global_f1' in round_metrics.server_metrics:
        avg_metrics['global_f1'] = round_metrics.server_metrics['global_f1']
    
    logger.log_round_end(round_id, avg_metrics)
    
    return round_metrics


def main():
    """主函数"""
    print("=" * 80)
    print("联邦学习训练 - 基于主动推理的Qwen3-VL攻击检测系统")
    print("=" * 80)
    
    # 解析参数
    args = parse_args()
    config = create_config_from_args(args)
    
    print("\n配置信息:")
    print(config)
    
    # 创建保存目录
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.save_dir, exist_ok=True)
    
    # 保存配置
    config.save(os.path.join(config.save_dir, 'config.json'))
    
    # 初始化日志记录器
    logger = FederatedLogger(config.log_dir, config.verbose)
    logger.log("联邦学习训练开始")
    logger.log(f"配置: {config.to_dict()}")
    
    # 初始化通信管理器
    comm_manager = CommunicationManager()
    
    # 初始化服务器和客户端
    server = init_server(config)
    clients = init_clients(config)
    
    # 主训练循环
    print("\n" + "=" * 80)
    print("开始联邦训练")
    print("=" * 80)
    
    best_f1 = 0.0
    best_round = 0
    
    for round_id in range(1, config.num_rounds + 1):
        # 执行一个联邦回合
        round_metrics = run_federated_round(
            round_id=round_id,
            clients=clients,
            server=server,
            comm_manager=comm_manager,
            logger=logger,
            config=config
        )
        
        # 添加到日志
        logger.add_round_metrics(round_metrics)
        
        # 计算平均F1分数
        avg_f1 = np.mean([cm.f1_score for cm in round_metrics.client_metrics.values()])
        
        # 保存最佳模型
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_round = round_id
            
            # 保存服务器模型
            server_model_path = os.path.join(config.save_dir, 'best_server_model.pth')
            server.save_model(server_model_path)
            logger.log(f"✓ 保存最佳服务器模型 (Round {round_id}, F1: {best_f1:.4f})")
            
            # 保存所有客户端模型
            for client in clients:
                client_model_path = os.path.join(
                    config.save_dir,
                    f'best_client_{client.client_id}_model.pth'
                )
                torch.save(client.get_model_state(), client_model_path)
        
        # 定期保存检查点
        if round_id % 5 == 0:
            checkpoint_path = os.path.join(
                config.save_dir,
                f'checkpoint_round_{round_id}.pth'
            )
            server.save_model(checkpoint_path)
            logger.log(f"✓ 保存检查点: Round {round_id}")
    
    # 训练完成
    print("\n" + "=" * 80)
    print("联邦训练完成!")
    print("=" * 80)
    print(f"最佳F1分数: {best_f1:.4f} (Round {best_round})")
    
    # 保存指标和绘制曲线
    logger.save_metrics()
    logger.plot_training_curves()
    
    # 最终评估
    print("\n最终评估:")
    for client in clients:
        eval_metrics = client.evaluate()
        print(f"  Client {client.client_id}: "
              f"Acc={eval_metrics['accuracy']:.4f}, "
              f"F1={eval_metrics['f1_score']:.4f}")
    
    logger.log("联邦学习训练结束")
    
    print("\n日志和模型已保存到:")
    print(f"  日志目录: {config.log_dir}")
    print(f"  模型目录: {config.save_dir}")


if __name__ == "__main__":
    main()
