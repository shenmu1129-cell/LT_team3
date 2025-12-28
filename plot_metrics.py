#!/usr/bin/env python3
"""
联邦学习训练指标可视化脚本

功能：
1. 绘制每轮客户端Loss、Accuracy、F1-score变化趋势
2. 绘制服务器聚合指标变化
3. 绘制客户端权重和自由能变化
4. 支持从JSON文件或实时日志读取数据

用法：
    python plot_metrics.py --metrics_file logs_federated/metrics_xxx.json
    python plot_metrics.py --log_dir logs_federated/
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional


# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class MetricsVisualizer:
    """训练指标可视化器"""
    
    def __init__(self, save_dir: str = './plots'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 存储每轮的指标
        self.rounds: List[int] = []
        self.client_metrics: Dict[int, Dict[str, List[float]]] = {}  # client_id -> {metric_name -> [values]}
        self.server_metrics: Dict[str, List[float]] = {}
        self.global_metrics: Dict[str, List[float]] = {
            'avg_loss': [],
            'avg_accuracy': [],
            'avg_f1': [],
            'avg_free_energy': []
        }
        self.raw_rounds_data: List[Dict] = []  # 保存原始数据
    
    def load_from_json(self, json_file: str) -> None:
        """从JSON文件加载指标数据"""
        print(f"加载指标文件: {json_file}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        rounds_data = data.get('rounds', [])
        self.raw_rounds_data = rounds_data  # 保存原始数据用于高级分析
        
        for round_data in rounds_data:
            if isinstance(round_data, dict):
                round_id = round_data.get('round_id', len(self.rounds))
                self.rounds.append(round_id)
                
                # 解析客户端指标
                client_metrics_list = round_data.get('client_metrics', [])
                round_losses = []
                round_accs = []
                round_f1s = []
                round_fes = []
                
                for cm in client_metrics_list:
                    if isinstance(cm, dict):
                        client_id = cm.get('client_id', 0)
                        
                        if client_id not in self.client_metrics:
                            self.client_metrics[client_id] = {
                                'loss': [], 'accuracy': [], 'f1_score': [],
                                'precision': [], 'recall': [], 'free_energy': [], 'weight': [],
                                'fpr': [], 'fnr': [], 'specificity': [],
                                'auc_roc': [], 'auc_pr': [],
                                'tp': [], 'tn': [], 'fp': [], 'fn': [],
                                # 5分类专有指标
                                'macro_f1': [], 'per_class_f1': []
                            }
                        
                        # 获取各项指标
                        loss = cm.get('loss', cm.get('distill_loss', cm.get('total_loss', 0)))
                        acc = cm.get('accuracy', 0)
                        f1 = cm.get('f1_score', 0)
                        precision = cm.get('precision', 0)
                        recall = cm.get('recall', 0)
                        fpr = cm.get('fpr', 0)
                        fnr = cm.get('fnr', 0)
                        specificity = cm.get('specificity', 0)
                        auc_roc = cm.get('auc_roc', 0)
                        auc_pr = cm.get('auc_pr', 0)
                        fe = cm.get('free_energy', 0)
                        w = cm.get('weight', 0)
                        
                        # 5分类专有指标
                        macro_f1 = cm.get('macro_f1', 0)
                        per_class_f1 = cm.get('per_class_f1', [0]*5)
                        
                        self.client_metrics[client_id]['loss'].append(loss)
                        self.client_metrics[client_id]['accuracy'].append(acc)
                        self.client_metrics[client_id]['f1_score'].append(f1)
                        self.client_metrics[client_id]['precision'].append(precision)
                        self.client_metrics[client_id]['recall'].append(recall)
                        self.client_metrics[client_id]['fpr'].append(fpr)
                        self.client_metrics[client_id]['fnr'].append(fnr)
                        self.client_metrics[client_id]['specificity'].append(specificity)
                        self.client_metrics[client_id]['auc_roc'].append(auc_roc)
                        self.client_metrics[client_id]['auc_pr'].append(auc_pr)
                        self.client_metrics[client_id]['free_energy'].append(fe)
                        self.client_metrics[client_id]['weight'].append(w)
                        
                        # 5分类专有指标
                        self.client_metrics[client_id]['macro_f1'].append(macro_f1)
                        self.client_metrics[client_id]['per_class_f1'].append(per_class_f1)
                        
                        # 混淆矩阵元素
                        self.client_metrics[client_id]['tp'].append(cm.get('tp', 0))
                        self.client_metrics[client_id]['tn'].append(cm.get('tn', 0))
                        self.client_metrics[client_id]['fp'].append(cm.get('fp', 0))
                        self.client_metrics[client_id]['fn'].append(cm.get('fn', 0))
                        
                        round_losses.append(loss)
                        round_accs.append(acc)
                        round_f1s.append(f1)
                        round_fes.append(fe)
                
                # 解析自由能和权重（从round级别）
                free_energies = round_data.get('free_energies', [])
                weights = round_data.get('weights', [])
                
                # 如果client_metrics中没有，从round级别获取
                for i, (fe, w) in enumerate(zip(free_energies, weights)):
                    if i in self.client_metrics:
                        if not self.client_metrics[i]['free_energy'] or self.client_metrics[i]['free_energy'][-1] == 0:
                            if self.client_metrics[i]['free_energy']:
                                self.client_metrics[i]['free_energy'][-1] = fe
                            else:
                                self.client_metrics[i]['free_energy'].append(fe)
                        if not self.client_metrics[i]['weight'] or self.client_metrics[i]['weight'][-1] == 0:
                            if self.client_metrics[i]['weight']:
                                self.client_metrics[i]['weight'][-1] = w
                            else:
                                self.client_metrics[i]['weight'].append(w)
                        round_fes.append(fe)
                
                # 计算全局平均
                if round_losses:
                    self.global_metrics['avg_loss'].append(np.mean(round_losses))
                if round_accs:
                    self.global_metrics['avg_accuracy'].append(np.mean(round_accs))
                if round_f1s:
                    self.global_metrics['avg_f1'].append(np.mean(round_f1s))
                if round_fes:
                    self.global_metrics['avg_free_energy'].append(np.mean(round_fes))
        
        print(f"加载完成: {len(self.rounds)} 轮, {len(self.client_metrics)} 个客户端")
    
    def add_round_metrics(self, round_id: int, client_metrics: List[Dict], 
                          free_energies: List[float], weights: List[float]) -> None:
        """手动添加一轮的指标（用于实时绘图）"""
        self.rounds.append(round_id)
        
        round_losses = []
        round_accs = []
        round_f1s = []
        
        for i, cm in enumerate(client_metrics):
            client_id = cm.get('client_id', i)
            
            if client_id not in self.client_metrics:
                self.client_metrics[client_id] = {
                    'loss': [], 'accuracy': [], 'f1_score': [],
                    'precision': [], 'recall': [], 'free_energy': [], 'weight': []
                }
            
            loss = cm.get('loss', cm.get('distill_loss', cm.get('total_loss', 0)))
            acc = cm.get('accuracy', 0)
            f1 = cm.get('f1_score', 0)
            
            self.client_metrics[client_id]['loss'].append(loss)
            self.client_metrics[client_id]['accuracy'].append(acc)
            self.client_metrics[client_id]['f1_score'].append(f1)
            self.client_metrics[client_id]['precision'].append(cm.get('precision', 0))
            self.client_metrics[client_id]['recall'].append(cm.get('recall', 0))
            
            if i < len(free_energies):
                self.client_metrics[client_id]['free_energy'].append(free_energies[i])
            if i < len(weights):
                self.client_metrics[client_id]['weight'].append(weights[i])
            
            round_losses.append(loss)
            round_accs.append(acc)
            round_f1s.append(f1)
        
        # 更新全局平均
        if round_losses:
            self.global_metrics['avg_loss'].append(np.mean(round_losses))
        if round_accs:
            self.global_metrics['avg_accuracy'].append(np.mean(round_accs))
        if round_f1s:
            self.global_metrics['avg_f1'].append(np.mean(round_f1s))
        if free_energies:
            self.global_metrics['avg_free_energy'].append(np.mean(free_energies))
    
    def plot_client_losses(self, save_name: str = 'client_losses.png') -> None:
        """绘制各客户端Loss变化曲线"""
        plt.figure(figsize=(12, 6))
        
        for client_id, metrics in self.client_metrics.items():
            if metrics['loss']:
                rounds = list(range(1, len(metrics['loss']) + 1))
                plt.plot(rounds, metrics['loss'], marker='o', label=f'Client {client_id}', linewidth=2, markersize=4)
        
        # 绘制平均Loss
        if self.global_metrics['avg_loss']:
            rounds = list(range(1, len(self.global_metrics['avg_loss']) + 1))
            plt.plot(rounds, self.global_metrics['avg_loss'], 'k--', 
                    label='Average', linewidth=3, marker='s', markersize=6)
        
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Client Loss over Federated Rounds', fontsize=14)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"保存: {save_path}")
    
    def plot_client_accuracy(self, save_name: str = 'client_accuracy.png') -> None:
        """绘制各客户端Accuracy变化曲线"""
        plt.figure(figsize=(12, 6))
        
        for client_id, metrics in self.client_metrics.items():
            if metrics['accuracy']:
                rounds = list(range(1, len(metrics['accuracy']) + 1))
                plt.plot(rounds, metrics['accuracy'], marker='o', label=f'Client {client_id}', linewidth=2, markersize=4)
        
        # 绘制平均Accuracy
        if self.global_metrics['avg_accuracy']:
            rounds = list(range(1, len(self.global_metrics['avg_accuracy']) + 1))
            plt.plot(rounds, self.global_metrics['avg_accuracy'], 'k--', 
                    label='Average', linewidth=3, marker='s', markersize=6)
        
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Client Accuracy over Federated Rounds', fontsize=14)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"保存: {save_path}")
    
    def plot_client_f1(self, save_name: str = 'client_f1_score.png') -> None:
        """绘制各客户端F1-score变化曲线"""
        plt.figure(figsize=(12, 6))
        
        for client_id, metrics in self.client_metrics.items():
            if metrics['f1_score']:
                rounds = list(range(1, len(metrics['f1_score']) + 1))
                plt.plot(rounds, metrics['f1_score'], marker='o', label=f'Client {client_id}', linewidth=2, markersize=4)
        
        # 绘制平均F1
        if self.global_metrics['avg_f1']:
            rounds = list(range(1, len(self.global_metrics['avg_f1']) + 1))
            plt.plot(rounds, self.global_metrics['avg_f1'], 'k--', 
                    label='Average', linewidth=3, marker='s', markersize=6)
        
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('F1-Score', fontsize=12)
        plt.title('Client F1-Score over Federated Rounds', fontsize=14)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"保存: {save_path}")
    
    def plot_free_energy(self, save_name: str = 'free_energy.png') -> None:
        """绘制各客户端自由能变化曲线"""
        plt.figure(figsize=(12, 6))
        
        for client_id, metrics in self.client_metrics.items():
            if metrics['free_energy']:
                rounds = list(range(1, len(metrics['free_energy']) + 1))
                plt.plot(rounds, metrics['free_energy'], marker='o', label=f'Client {client_id}', linewidth=2, markersize=4)
        
        # 绘制平均自由能
        if self.global_metrics['avg_free_energy']:
            rounds = list(range(1, len(self.global_metrics['avg_free_energy']) + 1))
            plt.plot(rounds, self.global_metrics['avg_free_energy'], 'k--', 
                    label='Average', linewidth=3, marker='s', markersize=6)
        
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Free Energy', fontsize=12)
        plt.title('Client Free Energy over Federated Rounds', fontsize=14)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"保存: {save_path}")
    
    def plot_client_weights(self, save_name: str = 'client_weights.png') -> None:
        """绘制各客户端权重变化曲线"""
        plt.figure(figsize=(12, 6))
        
        for client_id, metrics in self.client_metrics.items():
            if metrics['weight']:
                rounds = list(range(1, len(metrics['weight']) + 1))
                plt.plot(rounds, metrics['weight'], marker='o', label=f'Client {client_id}', linewidth=2, markersize=4)
        
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Weight', fontsize=12)
        plt.title('Client Aggregation Weights over Federated Rounds', fontsize=14)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"保存: {save_path}")
    
    def plot_combined_metrics(self, save_name: str = 'combined_metrics.png') -> None:
        """绘制组合指标图（2x2子图）"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.client_metrics)))
        
        # 子图1: Loss
        ax1 = axes[0, 0]
        for (client_id, metrics), color in zip(self.client_metrics.items(), colors):
            if metrics['loss']:
                rounds = list(range(1, len(metrics['loss']) + 1))
                ax1.plot(rounds, metrics['loss'], marker='o', color=color, 
                        label=f'Client {client_id}', linewidth=2, markersize=3)
        if self.global_metrics['avg_loss']:
            rounds = list(range(1, len(self.global_metrics['avg_loss']) + 1))
            ax1.plot(rounds, self.global_metrics['avg_loss'], 'k--', 
                    label='Average', linewidth=2.5, marker='s', markersize=5)
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss')
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 子图2: Accuracy
        ax2 = axes[0, 1]
        for (client_id, metrics), color in zip(self.client_metrics.items(), colors):
            if metrics['accuracy']:
                rounds = list(range(1, len(metrics['accuracy']) + 1))
                ax2.plot(rounds, metrics['accuracy'], marker='o', color=color, 
                        label=f'Client {client_id}', linewidth=2, markersize=3)
        if self.global_metrics['avg_accuracy']:
            rounds = list(range(1, len(self.global_metrics['avg_accuracy']) + 1))
            ax2.plot(rounds, self.global_metrics['avg_accuracy'], 'k--', 
                    label='Average', linewidth=2.5, marker='s', markersize=5)
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy')
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)
        
        # 子图3: F1-Score
        ax3 = axes[1, 0]
        for (client_id, metrics), color in zip(self.client_metrics.items(), colors):
            if metrics['f1_score']:
                rounds = list(range(1, len(metrics['f1_score']) + 1))
                ax3.plot(rounds, metrics['f1_score'], marker='o', color=color, 
                        label=f'Client {client_id}', linewidth=2, markersize=3)
        if self.global_metrics['avg_f1']:
            rounds = list(range(1, len(self.global_metrics['avg_f1']) + 1))
            ax3.plot(rounds, self.global_metrics['avg_f1'], 'k--', 
                    label='Average', linewidth=2.5, marker='s', markersize=5)
        ax3.set_xlabel('Round')
        ax3.set_ylabel('F1-Score')
        ax3.set_title('F1-Score')
        ax3.legend(loc='best', fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.05)
        
        # 子图4: Free Energy & Weights
        ax4 = axes[1, 1]
        ax4_twin = ax4.twinx()
        
        for (client_id, metrics), color in zip(self.client_metrics.items(), colors):
            if metrics['free_energy']:
                rounds = list(range(1, len(metrics['free_energy']) + 1))
                ax4.plot(rounds, metrics['free_energy'], marker='o', color=color, 
                        label=f'FE-C{client_id}', linewidth=2, markersize=3)
            if metrics['weight']:
                rounds = list(range(1, len(metrics['weight']) + 1))
                ax4_twin.plot(rounds, metrics['weight'], linestyle='--', color=color, 
                             label=f'W-C{client_id}', linewidth=1.5, alpha=0.7)
        
        ax4.set_xlabel('Round')
        ax4.set_ylabel('Free Energy', color='blue')
        ax4_twin.set_ylabel('Weight', color='red')
        ax4.set_title('Free Energy (solid) & Weights (dashed)')
        ax4.grid(True, alpha=0.3)
        
        # 合并图例
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=7)
        
        plt.suptitle('Federated Learning Training Metrics', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"保存: {save_path}")
    
    def plot_fpr_fnr(self, save_name: str = 'fpr_fnr.png') -> None:
        """绘制误报率和漏报率变化曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.client_metrics)))
        
        # FPR - 误报率
        ax1 = axes[0]
        for (client_id, metrics), color in zip(self.client_metrics.items(), colors):
            if 'fpr' in metrics and metrics['fpr']:
                rounds = list(range(1, len(metrics['fpr']) + 1))
                ax1.plot(rounds, metrics['fpr'], marker='o', color=color, 
                        label=f'Client {client_id}', linewidth=2, markersize=4)
        ax1.set_xlabel('Round')
        ax1.set_ylabel('False Positive Rate (误报率)')
        ax1.set_title('FPR - 正常样本被误判为攻击的比例')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.05)
        
        # FNR - 漏报率
        ax2 = axes[1]
        for (client_id, metrics), color in zip(self.client_metrics.items(), colors):
            if 'fnr' in metrics and metrics['fnr']:
                rounds = list(range(1, len(metrics['fnr']) + 1))
                ax2.plot(rounds, metrics['fnr'], marker='o', color=color, 
                        label=f'Client {client_id}', linewidth=2, markersize=4)
        ax2.set_xlabel('Round')
        ax2.set_ylabel('False Negative Rate (漏报率)')
        ax2.set_title('FNR - 攻击样本被漏检的比例')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"保存: {save_path}")
    
    def plot_auc_metrics(self, save_name: str = 'auc_metrics.png') -> None:
        """绘制AUC-ROC和AUC-PR变化曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.client_metrics)))
        
        # AUC-ROC
        ax1 = axes[0]
        has_auc_roc = False
        for (client_id, metrics), color in zip(self.client_metrics.items(), colors):
            if 'auc_roc' in metrics and metrics['auc_roc']:
                rounds = list(range(1, len(metrics['auc_roc']) + 1))
                ax1.plot(rounds, metrics['auc_roc'], marker='o', color=color, 
                        label=f'Client {client_id}', linewidth=2, markersize=4)
                has_auc_roc = True
        ax1.axhline(y=0.5, color='gray', linestyle='--', label='Random (0.5)')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('AUC-ROC')
        ax1.set_title('AUC-ROC - 分类器综合性能')
        if has_auc_roc:
            ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0.4, 1.05)
        
        # AUC-PR
        ax2 = axes[1]
        has_auc_pr = False
        for (client_id, metrics), color in zip(self.client_metrics.items(), colors):
            if 'auc_pr' in metrics and metrics['auc_pr']:
                rounds = list(range(1, len(metrics['auc_pr']) + 1))
                ax2.plot(rounds, metrics['auc_pr'], marker='o', color=color, 
                        label=f'Client {client_id}', linewidth=2, markersize=4)
                has_auc_pr = True
        ax2.set_xlabel('Round')
        ax2.set_ylabel('AUC-PR')
        ax2.set_title('AUC-PR - 不平衡数据下的性能')
        if has_auc_pr:
            ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"保存: {save_path}")
    
    def plot_confusion_matrix_trend(self, save_name: str = 'confusion_trend.png') -> None:
        """绘制混淆矩阵元素随训练的变化"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 汇总所有客户端的TP, TN, FP, FN
        rounds = []
        total_tp, total_tn, total_fp, total_fn = [], [], [], []
        
        for round_data in self.raw_rounds_data:
            if isinstance(round_data, dict):
                rounds.append(round_data.get('round_id', len(rounds)))
                
                round_tp, round_tn, round_fp, round_fn = 0, 0, 0, 0
                for cm in round_data.get('client_metrics', []):
                    if isinstance(cm, dict):
                        round_tp += cm.get('tp', 0)
                        round_tn += cm.get('tn', 0)
                        round_fp += cm.get('fp', 0)
                        round_fn += cm.get('fn', 0)
                
                total_tp.append(round_tp)
                total_tn.append(round_tn)
                total_fp.append(round_fp)
                total_fn.append(round_fn)
        
        if not rounds:
            plt.close()
            return
        
        x_rounds = list(range(1, len(rounds) + 1))
        
        # TP - True Positives
        axes[0, 0].plot(x_rounds, total_tp, 'g-o', linewidth=2, markersize=6)
        axes[0, 0].fill_between(x_rounds, total_tp, alpha=0.3, color='green')
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('True Positives (正确检测的攻击)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # TN - True Negatives
        axes[0, 1].plot(x_rounds, total_tn, 'b-o', linewidth=2, markersize=6)
        axes[0, 1].fill_between(x_rounds, total_tn, alpha=0.3, color='blue')
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('True Negatives (正确识别的正常)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # FP - False Positives (误报)
        axes[1, 0].plot(x_rounds, total_fp, 'r-o', linewidth=2, markersize=6)
        axes[1, 0].fill_between(x_rounds, total_fp, alpha=0.3, color='red')
        axes[1, 0].set_xlabel('Round')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('False Positives (误报 - 正常被判为攻击)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # FN - False Negatives (漏报)
        axes[1, 1].plot(x_rounds, total_fn, 'orange', marker='o', linewidth=2, markersize=6)
        axes[1, 1].fill_between(x_rounds, total_fn, alpha=0.3, color='orange')
        axes[1, 1].set_xlabel('Round')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('False Negatives (漏报 - 攻击被判为正常)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Confusion Matrix Elements over Training', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"保存: {save_path}")
    
    def plot_detection_vs_false_alarm(self, save_name: str = 'detection_vs_false_alarm.png') -> None:
        """绘制检测率 vs 误报率的权衡曲线"""
        plt.figure(figsize=(10, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.client_metrics)))
        
        for (client_id, metrics), color in zip(self.client_metrics.items(), colors):
            if metrics['recall'] and 'fpr' in metrics and metrics['fpr']:
                # recall就是检测率
                detection_rates = metrics['recall']
                false_alarm_rates = metrics['fpr']
                
                # 绘制曲线，用箭头表示训练方向
                plt.plot(false_alarm_rates, detection_rates, 'o-', color=color, 
                        label=f'Client {client_id}', linewidth=2, markersize=6)
                
                # 标记起点和终点
                if len(detection_rates) > 0:
                    plt.annotate('Start', (false_alarm_rates[0], detection_rates[0]), 
                               textcoords="offset points", xytext=(5,5), fontsize=8)
                    plt.annotate('End', (false_alarm_rates[-1], detection_rates[-1]), 
                               textcoords="offset points", xytext=(5,5), fontsize=8)
        
        # 理想点
        plt.scatter([0], [1], s=200, c='gold', marker='*', edgecolors='black', 
                   label='Ideal Point (0,1)', zorder=5)
        
        # 随机分类器线
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        
        plt.xlabel('False Positive Rate (误报率)', fontsize=12)
        plt.ylabel('Detection Rate / Recall (检测率)', fontsize=12)
        plt.title('Detection Rate vs False Alarm Rate Trade-off', fontsize=14)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"保存: {save_path}")
    
    def plot_federated_convergence(self, save_name: str = 'federated_convergence.png') -> None:
        """绘制联邦学习收敛分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Loss收敛曲线
        ax1 = axes[0, 0]
        if self.global_metrics['avg_loss']:
            rounds = list(range(1, len(self.global_metrics['avg_loss']) + 1))
            ax1.plot(rounds, self.global_metrics['avg_loss'], 'b-o', linewidth=2, markersize=6)
            ax1.fill_between(rounds, self.global_metrics['avg_loss'], alpha=0.2)
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Average Loss')
        ax1.set_title('Loss Convergence')
        ax1.grid(True, alpha=0.3)
        
        # 2. 客户端间Loss方差
        ax2 = axes[0, 1]
        loss_variance = []
        for round_data in self.raw_rounds_data:
            if isinstance(round_data, dict):
                losses = [cm.get('loss', cm.get('total_loss', 0)) 
                         for cm in round_data.get('client_metrics', [])
                         if isinstance(cm, dict)]
                if losses:
                    loss_variance.append(np.var(losses))
        
        if loss_variance:
            rounds = list(range(1, len(loss_variance) + 1))
            ax2.plot(rounds, loss_variance, 'r-o', linewidth=2, markersize=6)
            ax2.fill_between(rounds, loss_variance, alpha=0.2, color='red')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Loss Variance')
        ax2.set_title('Client Heterogeneity (Loss Variance)')
        ax2.grid(True, alpha=0.3)
        
        # 3. 权重分布变化
        ax3 = axes[1, 0]
        for client_id, metrics in self.client_metrics.items():
            if metrics['weight']:
                rounds = list(range(1, len(metrics['weight']) + 1))
                ax3.plot(rounds, metrics['weight'], '-o', label=f'Client {client_id}', 
                        linewidth=2, markersize=4, alpha=0.7)
        ax3.axhline(y=1/len(self.client_metrics) if self.client_metrics else 0.33, 
                   color='gray', linestyle='--', label='Uniform Weight')
        ax3.set_xlabel('Round')
        ax3.set_ylabel('Aggregation Weight')
        ax3.set_title('Client Weight Evolution')
        ax3.legend(loc='best', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. 性能提升率
        ax4 = axes[1, 1]
        if len(self.global_metrics['avg_f1']) > 1:
            rounds = list(range(2, len(self.global_metrics['avg_f1']) + 1))
            improvement = [self.global_metrics['avg_f1'][i] - self.global_metrics['avg_f1'][i-1] 
                          for i in range(1, len(self.global_metrics['avg_f1']))]
            colors = ['green' if x > 0 else 'red' for x in improvement]
            ax4.bar(rounds, improvement, color=colors, alpha=0.7, edgecolor='black')
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4.set_xlabel('Round')
        ax4.set_ylabel('F1 Improvement')
        ax4.set_title('Per-Round Performance Improvement')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Federated Learning Convergence Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"保存: {save_path}")
    
    def plot_per_class_f1(self, save_name: str = 'per_class_f1.png') -> None:
        """绘制5分类每个类别的F1-score变化"""
        CLASS_NAMES = ['Normal', 'Adv Patch', 'Sensor Spoof', 'Physical', 'Data Poison']
        
        # 检查是否有per_class_f1数据
        has_per_class_data = False
        for client_id, metrics in self.client_metrics.items():
            if 'per_class_f1' in metrics and metrics['per_class_f1']:
                has_per_class_data = True
                break
        
        if not has_per_class_data:
            print("跳过 per_class_f1 图表: 没有每类别F1数据")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        colors = plt.cm.Set2(np.linspace(0, 1, 5))
        
        # 为每个类别绘制一个子图
        for class_idx in range(5):
            ax = axes[class_idx]
            
            for client_id, metrics in self.client_metrics.items():
                if 'per_class_f1' in metrics and metrics['per_class_f1']:
                    # per_class_f1 是每轮的5个值的列表
                    class_f1_over_rounds = []
                    for round_f1 in metrics['per_class_f1']:
                        if isinstance(round_f1, list) and len(round_f1) > class_idx:
                            class_f1_over_rounds.append(round_f1[class_idx])
                    
                    if class_f1_over_rounds:
                        rounds = list(range(1, len(class_f1_over_rounds) + 1))
                        ax.plot(rounds, class_f1_over_rounds, 'o-', 
                               label=f'Client {client_id}', linewidth=2, markersize=4)
            
            ax.set_xlabel('Round')
            ax.set_ylabel('F1-Score')
            ax.set_title(f'{CLASS_NAMES[class_idx]}', fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.05)
        
        # 隐藏最后一个子图（因为只有5个类）
        axes[5].axis('off')
        
        plt.suptitle('Per-Class F1-Score over Training Rounds', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"保存: {save_path}")
    
    def plot_multiclass_summary(self, save_name: str = 'multiclass_summary.png') -> None:
        """绘制5分类汇总图表"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.client_metrics)))
        
        # 1. Macro F1
        ax1 = axes[0, 0]
        for (client_id, metrics), color in zip(self.client_metrics.items(), colors):
            if 'macro_f1' in metrics and metrics['macro_f1']:
                rounds = list(range(1, len(metrics['macro_f1']) + 1))
                ax1.plot(rounds, metrics['macro_f1'], marker='o', color=color, 
                        label=f'Client {client_id}', linewidth=2, markersize=4)
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Macro F1-Score')
        ax1.set_title('Macro F1-Score (5-Class Average)')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.05)
        
        # 2. 5分类 Accuracy
        ax2 = axes[0, 1]
        for (client_id, metrics), color in zip(self.client_metrics.items(), colors):
            if metrics['accuracy']:
                rounds = list(range(1, len(metrics['accuracy']) + 1))
                ax2.plot(rounds, metrics['accuracy'], marker='o', color=color, 
                        label=f'Client {client_id}', linewidth=2, markersize=4)
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('5-Class Classification Accuracy')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)
        
        # 3. Binary视角的FPR/FNR
        ax3 = axes[1, 0]
        for (client_id, metrics), color in zip(self.client_metrics.items(), colors):
            if 'fpr' in metrics and metrics['fpr']:
                rounds = list(range(1, len(metrics['fpr']) + 1))
                ax3.plot(rounds, metrics['fpr'], marker='o', color=color, 
                        linestyle='-', label=f'FPR-Client{client_id}', linewidth=2, markersize=4)
            if 'fnr' in metrics and metrics['fnr']:
                rounds = list(range(1, len(metrics['fnr']) + 1))
                ax3.plot(rounds, metrics['fnr'], marker='s', color=color, 
                        linestyle='--', label=f'FNR-Client{client_id}', linewidth=2, markersize=4)
        ax3.set_xlabel('Round')
        ax3.set_ylabel('Rate')
        ax3.set_title('FPR & FNR (Binary: Normal vs Attack)')
        ax3.legend(loc='best', fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.05)
        
        # 4. Loss
        ax4 = axes[1, 1]
        for (client_id, metrics), color in zip(self.client_metrics.items(), colors):
            if metrics['loss']:
                rounds = list(range(1, len(metrics['loss']) + 1))
                ax4.plot(rounds, metrics['loss'], marker='o', color=color, 
                        label=f'Client {client_id}', linewidth=2, markersize=4)
        ax4.set_xlabel('Round')
        ax4.set_ylabel('Loss')
        ax4.set_title('Training Loss')
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('5-Class Classification Summary', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"保存: {save_path}")
    
    def plot_all(self) -> None:
        """绘制所有图表"""
        print("\n开始绘制训练指标图表...")
        
        # 基础指标
        self.plot_client_losses()
        self.plot_client_accuracy()
        self.plot_client_f1()
        
        # 联邦学习指标
        self.plot_free_energy()
        self.plot_client_weights()
        
        # 误报漏报分析
        self.plot_fpr_fnr()
        
        # AUC指标（如果有）
        self.plot_auc_metrics()
        
        # 混淆矩阵趋势
        self.plot_confusion_matrix_trend()
        
        # 检测率vs误报率权衡
        self.plot_detection_vs_false_alarm()
        
        # 联邦学习收敛分析
        self.plot_federated_convergence()
        
        # 5分类专有图表
        self.plot_per_class_f1()
        self.plot_multiclass_summary()
        
        # 组合图
        self.plot_combined_metrics()
        
        print(f"\n所有图表已保存到: {self.save_dir}")
    
    def print_summary(self) -> None:
        """打印训练摘要"""
        print("\n" + "="*60)
        print("训练指标摘要")
        print("="*60)
        
        print(f"总轮数: {len(self.rounds)}")
        print(f"客户端数: {len(self.client_metrics)}")
        
        if self.global_metrics['avg_loss']:
            print(f"\n平均Loss:")
            print(f"  初始: {self.global_metrics['avg_loss'][0]:.4f}")
            print(f"  最终: {self.global_metrics['avg_loss'][-1]:.4f}")
            print(f"  最小: {min(self.global_metrics['avg_loss']):.4f}")
        
        if self.global_metrics['avg_accuracy']:
            print(f"\n平均Accuracy:")
            print(f"  初始: {self.global_metrics['avg_accuracy'][0]:.4f}")
            print(f"  最终: {self.global_metrics['avg_accuracy'][-1]:.4f}")
            print(f"  最大: {max(self.global_metrics['avg_accuracy']):.4f}")
        
        if self.global_metrics['avg_f1']:
            print(f"\n平均F1-Score:")
            print(f"  初始: {self.global_metrics['avg_f1'][0]:.4f}")
            print(f"  最终: {self.global_metrics['avg_f1'][-1]:.4f}")
            print(f"  最大: {max(self.global_metrics['avg_f1']):.4f}")
        
        print("="*60)


def find_latest_metrics_file(log_dir: str) -> Optional[str]:
    """找到最新的metrics JSON文件"""
    if not os.path.exists(log_dir):
        return None
    
    json_files = [f for f in os.listdir(log_dir) if f.startswith('metrics_') and f.endswith('.json')]
    
    if not json_files:
        return None
    
    # 按修改时间排序
    json_files.sort(key=lambda x: os.path.getmtime(os.path.join(log_dir, x)), reverse=True)
    
    return os.path.join(log_dir, json_files[0])


def main():
    parser = argparse.ArgumentParser(description='联邦学习训练指标可视化')
    
    parser.add_argument('--metrics_file', type=str, default=None,
                        help='指标JSON文件路径')
    parser.add_argument('--log_dir', type=str, default='logs_federated',
                        help='日志目录（自动查找最新的metrics文件）')
    parser.add_argument('--save_dir', type=str, default='plots',
                        help='图表保存目录')
    parser.add_argument('--show', action='store_true',
                        help='显示图表（默认只保存）')
    
    args = parser.parse_args()
    
    # 确定metrics文件
    if args.metrics_file:
        metrics_file = args.metrics_file
    else:
        metrics_file = find_latest_metrics_file(args.log_dir)
        if metrics_file is None:
            print(f"错误: 在 {args.log_dir} 中找不到metrics文件")
            print("请使用 --metrics_file 指定文件路径")
            return
    
    if not os.path.exists(metrics_file):
        print(f"错误: 文件不存在 - {metrics_file}")
        return
    
    # 创建可视化器
    visualizer = MetricsVisualizer(save_dir=args.save_dir)
    
    # 加载数据
    visualizer.load_from_json(metrics_file)
    
    # 打印摘要
    visualizer.print_summary()
    
    # 绘制所有图表
    visualizer.plot_all()
    
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
