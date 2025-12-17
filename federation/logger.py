"""
日志记录模块

管理联邦学习训练过程中的指标记录和日志输出。
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
import json
import os
from datetime import datetime
import numpy as np


@dataclass
class ClientMetrics:
    """客户端指标"""
    client_id: int
    free_energy: float
    weight: float
    local_loss: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    num_samples: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class RoundMetrics:
    """单轮训练指标"""
    round_id: int
    client_metrics: Dict[int, ClientMetrics] = field(default_factory=dict)
    server_metrics: Dict[str, float] = field(default_factory=dict)
    communication_mb: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'round_id': self.round_id,
            'client_metrics': {
                cid: cm.to_dict() for cid, cm in self.client_metrics.items()
            },
            'server_metrics': self.server_metrics,
            'communication_mb': self.communication_mb,
            'timestamp': self.timestamp
        }


class FederatedLogger:
    """联邦学习日志记录器"""
    
    def __init__(self, log_dir: str, verbose: bool = True):
        """
        Args:
            log_dir: 日志保存目录
            verbose: 是否打印详细日志
        """
        self.log_dir = log_dir
        self.verbose = verbose
        self.round_metrics_list: List[RoundMetrics] = []
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"federated_training_{timestamp}.log")
        self.metrics_file = os.path.join(log_dir, f"metrics_{timestamp}.json")
        
        # 初始化日志文件
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"Federated Learning Training Log\n")
            f.write(f"Started at: {timestamp}\n")
            f.write("=" * 80 + "\n\n")
    
    def log(self, message: str, level: str = "INFO") -> None:
        """
        记录日志消息
        
        Args:
            message: 日志消息
            level: 日志级别 (INFO, WARNING, ERROR)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] [{level}] {message}\n"
        
        # 写入文件
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_line)
        
        # 打印到控制台
        if self.verbose:
            print(log_line.rstrip())
    
    def log_round_start(self, round_id: int, num_clients: int) -> None:
        """记录回合开始"""
        self.log("=" * 80)
        self.log(f"Round {round_id} Started - {num_clients} clients")
        self.log("=" * 80)
    
    def log_client_metrics(self, client_metrics: ClientMetrics) -> None:
        """记录客户端指标"""
        msg = (
            f"Client {client_metrics.client_id}: "
            f"F={client_metrics.free_energy:.4f}, "
            f"w={client_metrics.weight:.4f}, "
            f"loss={client_metrics.local_loss:.4f}, "
            f"acc={client_metrics.accuracy:.4f}, "
            f"f1={client_metrics.f1_score:.4f}"
        )
        self.log(msg)
    
    def log_server_aggregation(
        self,
        free_energies: List[float],
        weights: np.ndarray,
        global_logits_stats: Dict[str, float]
    ) -> None:
        """记录服务器聚合信息"""
        self.log("Server Aggregation:")
        self.log(f"  Free Energies: {[f'{f:.4f}' for f in free_energies]}")
        self.log(f"  Weights: {[f'{w:.4f}' for w in weights]}")
        self.log(f"  Global Logits Stats:")
        for key, value in global_logits_stats.items():
            self.log(f"    {key}: {value:.4f}")
    
    def log_communication_stats(self, comm_stats: Dict[str, float]) -> None:
        """记录通信统计"""
        self.log("Communication Stats:")
        self.log(f"  Round Sent: {comm_stats['round_sent_mb']:.4f} MB")
        self.log(f"  Round Received: {comm_stats['round_received_mb']:.4f} MB")
        self.log(f"  Round Total: {comm_stats['round_mb']:.4f} MB")
        self.log(f"  Cumulative Total: {comm_stats['total_mb']:.4f} MB")
    
    def log_round_end(self, round_id: int, avg_metrics: Dict[str, float]) -> None:
        """记录回合结束"""
        self.log(f"Round {round_id} Completed")
        self.log(f"  Average Metrics:")
        for key, value in avg_metrics.items():
            self.log(f"    {key}: {value:.4f}")
        self.log("=" * 80 + "\n")
    
    def add_round_metrics(self, round_metrics: RoundMetrics) -> None:
        """添加回合指标"""
        self.round_metrics_list.append(round_metrics)
    
    def save_metrics(self) -> None:
        """保存所有指标到JSON文件"""
        metrics_data = {
            'rounds': [rm.to_dict() for rm in self.round_metrics_list],
            'summary': self._compute_summary()
        }
        
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=4, ensure_ascii=False)
        
        self.log(f"Metrics saved to: {self.metrics_file}")
    
    def _compute_summary(self) -> Dict[str, Any]:
        """计算训练摘要统计"""
        if not self.round_metrics_list:
            return {}
        
        # 收集所有客户端的指标
        all_accuracies = []
        all_f1_scores = []
        all_free_energies = []
        
        for rm in self.round_metrics_list:
            for cm in rm.client_metrics.values():
                all_accuracies.append(cm.accuracy)
                all_f1_scores.append(cm.f1_score)
                all_free_energies.append(cm.free_energy)
        
        return {
            'total_rounds': len(self.round_metrics_list),
            'avg_accuracy': float(np.mean(all_accuracies)),
            'avg_f1_score': float(np.mean(all_f1_scores)),
            'avg_free_energy': float(np.mean(all_free_energies)),
            'final_round_accuracy': float(np.mean([
                cm.accuracy for cm in self.round_metrics_list[-1].client_metrics.values()
            ])),
            'final_round_f1': float(np.mean([
                cm.f1_score for cm in self.round_metrics_list[-1].client_metrics.values()
            ]))
        }
    
    def plot_training_curves(self) -> None:
        """绘制训练曲线"""
        try:
            import matplotlib.pyplot as plt
            
            if not self.round_metrics_list:
                self.log("No metrics to plot", "WARNING")
                return
            
            rounds = [rm.round_id for rm in self.round_metrics_list]
            
            # 收集每轮的平均指标
            avg_accuracies = []
            avg_f1_scores = []
            avg_free_energies = []
            avg_weights = []
            
            for rm in self.round_metrics_list:
                metrics_list = list(rm.client_metrics.values())
                avg_accuracies.append(np.mean([m.accuracy for m in metrics_list]))
                avg_f1_scores.append(np.mean([m.f1_score for m in metrics_list]))
                avg_free_energies.append(np.mean([m.free_energy for m in metrics_list]))
                avg_weights.append(np.mean([m.weight for m in metrics_list]))
            
            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 准确率和F1
            axes[0, 0].plot(rounds, avg_accuracies, 'b-o', label='Accuracy')
            axes[0, 0].plot(rounds, avg_f1_scores, 'r-s', label='F1 Score')
            axes[0, 0].set_xlabel('Round')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].set_title('Detection Performance')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # 自由能
            axes[0, 1].plot(rounds, avg_free_energies, 'g-^', label='Free Energy')
            axes[0, 1].set_xlabel('Round')
            axes[0, 1].set_ylabel('Free Energy')
            axes[0, 1].set_title('Average Free Energy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # 权重
            axes[1, 0].plot(rounds, avg_weights, 'm-d', label='Weight')
            axes[1, 0].set_xlabel('Round')
            axes[1, 0].set_ylabel('Weight')
            axes[1, 0].set_title('Average Client Weight')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # 通信量
            comm_mbs = [rm.communication_mb for rm in self.round_metrics_list]
            axes[1, 1].plot(rounds, comm_mbs, 'c-p', label='Communication')
            axes[1, 1].set_xlabel('Round')
            axes[1, 1].set_ylabel('Communication (MB)')
            axes[1, 1].set_title('Communication per Round')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            plot_path = os.path.join(self.log_dir, 'training_curves.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.log(f"Training curves saved to: {plot_path}")
            
        except ImportError:
            self.log("matplotlib not available, skipping plot", "WARNING")
        except Exception as e:
            self.log(f"Error plotting curves: {e}", "ERROR")
