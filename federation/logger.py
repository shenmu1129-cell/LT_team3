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
    free_energies: List[float] = field(default_factory=list)
    weights: List[float] = field(default_factory=list)
    extra_metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        # 处理client_metrics，可能是dict或list
        if isinstance(self.client_metrics, dict):
            cm_list = [
                cm.to_dict() if hasattr(cm, 'to_dict') else cm 
                for cm in self.client_metrics.values()
            ]
        else:
            cm_list = [
                cm.to_dict() if hasattr(cm, 'to_dict') else cm 
                for cm in self.client_metrics
            ]
        
        return {
            'round_id': self.round_id,
            'client_metrics': cm_list,
            'server_metrics': self.server_metrics,
            'communication_mb': self.communication_mb,
            'free_energies': self.free_energies,
            'weights': self.weights,
            'extra_metrics': self.extra_metrics,
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
    
    def log_client_metrics(self, client_metrics: Any = None, **kwargs) -> None:
        """
        记录客户端指标
        
        可以传入 ClientMetrics 对象，或者直接传入关键字参数 (client_id, free_energy, weight, loss, accuracy, f1_score, num_samples)
        """
        if client_metrics is not None and hasattr(client_metrics, 'client_id'):
            # 处理传入 ClientMetrics 对象的情况
            cid = client_metrics.client_id
            fe = client_metrics.free_energy
            w = client_metrics.weight
            loss = client_metrics.local_loss
            acc = client_metrics.accuracy
            f1 = client_metrics.f1_score
            n = getattr(client_metrics, 'num_samples', 0)
        else:
            # 处理传入关键字参数的情况
            cid = kwargs.get('client_id', 'unknown')
            fe = kwargs.get('free_energy', 0.0)
            w = kwargs.get('weight', 0.0)
            # 兼容 loss 或 local_loss
            loss = kwargs.get('loss', kwargs.get('local_loss', 0.0))
            acc = kwargs.get('accuracy', 0.0)
            f1 = kwargs.get('f1_score', kwargs.get('f1', 0.0))
            n = kwargs.get('num_samples', 0)

        msg = (
            f"Client {cid}: "
            f"F={fe:.4f}, "
            f"w={w:.4f}, "
            f"loss={loss:.4f}, "
            f"acc={acc:.4f}, "
            f"f1={f1:.4f}, "
            f"n={n}"
        )
        self.log(msg)
    
    def log_server_aggregation(
        self,
        weights: np.ndarray,
        free_energies: List[float],
        global_logits_stats: Dict[str, float]
    ) -> None:
        """记录服务器聚合信息"""
        self.log("Server Aggregation:")
        self.log(f"  Free Energies: {[f'{f:.4f}' for f in free_energies]}")
        self.log(f"  Weights: {[f'{w:.4f}' for w in weights]}")
        self.log(f"  Global Logits Stats:")
        # for key, value in global_logits_stats.items():
        #     self.log(f"    {key}: {value:.4f}")
        for key, value in global_logits_stats.items():
            if isinstance(value, (list, tuple)):
                self.log(f"    {key}: {value}")
            elif isinstance(value, (int, float)):
                self.log(f"    {key}: {value:.4f}")
            else:
                self.log(f"    {key}: {value}")
    
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
    
    def add_round_metrics(self, round_metrics) -> None:
        """添加回合指标，支持RoundMetrics对象或字典"""
        if isinstance(round_metrics, RoundMetrics):
            self.round_metrics_list.append(round_metrics)
        elif isinstance(round_metrics, dict):
            # 从字典创建RoundMetrics对象
            rm = RoundMetrics(
                round_id=round_metrics.get('round_id', len(self.round_metrics_list)),
                free_energies=round_metrics.get('free_energies', []),
                weights=round_metrics.get('weights', [])
            )
            known_keys = {
                'round_id', 'client_metrics', 'server_metrics',
                'communication', 'communication_mb', 'free_energies',
                'weights', 'avg_metrics'
            }
            rm.extra_metrics = {
                k: v for k, v in round_metrics.items()
                if k not in known_keys
            }
            rm.server_metrics = round_metrics.get('server_metrics', {})
            if 'communication' in round_metrics:
                rm.communication_mb = round_metrics['communication'].get('round_mb', 0.0)
            else:
                rm.communication_mb = round_metrics.get('communication_mb', 0.0)
            
            # 处理client_metrics
            client_metrics_list = round_metrics.get('client_metrics', [])
            for i, cm in enumerate(client_metrics_list):
                if isinstance(cm, dict):
                    client_id = cm.get('client_id', i)
                    rm.client_metrics[client_id] = ClientMetrics(
                        client_id=client_id,
                        free_energy=cm.get('free_energy', 0.0),
                        weight=cm.get('weight', 0.0),
                        local_loss=cm.get('loss', cm.get('distill_loss', cm.get('total_loss', 0.0))),
                        accuracy=cm.get('accuracy', 0.0),
                        precision=cm.get('precision', 0.0),
                        recall=cm.get('recall', 0.0),
                        f1_score=cm.get('f1_score', 0.0),
                        num_samples=cm.get('num_samples', 0)
                    )
                elif isinstance(cm, ClientMetrics):
                    rm.client_metrics[cm.client_id] = cm
            
            # 如果free_energies/weights为空，尝试从client_metrics中获取
            if not rm.free_energies and rm.client_metrics:
                rm.free_energies = [cm.free_energy for cm in rm.client_metrics.values()]
            if not rm.weights and rm.client_metrics:
                rm.weights = [cm.weight for cm in rm.client_metrics.values()]
            
            self.round_metrics_list.append(rm)
        else:
            self.log(f"Warning: Unknown round_metrics type: {type(round_metrics)}", "WARNING")
    
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
            if isinstance(rm.client_metrics, dict):
                for cm in rm.client_metrics.values():
                    if hasattr(cm, 'accuracy'):
                        all_accuracies.append(cm.accuracy)
                        all_f1_scores.append(cm.f1_score)
                        all_free_energies.append(cm.free_energy)
                    elif isinstance(cm, dict):
                        all_accuracies.append(cm.get('accuracy', 0))
                        all_f1_scores.append(cm.get('f1_score', 0))
                        all_free_energies.append(cm.get('free_energy', 0))
        
        if not all_accuracies:
            return {'total_rounds': len(self.round_metrics_list)}
        
        # 获取最后一轮的指标
        last_rm = self.round_metrics_list[-1]
        if isinstance(last_rm.client_metrics, dict) and last_rm.client_metrics:
            last_accs = []
            last_f1s = []
            for cm in last_rm.client_metrics.values():
                if hasattr(cm, 'accuracy'):
                    last_accs.append(cm.accuracy)
                    last_f1s.append(cm.f1_score)
                elif isinstance(cm, dict):
                    last_accs.append(cm.get('accuracy', 0))
                    last_f1s.append(cm.get('f1_score', 0))
            final_acc = float(np.mean(last_accs)) if last_accs else 0
            final_f1 = float(np.mean(last_f1s)) if last_f1s else 0
        else:
            final_acc = 0
            final_f1 = 0
        
        return {
            'total_rounds': len(self.round_metrics_list),
            'avg_accuracy': float(np.mean(all_accuracies)) if all_accuracies else 0,
            'avg_f1_score': float(np.mean(all_f1_scores)) if all_f1_scores else 0,
            'avg_free_energy': float(np.mean(all_free_energies)) if all_free_energies else 0,
            'final_round_accuracy': final_acc,
            'final_round_f1': final_f1
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
