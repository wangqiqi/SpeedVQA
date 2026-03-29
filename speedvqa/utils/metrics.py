"""
SpeedVQA训练指标计算系统

实现准确率、精确率、召回率、F1计算等指标，
支持多类别和二分类任务。
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class MetricsCalculator:
    """指标计算器"""
    
    def __init__(self, num_classes: int = 2, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f'Class_{i}' for i in range(num_classes)]
        
        # 累积统计
        self.reset()
    
    def reset(self):
        """重置累积统计"""
        self.all_predictions = []
        self.all_targets = []
        self.all_probabilities = []
        self.batch_losses = []
    
    def update(self, predictions: Union[torch.Tensor, np.ndarray], 
               targets: Union[torch.Tensor, np.ndarray],
               probabilities: Optional[Union[torch.Tensor, np.ndarray]] = None,
               loss: Optional[float] = None):
        """
        更新指标统计
        
        Args:
            predictions: 预测结果 [batch_size]
            targets: 真实标签 [batch_size]
            probabilities: 预测概率 [batch_size, num_classes] (可选)
            loss: 批次损失 (可选)
        """
        # 转换为numpy数组
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        if probabilities is not None and isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.detach().cpu().numpy()
        
        # 累积数据
        self.all_predictions.extend(predictions.flatten())
        self.all_targets.extend(targets.flatten())
        
        if probabilities is not None:
            self.all_probabilities.extend(probabilities)
        
        if loss is not None:
            self.batch_losses.append(loss)
    
    def compute_basic_metrics(self) -> Dict[str, float]:
        """计算基本分类指标"""
        if not self.all_predictions:
            return {}
        
        predictions = np.array(self.all_predictions)
        targets = np.array(self.all_targets)
        
        # 基本指标
        accuracy = accuracy_score(targets, predictions)
        
        # 多类别指标
        if self.num_classes > 2:
            precision = precision_score(targets, predictions, average='weighted', zero_division=0)
            recall = recall_score(targets, predictions, average='weighted', zero_division=0)
            f1 = f1_score(targets, predictions, average='weighted', zero_division=0)
            
            # 每类别指标
            precision_per_class = precision_score(targets, predictions, average=None, zero_division=0)
            recall_per_class = recall_score(targets, predictions, average=None, zero_division=0)
            f1_per_class = f1_score(targets, predictions, average=None, zero_division=0)
        else:
            # 二分类
            precision = precision_score(targets, predictions, zero_division=0)
            recall = recall_score(targets, predictions, zero_division=0)
            f1 = f1_score(targets, predictions, zero_division=0)
            
            precision_per_class = [precision]
            recall_per_class = [recall]
            f1_per_class = [f1]
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'num_samples': len(predictions)
        }
        
        # 添加每类别指标
        for i, class_name in enumerate(self.class_names):
            if i < len(precision_per_class):
                metrics[f'precision_{class_name}'] = precision_per_class[i]
                metrics[f'recall_{class_name}'] = recall_per_class[i]
                metrics[f'f1_{class_name}'] = f1_per_class[i]
        
        return metrics
    
    def compute_advanced_metrics(self) -> Dict[str, float]:
        """计算高级指标（需要概率）"""
        if not self.all_probabilities or not self.all_predictions:
            return {}
        
        targets = np.array(self.all_targets)
        probabilities = np.array(self.all_probabilities)
        
        metrics = {}
        
        try:
            if self.num_classes == 2:
                # 二分类AUC
                if probabilities.ndim == 2:
                    pos_probs = probabilities[:, 1]
                else:
                    pos_probs = probabilities
                
                auc_roc = roc_auc_score(targets, pos_probs)
                auc_pr = average_precision_score(targets, pos_probs)
                
                metrics.update({
                    'auc_roc': auc_roc,
                    'auc_pr': auc_pr
                })
            else:
                # 多分类AUC (one-vs-rest)
                auc_roc = roc_auc_score(targets, probabilities, multi_class='ovr', average='weighted')
                metrics['auc_roc'] = auc_roc
        
        except Exception:
            # AUC计算可能失败（例如只有一个类别）
            pass
        
        return metrics
    
    def compute_confusion_matrix(self) -> np.ndarray:
        """计算混淆矩阵"""
        if not self.all_predictions:
            return np.array([])
        
        predictions = np.array(self.all_predictions)
        targets = np.array(self.all_targets)
        
        return confusion_matrix(targets, predictions)
    
    def get_classification_report(self) -> str:
        """获取分类报告"""
        if not self.all_predictions:
            return "No predictions available"
        
        predictions = np.array(self.all_predictions)
        targets = np.array(self.all_targets)
        
        return classification_report(
            targets, predictions, 
            target_names=self.class_names,
            zero_division=0
        )
    
    def compute_all_metrics(self) -> Dict[str, Any]:
        """计算所有指标"""
        basic_metrics = self.compute_basic_metrics()
        advanced_metrics = self.compute_advanced_metrics()
        
        # 平均损失
        avg_loss = np.mean(self.batch_losses) if self.batch_losses else 0.0
        
        all_metrics = {
            **basic_metrics,
            **advanced_metrics,
            'avg_loss': avg_loss,
            'confusion_matrix': self.compute_confusion_matrix().tolist(),
            'classification_report': self.get_classification_report()
        }
        
        return all_metrics
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None, 
                            normalize: bool = False) -> Optional[plt.Figure]:
        """绘制混淆矩阵"""
        cm = self.compute_confusion_matrix()
        if cm.size == 0:
            return None
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_roc_curve(self, save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """绘制ROC曲线（仅二分类）"""
        if self.num_classes != 2 or not self.all_probabilities:
            return None
        
        targets = np.array(self.all_targets)
        probabilities = np.array(self.all_probabilities)
        
        if probabilities.ndim == 2:
            pos_probs = probabilities[:, 1]
        else:
            pos_probs = probabilities
        
        fpr, tpr, _ = roc_curve(targets, pos_probs)
        auc = roc_auc_score(targets, pos_probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, save_dir: str, experiment_name: str = 'experiment'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        
        # 历史记录
        self.train_history = {
            'epoch': [],
            'loss': [],
            'accuracy': [],
            'lr': []
        }
        
        self.val_history = {
            'epoch': [],
            'loss': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': []
        }
        
        self.best_metrics = {}
        self.current_epoch = 0
    
    def log_train_metrics(self, epoch: int, metrics: Dict[str, float], lr: float):
        """记录训练指标"""
        self.current_epoch = epoch
        
        self.train_history['epoch'].append(epoch)
        self.train_history['loss'].append(metrics.get('loss', 0.0))
        self.train_history['accuracy'].append(metrics.get('accuracy', 0.0))
        self.train_history['lr'].append(lr)
    
    def log_val_metrics(self, epoch: int, metrics: Dict[str, float]):
        """记录验证指标"""
        self.val_history['epoch'].append(epoch)
        self.val_history['loss'].append(metrics.get('loss', 0.0))
        self.val_history['accuracy'].append(metrics.get('accuracy', 0.0))
        self.val_history['precision'].append(metrics.get('precision', 0.0))
        self.val_history['recall'].append(metrics.get('recall', 0.0))
        self.val_history['f1_score'].append(metrics.get('f1_score', 0.0))
        
        # 更新最佳指标
        for key, value in metrics.items():
            if key not in self.best_metrics:
                self.best_metrics[key] = {'value': value, 'epoch': epoch}
            else:
                # 对于损失，越小越好；对于其他指标，越大越好
                if key == 'loss':
                    if value < self.best_metrics[key]['value']:
                        self.best_metrics[key] = {'value': value, 'epoch': epoch}
                else:
                    if value > self.best_metrics[key]['value']:
                        self.best_metrics[key] = {'value': value, 'epoch': epoch}
    
    def plot_training_curves(self, save_path: Optional[str] = None) -> plt.Figure:
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Curves - {self.experiment_name}', fontsize=16)
        
        # 损失曲线
        axes[0, 0].plot(self.train_history['epoch'], self.train_history['loss'], 
                       label='Train Loss', color='blue')
        if self.val_history['epoch']:
            axes[0, 0].plot(self.val_history['epoch'], self.val_history['loss'], 
                           label='Val Loss', color='red')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 准确率曲线
        axes[0, 1].plot(self.train_history['epoch'], self.train_history['accuracy'], 
                       label='Train Accuracy', color='blue')
        if self.val_history['epoch']:
            axes[0, 1].plot(self.val_history['epoch'], self.val_history['accuracy'], 
                           label='Val Accuracy', color='red')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 学习率曲线
        axes[1, 0].plot(self.train_history['epoch'], self.train_history['lr'], 
                       color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # F1分数曲线
        if self.val_history['epoch']:
            axes[1, 1].plot(self.val_history['epoch'], self.val_history['f1_score'], 
                           label='Val F1', color='purple')
            axes[1, 1].plot(self.val_history['epoch'], self.val_history['precision'], 
                           label='Val Precision', color='orange')
            axes[1, 1].plot(self.val_history['epoch'], self.val_history['recall'], 
                           label='Val Recall', color='brown')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Validation Metrics')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_metrics_history(self):
        """保存指标历史"""
        import json
        
        history_data = {
            'experiment_name': self.experiment_name,
            'current_epoch': self.current_epoch,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_metrics': self.best_metrics
        }
        
        save_path = self.save_dir / f'{self.experiment_name}_metrics_history.json'
        with open(save_path, 'w') as f:
            json.dump(history_data, f, indent=2)
    
    def load_metrics_history(self, history_path: str):
        """加载指标历史"""
        import json
        
        with open(history_path, 'r') as f:
            history_data = json.load(f)
        
        self.experiment_name = history_data.get('experiment_name', 'experiment')
        self.current_epoch = history_data.get('current_epoch', 0)
        self.train_history = history_data.get('train_history', {})
        self.val_history = history_data.get('val_history', {})
        self.best_metrics = history_data.get('best_metrics', {})
    
    def get_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        summary = {
            'experiment_name': self.experiment_name,
            'total_epochs': self.current_epoch,
            'best_metrics': self.best_metrics
        }
        
        if self.train_history['loss']:
            summary['final_train_loss'] = self.train_history['loss'][-1]
            summary['final_train_accuracy'] = self.train_history['accuracy'][-1]
        
        if self.val_history['loss']:
            summary['final_val_loss'] = self.val_history['loss'][-1]
            summary['final_val_accuracy'] = self.val_history['accuracy'][-1]
            summary['final_val_f1'] = self.val_history['f1_score'][-1]
        
        return summary


def calculate_metrics(predictions: Union[torch.Tensor, np.ndarray],
                     targets: Union[torch.Tensor, np.ndarray],
                     probabilities: Optional[Union[torch.Tensor, np.ndarray]] = None,
                     num_classes: int = 2,
                     class_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    计算分类指标（便捷函数）
    
    Args:
        predictions: 预测结果
        targets: 真实标签
        probabilities: 预测概率（可选）
        num_classes: 类别数量
        class_names: 类别名称
        
    Returns:
        metrics: 指标字典
    """
    calculator = MetricsCalculator(num_classes, class_names)
    calculator.update(predictions, targets, probabilities)
    return calculator.compute_all_metrics()


if __name__ == '__main__':
    # 测试指标计算
    print("=== SpeedVQA Metrics Test ===")
    
    # 模拟数据
    np.random.seed(42)
    n_samples = 1000
    
    # 二分类测试
    targets = np.random.randint(0, 2, n_samples)
    predictions = np.random.randint(0, 2, n_samples)
    probabilities = np.random.rand(n_samples, 2)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    
    print("Testing binary classification metrics:")
    metrics = calculate_metrics(predictions, targets, probabilities, 
                              num_classes=2, class_names=['NO', 'YES'])
    
    print(f"✓ Accuracy: {metrics['accuracy']:.4f}")
    print(f"✓ Precision: {metrics['precision']:.4f}")
    print(f"✓ Recall: {metrics['recall']:.4f}")
    print(f"✓ F1 Score: {metrics['f1_score']:.4f}")
    if 'auc_roc' in metrics:
        print(f"✓ AUC-ROC: {metrics['auc_roc']:.4f}")
    
    # 测试性能监控器
    print("\nTesting performance monitor:")
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    monitor = PerformanceMonitor(temp_dir, 'test_experiment')
    
    # 模拟训练过程
    for epoch in range(5):
        train_metrics = {'loss': 1.0 - epoch * 0.1, 'accuracy': 0.5 + epoch * 0.1}
        val_metrics = {'loss': 0.9 - epoch * 0.08, 'accuracy': 0.6 + epoch * 0.08, 
                      'precision': 0.55 + epoch * 0.09, 'recall': 0.58 + epoch * 0.07,
                      'f1_score': 0.56 + epoch * 0.08}
        
        monitor.log_train_metrics(epoch, train_metrics, 0.001 * (0.9 ** epoch))
        monitor.log_val_metrics(epoch, val_metrics)
    
    summary = monitor.get_summary()
    print(f"✓ Training summary: {summary['total_epochs']} epochs")
    print(f"✓ Best val accuracy: {summary['best_metrics']['accuracy']['value']:.4f} at epoch {summary['best_metrics']['accuracy']['epoch']}")
    
    # 清理
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("\n✓ Metrics test completed!")