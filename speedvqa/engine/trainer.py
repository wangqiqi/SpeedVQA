"""
SpeedVQA可配置训练器

实现完全可配置的训练系统，支持混合精度、梯度累积、早停等功能。
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Any, Optional
import logging
from tqdm import tqdm

from ..utils.artifact_paths import resolve_train_save_dir


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 15, min_delta: float = 0.001, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
        self.compare = self._get_compare_fn()
    
    def _get_compare_fn(self):
        if self.mode == 'min':
            return lambda current, best: current < best - self.min_delta
        else:  # mode == 'max'
            return lambda current, best: current > best + self.min_delta
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.compare(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class ConfigurableTrainer:
    """可配置训练器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.train_config = config.get('train', {})
        
        # 保存目录（仓库内强制落在 runs/exports/cache 下，避免根目录误写）
        _raw_save = self.train_config.get('save_dir', './runs/train')
        _exp = self.train_config.get('experiment_name', 'speedvqa_exp')
        self.save_dir, self._save_dir_coerced = resolve_train_save_dir(_raw_save, _exp)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志（必须在device setup之前）
        self._setup_logging()
        
        self.device = self._setup_device()
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = None
        self.train_losses = []
        self.val_metrics = []
        
        # 混合精度训练
        self.use_amp = self.train_config.get('strategy', {}).get('mixed_precision', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # 梯度累积
        self.gradient_accumulation_steps = self.train_config.get('strategy', {}).get('gradient_accumulation_steps', 1)
        self.max_grad_norm = self.train_config.get('strategy', {}).get('max_grad_norm', 1.0)
        
        # 早停
        early_stopping_config = self.train_config.get('strategy', {}).get('early_stopping', {})
        if early_stopping_config.get('enabled', True):
            self.early_stopping = EarlyStopping(
                patience=early_stopping_config.get('patience', 15),
                min_delta=early_stopping_config.get('min_delta', 0.001),
                mode='min'  # 假设监控损失，越小越好
            )
        else:
            self.early_stopping = None
        
        # 日志配置
        self.log_interval = self.train_config.get('logging', {}).get('log_interval', 100)
        self.save_checkpoint_interval = self.train_config.get('logging', {}).get('save_checkpoint_interval', 10)
        
        self.logger.info(f"Trainer initialized with device: {self.device}")
        self.logger.info(f"Mixed precision: {self.use_amp}")
        self.logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        if getattr(self, '_save_dir_coerced', False):
            self.logger.info(
                "训练检查点目录已从 save_dir=%r 规范到 %s（产物应在 runs/ 下，见 docs/02_使用.md）",
                _raw_save,
                self.save_dir,
            )
    
    def _setup_device(self) -> torch.device:
        """设置训练设备"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            self.logger.info("Using CPU")
        
        return device
    
    def _setup_logging(self):
        """设置日志"""
        log_file = self.save_dir / 'training.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('SpeedVQA-Trainer')
    
    def _create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """创建优化器"""
        optimizer_config = self.train_config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'adamw').lower()
        
        lr = optimizer_config.get('lr', 0.001)
        weight_decay = optimizer_config.get('weight_decay', 0.0005)
        
        if optimizer_type == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=optimizer_config.get('betas', [0.9, 0.999]),
                eps=optimizer_config.get('eps', 1e-8)
            )
        elif optimizer_type == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=optimizer_config.get('betas', [0.9, 0.999]),
                eps=optimizer_config.get('eps', 1e-8)
            )
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=optimizer_config.get('momentum', 0.9)
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        self.logger.info(f"Created {optimizer_type.upper()} optimizer with lr={lr}")
        return optimizer
    
    def _create_scheduler(self, optimizer: optim.Optimizer, num_training_steps: int):
        """创建学习率调度器"""
        scheduler_config = self.train_config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'cosine').lower()
        
        if scheduler_type == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=num_training_steps,
                eta_min=scheduler_config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'step':
            from torch.optim.lr_scheduler import StepLR
            scheduler = StepLR(
                optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_type == 'plateau':
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 10),
                min_lr=scheduler_config.get('min_lr', 1e-6)
            )
        else:
            scheduler = None
        
        if scheduler:
            self.logger.info(f"Created {scheduler_type.upper()} scheduler")
        
        return scheduler
    
    def _create_loss_function(self) -> nn.Module:
        """创建损失函数"""
        loss_config = self.config.get('model', {}).get('loss', {})
        loss_type = loss_config.get('type', 'cross_entropy').lower()
        
        if loss_type == 'cross_entropy':
            weight = loss_config.get('weight', None)
            if weight:
                weight = torch.tensor(weight, dtype=torch.float32, device=self.device)
            
            loss_fn = nn.CrossEntropyLoss(
                weight=weight,
                label_smoothing=loss_config.get('label_smoothing', 0.0)
            )
        elif loss_type == 'focal_loss':
            # 简单的Focal Loss实现
            class FocalLoss(nn.Module):
                def __init__(self, alpha=1, gamma=2):
                    super().__init__()
                    self.alpha = alpha
                    self.gamma = gamma
                    self.ce_loss = nn.CrossEntropyLoss(reduction='none')
                
                def forward(self, inputs, targets):
                    ce_loss = self.ce_loss(inputs, targets)
                    pt = torch.exp(-ce_loss)
                    focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                    return focal_loss.mean()
            
            loss_fn = FocalLoss(
                alpha=loss_config.get('alpha', 1),
                gamma=loss_config.get('gamma', 2)
            )
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        self.logger.info(f"Created {loss_type.upper()} loss function")
        return loss_fn
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   optimizer: optim.Optimizer, loss_fn: nn.Module, 
                   scheduler=None) -> Dict[str, float]:
        """训练一个epoch"""
        model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # 移动数据到设备
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # 前向传播
            with autocast(enabled=self.use_amp):
                outputs = model(batch)
                loss = loss_fn(outputs['logits'], batch['label'])
                
                # 梯度累积
                loss = loss / self.gradient_accumulation_steps
            
            # 反向传播
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 更新参数
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    # 梯度裁剪
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                    
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                    optimizer.step()
                
                optimizer.zero_grad()
                
                if scheduler and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step()
                
                self.global_step += 1
            
            # 记录损失
            total_loss += loss.item() * self.gradient_accumulation_steps
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # 日志记录
            if self.global_step % self.log_interval == 0:
                self.logger.info(
                    f'Step {self.global_step}: loss={loss.item() * self.gradient_accumulation_steps:.4f}, '
                    f'lr={optimizer.param_groups[0]["lr"]:.2e}'
                )
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return {'train_loss': avg_loss}
    
    def validate(self, model: nn.Module, val_loader: DataLoader, 
                loss_fn: nn.Module) -> Dict[str, float]:
        """验证模型"""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                # 移动数据到设备
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # 前向传播
                outputs = model(batch)
                loss = loss_fn(outputs['logits'], batch['label'])
                
                # 计算指标
                predictions = torch.argmax(outputs['logits'], dim=1)
                correct += (predictions == batch['label']).sum().item()
                total += batch['label'].size(0)
                
                total_loss += loss.item()
                
                # 收集预测和标签用于详细指标计算
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch['label'].cpu().numpy())
        
        # 计算指标
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        # 计算详细指标
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='weighted', zero_division=0
        )
        
        metrics = {
            'val_loss': avg_loss,
            'val_accuracy': accuracy,
            'val_precision': precision,
            'val_recall': recall,
            'val_f1': f1
        }
        
        self.val_metrics.append(metrics)
        
        self.logger.info(
            f'Validation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, '
            f'F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}'
        )
        
        return metrics
    
    def save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, 
                       scheduler, metrics: Dict[str, float], is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics
        }
        
        # 保存最新检查点
        checkpoint_path = self.save_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳检查点
        if is_best:
            best_checkpoint_path = self.save_dir / 'best_checkpoint.pth'
            torch.save(checkpoint, best_checkpoint_path)
            self.logger.info(f'Saved best checkpoint with metric: {metrics}')
        
        # 保存周期性检查点
        if self.current_epoch % self.save_checkpoint_interval == 0:
            epoch_checkpoint_path = self.save_dir / f'checkpoint_epoch_{self.current_epoch}.pth'
            torch.save(checkpoint, epoch_checkpoint_path)
        
        self.logger.info(f'Saved checkpoint at epoch {self.current_epoch}')
    
    def load_checkpoint(self, checkpoint_path: str, model: nn.Module, 
                       optimizer: optim.Optimizer = None, scheduler=None) -> Dict[str, Any]:
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_metrics = checkpoint.get('val_metrics', [])
        
        self.logger.info(f'Loaded checkpoint from epoch {self.current_epoch}')
        
        return checkpoint
    
    def train(self, model: nn.Module, train_loader: DataLoader, 
             val_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """完整训练流程"""
        self.logger.info("Starting training...")
        
        # 移动模型到设备
        model = model.to(self.device)
        
        # 创建优化器、调度器和损失函数
        optimizer = self._create_optimizer(model)
        
        num_epochs = self.train_config.get('epochs', 100)
        num_training_steps = len(train_loader) * num_epochs // self.gradient_accumulation_steps
        scheduler = self._create_scheduler(optimizer, num_training_steps)
        
        loss_fn = self._create_loss_function()
        
        # 恢复训练（如果指定）
        resume_path = self.train_config.get('resume')
        if resume_path and os.path.exists(resume_path):
            self.load_checkpoint(resume_path, model, optimizer, scheduler)
        
        # 训练循环
        start_time = time.time()
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # 训练
            train_metrics = self.train_epoch(model, train_loader, optimizer, loss_fn, scheduler)
            
            # 验证
            val_metrics = {}
            if val_loader:
                val_metrics = self.validate(model, val_loader, loss_fn)
                
                # 学习率调度（对于ReduceLROnPlateau）
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics['val_loss'])
            
            # 合并指标
            epoch_metrics = {**train_metrics, **val_metrics}
            
            # 检查是否为最佳模型
            is_best = False
            if val_loader:
                current_metric = val_metrics.get('val_loss', float('inf'))
                if self.best_metric is None or current_metric < self.best_metric:
                    self.best_metric = current_metric
                    is_best = True
            
            # 保存检查点
            self.save_checkpoint(model, optimizer, scheduler, epoch_metrics, is_best)
            
            # 早停检查
            if self.early_stopping and val_loader:
                if self.early_stopping(val_metrics['val_loss']):
                    self.logger.info(f'Early stopping triggered at epoch {epoch}')
                    break
            
            # 记录epoch总结
            self.logger.info(
                f'Epoch {epoch} completed - Train Loss: {train_metrics["train_loss"]:.4f}'
                + (f', Val Loss: {val_metrics["val_loss"]:.4f}' if val_loader else '')
            )
        
        total_time = time.time() - start_time
        self.logger.info(f'Training completed in {total_time:.2f} seconds')
        
        return {
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics,
            'best_metric': self.best_metric,
            'total_epochs': self.current_epoch,
            'total_time': total_time
        }


def train(config: Dict[str, Any], model: nn.Module, train_loader: DataLoader, 
         val_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
    """训练函数（简化接口）"""
    trainer = ConfigurableTrainer(config)
    return trainer.train(model, train_loader, val_loader)


if __name__ == '__main__':
    # 测试训练器
    print("Testing ConfigurableTrainer...")
    
    # 创建测试配置
    test_config = {
        'train': {
            'epochs': 2,
            'save_dir': './test_runs',
            'optimizer': {
                'type': 'adamw',
                'lr': 0.001,
                'weight_decay': 0.0005
            },
            'scheduler': {
                'type': 'cosine',
                'min_lr': 1e-6
            },
            'strategy': {
                'mixed_precision': False,  # 关闭混合精度以避免CUDA依赖
                'gradient_accumulation_steps': 1,
                'max_grad_norm': 1.0,
                'early_stopping': {
                    'enabled': True,
                    'patience': 5
                }
            },
            'logging': {
                'log_interval': 10
            }
        },
        'model': {
            'loss': {
                'type': 'cross_entropy'
            }
        }
    }
    
    trainer = ConfigurableTrainer(test_config)
    print("✓ ConfigurableTrainer created successfully")
    print(f"✓ Device: {trainer.device}")
    print(f"✓ Mixed precision: {trainer.use_amp}")
    print(f"✓ Gradient accumulation steps: {trainer.gradient_accumulation_steps}")
    print("✓ ConfigurableTrainer test completed!")