"""
属性测试 4: 训练监控完整性
验证需求: 需求 2.2

使用Hypothesis进行属性测试，验证训练监控系统的完整性和正确性。
"""

import pytest
import tempfile
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any
from hypothesis import given, strategies as st, settings
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, invariant

from speedvqa.engine.trainer import ConfigurableTrainer, EarlyStopping


class MockModel(nn.Module):
    """用于测试的简单模型"""
    
    def __init__(self, input_dim: int = 100, num_classes: int = 2):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)
        self.config = {'name': 'mock_model'}
    
    def forward(self, batch):
        # 模拟SpeedVQA模型的输出格式
        batch_size = batch['label'].size(0)
        dev = batch['label'].device
        fake_features = torch.randn(batch_size, 100, device=dev)
        logits = self.classifier(fake_features)
        
        return {
            'logits': logits,
            'vision_features': fake_features[:, :50],
            'text_features': fake_features[:, 50:],
            'fused_features': fake_features
        }
    
    def get_model_info(self):
        return {
            'model_name': 'mock_model',
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


class MockDataLoader:
    """用于测试的简单数据加载器"""
    
    def __init__(self, num_batches: int = 5, batch_size: int = 4):
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.current_batch = 0
    
    def __iter__(self):
        self.current_batch = 0
        return self
    
    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration
        
        batch = {
            'image': torch.randn(self.batch_size, 3, 224, 224),
            'input_ids': torch.randint(0, 1000, (self.batch_size, 128)),
            'attention_mask': torch.ones(self.batch_size, 128),
            'label': torch.randint(0, 2, (self.batch_size,)),
            'question': ['测试问题'] * self.batch_size,
            'answer': ['YES'] * self.batch_size
        }
        
        self.current_batch += 1
        return batch
    
    def __len__(self):
        return self.num_batches


class TestTrainingMonitoringProperties:
    """属性测试 4: 训练监控完整性"""
    
    def setup_method(self):
        """测试设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.save_dir = Path(self.temp_dir) / 'test_training'
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def teardown_method(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_config(self, **overrides) -> Dict[str, Any]:
        """创建测试配置"""
        config = {
            'train': {
                'epochs': 3,
                'save_dir': str(self.save_dir),
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
                    'mixed_precision': False,  # 避免CUDA依赖
                    'gradient_accumulation_steps': 1,
                    'max_grad_norm': 1.0,
                    'early_stopping': {
                        'enabled': True,
                        'patience': 5,
                        'min_delta': 0.001
                    }
                },
                'logging': {
                    'log_interval': 2,
                    'save_checkpoint_interval': 2
                }
            },
            'model': {
                'loss': {
                    'type': 'cross_entropy',
                    'label_smoothing': 0.0
                }
            }
        }
        
        # 应用覆盖参数
        if overrides:
            from speedvqa.utils.config import ConfigManager
            config_manager = ConfigManager()
            config_manager.config = config
            config_manager.update_config(overrides)
            config = config_manager.config
        
        return config
    
    @given(
        epochs=st.integers(min_value=1, max_value=5),
        log_interval=st.integers(min_value=1, max_value=3),
        patience=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=10, deadline=10000)
    def test_training_monitoring_consistency(self, epochs, log_interval, patience):
        """
        属性 4: 训练监控完整性
        验证需求: 需求 2.2
        
        测试属性:
        1. 训练损失应该被正确记录和存储
        2. 验证指标应该在每个epoch后更新
        3. 检查点保存应该包含完整的训练状态
        4. 早停机制应该在满足条件时正确触发
        """
        config = self.create_test_config(
            train={
                'epochs': epochs,
                'logging': {
                    'log_interval': log_interval,
                    'save_checkpoint_interval': 1
                },
                'strategy': {
                    'early_stopping': {
                        'enabled': True,
                        'patience': patience,
                        'min_delta': 0.001
                    }
                }
            }
        )
        
        trainer = ConfigurableTrainer(config)
        model = MockModel()
        train_loader = MockDataLoader(num_batches=3, batch_size=2)
        val_loader = MockDataLoader(num_batches=2, batch_size=2)
        
        # 执行训练
        results = trainer.train(model, train_loader, val_loader)
        
        # 属性1: 训练损失记录完整性
        assert len(trainer.train_losses) > 0, "训练损失应该被记录"
        assert all(isinstance(loss, (int, float)) for loss in trainer.train_losses), "损失值应该是数值类型"
        assert all(loss >= 0 for loss in trainer.train_losses), "损失值应该非负"
        
        # 属性2: 验证指标更新完整性
        assert len(trainer.val_metrics) > 0, "验证指标应该被记录"
        for metrics in trainer.val_metrics:
            required_keys = ['val_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_f1']
            for key in required_keys:
                assert key in metrics, f"验证指标应该包含 {key}"
                assert isinstance(metrics[key], (int, float)), f"{key} 应该是数值类型"
        
        # 属性3: 检查点保存完整性
        checkpoint_files = list(self.save_dir.glob('*.pth'))
        assert len(checkpoint_files) > 0, "应该保存检查点文件"
        
        # 验证最新检查点的完整性
        latest_checkpoint = self.save_dir / 'latest_checkpoint.pth'
        if latest_checkpoint.exists():
            checkpoint = torch.load(latest_checkpoint, map_location='cpu')
            required_checkpoint_keys = [
                'epoch', 'global_step', 'model_state_dict', 
                'optimizer_state_dict', 'metrics', 'config'
            ]
            for key in required_checkpoint_keys:
                assert key in checkpoint, f"检查点应该包含 {key}"
        
        # 属性4: 训练结果一致性
        assert 'train_losses' in results, "结果应该包含训练损失"
        assert 'val_metrics' in results, "结果应该包含验证指标"
        assert results['train_losses'] == trainer.train_losses, "返回的训练损失应该与trainer一致"
        assert results['val_metrics'] == trainer.val_metrics, "返回的验证指标应该与trainer一致"
    
    @given(
        patience=st.integers(min_value=1, max_value=5),
        min_delta=st.floats(min_value=0.0001, max_value=0.01)
    )
    @settings(max_examples=8, deadline=5000)
    def test_early_stopping_mechanism(self, patience, min_delta):
        """
        属性测试: 早停机制正确性
        验证早停机制在不同参数下的行为一致性
        """
        early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, mode='min')
        
        # 模拟损失序列：先下降后上升
        loss_sequence = [1.0, 0.8, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
        
        should_stop = False
        stop_epoch = None
        
        for epoch, loss in enumerate(loss_sequence):
            if early_stopping(loss):
                should_stop = True
                stop_epoch = epoch
                break
        
        # 验证早停行为
        if should_stop:
            assert stop_epoch is not None, "如果触发早停，应该记录停止的epoch"
            assert stop_epoch >= patience, f"早停应该在至少 {patience} 个epoch后触发"
        
        # 验证最佳分数记录
        assert early_stopping.best_score is not None, "应该记录最佳分数"
        assert early_stopping.best_score <= min(loss_sequence[:stop_epoch+1] if stop_epoch else loss_sequence), "最佳分数应该是历史最小值"
    
    @given(
        optimizer_type=st.sampled_from(['adamw', 'adam', 'sgd']),
        lr=st.floats(min_value=1e-5, max_value=1e-2),
        weight_decay=st.floats(min_value=0.0, max_value=1e-3)
    )
    @settings(max_examples=6, deadline=8000)
    def test_optimizer_creation_consistency(self, optimizer_type, lr, weight_decay):
        """
        属性测试: 优化器创建一致性
        验证不同优化器配置的创建和参数设置
        """
        config = self.create_test_config(
            train={
                'optimizer': {
                    'type': optimizer_type,
                    'lr': lr,
                    'weight_decay': weight_decay
                }
            }
        )
        
        trainer = ConfigurableTrainer(config)
        model = MockModel()
        
        optimizer = trainer._create_optimizer(model)
        
        # 验证优化器类型
        expected_types = {
            'adamw': torch.optim.AdamW,
            'adam': torch.optim.Adam,
            'sgd': torch.optim.SGD
        }
        assert isinstance(optimizer, expected_types[optimizer_type]), f"优化器类型应该是 {expected_types[optimizer_type]}"
        
        # 验证学习率设置
        assert abs(optimizer.param_groups[0]['lr'] - lr) < 1e-8, "学习率应该正确设置"
        
        # 验证权重衰减设置
        assert abs(optimizer.param_groups[0]['weight_decay'] - weight_decay) < 1e-8, "权重衰减应该正确设置"
    
    @given(
        scheduler_type=st.sampled_from(['cosine', 'step', 'plateau']),
        num_steps=st.integers(min_value=10, max_value=100)
    )
    @settings(max_examples=6, deadline=6000)
    def test_scheduler_creation_consistency(self, scheduler_type, num_steps):
        """
        属性测试: 调度器创建一致性
        验证不同调度器配置的创建和行为
        """
        config = self.create_test_config(
            train={
                'scheduler': {
                    'type': scheduler_type,
                    'min_lr': 1e-6
                }
            }
        )
        
        trainer = ConfigurableTrainer(config)
        model = MockModel()
        optimizer = trainer._create_optimizer(model)
        
        scheduler = trainer._create_scheduler(optimizer, num_steps)
        
        if scheduler_type == 'cosine':
            assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        elif scheduler_type == 'step':
            assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
        elif scheduler_type == 'plateau':
            assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    
    def test_loss_function_consistency(self):
        """
        测试损失函数创建的一致性
        """
        loss_types = ['cross_entropy', 'focal_loss']
        
        for loss_type in loss_types:
            config = self.create_test_config(
                model={
                    'loss': {
                        'type': loss_type,
                        'label_smoothing': 0.1 if loss_type == 'cross_entropy' else 0.0
                    }
                }
            )
            
            trainer = ConfigurableTrainer(config)
            loss_fn = trainer._create_loss_function()
            
            # 测试损失函数
            batch_size = 4
            num_classes = 2
            logits = torch.randn(batch_size, num_classes)
            targets = torch.randint(0, num_classes, (batch_size,))
            
            loss = loss_fn(logits, targets)
            
            assert isinstance(loss, torch.Tensor), "损失应该是tensor"
            assert loss.dim() == 0, "损失应该是标量"
            assert loss.item() >= 0, "损失值应该非负"
    
    def test_checkpoint_save_load_consistency(self):
        """
        测试检查点保存和加载的一致性
        """
        config = self.create_test_config()
        trainer = ConfigurableTrainer(config)
        model = MockModel()
        optimizer = trainer._create_optimizer(model)
        scheduler = trainer._create_scheduler(optimizer, 100)
        
        # 模拟训练状态
        trainer.current_epoch = 5
        trainer.global_step = 50
        trainer.train_losses = [1.0, 0.8, 0.6]
        trainer.val_metrics = [
            {'val_loss': 0.9, 'val_accuracy': 0.7},
            {'val_loss': 0.7, 'val_accuracy': 0.8}
        ]
        
        test_metrics = {'val_loss': 0.5, 'val_accuracy': 0.9}
        
        # 保存检查点
        trainer.save_checkpoint(model, optimizer, scheduler, test_metrics, is_best=True)
        
        # 验证文件存在
        assert (trainer.save_dir / 'latest_checkpoint.pth').exists(), "应该保存最新检查点"
        assert (trainer.save_dir / 'best_checkpoint.pth').exists(), "应该保存最佳检查点"
        
        # 创建新的trainer和模型来测试加载
        new_trainer = ConfigurableTrainer(config)
        new_model = MockModel()
        new_optimizer = new_trainer._create_optimizer(new_model)
        new_scheduler = new_trainer._create_scheduler(new_optimizer, 100)
        
        # 加载检查点
        new_trainer.load_checkpoint(
            str(trainer.save_dir / 'latest_checkpoint.pth'),
            new_model, new_optimizer, new_scheduler,
        )
        
        # 验证状态恢复
        assert new_trainer.current_epoch == 5, "epoch应该正确恢复"
        assert new_trainer.global_step == 50, "global_step应该正确恢复"
        assert new_trainer.train_losses == [1.0, 0.8, 0.6], "训练损失应该正确恢复"
        assert len(new_trainer.val_metrics) == 2, "验证指标应该正确恢复"


class TrainingMonitoringStateMachine(RuleBasedStateMachine):
    """
    基于状态的属性测试：训练监控状态一致性
    """
    
    def __init__(self):
        super().__init__()
        self.temp_dir = None
        self.trainer = None
        self.model = None
        self.train_losses_count = 0
        self.val_metrics_count = 0
        self.epochs_completed = 0
    
    @initialize()
    def setup(self):
        """初始化测试环境"""
        import tempfile
        self.temp_dir = tempfile.mkdtemp()
        save_dir = Path(self.temp_dir) / 'state_test'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        config = {
            'train': {
                'epochs': 10,
                'save_dir': str(save_dir),
                'optimizer': {'type': 'adamw', 'lr': 0.001},
                'strategy': {'mixed_precision': False},
                'logging': {'log_interval': 1, 'save_checkpoint_interval': 2}
            },
            'model': {'loss': {'type': 'cross_entropy'}}
        }
        
        self.trainer = ConfigurableTrainer(config)
        self.model = MockModel()
    
    @rule()
    def simulate_training_step(self):
        """模拟训练步骤"""
        if self.trainer and self.model:
            # 模拟添加训练损失
            fake_loss = torch.rand(1).item()
            self.trainer.train_losses.append(fake_loss)
            self.train_losses_count += 1
    
    @rule()
    def simulate_validation_step(self):
        """模拟验证步骤"""
        if self.trainer:
            # 模拟添加验证指标
            fake_metrics = {
                'val_loss': torch.rand(1).item(),
                'val_accuracy': torch.rand(1).item(),
                'val_precision': torch.rand(1).item(),
                'val_recall': torch.rand(1).item(),
                'val_f1': torch.rand(1).item()
            }
            self.trainer.val_metrics.append(fake_metrics)
            self.val_metrics_count += 1
    
    @rule()
    def simulate_epoch_completion(self):
        """模拟epoch完成"""
        if self.trainer:
            self.trainer.current_epoch += 1
            self.epochs_completed += 1
    
    @invariant()
    def training_state_consistency(self):
        """不变量：训练状态一致性"""
        if self.trainer:
            # 训练损失数量应该非负
            assert len(self.trainer.train_losses) >= 0
            assert len(self.trainer.train_losses) == self.train_losses_count
            
            # 验证指标数量应该非负
            assert len(self.trainer.val_metrics) >= 0
            assert len(self.trainer.val_metrics) == self.val_metrics_count
            
            # epoch数量应该一致
            assert self.trainer.current_epoch >= 0
            assert self.trainer.current_epoch == self.epochs_completed
            
            # 所有损失值应该是有效数值
            for loss in self.trainer.train_losses:
                assert isinstance(loss, (int, float))
                assert loss >= 0
            
            # 所有验证指标应该是有效数值
            for metrics in self.trainer.val_metrics:
                for key, value in metrics.items():
                    assert isinstance(value, (int, float))
                    if 'accuracy' in key or 'precision' in key or 'recall' in key or 'f1' in key:
                        assert 0 <= value <= 1  # 这些指标应该在0-1之间
    
    def teardown(self):
        """清理"""
        if self.temp_dir:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)


# 运行状态机测试
TestTrainingMonitoringState = TrainingMonitoringStateMachine.TestCase


if __name__ == '__main__':
    # 运行属性测试
    pytest.main([__file__, '-v'])