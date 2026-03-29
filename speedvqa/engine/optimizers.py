"""
SpeedVQA优化器和调度器构建系统

支持AdamW/Adam/SGD优化器和Cosine/Step/Plateau学习率调度，
实现配置驱动的构建函数。
"""

import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, 
    StepLR, 
    ReduceLROnPlateau,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    MultiStepLR
)
from typing import Dict, Any, Optional


class OptimizerFactory:
    """优化器工厂"""
    
    SUPPORTED_OPTIMIZERS = {
        'adamw': optim.AdamW,
        'adam': optim.Adam,
        'sgd': optim.SGD,
        'rmsprop': optim.RMSprop,
        'adagrad': optim.Adagrad
    }
    
    @classmethod
    def create_optimizer(cls, model_parameters, config: Dict[str, Any]) -> optim.Optimizer:
        """
        创建优化器
        
        Args:
            model_parameters: 模型参数
            config: 优化器配置
            
        Returns:
            optimizer: 优化器实例
        """
        optimizer_type = config.get('type', 'adamw').lower()
        
        if optimizer_type not in cls.SUPPORTED_OPTIMIZERS:
            raise ValueError(
                f"Unsupported optimizer type: {optimizer_type}. "
                f"Supported: {list(cls.SUPPORTED_OPTIMIZERS.keys())}"
            )
        
        optimizer_class = cls.SUPPORTED_OPTIMIZERS[optimizer_type]
        
        # 基本参数
        lr = config.get('lr', 0.001)
        weight_decay = config.get('weight_decay', 0.0)
        
        # 根据优化器类型设置特定参数
        if optimizer_type in ['adamw', 'adam']:
            optimizer = optimizer_class(
                model_parameters,
                lr=lr,
                weight_decay=weight_decay,
                betas=config.get('betas', [0.9, 0.999]),
                eps=config.get('eps', 1e-8),
                amsgrad=config.get('amsgrad', False)
            )
        elif optimizer_type == 'sgd':
            optimizer = optimizer_class(
                model_parameters,
                lr=lr,
                weight_decay=weight_decay,
                momentum=config.get('momentum', 0.9),
                dampening=config.get('dampening', 0),
                nesterov=config.get('nesterov', False)
            )
        elif optimizer_type == 'rmsprop':
            optimizer = optimizer_class(
                model_parameters,
                lr=lr,
                weight_decay=weight_decay,
                alpha=config.get('alpha', 0.99),
                eps=config.get('eps', 1e-8),
                momentum=config.get('momentum', 0),
                centered=config.get('centered', False)
            )
        elif optimizer_type == 'adagrad':
            optimizer = optimizer_class(
                model_parameters,
                lr=lr,
                weight_decay=weight_decay,
                lr_decay=config.get('lr_decay', 0),
                eps=config.get('eps', 1e-10)
            )
        
        return optimizer
    
    @classmethod
    def get_optimizer_info(cls, optimizer: optim.Optimizer) -> Dict[str, Any]:
        """获取优化器信息"""
        return {
            'type': optimizer.__class__.__name__,
            'param_groups': len(optimizer.param_groups),
            'lr': optimizer.param_groups[0]['lr'],
            'weight_decay': optimizer.param_groups[0].get('weight_decay', 0),
            'total_params': sum(len(group['params']) for group in optimizer.param_groups)
        }


class SchedulerFactory:
    """学习率调度器工厂"""
    
    SUPPORTED_SCHEDULERS = {
        'cosine': CosineAnnealingLR,
        'step': StepLR,
        'plateau': ReduceLROnPlateau,
        'cosine_warm_restarts': CosineAnnealingWarmRestarts,
        'exponential': ExponentialLR,
        'multistep': MultiStepLR
    }
    
    @classmethod
    def create_scheduler(cls, optimizer: optim.Optimizer, config: Dict[str, Any], 
                        num_training_steps: Optional[int] = None) -> Optional[object]:
        """
        创建学习率调度器
        
        Args:
            optimizer: 优化器实例
            config: 调度器配置
            num_training_steps: 总训练步数（某些调度器需要）
            
        Returns:
            scheduler: 调度器实例或None
        """
        scheduler_type = config.get('type', 'cosine').lower()
        
        if scheduler_type == 'none' or not config.get('enabled', True):
            return None
        
        if scheduler_type not in cls.SUPPORTED_SCHEDULERS:
            raise ValueError(
                f"Unsupported scheduler type: {scheduler_type}. "
                f"Supported: {list(cls.SUPPORTED_SCHEDULERS.keys())}"
            )
        
        scheduler_class = cls.SUPPORTED_SCHEDULERS[scheduler_type]
        
        # 根据调度器类型创建
        if scheduler_type == 'cosine':
            T_max = config.get('T_max', num_training_steps or 100)
            eta_min = config.get('min_lr', 1e-6)
            scheduler = scheduler_class(optimizer, T_max=T_max, eta_min=eta_min)
            
        elif scheduler_type == 'step':
            step_size = config.get('step_size', 30)
            gamma = config.get('gamma', 0.1)
            scheduler = scheduler_class(optimizer, step_size=step_size, gamma=gamma)
            
        elif scheduler_type == 'plateau':
            mode = config.get('mode', 'min')
            factor = config.get('factor', 0.5)
            patience = config.get('patience', 10)
            min_lr = config.get('min_lr', 1e-6)
            threshold = config.get('threshold', 1e-4)
            scheduler = scheduler_class(
                optimizer, mode=mode, factor=factor, patience=patience,
                min_lr=min_lr, threshold=threshold
            )
            
        elif scheduler_type == 'cosine_warm_restarts':
            T_0 = config.get('T_0', 10)
            T_mult = config.get('T_mult', 1)
            eta_min = config.get('min_lr', 1e-6)
            scheduler = scheduler_class(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)
            
        elif scheduler_type == 'exponential':
            gamma = config.get('gamma', 0.95)
            scheduler = scheduler_class(optimizer, gamma=gamma)
            
        elif scheduler_type == 'multistep':
            milestones = config.get('milestones', [30, 60, 90])
            gamma = config.get('gamma', 0.1)
            scheduler = scheduler_class(optimizer, milestones=milestones, gamma=gamma)
        
        return scheduler
    
    @classmethod
    def create_warmup_scheduler(cls, optimizer: optim.Optimizer, warmup_config: Dict[str, Any],
                               main_scheduler: Optional[object] = None) -> Optional[object]:
        """
        创建预热调度器
        
        Args:
            optimizer: 优化器
            warmup_config: 预热配置
            main_scheduler: 主调度器
            
        Returns:
            warmup_scheduler: 预热调度器
        """
        if not warmup_config.get('enabled', False):
            return main_scheduler
        
        warmup_epochs = warmup_config.get('epochs', 5)
        warmup_lr = warmup_config.get('start_lr', 1e-6)
        
        # 简单的线性预热实现
        class LinearWarmupScheduler:
            def __init__(self, optimizer, warmup_epochs, warmup_lr, main_scheduler=None):
                self.optimizer = optimizer
                self.warmup_epochs = warmup_epochs
                self.warmup_lr = warmup_lr
                self.main_scheduler = main_scheduler
                self.current_epoch = 0
                self.base_lrs = [group['lr'] for group in optimizer.param_groups]
            
            def step(self, epoch=None):
                if epoch is not None:
                    self.current_epoch = epoch
                else:
                    self.current_epoch += 1
                
                if self.current_epoch < self.warmup_epochs:
                    # 预热阶段：线性增长
                    lr_scale = (self.current_epoch + 1) / self.warmup_epochs
                    for i, param_group in enumerate(self.optimizer.param_groups):
                        param_group['lr'] = self.warmup_lr + (self.base_lrs[i] - self.warmup_lr) * lr_scale
                elif self.main_scheduler:
                    # 预热结束，使用主调度器
                    self.main_scheduler.step()
            
            def state_dict(self):
                state = {
                    'current_epoch': self.current_epoch,
                    'base_lrs': self.base_lrs
                }
                if self.main_scheduler:
                    state['main_scheduler'] = self.main_scheduler.state_dict()
                return state
            
            def load_state_dict(self, state_dict):
                self.current_epoch = state_dict['current_epoch']
                self.base_lrs = state_dict['base_lrs']
                if self.main_scheduler and 'main_scheduler' in state_dict:
                    self.main_scheduler.load_state_dict(state_dict['main_scheduler'])
        
        return LinearWarmupScheduler(optimizer, warmup_epochs, warmup_lr, main_scheduler)
    
    @classmethod
    def get_scheduler_info(cls, scheduler) -> Dict[str, Any]:
        """获取调度器信息"""
        if scheduler is None:
            return {'type': 'None', 'enabled': False}
        
        info = {
            'type': scheduler.__class__.__name__,
            'enabled': True
        }
        
        # 添加特定调度器的信息
        if hasattr(scheduler, 'T_max'):
            info['T_max'] = scheduler.T_max
        if hasattr(scheduler, 'eta_min'):
            info['eta_min'] = scheduler.eta_min
        if hasattr(scheduler, 'step_size'):
            info['step_size'] = scheduler.step_size
        if hasattr(scheduler, 'gamma'):
            info['gamma'] = scheduler.gamma
        
        return info


class OptimizationBuilder:
    """优化组件构建器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.optimizer_config = config.get('optimizer', {})
        self.scheduler_config = config.get('scheduler', {})
        self.warmup_config = config.get('warmup', {})
    
    def build_optimizer(self, model_parameters) -> optim.Optimizer:
        """构建优化器"""
        return OptimizerFactory.create_optimizer(model_parameters, self.optimizer_config)
    
    def build_scheduler(self, optimizer: optim.Optimizer, 
                       num_training_steps: Optional[int] = None) -> Optional[object]:
        """构建调度器"""
        # 创建主调度器
        main_scheduler = SchedulerFactory.create_scheduler(
            optimizer, self.scheduler_config, num_training_steps
        )
        
        # 如果启用预热，创建预热调度器
        if self.warmup_config.get('enabled', False):
            return SchedulerFactory.create_warmup_scheduler(
                optimizer, self.warmup_config, main_scheduler
            )
        
        return main_scheduler
    
    def build_optimization_components(self, model_parameters, 
                                    num_training_steps: Optional[int] = None) -> Dict[str, Any]:
        """构建完整的优化组件"""
        optimizer = self.build_optimizer(model_parameters)
        scheduler = self.build_scheduler(optimizer, num_training_steps)
        
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
            'optimizer_info': OptimizerFactory.get_optimizer_info(optimizer),
            'scheduler_info': SchedulerFactory.get_scheduler_info(scheduler)
        }
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """获取优化配置摘要"""
        return {
            'optimizer': {
                'type': self.optimizer_config.get('type', 'adamw'),
                'lr': self.optimizer_config.get('lr', 0.001),
                'weight_decay': self.optimizer_config.get('weight_decay', 0.0)
            },
            'scheduler': {
                'type': self.scheduler_config.get('type', 'cosine'),
                'enabled': self.scheduler_config.get('enabled', True)
            },
            'warmup': {
                'enabled': self.warmup_config.get('enabled', False),
                'epochs': self.warmup_config.get('epochs', 5)
            }
        }


def build_optimizer(model_parameters, config: Dict[str, Any]) -> optim.Optimizer:
    """构建优化器（简化接口）"""
    return OptimizerFactory.create_optimizer(model_parameters, config)


def build_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any], 
                   num_training_steps: Optional[int] = None) -> Optional[object]:
    """构建调度器（简化接口）"""
    return SchedulerFactory.create_scheduler(optimizer, config, num_training_steps)


def build_optimization_components(model_parameters, config: Dict[str, Any],
                                num_training_steps: Optional[int] = None) -> Dict[str, Any]:
    """构建完整优化组件（简化接口）"""
    builder = OptimizationBuilder(config)
    return builder.build_optimization_components(model_parameters, num_training_steps)


if __name__ == '__main__':
    # 测试优化器和调度器构建
    print("=== SpeedVQA Optimization Components Test ===")
    
    # 创建测试模型
    import torch.nn as nn
    test_model = nn.Linear(100, 2)
    
    # 测试不同优化器
    optimizer_configs = [
        {'type': 'adamw', 'lr': 0.001, 'weight_decay': 0.0005},
        {'type': 'adam', 'lr': 0.001, 'betas': [0.9, 0.999]},
        {'type': 'sgd', 'lr': 0.01, 'momentum': 0.9, 'weight_decay': 0.0001}
    ]
    
    for config in optimizer_configs:
        try:
            optimizer = build_optimizer(test_model.parameters(), config)
            info = OptimizerFactory.get_optimizer_info(optimizer)
            print(f"✓ {config['type'].upper()} optimizer: {info}")
        except Exception as e:
            print(f"✗ {config['type'].upper()} optimizer failed: {e}")
    
    # 测试不同调度器
    scheduler_configs = [
        {'type': 'cosine', 'min_lr': 1e-6},
        {'type': 'step', 'step_size': 30, 'gamma': 0.1},
        {'type': 'plateau', 'patience': 10, 'factor': 0.5}
    ]
    
    optimizer = build_optimizer(test_model.parameters(), {'type': 'adamw', 'lr': 0.001})
    
    for config in scheduler_configs:
        try:
            scheduler = build_scheduler(optimizer, config, num_training_steps=100)
            info = SchedulerFactory.get_scheduler_info(scheduler)
            print(f"✓ {config['type'].upper()} scheduler: {info}")
        except Exception as e:
            print(f"✗ {config['type'].upper()} scheduler failed: {e}")
    
    # 测试完整构建
    print("\nTesting complete optimization components:")
    full_config = {
        'optimizer': {'type': 'adamw', 'lr': 0.001, 'weight_decay': 0.0005},
        'scheduler': {'type': 'cosine', 'min_lr': 1e-6},
        'warmup': {'enabled': True, 'epochs': 5, 'start_lr': 1e-6}
    }
    
    try:
        components = build_optimization_components(test_model.parameters(), full_config, 1000)
        print("✓ Complete components built successfully")
        print(f"  Optimizer: {components['optimizer_info']['type']}")
        print(f"  Scheduler: {components['scheduler_info']['type']}")
    except Exception as e:
        print(f"✗ Complete components failed: {e}")
    
    print("\n✓ Optimization components test completed!")