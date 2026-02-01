"""
SpeedVQA训练和推理引擎

包含训练器、验证器、推理器等核心执行组件。
"""

from .trainer import train, ConfigurableTrainer
from .optimizers import (
    OptimizerFactory,
    SchedulerFactory,
    OptimizationBuilder,
    build_optimizer,
    build_scheduler,
    build_optimization_components
)

__all__ = [
    'train',
    'ConfigurableTrainer',
    'OptimizerFactory',
    'SchedulerFactory',
    'OptimizationBuilder',
    'build_optimizer',
    'build_scheduler',
    'build_optimization_components'
]