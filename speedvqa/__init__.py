"""
SpeedVQA: 极速视觉问答系统

一个专为YES/NO问答任务设计的完整解决方案，采用模块化全栈架构，
专注于T4显卡上的毫秒级推理性能。

主要特性:
- 完全支持X-AnyLabeling标注格式
- MobileNetV3 + DistilBERT + MLP轻量化架构
- 完全可配置的YAML驱动系统
- TensorRT优化，T4显卡<50ms推理
- 模块化设计，易于扩展

使用示例:
    from speedvqa.utils.config import load_config
    cfg = load_config()  # 默认 speedvqa/configs/default.yaml
"""

__version__ = '1.0.0'
__author__ = 'SpeedVQA Team'

# 导入当前可用的接口
from .data.datasets import VQADataset, build_dataset, split_dataset
from .data.validators import DataValidator, XAnyLabelingAdapter
from .utils.config import load_config, ConfigManager
from .models.factory import build_model, build_model_from_preset, get_model_presets

__all__ = [
    'VQADataset',
    'build_dataset',
    'split_dataset',
    'DataValidator',
    'XAnyLabelingAdapter',
    'load_config',
    'ConfigManager',
    'build_model',
    'build_model_from_preset',
    'get_model_presets'
]