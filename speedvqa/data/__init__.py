"""
SpeedVQA数据处理组件

包含数据加载、预处理、验证等功能，完全支持X-AnyLabeling格式。
"""

from .datasets import VQADataset, build_dataset, split_dataset
from .transforms import build_transforms
from .validators import DataValidator, XAnyLabelingAdapter

__all__ = [
    'VQADataset',
    'build_dataset', 
    'split_dataset',
    'build_transforms',
    'DataValidator',
    'XAnyLabelingAdapter'
]