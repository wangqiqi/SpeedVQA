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

公开 API（本模块 ``__all__``）:
    顶层 ``import speedvqa`` 仅导出 **数据、校验、配置、模型工厂**，便于轻量依赖与快速搭数据集/建模型。

推荐按子模块导入（训练 / 导出 / 推理）:
    - 训练: ``speedvqa.engine`` — 如 ``from speedvqa.engine import train, ConfigurableTrainer``
    - 超参: ``speedvqa.engine.hyperparameter_optimizer`` 等
    - 导出: ``speedvqa.export`` — 如 ``ModelExporter``（见 ``speedvqa/export/README.md``）
    - 推理: ``speedvqa.inference`` — 如 ``ROIInferencer``（见 ``speedvqa/inference/README.md``）
    - 一键 CLI: ``python -m speedvqa.scripts.onekey_train``（及 predict / export）

可运行示例与 pytest 用例位于仓库内 ``speedvqa/examples/``、``speedvqa/tests/``（发布 wheel 中不包含二者，见根目录 ``pyproject.toml``）。

使用示例:
    from speedvqa import load_config, build_model
    cfg = load_config()  # 默认 speedvqa/configs/default.yaml
    model = build_model(cfg)
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