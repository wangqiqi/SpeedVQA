"""
ROI推理模块

支持PyTorch/ONNX/TensorRT模型加载和推理
"""

from .inferencer import ROIInferencer

__all__ = ['ROIInferencer']
