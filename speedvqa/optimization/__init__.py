"""
Optimization module for SpeedVQA.

Provides TensorRT optimization, quantization, and performance tuning capabilities.
"""

from .tensorrt_optimizer import TensorRTOptimizer

__all__ = ['TensorRTOptimizer']
