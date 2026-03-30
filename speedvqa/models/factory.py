"""
SpeedVQA模型工厂

提供模型构建工厂函数，支持动态架构配置和YAML配置文件解析。
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union
from pathlib import Path

from .speedvqa import (
    SpeedVQAModel,
    VisionEncoder,
    TextEncoder,
    MultiModalFusion,
    MLPClassifier
)
from ..utils.config import load_config, ConfigManager


class ModelFactory:
    """模型工厂类"""
    
    # 支持的视觉backbone
    SUPPORTED_VISION_BACKBONES = {
        'mobilenet_v3_small': 'MobileNetV3-Small',
        'mobilenet_v3_large': 'MobileNetV3-Large', 
        'resnet50': 'ResNet50'
    }
    
    # 支持的文本编码器
    SUPPORTED_TEXT_ENCODERS = {
        'distilbert-base-uncased': 'DistilBERT Base Uncased',
        'bert-base-uncased': 'BERT Base Uncased',
        'distilbert-base-multilingual-cased': 'DistilBERT Multilingual'
    }
    
    # 支持的融合方法
    SUPPORTED_FUSION_METHODS = {
        'concat': 'Concatenation Fusion',
        'attention': 'Attention Fusion',
        'bilinear': 'Bilinear Fusion',
        'film': 'FiLM / text-conditioned visual modulation + concat MLP',
        'cross_attn': 'Single-token cross-attention (text Q, vision KV)',
    }
    
    # 支持的激活函数
    SUPPORTED_ACTIVATIONS = {
        'relu': 'ReLU',
        'gelu': 'GELU',
        'swish': 'Swish/SiLU'
    }
    
    @classmethod
    def create_model(cls, config: Union[str, Dict[str, Any]], **kwargs) -> SpeedVQAModel:
        """
        创建SpeedVQA模型
        
        Args:
            config: 配置文件路径或配置字典
            **kwargs: 额外的配置覆盖参数
            
        Returns:
            model: SpeedVQA模型实例
        """
        # 加载配置
        if isinstance(config, str):
            config = load_config(config, **kwargs)
        elif isinstance(config, dict):
            if kwargs:
                # 合并额外参数
                config_manager = ConfigManager()
                config_manager.config = config
                config_manager.update_config(kwargs)
                config = config_manager.config
        else:
            raise ValueError("Config must be a file path or dictionary")
        
        # 验证配置
        cls._validate_config(config)
        
        # 提取模型配置
        model_config = config.get('model', {})
        
        # 创建模型
        model = SpeedVQAModel(model_config)
        
        return model
    
    @classmethod
    def create_vision_encoder(cls, config: Dict[str, Any]) -> VisionEncoder:
        """创建视觉编码器"""
        cls._validate_vision_config(config)
        return VisionEncoder(config)
    
    @classmethod
    def create_text_encoder(cls, config: Dict[str, Any]) -> TextEncoder:
        """创建文本编码器"""
        cls._validate_text_config(config)
        return TextEncoder(config)
    
    @classmethod
    def create_fusion_layer(cls, config: Dict[str, Any]) -> MultiModalFusion:
        """创建融合层"""
        cls._validate_fusion_config(config)
        return MultiModalFusion(config)
    
    @classmethod
    def create_classifier(cls, config: Dict[str, Any]) -> MLPClassifier:
        """创建分类器"""
        cls._validate_classifier_config(config)
        return MLPClassifier(config)
    
    @classmethod
    def _validate_config(cls, config: Dict[str, Any]):
        """验证完整配置"""
        if 'model' not in config:
            raise ValueError("Missing 'model' section in config")
        
        model_config = config['model']
        
        # 验证各组件配置
        if 'vision' in model_config:
            cls._validate_vision_config(model_config['vision'])
        
        if 'text' in model_config:
            cls._validate_text_config(model_config['text'])
        
        if 'fusion' in model_config:
            cls._validate_fusion_config(model_config['fusion'])
        
        if 'classifier' in model_config:
            cls._validate_classifier_config(model_config['classifier'])
    
    @classmethod
    def _validate_vision_config(cls, config: Dict[str, Any]):
        """验证视觉编码器配置"""
        backbone = config.get('backbone', 'mobilenet_v3_small')
        if backbone not in cls.SUPPORTED_VISION_BACKBONES:
            raise ValueError(
                f"Unsupported vision backbone: {backbone}. "
                f"Supported: {list(cls.SUPPORTED_VISION_BACKBONES.keys())}"
            )
        
        feature_dim = config.get('feature_dim', 1024)
        if not isinstance(feature_dim, int) or feature_dim <= 0:
            raise ValueError("feature_dim must be a positive integer")
        
        dropout = config.get('dropout', 0.1)
        if not 0 <= dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")
    
    @classmethod
    def _validate_text_config(cls, config: Dict[str, Any]):
        """验证文本编码器配置"""
        encoder = config.get('encoder', 'distilbert-base-uncased')
        if encoder not in cls.SUPPORTED_TEXT_ENCODERS:
            print(f"Warning: Text encoder '{encoder}' not in supported list. "
                  f"Supported: {list(cls.SUPPORTED_TEXT_ENCODERS.keys())}")
        
        max_length = config.get('max_length', 128)
        if not isinstance(max_length, int) or max_length <= 0:
            raise ValueError("max_length must be a positive integer")
        
        feature_dim = config.get('feature_dim', 768)
        if not isinstance(feature_dim, int) or feature_dim <= 0:
            raise ValueError("feature_dim must be a positive integer")
    
    @classmethod
    def _validate_fusion_config(cls, config: Dict[str, Any]):
        """验证融合层配置"""
        method = config.get('method', 'concat')
        if method not in cls.SUPPORTED_FUSION_METHODS:
            raise ValueError(
                f"Unsupported fusion method: {method}. "
                f"Supported: {list(cls.SUPPORTED_FUSION_METHODS.keys())}"
            )
        
        hidden_dim = config.get('hidden_dim', 1792)
        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        
        dropout = config.get('dropout', 0.3)
        if not 0 <= dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")
    
    @classmethod
    def _validate_classifier_config(cls, config: Dict[str, Any]):
        """验证分类器配置"""
        hidden_dims = config.get('hidden_dims', [512, 256])
        if not isinstance(hidden_dims, list):
            raise ValueError("hidden_dims must be a list")
        
        for dim in hidden_dims:
            if not isinstance(dim, int) or dim <= 0:
                raise ValueError("All hidden dimensions must be positive integers")
        
        num_classes = config.get('num_classes', 2)
        if not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError("num_classes must be a positive integer")
        
        activation = config.get('activation', 'relu')
        if activation not in cls.SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation: {activation}. "
                f"Supported: {list(cls.SUPPORTED_ACTIVATIONS.keys())}"
            )
        
        dropout = config.get('dropout', 0.2)
        if not 0 <= dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")
    
    @classmethod
    def get_supported_components(cls) -> Dict[str, Dict[str, str]]:
        """获取支持的组件列表"""
        return {
            'vision_backbones': cls.SUPPORTED_VISION_BACKBONES,
            'text_encoders': cls.SUPPORTED_TEXT_ENCODERS,
            'fusion_methods': cls.SUPPORTED_FUSION_METHODS,
            'activations': cls.SUPPORTED_ACTIVATIONS
        }
    
    @classmethod
    def create_model_from_preset(cls, preset: str, **kwargs) -> SpeedVQAModel:
        """
        从预设配置创建模型
        
        Args:
            preset: 预设名称 ('small', 'base', 'large')
            **kwargs: 额外配置覆盖
            
        Returns:
            model: SpeedVQA模型实例
        """
        presets = cls.get_model_presets()
        
        if preset not in presets:
            raise ValueError(
                f"Unknown preset: {preset}. "
                f"Available presets: {list(presets.keys())}"
            )
        
        config = presets[preset].copy()
        
        # 应用额外配置
        if kwargs:
            config_manager = ConfigManager()
            config_manager.config = {'model': config}
            config_manager.update_config({'model': kwargs})
            config = config_manager.config['model']
        
        return SpeedVQAModel(config)
    
    @classmethod
    def get_model_presets(cls) -> Dict[str, Dict[str, Any]]:
        """获取模型预设配置"""
        return {
            'small': {
                'name': 'speedvqa_small',
                'vision': {
                    'backbone': 'mobilenet_v3_small',
                    'pretrained': True,
                    'feature_dim': 512,
                    'dropout': 0.1
                },
                'text': {
                    'encoder': 'distilbert-base-uncased',
                    'max_length': 64,
                    'feature_dim': 512,
                    'freeze_encoder': False
                },
                'fusion': {
                    'method': 'concat',
                    'hidden_dim': 1024,
                    'dropout': 0.2,
                    'use_layer_norm': True
                },
                'classifier': {
                    'hidden_dims': [256],
                    'num_classes': 2,
                    'dropout': 0.1,
                    'activation': 'relu'
                }
            },
            'base': {
                'name': 'speedvqa_base',
                'vision': {
                    'backbone': 'mobilenet_v3_small',
                    'pretrained': True,
                    'feature_dim': 1024,
                    'dropout': 0.1
                },
                'text': {
                    'encoder': 'distilbert-base-uncased',
                    'max_length': 128,
                    'feature_dim': 768,
                    'freeze_encoder': False
                },
                'fusion': {
                    'method': 'concat',
                    'hidden_dim': 1792,
                    'dropout': 0.3,
                    'use_layer_norm': True
                },
                'classifier': {
                    'hidden_dims': [512, 256],
                    'num_classes': 2,
                    'dropout': 0.2,
                    'activation': 'relu'
                }
            },
            'large': {
                'name': 'speedvqa_large',
                'vision': {
                    'backbone': 'mobilenet_v3_large',
                    'pretrained': True,
                    'feature_dim': 1280,
                    'dropout': 0.1
                },
                'text': {
                    'encoder': 'distilbert-base-uncased',
                    'max_length': 256,
                    'feature_dim': 768,
                    'freeze_encoder': False
                },
                'fusion': {
                    'method': 'attention',
                    'hidden_dim': 1024,
                    'dropout': 0.3,
                    'use_layer_norm': True
                },
                'classifier': {
                    'hidden_dims': [1024, 512, 256],
                    'num_classes': 2,
                    'dropout': 0.3,
                    'activation': 'gelu'
                }
            }
        }
    
    @classmethod
    def save_model_config(cls, model: SpeedVQAModel, save_path: str):
        """保存模型配置"""
        config = {
            'model': model.config,
            'model_info': model.get_model_info()
        }
        
        config_manager = ConfigManager()
        config_manager.config = config
        config_manager.save_config(save_path)
    
    @classmethod
    def load_model_from_checkpoint(cls, checkpoint_path: str, config_path: Optional[str] = None) -> SpeedVQAModel:
        """
        从检查点加载模型
        
        Args:
            checkpoint_path: 检查点文件路径
            config_path: 配置文件路径（可选）
            
        Returns:
            model: 加载的模型
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 尝试从检查点获取配置
        if 'config' in checkpoint:
            model_config = checkpoint['config']
        elif config_path:
            config = load_config(config_path)
            model_config = config.get('model', {})
        else:
            raise ValueError("No model config found in checkpoint and no config_path provided")
        
        # 创建模型
        model = SpeedVQAModel(model_config)
        
        # 加载权重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model


# 便捷函数
def build_model(config: Union[str, Dict[str, Any]], **kwargs) -> SpeedVQAModel:
    """构建SpeedVQA模型（简化接口）"""
    return ModelFactory.create_model(config, **kwargs)


def build_model_from_preset(preset: str, **kwargs) -> SpeedVQAModel:
    """从预设构建模型（简化接口）"""
    return ModelFactory.create_model_from_preset(preset, **kwargs)


def get_model_presets() -> Dict[str, Dict[str, Any]]:
    """获取模型预设（简化接口）"""
    return ModelFactory.get_model_presets()


def get_supported_components() -> Dict[str, Dict[str, str]]:
    """获取支持的组件（简化接口）"""
    return ModelFactory.get_supported_components()


if __name__ == '__main__':
    # 测试模型工厂
    print("=== SpeedVQA Model Factory Test ===")
    
    # 测试预设模型
    presets = get_model_presets()
    print(f"Available presets: {list(presets.keys())}")
    
    for preset_name in presets.keys():
        print(f"\nTesting {preset_name} preset:")
        try:
            model = build_model_from_preset(preset_name)
            info = model.get_model_info()
            print(f"  ✓ Model: {info['model_name']}")
            print(f"  ✓ Parameters: {info['total_parameters']:,}")
            print(f"  ✓ Size: {info['total_parameters'] * 4 / 1024 / 1024:.1f} MB")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    # 测试支持的组件
    print(f"\nSupported components:")
    components = get_supported_components()
    for component_type, component_list in components.items():
        print(f"  {component_type}: {list(component_list.keys())}")
    
    print("\n✓ Model factory test completed!")