#!/usr/bin/env python3
"""
SpeedVQA模型工厂测试

测试模型配置系统和工厂函数的功能。
"""

import os
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.chdir(_REPO_ROOT)

import yaml

from speedvqa.models.factory import (
    ModelFactory,
    get_model_presets,
    get_supported_components,
)
from speedvqa.utils.config import create_minimal_config


def test_supported_components():
    """测试支持的组件列表"""
    print("=== 测试支持的组件 ===")

    components = get_supported_components()

    expected_components = ['vision_backbones', 'text_encoders', 'fusion_methods', 'activations']
    for component_type in expected_components:
        assert component_type in components
        assert len(components[component_type]) > 0
        print(f"✓ {component_type}: {list(components[component_type].keys())}")

    print("✓ 支持的组件测试通过")


def test_model_presets():
    """测试模型预设"""
    print("\n=== 测试模型预设 ===")

    presets = get_model_presets()

    expected_presets = ['small', 'base', 'large']
    for preset in expected_presets:
        assert preset in presets
        preset_config = presets[preset]

        assert 'name' in preset_config
        assert 'vision' in preset_config
        assert 'text' in preset_config
        assert 'fusion' in preset_config
        assert 'classifier' in preset_config

        print(f"✓ {preset} 预设配置完整")

    print("✓ 模型预设测试通过")


def test_config_validation():
    """测试配置验证"""
    print("\n=== 测试配置验证 ===")

    valid_config = {
        'model': {
            'vision': {
                'backbone': 'mobilenet_v3_small',
                'feature_dim': 1024,
                'dropout': 0.1
            },
            'text': {
                'encoder': 'distilbert-base-uncased',
                'max_length': 128,
                'feature_dim': 768
            },
            'fusion': {
                'method': 'concat',
                'hidden_dim': 1792,
                'dropout': 0.3
            },
            'classifier': {
                'hidden_dims': [512, 256],
                'num_classes': 2,
                'activation': 'relu'
            }
        }
    }

    try:
        ModelFactory._validate_config(valid_config)
        print("✓ 有效配置验证通过")
    except Exception as e:
        print(f"✗ 有效配置验证失败: {e}")
        return False

    invalid_configs = [
        {},
        {
            'model': {
                'vision': {'backbone': 'invalid_backbone'}
            }
        },
        {
            'model': {
                'fusion': {'method': 'invalid_method'}
            }
        },
        {
            'model': {
                'classifier': {'activation': 'invalid_activation'}
            }
        }
    ]

    for i, invalid_config in enumerate(invalid_configs):
        try:
            ModelFactory._validate_config(invalid_config)
            print(f"✗ 无效配置 {i+1} 应该失败但通过了")
            return False
        except (ValueError, KeyError):
            print(f"✓ 无效配置 {i+1} 正确被拒绝")

    print("✓ 配置验证测试通过")
    return True


def test_yaml_config_loading():
    """测试YAML配置文件加载"""
    print("\n=== 测试YAML配置文件加载 ===")

    config_data = {
        'model': {
            'name': 'test_model',
            'vision': {
                'backbone': 'mobilenet_v3_small',
                'pretrained': False,
                'feature_dim': 512,
                'dropout': 0.1
            },
            'text': {
                'encoder': 'distilbert-base-uncased',
                'max_length': 64,
                'feature_dim': 512
            },
            'fusion': {
                'method': 'concat',
                'hidden_dim': 1024,
                'dropout': 0.2
            },
            'classifier': {
                'hidden_dims': [256],
                'num_classes': 2,
                'activation': 'relu'
            }
        },
        'data': {
            'dataset_path': './test_dataset'
        },
        'train': {
            'epochs': 10,
            'optimizer': {
                'lr': 0.001
            }
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_config_path = f.name

    try:
        from speedvqa.utils.config import load_config
        loaded_config = load_config(temp_config_path)

        assert 'model' in loaded_config
        assert loaded_config['model']['name'] == 'test_model'
        assert loaded_config['model']['vision']['backbone'] == 'mobilenet_v3_small'

        print("✓ YAML配置文件加载成功")

        ModelFactory._validate_config(loaded_config)
        print("✓ 加载的配置验证通过")

        return True

    except Exception as e:
        print(f"✗ YAML配置文件加载失败: {e}")
        return False

    finally:
        Path(temp_config_path).unlink(missing_ok=True)


def test_component_creation():
    """测试组件创建"""
    print("\n=== 测试组件创建 ===")

    try:
        vision_config = {
            'backbone': 'mobilenet_v3_small',
            'pretrained': False,
            'feature_dim': 1024,
            'dropout': 0.1
        }
        vision_encoder = ModelFactory.create_vision_encoder(vision_config)
        assert vision_encoder.feature_dim == 1024
        print("✓ 视觉编码器创建成功")

        fusion_config = {
            'vision_dim': 1024,
            'text_dim': 768,
            'method': 'concat',
            'hidden_dim': 1792,
            'dropout': 0.3
        }
        fusion_layer = ModelFactory.create_fusion_layer(fusion_config)
        assert fusion_layer.output_dim == 1792
        print("✓ 融合层创建成功")

        classifier_config = {
            'input_dim': 1792,
            'hidden_dims': [512, 256],
            'num_classes': 2,
            'activation': 'relu'
        }
        ModelFactory.create_classifier(classifier_config)
        print("✓ 分类器创建成功")

        return True

    except Exception as e:
        print(f"✗ 组件创建失败: {e}")
        return False


def test_config_override():
    """测试配置覆盖"""
    print("\n=== 测试配置覆盖 ===")

    base_config = {
        'model': {
            'vision': {
                'backbone': 'mobilenet_v3_small',
                'feature_dim': 1024
            },
            'classifier': {
                'num_classes': 2
            }
        }
    }

    overrides = {
        'model': {
            'vision': {
                'feature_dim': 512
            },
            'classifier': {
                'num_classes': 5
            }
        }
    }

    try:
        from speedvqa.utils.config import ConfigManager
        config_manager = ConfigManager()
        config_manager.config = base_config
        config_manager.update_config(overrides)

        merged_config = config_manager.config

        assert merged_config['model']['vision']['feature_dim'] == 512
        assert merged_config['model']['classifier']['num_classes'] == 5
        assert merged_config['model']['vision']['backbone'] == 'mobilenet_v3_small'

        print("✓ 配置覆盖测试通过")
        return True

    except Exception as e:
        print(f"✗ 配置覆盖测试失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("SpeedVQA模型配置系统测试")
    print("=" * 50)

    tests = [
        test_supported_components,
        test_model_presets,
        test_config_validation,
        test_yaml_config_loading,
        test_component_creation,
        test_config_override
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result if result is not None else True)
        except Exception as e:
            print(f"✗ 测试异常: {e}")
            results.append(False)

    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"测试结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有测试通过！模型配置系统工作正常。")
        return True
    print("❌ 部分测试失败，请检查实现。")
    return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
