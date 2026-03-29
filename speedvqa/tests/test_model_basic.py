#!/usr/bin/env python3
"""
SpeedVQA模型基础测试

测试模型架构的基本功能，不依赖外部预训练模型。
"""

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.chdir(_REPO_ROOT)

import torch
import torch.nn as nn


class MockTextEncoder(nn.Module):
    """模拟文本编码器用于测试"""

    def __init__(self, config):
        super().__init__()
        embed_dim = config.get('feature_dim', 768)

        self.embedding = nn.Embedding(1000, embed_dim)
        self.feature_dim = embed_dim

    def forward(self, input_ids, attention_mask):
        embeddings = self.embedding(input_ids)
        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask


def test_vision_encoder():
    """测试视觉编码器"""
    print("=== 测试视觉编码器 ===")

    from speedvqa.models.speedvqa import VisionEncoder

    config = {
        'backbone': 'mobilenet_v3_small',
        'pretrained': False,
        'feature_dim': 1024,
        'dropout': 0.1
    }

    vision_encoder = VisionEncoder(config)

    batch_size = 2
    test_images = torch.randn(batch_size, 3, 224, 224)

    features = vision_encoder(test_images)

    print(f"✓ 输入图像形状: {test_images.shape}")
    print(f"✓ 输出特征形状: {features.shape}")
    print(f"✓ 特征维度: {vision_encoder.feature_dim}")

    assert features.shape == (batch_size, config['feature_dim'])
    print("✓ 视觉编码器测试通过")

    return vision_encoder


def test_fusion_layers():
    """测试融合层"""
    print("\n=== 测试多模态融合层 ===")

    from speedvqa.models.speedvqa import MultiModalFusion

    vision_dim = 1024
    text_dim = 768
    batch_size = 2

    fusion_methods = ['concat', 'attention', 'bilinear']

    for method in fusion_methods:
        print(f"\n测试 {method} 融合:")

        config = {
            'vision_dim': vision_dim,
            'text_dim': text_dim,
            'method': method,
            'hidden_dim': 1792 if method == 'concat' else 512,
            'dropout': 0.3,
            'use_layer_norm': True
        }

        fusion = MultiModalFusion(config)

        vision_features = torch.randn(batch_size, vision_dim)
        text_features = torch.randn(batch_size, text_dim)

        fused_features = fusion(vision_features, text_features)

        print(f"  ✓ 视觉特征形状: {vision_features.shape}")
        print(f"  ✓ 文本特征形状: {text_features.shape}")
        print(f"  ✓ 融合特征形状: {fused_features.shape}")
        print(f"  ✓ 输出维度: {fusion.output_dim}")

        assert fused_features.shape == (batch_size, fusion.output_dim)
        print(f"  ✓ {method} 融合测试通过")


def test_classifier():
    """测试分类器"""
    print("\n=== 测试MLP分类器 ===")

    from speedvqa.models.speedvqa import MLPClassifier

    config = {
        'input_dim': 1792,
        'hidden_dims': [512, 256],
        'num_classes': 2,
        'dropout': 0.2,
        'activation': 'relu'
    }

    classifier = MLPClassifier(config)

    batch_size = 2
    test_features = torch.randn(batch_size, config['input_dim'])

    logits = classifier(test_features)

    print(f"✓ 输入特征形状: {test_features.shape}")
    print(f"✓ 输出logits形状: {logits.shape}")

    assert logits.shape == (batch_size, config['num_classes'])
    print("✓ MLP分类器测试通过")

    return classifier


def test_complete_model():
    """测试完整模型（使用模拟组件）"""
    print("\n=== 测试完整SpeedVQA模型 ===")

    config = {
        'model': {
            'name': 'speedvqa_test',
            'vision': {
                'backbone': 'mobilenet_v3_small',
                'pretrained': False,
                'feature_dim': 1024,
                'dropout': 0.1
            },
            'text': {
                'encoder': 'mock',
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
        }
    }

    from speedvqa.models.speedvqa import VisionEncoder, MultiModalFusion, MLPClassifier

    vision_encoder = VisionEncoder(config['model']['vision'])
    text_encoder = MockTextEncoder(config['model']['text'])

    fusion_config = config['model']['fusion'].copy()
    fusion_config.update({
        'vision_dim': vision_encoder.feature_dim,
        'text_dim': text_encoder.feature_dim
    })
    fusion = MultiModalFusion(fusion_config)

    classifier_config = config['model']['classifier'].copy()
    classifier_config.update({
        'input_dim': fusion.output_dim
    })
    classifier = MLPClassifier(classifier_config)

    batch_size = 2
    test_batch = {
        'image': torch.randn(batch_size, 3, 224, 224),
        'input_ids': torch.randint(0, 1000, (batch_size, 128)),
        'attention_mask': torch.ones(batch_size, 128)
    }

    vision_features = vision_encoder(test_batch['image'])
    text_features = text_encoder(test_batch['input_ids'], test_batch['attention_mask'])
    fused_features = fusion(vision_features, text_features)
    logits = classifier(fused_features)

    print(f"✓ 视觉特征形状: {vision_features.shape}")
    print(f"✓ 文本特征形状: {text_features.shape}")
    print(f"✓ 融合特征形状: {fused_features.shape}")
    print(f"✓ 最终logits形状: {logits.shape}")

    assert vision_features.shape == (batch_size, 1024)
    assert text_features.shape == (batch_size, 768)
    assert fused_features.shape == (batch_size, 1792)
    assert logits.shape == (batch_size, 2)

    print("✓ 完整模型流程测试通过")

    total_params = (
        sum(p.numel() for p in vision_encoder.parameters()) +
        sum(p.numel() for p in text_encoder.parameters()) +
        sum(p.numel() for p in fusion.parameters()) +
        sum(p.numel() for p in classifier.parameters())
    )

    print(f"✓ 模型总参数量: {total_params:,}")
    print(f"✓ 模型大小估计: {total_params * 4 / 1024 / 1024:.1f} MB (FP32)")


def main():
    """运行所有测试"""
    print("SpeedVQA模型架构测试")
    print("=" * 50)

    try:
        test_vision_encoder()
        test_fusion_layers()
        test_classifier()
        test_complete_model()

        print("\n" + "=" * 50)
        print("🎉 所有模型组件测试通过！")
        print("SpeedVQA模型架构实现正确。")

        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
