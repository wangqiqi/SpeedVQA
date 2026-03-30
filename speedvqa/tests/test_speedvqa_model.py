"""
SpeedVQA模型单元测试

测试模型前向传播功能和不同融合方法的正确性。
验证需求: 需求 2.1
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict, Any

from speedvqa.models.speedvqa import (
    VisionEncoder, 
    MultiModalFusion, 
    MLPClassifier
)


class MockTextEncoder(nn.Module):
    """模拟文本编码器用于测试"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        vocab_size = 1000
        embed_dim = config.get('feature_dim', 768)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.feature_dim = embed_dim
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # 简单的嵌入和平均池化
        embeddings = self.embedding(input_ids)
        # 使用attention_mask进行加权平均
        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask


class TestVisionEncoder:
    """视觉编码器单元测试"""
    
    def test_mobilenet_v3_small(self):
        """测试MobileNetV3-Small backbone"""
        config = {
            'backbone': 'mobilenet_v3_small',
            'pretrained': False,
            'feature_dim': 1024,
            'dropout': 0.1
        }
        
        encoder = VisionEncoder(config)
        
        # 测试输入
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        
        # 前向传播
        features = encoder(images)
        
        # 验证输出形状
        assert features.shape == (batch_size, 1024)
        assert encoder.feature_dim == 1024
        
        # 验证输出是有限的数值
        assert torch.isfinite(features).all()
    
    def test_mobilenet_v3_large(self):
        """测试MobileNetV3-Large backbone"""
        config = {
            'backbone': 'mobilenet_v3_large',
            'pretrained': False,
            'feature_dim': 512,
            'dropout': 0.2
        }
        
        encoder = VisionEncoder(config)
        
        batch_size = 3
        images = torch.randn(batch_size, 3, 224, 224)
        features = encoder(images)
        
        assert features.shape == (batch_size, 512)
        assert encoder.feature_dim == 512
    
    def test_resnet50(self):
        """测试ResNet50 backbone"""
        config = {
            'backbone': 'resnet50',
            'pretrained': False,
            'feature_dim': 2048,
            'dropout': 0.0
        }
        
        encoder = VisionEncoder(config)
        
        batch_size = 1
        images = torch.randn(batch_size, 3, 224, 224)
        features = encoder(images)
        
        assert features.shape == (batch_size, 2048)
        assert encoder.feature_dim == 2048
    
    def test_invalid_backbone(self):
        """测试无效的backbone"""
        config = {
            'backbone': 'invalid_backbone',
            'pretrained': False,
            'feature_dim': 1024
        }
        
        with pytest.raises(ValueError, match="Unsupported backbone"):
            VisionEncoder(config)
    
    def test_different_input_sizes(self):
        """测试不同输入尺寸"""
        config = {
            'backbone': 'mobilenet_v3_small',
            'pretrained': False,
            'feature_dim': 1024
        }
        
        encoder = VisionEncoder(config)
        
        # 测试不同尺寸的输入
        test_sizes = [(224, 224), (256, 256), (320, 320)]
        batch_size = 2
        
        for h, w in test_sizes:
            images = torch.randn(batch_size, 3, h, w)
            features = encoder(images)
            assert features.shape == (batch_size, 1024)


class TestMultiModalFusion:
    """多模态融合层单元测试"""
    
    def setup_method(self):
        """测试设置"""
        self.vision_dim = 1024
        self.text_dim = 768
        self.batch_size = 2
        
        self.vision_features = torch.randn(self.batch_size, self.vision_dim)
        self.text_features = torch.randn(self.batch_size, self.text_dim)
    
    def test_concat_fusion(self):
        """测试拼接融合"""
        config = {
            'vision_dim': self.vision_dim,
            'text_dim': self.text_dim,
            'method': 'concat',
            'hidden_dim': 1792,
            'dropout': 0.3,
            'use_layer_norm': True
        }
        
        fusion = MultiModalFusion(config)
        fused_features = fusion(self.vision_features, self.text_features)
        
        assert fused_features.shape == (self.batch_size, 1792)
        assert fusion.output_dim == 1792
        assert torch.isfinite(fused_features).all()
    
    def test_attention_fusion(self):
        """测试注意力融合"""
        config = {
            'vision_dim': self.vision_dim,
            'text_dim': self.text_dim,
            'method': 'attention',
            'hidden_dim': 512,
            'dropout': 0.2,
            'use_layer_norm': True
        }
        
        fusion = MultiModalFusion(config)
        fused_features = fusion(self.vision_features, self.text_features)
        
        assert fused_features.shape == (self.batch_size, 512)
        assert fusion.output_dim == 512
        assert torch.isfinite(fused_features).all()
    
    def test_bilinear_fusion(self):
        """测试双线性融合"""
        config = {
            'vision_dim': self.vision_dim,
            'text_dim': self.text_dim,
            'method': 'bilinear',
            'hidden_dim': 256,
            'dropout': 0.1,
            'use_layer_norm': False
        }
        
        fusion = MultiModalFusion(config)
        fused_features = fusion(self.vision_features, self.text_features)
        
        assert fused_features.shape == (self.batch_size, 256)
        assert fusion.output_dim == 256
        assert torch.isfinite(fused_features).all()
    
    def test_film_fusion(self):
        """FiLM：文本调制视觉后拼接"""
        config = {
            'vision_dim': self.vision_dim,
            'text_dim': self.text_dim,
            'method': 'film',
            'hidden_dim': self.vision_dim + self.text_dim,
            'dropout': 0.1,
            'use_layer_norm': True,
        }
        fusion = MultiModalFusion(config)
        fused = fusion(self.vision_features, self.text_features)
        assert fused.shape == (self.batch_size, config['hidden_dim'])
        assert torch.isfinite(fused).all()

    def test_cross_attn_fusion(self):
        """跨模态单 token 注意力"""
        config = {
            'vision_dim': self.vision_dim,
            'text_dim': self.text_dim,
            'method': 'cross_attn',
            'hidden_dim': 256,
            'dropout': 0.1,
            'use_layer_norm': True,
        }
        fusion = MultiModalFusion(config)
        fused = fusion(self.vision_features, self.text_features)
        assert fused.shape == (self.batch_size, 256)
        assert torch.isfinite(fused).all()

    def test_invalid_fusion_method(self):
        """测试无效的融合方法"""
        config = {
            'vision_dim': self.vision_dim,
            'text_dim': self.text_dim,
            'method': 'invalid_method',
            'hidden_dim': 512
        }
        
        with pytest.raises(ValueError, match="Unsupported fusion method"):
            MultiModalFusion(config)
    
    def test_fusion_without_layer_norm(self):
        """测试不使用LayerNorm的融合"""
        config = {
            'vision_dim': self.vision_dim,
            'text_dim': self.text_dim,
            'method': 'concat',
            'hidden_dim': 1792,
            'use_layer_norm': False
        }
        
        fusion = MultiModalFusion(config)
        fused_features = fusion(self.vision_features, self.text_features)
        
        assert fused_features.shape == (self.batch_size, 1792)


class TestMLPClassifier:
    """MLP分类器单元测试"""
    
    def test_basic_classifier(self):
        """测试基本分类器"""
        config = {
            'input_dim': 1792,
            'hidden_dims': [512, 256],
            'num_classes': 2,
            'dropout': 0.2,
            'activation': 'relu'
        }
        
        classifier = MLPClassifier(config)
        
        batch_size = 3
        features = torch.randn(batch_size, 1792)
        logits = classifier(features)
        
        assert logits.shape == (batch_size, 2)
        assert torch.isfinite(logits).all()
    
    def test_different_activations(self):
        """测试不同激活函数"""
        activations = ['relu', 'gelu', 'swish']
        
        for activation in activations:
            config = {
                'input_dim': 512,
                'hidden_dims': [256],
                'num_classes': 2,
                'activation': activation
            }
            
            classifier = MLPClassifier(config)
            features = torch.randn(2, 512)
            logits = classifier(features)
            
            assert logits.shape == (2, 2)
    
    def test_invalid_activation(self):
        """测试无效激活函数"""
        config = {
            'input_dim': 512,
            'hidden_dims': [256],
            'num_classes': 2,
            'activation': 'invalid_activation'
        }
        
        with pytest.raises(ValueError, match="Unsupported activation"):
            MLPClassifier(config)
    
    def test_single_layer_classifier(self):
        """测试单层分类器"""
        config = {
            'input_dim': 1024,
            'hidden_dims': [],  # 无隐藏层
            'num_classes': 3,
            'dropout': 0.0
        }
        
        classifier = MLPClassifier(config)
        features = torch.randn(2, 1024)
        logits = classifier(features)
        
        assert logits.shape == (2, 3)
    
    def test_multi_layer_classifier(self):
        """测试多层分类器"""
        config = {
            'input_dim': 2048,
            'hidden_dims': [1024, 512, 256, 128],
            'num_classes': 10,
            'dropout': 0.5
        }
        
        classifier = MLPClassifier(config)
        features = torch.randn(1, 2048)
        logits = classifier(features)
        
        assert logits.shape == (1, 10)


class TestSpeedVQAModelIntegration:
    """SpeedVQA完整模型集成测试"""
    
    def setup_method(self):
        """测试设置"""
        self.config = {
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
        
        self.batch_size = 2
        self.test_batch = {
            'image': torch.randn(self.batch_size, 3, 224, 224),
            'input_ids': torch.randint(0, 1000, (self.batch_size, 128)),
            'attention_mask': torch.ones(self.batch_size, 128)
        }
    
    def create_mock_model(self):
        """创建使用模拟文本编码器的模型"""
        vision_encoder = VisionEncoder(self.config['vision'])
        text_encoder = MockTextEncoder(self.config['text'])
        
        # 更新融合配置
        fusion_config = self.config['fusion'].copy()
        fusion_config.update({
            'vision_dim': vision_encoder.feature_dim,
            'text_dim': text_encoder.feature_dim
        })
        fusion = MultiModalFusion(fusion_config)
        
        # 更新分类器配置
        classifier_config = self.config['classifier'].copy()
        classifier_config.update({
            'input_dim': fusion.output_dim
        })
        classifier = MLPClassifier(classifier_config)
        
        return vision_encoder, text_encoder, fusion, classifier
    
    def test_forward_pass(self):
        """测试前向传播"""
        vision_encoder, text_encoder, fusion, classifier = self.create_mock_model()
        
        # 前向传播
        vision_features = vision_encoder(self.test_batch['image'])
        text_features = text_encoder(
            self.test_batch['input_ids'], 
            self.test_batch['attention_mask']
        )
        fused_features = fusion(vision_features, text_features)
        logits = classifier(fused_features)
        
        # 验证形状
        assert vision_features.shape == (self.batch_size, 1024)
        assert text_features.shape == (self.batch_size, 768)
        assert fused_features.shape == (self.batch_size, 1792)
        assert logits.shape == (self.batch_size, 2)
        
        # 验证数值有效性
        assert torch.isfinite(vision_features).all()
        assert torch.isfinite(text_features).all()
        assert torch.isfinite(fused_features).all()
        assert torch.isfinite(logits).all()
    
    def test_different_fusion_methods(self):
        """测试不同融合方法的正确性"""
        fusion_methods = ['concat', 'attention', 'bilinear', 'film', 'cross_attn']

        for method in fusion_methods:
            # 更新配置
            config = self.config.copy()
            config['fusion']['method'] = method
            if method == 'concat':
                config['fusion']['hidden_dim'] = 1792
            elif method == 'film':
                config['fusion']['hidden_dim'] = 1792
            else:
                config['fusion']['hidden_dim'] = 512
            
            vision_encoder = VisionEncoder(config['vision'])
            text_encoder = MockTextEncoder(config['text'])
            
            fusion_config = config['fusion'].copy()
            fusion_config.update({
                'vision_dim': vision_encoder.feature_dim,
                'text_dim': text_encoder.feature_dim
            })
            fusion = MultiModalFusion(fusion_config)
            
            classifier_config = config['classifier'].copy()
            classifier_config.update({
                'input_dim': fusion.output_dim
            })
            classifier = MLPClassifier(classifier_config)
            
            # 测试前向传播
            vision_features = vision_encoder(self.test_batch['image'])
            text_features = text_encoder(
                self.test_batch['input_ids'], 
                self.test_batch['attention_mask']
            )
            fused_features = fusion(vision_features, text_features)
            logits = classifier(fused_features)
            
            # 验证输出
            if method in ('concat', 'film'):
                expected_fusion_dim = 1792
            else:
                expected_fusion_dim = 512
            assert fused_features.shape == (self.batch_size, expected_fusion_dim)
            assert logits.shape == (self.batch_size, 2)
            assert torch.isfinite(logits).all()
    
    def test_model_info(self):
        """测试模型信息获取"""
        vision_encoder, text_encoder, fusion, classifier = self.create_mock_model()
        
        # 计算参数数量
        total_params = (
            sum(p.numel() for p in vision_encoder.parameters()) +
            sum(p.numel() for p in text_encoder.parameters()) +
            sum(p.numel() for p in fusion.parameters()) +
            sum(p.numel() for p in classifier.parameters())
        )
        
        trainable_params = (
            sum(p.numel() for p in vision_encoder.parameters() if p.requires_grad) +
            sum(p.numel() for p in text_encoder.parameters() if p.requires_grad) +
            sum(p.numel() for p in fusion.parameters() if p.requires_grad) +
            sum(p.numel() for p in classifier.parameters() if p.requires_grad)
        )
        
        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params <= total_params
    
    def test_gradient_flow(self):
        """测试梯度流"""
        vision_encoder, text_encoder, fusion, classifier = self.create_mock_model()
        
        # 前向传播
        vision_features = vision_encoder(self.test_batch['image'])
        text_features = text_encoder(
            self.test_batch['input_ids'], 
            self.test_batch['attention_mask']
        )
        fused_features = fusion(vision_features, text_features)
        logits = classifier(fused_features)
        
        # 计算损失
        targets = torch.randint(0, 2, (self.batch_size,))
        loss = nn.CrossEntropyLoss()(logits, targets)
        
        # 反向传播
        loss.backward()
        
        # 检查梯度
        for name, param in vision_encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Invalid gradient for {name}"
        
        for name, param in text_encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Invalid gradient for {name}"
    
    def test_eval_mode(self):
        """测试评估模式"""
        vision_encoder, text_encoder, fusion, classifier = self.create_mock_model()
        
        # 设置为评估模式
        vision_encoder.eval()
        text_encoder.eval()
        fusion.eval()
        classifier.eval()
        
        with torch.no_grad():
            vision_features = vision_encoder(self.test_batch['image'])
            text_features = text_encoder(
                self.test_batch['input_ids'], 
                self.test_batch['attention_mask']
            )
            fused_features = fusion(vision_features, text_features)
            logits = classifier(fused_features)
            
            # 验证输出
            assert logits.shape == (self.batch_size, 2)
            assert torch.isfinite(logits).all()
    
    def test_batch_size_flexibility(self):
        """测试不同批次大小的灵活性"""
        vision_encoder, text_encoder, fusion, classifier = self.create_mock_model()
        
        batch_sizes = [1, 3, 8, 16]
        
        for batch_size in batch_sizes:
            test_batch = {
                'image': torch.randn(batch_size, 3, 224, 224),
                'input_ids': torch.randint(0, 1000, (batch_size, 128)),
                'attention_mask': torch.ones(batch_size, 128)
            }
            
            vision_features = vision_encoder(test_batch['image'])
            text_features = text_encoder(
                test_batch['input_ids'], 
                test_batch['attention_mask']
            )
            fused_features = fusion(vision_features, text_features)
            logits = classifier(fused_features)
            
            assert logits.shape == (batch_size, 2)


if __name__ == '__main__':
    # 运行单元测试
    pytest.main([__file__, '-v'])