"""
SpeedVQA核心模型架构

MobileNetV3 + DistilBERT + MLP的轻量化多模态架构，
专为T4显卡<50ms推理优化设计。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict
from transformers import AutoModel, AutoConfig
import torchvision.models as models


class VisionEncoder(nn.Module):
    """视觉编码器 - 基于MobileNetV3"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        backbone_name = config.get('backbone', 'mobilenet_v3_small')
        pretrained = config.get('pretrained', True)
        feature_dim = config.get('feature_dim', 1024)
        dropout = config.get('dropout', 0.1)
        
        # 加载预训练的MobileNetV3
        if backbone_name == 'mobilenet_v3_small':
            if pretrained:
                weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            else:
                weights = None
            self.backbone = models.mobilenet_v3_small(weights=weights)
            backbone_feature_dim = 576  # MobileNetV3-Small的特征维度
        elif backbone_name == 'mobilenet_v3_large':
            if pretrained:
                weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
            else:
                weights = None
            self.backbone = models.mobilenet_v3_large(weights=weights)
            backbone_feature_dim = 960  # MobileNetV3-Large的特征维度
        elif backbone_name == 'resnet50':
            if pretrained:
                weights = models.ResNet50_Weights.IMAGENET1K_V1
            else:
                weights = None
            self.backbone = models.resnet50(weights=weights)
            backbone_feature_dim = 2048  # ResNet50的特征维度
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # 移除分类头，只保留特征提取部分
        if 'mobilenet' in backbone_name:
            # MobileNetV3的特征提取
            self.backbone.classifier = nn.Identity()
        elif 'resnet' in backbone_name:
            # ResNet的特征提取
            self.backbone.fc = nn.Identity()
        
        # 特征投影层
        self.feature_projection = nn.Sequential(
            nn.Linear(backbone_feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.feature_dim = feature_dim
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            images: [batch_size, 3, height, width]
            
        Returns:
            features: [batch_size, feature_dim]
        """
        # 提取视觉特征
        features = self.backbone(images)
        
        # 如果backbone返回的是特征图而不是向量，需要进行全局平均池化
        if len(features.shape) == 4:  # [B, C, H, W]
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.flatten(1)
        elif len(features.shape) == 1:  # 单个样本的情况
            features = features.unsqueeze(0)
        
        # 特征投影
        features = self.feature_projection(features)
        
        return features


class TextEncoder(nn.Module):
    """文本编码器 - 基于DistilBERT"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        encoder_name = config.get('encoder', 'distilbert-base-uncased')
        max_length = config.get('max_length', 128)
        feature_dim = config.get('feature_dim', 768)
        freeze_encoder = config.get('freeze_encoder', False)
        
        # 加载预训练的文本编码器
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.encoder_config = AutoConfig.from_pretrained(encoder_name)
        
        # 是否冻结编码器参数
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # 特征投影层（如果需要调整维度）
        encoder_hidden_size = self.encoder_config.hidden_size
        if encoder_hidden_size != feature_dim:
            self.feature_projection = nn.Linear(encoder_hidden_size, feature_dim)
        else:
            self.feature_projection = nn.Identity()
        
        self.max_length = max_length
        self.feature_dim = feature_dim
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: [batch_size, seq_length]
            attention_mask: [batch_size, seq_length]
            
        Returns:
            features: [batch_size, feature_dim]
        """
        # 文本编码
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 使用[CLS] token的表示或平均池化
        if hasattr(outputs, 'last_hidden_state'):
            # 使用[CLS] token (第一个token)
            features = outputs.last_hidden_state[:, 0, :]
        else:
            # 备用方案：平均池化
            features = outputs.last_hidden_state.mean(dim=1)
        
        # 特征投影
        features = self.feature_projection(features)
        
        return features


class MultiModalFusion(nn.Module):
    """多模态融合层"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        vision_dim = config.get('vision_dim', 1024)
        text_dim = config.get('text_dim', 768)
        fusion_method = config.get('method', 'concat')
        hidden_dim = config.get('hidden_dim', vision_dim + text_dim)
        dropout = config.get('dropout', 0.3)
        use_layer_norm = config.get('use_layer_norm', True)
        
        self.fusion_method = fusion_method
        
        if fusion_method == 'concat':
            # 简单拼接
            self.fusion = nn.Sequential(
                nn.Linear(vision_dim + text_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
            self.output_dim = hidden_dim
            
        elif fusion_method == 'attention':
            # 注意力融合
            self.vision_proj = nn.Linear(vision_dim, hidden_dim)
            self.text_proj = nn.Linear(text_dim, hidden_dim)
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=dropout)
            self.layer_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
            self.output_dim = hidden_dim
            
        elif fusion_method == 'bilinear':
            # 双线性融合
            self.bilinear = nn.Bilinear(vision_dim, text_dim, hidden_dim)
            self.layer_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
            self.dropout = nn.Dropout(dropout)
            self.output_dim = hidden_dim

        elif fusion_method == 'film':
            # FiLM：由文本产生对视觉特征的 scale/shift，再与文本拼接后过 MLP（Phase A / plan.md A1）
            self.film_linear = nn.Linear(text_dim, 2 * vision_dim)
            self.fusion = nn.Sequential(
                nn.Linear(vision_dim + text_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.output_dim = hidden_dim

        elif fusion_method == 'cross_attn':
            # 单层跨模态注意力：文本为 Q，视觉为 K/V（单 token，Phase A / plan.md A2）
            self.hidden_dim = hidden_dim
            self.q_proj = nn.Linear(text_dim, hidden_dim)
            self.k_proj = nn.Linear(vision_dim, hidden_dim)
            self.v_proj = nn.Linear(vision_dim, hidden_dim)
            self.scale = hidden_dim ** -0.5
            self.out_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
            self.dropout = nn.Dropout(dropout)
            fused_dropout = dropout
            self.out_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(fused_dropout),
            )
            self.output_dim = hidden_dim

        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")
    
    def forward(self, vision_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            vision_features: [batch_size, vision_dim]
            text_features: [batch_size, text_dim]
            
        Returns:
            fused_features: [batch_size, output_dim]
        """
        if self.fusion_method == 'concat':
            # 拼接融合
            combined = torch.cat([vision_features, text_features], dim=1)
            fused_features = self.fusion(combined)
            
        elif self.fusion_method == 'attention':
            # 注意力融合
            vision_proj = self.vision_proj(vision_features).unsqueeze(1)  # [B, 1, H]
            text_proj = self.text_proj(text_features).unsqueeze(1)        # [B, 1, H]
            
            # 将视觉和文本特征作为序列
            sequence = torch.cat([vision_proj, text_proj], dim=1)  # [B, 2, H]
            sequence = sequence.transpose(0, 1)  # [2, B, H]
            
            # 自注意力
            attended, _ = self.attention(sequence, sequence, sequence)
            
            # 平均池化得到最终特征
            fused_features = attended.mean(dim=0)  # [B, H]
            fused_features = self.layer_norm(fused_features)
            
        elif self.fusion_method == 'bilinear':
            # 双线性融合
            fused_features = self.bilinear(vision_features, text_features)
            fused_features = self.layer_norm(fused_features)
            fused_features = F.relu(fused_features)
            fused_features = self.dropout(fused_features)

        elif self.fusion_method == 'film':
            gb = self.film_linear(text_features)
            gamma, beta = gb.chunk(2, dim=-1)
            modulated_vision = (1.0 + torch.tanh(gamma)) * vision_features + beta
            combined = torch.cat([modulated_vision, text_features], dim=1)
            fused_features = self.fusion(combined)

        elif self.fusion_method == 'cross_attn':
            # Q: [B,1,H], K/V: [B,1,H]
            q = self.q_proj(text_features).unsqueeze(1)
            k = self.k_proj(vision_features).unsqueeze(1)
            v = self.v_proj(vision_features).unsqueeze(1)
            attn = torch.softmax(
                torch.bmm(q, k.transpose(1, 2)) * self.scale,
                dim=-1,
            )
            fused_features = torch.bmm(attn, v).squeeze(1)
            fused_features = self.out_norm(fused_features)
            fused_features = self.out_mlp(fused_features)

        return fused_features


class MLPClassifier(nn.Module):
    """MLP分类器头"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        input_dim = config.get('input_dim', 1792)
        hidden_dims = config.get('hidden_dims', [512, 256])
        num_classes = config.get('num_classes', 2)
        dropout = config.get('dropout', 0.2)
        activation = config.get('activation', 'relu')
        
        def _act_relu() -> nn.Module:
            return nn.ReLU(inplace=True)

        def _act_gelu() -> nn.Module:
            return nn.GELU()

        def _act_swish() -> nn.Module:
            return nn.SiLU(inplace=True)

        _act_factories = {
            'relu': _act_relu,
            'gelu': _act_gelu,
            'swish': _act_swish,
        }
        if activation not in _act_factories:
            raise ValueError(f"Unsupported activation: {activation}")
        act_fn = _act_factories[activation]
        
        # 构建MLP层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                act_fn(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: [batch_size, input_dim]
            
        Returns:
            logits: [batch_size, num_classes]
        """
        return self.classifier(features)


class SpeedVQAModel(nn.Module):
    """
    SpeedVQA完整模型
    
    MobileNetV3 + DistilBERT + MLP架构，专为YES/NO问答优化
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # 提取各组件配置
        vision_config = config.get('vision', {})
        text_config = config.get('text', {})
        fusion_config = config.get('fusion', {})
        classifier_config = config.get('classifier', {})
        
        # 初始化各组件
        self.vision_encoder = VisionEncoder(vision_config)
        self.text_encoder = TextEncoder(text_config)
        
        # 更新融合层配置
        fusion_config.update({
            'vision_dim': self.vision_encoder.feature_dim,
            'text_dim': self.text_encoder.feature_dim
        })
        self.fusion = MultiModalFusion(fusion_config)
        
        # 更新分类器配置
        classifier_config.update({
            'input_dim': self.fusion.output_dim
        })
        self.classifier = MLPClassifier(classifier_config)
        
        # 模型信息
        self.model_name = config.get('name', 'speedvqa')
        self.num_classes = classifier_config.get('num_classes', 2)
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            batch: 包含以下键的字典
                - image: [batch_size, 3, height, width]
                - input_ids: [batch_size, seq_length]
                - attention_mask: [batch_size, seq_length]
                
        Returns:
            outputs: 包含以下键的字典
                - logits: [batch_size, num_classes]
                - probabilities: [batch_size, num_classes] (如果需要)
        """
        # 提取视觉特征
        vision_features = self.vision_encoder(batch['image'])
        
        # 提取文本特征
        text_features = self.text_encoder(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        
        # 多模态融合
        fused_features = self.fusion(vision_features, text_features)
        
        # 分类预测
        logits = self.classifier(fused_features)
        
        outputs = {
            'logits': logits,
            'vision_features': vision_features,
            'text_features': text_features,
            'fused_features': fused_features
        }
        
        return outputs
    
    def predict(self, batch: Dict[str, torch.Tensor], return_probabilities: bool = True) -> Dict[str, torch.Tensor]:
        """
        预测接口
        
        Args:
            batch: 输入批次
            return_probabilities: 是否返回概率
            
        Returns:
            predictions: 预测结果
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(batch)
            logits = outputs['logits']
            
            # 预测类别
            predictions = torch.argmax(logits, dim=1)
            
            result = {
                'predictions': predictions,
                'logits': logits
            }
            
            if return_probabilities:
                probabilities = F.softmax(logits, dim=1)
                result['probabilities'] = probabilities
                result['confidence'] = torch.max(probabilities, dim=1)[0]
            
            return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'vision_backbone': self.config.get('vision', {}).get('backbone', 'unknown'),
            'text_encoder': self.config.get('text', {}).get('encoder', 'unknown'),
            'fusion_method': self.config.get('fusion', {}).get('method', 'unknown')
        }


class SpeedVQAOnnxWrapper(nn.Module):
    """
    供 torch.onnx.export 使用：接受 image、input_ids、attention_mask 三个参数，返回 logits。
    ``SpeedVQAModel.forward(batch: Dict)`` 的字典接口不适用于 ONNX 追踪。
    """

    def __init__(self, model: SpeedVQAModel):
        super().__init__()
        self.model = model

    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.model(
            {
                'image': image,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            }
        )
        return out['logits']


def build_speedvqa_model(config: Dict[str, Any]) -> SpeedVQAModel:
    """
    构建SpeedVQA模型（YOLO风格的简单接口）
    
    Args:
        config: 模型配置
        
    Returns:
        model: SpeedVQA模型实例
    """
    model_config = config.get('model', {})
    model = SpeedVQAModel(model_config)
    
    # 打印模型信息
    info = model.get_model_info()
    print("\n=== SpeedVQA Model Info ===")
    print(f"Model Name: {info['model_name']}")
    print(f"Vision Backbone: {info['vision_backbone']}")
    print(f"Text Encoder: {info['text_encoder']}")
    print(f"Fusion Method: {info['fusion_method']}")
    print(f"Total Parameters: {info['total_parameters']:,}")
    print(f"Trainable Parameters: {info['trainable_parameters']:,}")
    print(f"Model Size: {info['total_parameters'] * 4 / 1024 / 1024:.1f} MB (FP32)")
    
    return model


if __name__ == '__main__':
    # 测试模型构建
    from speedvqa.utils.config import get_default_config
    
    config = get_default_config()
    model = build_speedvqa_model(config)
    
    # 测试前向传播
    batch_size = 2
    test_batch = {
        'image': torch.randn(batch_size, 3, 224, 224),
        'input_ids': torch.randint(0, 1000, (batch_size, 128)),
        'attention_mask': torch.ones(batch_size, 128)
    }
    
    outputs = model(test_batch)
    print("\nTest forward pass:")
    print(f"Input image shape: {test_batch['image'].shape}")
    print(f"Input text shape: {test_batch['input_ids'].shape}")
    print(f"Output logits shape: {outputs['logits'].shape}")
    print(f"Vision features shape: {outputs['vision_features'].shape}")
    print(f"Text features shape: {outputs['text_features'].shape}")
    print(f"Fused features shape: {outputs['fused_features'].shape}")
    
    print("\n✓ SpeedVQA model test completed successfully!")