#!/usr/bin/env python3
"""
ModelExporter使用示例

演示如何使用ModelExporter导出SpeedVQA模型为不同格式。
"""

import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import shutil

# 添加项目路径
import sys
sys.path.append('.')

from speedvqa.export.exporter import ModelExporter, export_model
from speedvqa.utils.config import get_default_config


class MockSpeedVQAModel(nn.Module):
    """
    模拟的SpeedVQA模型，用于演示导出功能
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 简化的组件
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.text_encoder = nn.Sequential(
            nn.Embedding(30522, 768),  # DistilBERT词汇表大小
            nn.LSTM(768, 384, batch_first=True),
            nn.Dropout(0.1)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(1024 + 384, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.LayerNorm(512)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )
    
    def forward(self, batch):
        """前向传播"""
        # 视觉特征
        vision_features = self.vision_encoder(batch['image'])
        
        # 文本特征
        text_embed = self.text_encoder[0](batch['input_ids'])
        lstm_out, _ = self.text_encoder[1](text_embed)
        text_features = lstm_out[:, -1, :]  # 使用最后一个时间步
        text_features = self.text_encoder[2](text_features)  # Dropout
        
        # 融合
        fused_features = torch.cat([vision_features, text_features], dim=1)
        fused_features = self.fusion(fused_features)
        
        # 分类
        logits = self.classifier(fused_features)
        
        return {
            'logits': logits,
            'vision_features': vision_features,
            'text_features': text_features,
            'fused_features': fused_features
        }
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'MockSpeedVQA',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'vision_backbone': 'simple_cnn',
            'text_encoder': 'simple_lstm',
            'fusion_method': 'concat'
        }


def create_sample_data(batch_size=4):
    """创建示例数据"""
    return {
        'image': torch.randn(batch_size, 3, 224, 224),
        'input_ids': torch.randint(0, 30522, (batch_size, 128)),
        'attention_mask': torch.ones(batch_size, 128)
    }


def demonstrate_pytorch_export():
    """演示PyTorch格式导出"""
    print("=== PyTorch Export Demo ===")
    
    # 创建配置
    config = {
        'model': {
            'name': 'mock_speedvqa',
            'vision': {'feature_dim': 1024},
            'text': {'feature_dim': 384, 'max_length': 128},
            'fusion': {'hidden_dim': 1408},
            'classifier': {'num_classes': 2}
        },
        'data': {'image': {'size': [224, 224]}},
        'inference': {'device': 'cpu'},
        'export': {
            'validation': {'enabled': True, 'tolerance': 1e-3},
            'benchmark': {'enabled': True}
        }
    }
    
    # 创建模型
    model = MockSpeedVQAModel(config['model'])
    model.eval()
    
    # 创建导出器
    exporter = ModelExporter(config)
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    try:
        save_path = Path(temp_dir) / 'mock_speedvqa.pt'
        
        print(f"Exporting model to: {save_path}")
        
        # 导出模型
        result = exporter.export_pytorch(model, str(save_path), include_optimizer=False)
        
        # 显示结果
        print(f"✓ Export Success: {result.success}")
        print(f"  File Size: {result.model_size_mb:.2f} MB")
        print(f"  Export Time: {result.export_time_s:.3f} seconds")
        
        if result.validation_result:
            val_result = result.validation_result
            print(f"  Validation: {'✓' if val_result['success'] else '✗'}")
            print(f"  Inference Time: {val_result['inference_time_ms']:.2f} ms")
            print(f"  Numerical Accuracy: {val_result['numerical_accuracy']:.6f}")
        
        # 验证加载
        print("\nTesting model loading...")
        checkpoint = torch.load(save_path, map_location='cpu')
        
        loaded_model = MockSpeedVQAModel(checkpoint['model_config'])
        loaded_model.load_state_dict(checkpoint['model_state_dict'])
        loaded_model.eval()
        
        # 测试推理
        test_data = create_sample_data(2)
        with torch.no_grad():
            original_output = model(test_data)
            loaded_output = loaded_model(test_data)
        
        # 比较输出
        diff = torch.max(torch.abs(original_output['logits'] - loaded_output['logits'])).item()
        print(f"✓ Model loaded successfully, max difference: {diff:.8f}")
        
    finally:
        shutil.rmtree(temp_dir)


def demonstrate_onnx_export():
    """演示ONNX格式导出"""
    print("\n=== ONNX Export Demo ===")
    
    try:
        import onnx
        import onnxruntime as ort
        onnx_available = True
    except ImportError:
        print("❌ ONNX not available, skipping ONNX export demo")
        return
    
    # 创建配置
    config = {
        'model': {
            'name': 'mock_speedvqa',
            'vision': {'feature_dim': 1024},
            'text': {'feature_dim': 384, 'max_length': 128},
            'fusion': {'hidden_dim': 1408},
            'classifier': {'num_classes': 2}
        },
        'data': {'image': {'size': [224, 224]}},
        'inference': {'device': 'cpu'},
        'export': {
            'validation': {'enabled': True, 'tolerance': 1e-2}  # ONNX容差稍大
        }
    }
    
    # 创建模型
    model = MockSpeedVQAModel(config['model'])
    model.eval()
    
    # 创建导出器
    exporter = ModelExporter(config)
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    try:
        save_path = Path(temp_dir) / 'mock_speedvqa.onnx'
        
        print(f"Exporting model to: {save_path}")
        
        # 导出模型
        result = exporter.export_onnx(
            model, 
            str(save_path),
            input_shapes={
                'image': (1, 3, 224, 224),
                'input_ids': (1, 128),
                'attention_mask': (1, 128)
            },
            opset_version=11
        )
        
        # 显示结果
        print(f"✓ Export Success: {result.success}")
        if result.success:
            print(f"  File Size: {result.model_size_mb:.2f} MB")
            print(f"  Export Time: {result.export_time_s:.3f} seconds")
            
            if result.validation_result:
                val_result = result.validation_result
                print(f"  Validation: {'✓' if val_result['success'] else '✗'}")
                print(f"  Inference Time: {val_result['inference_time_ms']:.2f} ms")
                print(f"  Numerical Accuracy: {val_result['numerical_accuracy']:.6f}")
        else:
            print(f"❌ Export failed: {result.error_message}")
        
    finally:
        shutil.rmtree(temp_dir)


def demonstrate_all_formats_export():
    """演示导出所有格式"""
    print("\n=== All Formats Export Demo ===")
    
    # 创建配置
    config = {
        'model': {
            'name': 'mock_speedvqa',
            'vision': {'feature_dim': 1024},
            'text': {'feature_dim': 384, 'max_length': 128},
            'fusion': {'hidden_dim': 1408},
            'classifier': {'num_classes': 2}
        },
        'data': {'image': {'size': [224, 224]}},
        'inference': {'device': 'cpu'},
        'export': {
            'validation': {'enabled': True, 'tolerance': 1e-3}
        }
    }
    
    # 创建模型
    model = MockSpeedVQAModel(config['model'])
    model.eval()
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    try:
        print(f"Export directory: {temp_dir}")
        
        # 使用便捷函数导出所有格式
        results = export_model(
            model=model,
            output_dir=temp_dir,
            model_name='mock_speedvqa_all',
            config=config,
            formats=['pytorch', 'onnx']  # 只导出可用的格式
        )
        
        print("\nExport Summary:")
        for format_name, result in results.items():
            status = "✓" if result.success else "✗"
            print(f"  {status} {format_name.upper()}: {result.model_size_mb:.2f} MB")
            if not result.success:
                print(f"    Error: {result.error_message}")
        
    finally:
        shutil.rmtree(temp_dir)


def demonstrate_performance_benchmark():
    """演示性能基准测试"""
    print("\n=== Performance Benchmark Demo ===")
    
    # 创建配置
    config = {
        'model': {
            'name': 'mock_speedvqa',
            'vision': {'feature_dim': 1024},
            'text': {'feature_dim': 384, 'max_length': 128},
            'fusion': {'hidden_dim': 1408},
            'classifier': {'num_classes': 2}
        },
        'data': {'image': {'size': [224, 224]}},
        'inference': {'device': 'cpu'},
        'export': {
            'validation': {'enabled': False},  # 跳过验证以加快速度
            'benchmark': {'enabled': True}
        }
    }
    
    # 创建模型
    model = MockSpeedVQAModel(config['model'])
    model.eval()
    
    # 创建导出器
    exporter = ModelExporter(config)
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    try:
        # 导出PyTorch模型
        pt_path = Path(temp_dir) / 'benchmark_model.pt'
        pt_result = exporter.export_pytorch(model, str(pt_path))
        
        if pt_result.success:
            print("Running performance benchmark...")
            
            # 运行基准测试
            model_paths = {'pytorch': str(pt_path)}
            benchmark_results = exporter.benchmark_exported_models(model_paths, num_iterations=50)
            
            print("\nBenchmark Results:")
            for format_name, metrics in benchmark_results.items():
                if 'error' not in metrics:
                    print(f"  {format_name.upper()}:")
                    print(f"    Average Inference Time: {metrics['avg_inference_time_ms']:.2f} ms")
                    print(f"    Throughput: {metrics['throughput_fps']:.1f} FPS")
                    print(f"    Total Time: {metrics['total_time_s']:.3f} s")
                else:
                    print(f"  {format_name.upper()}: Error - {metrics['error']}")
        
    finally:
        shutil.rmtree(temp_dir)


def main():
    """主函数"""
    print("🚀 SpeedVQA ModelExporter Demo")
    print("=" * 50)
    
    try:
        # 演示各种导出功能
        demonstrate_pytorch_export()
        demonstrate_onnx_export()
        demonstrate_all_formats_export()
        demonstrate_performance_benchmark()
        
        print("\n🎉 All demos completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()