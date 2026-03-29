#!/usr/bin/env python3
"""
SpeedVQA性能基准测试示例

演示如何使用增强的ModelExporter进行全面的性能基准测试，
包括推理速度、内存使用、结果一致性验证和性能比较分析。
"""

import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import shutil
import numpy as np

import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from speedvqa.export.exporter import ModelExporter


class MockSpeedVQAModel(nn.Module):
    """
    模拟的SpeedVQA模型，用于演示性能基准测试功能
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


def demonstrate_basic_performance_benchmark():
    """演示基本性能基准测试"""
    print("=== Basic Performance Benchmark Demo ===")
    
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
            'benchmark': {
                'enabled': True,
                'warmup_iterations': 5,
                'test_iterations': 20
            },
            'benchmark_report_dir': './benchmark_reports'
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
        print(f"Export directory: {temp_dir}")
        
        # 导出PyTorch模型
        pt_path = Path(temp_dir) / 'benchmark_model.pt'
        pt_result = exporter.export_pytorch(model, str(pt_path))
        
        if pt_result.success:
            print(f"✓ PyTorch model exported: {pt_result.model_size_mb:.2f} MB")
            
            # 运行基本基准测试
            model_paths = {'pytorch': str(pt_path)}
            
            print("\nRunning basic performance benchmark...")
            benchmark_results = exporter.benchmark_exported_models(
                model_paths, 
                num_iterations=20,
                batch_sizes=[1, 4],
                warmup_iterations=5
            )
            
            # 显示结果摘要
            print("\n=== Benchmark Results Summary ===")
            summary = benchmark_results.get('summary', {})
            print(f"Formats tested: {summary.get('total_formats', 0)}")
            print(f"Batch sizes: {summary.get('batch_sizes_tested', [])}")
            print(f"Iterations per test: {summary.get('iterations_per_test', 0)}")
            
            # 显示详细结果
            for batch_key, batch_results in benchmark_results.get('detailed_results', {}).items():
                batch_size = batch_key.split('_')[1]
                print(f"\n--- Batch Size {batch_size} ---")
                
                for format_name, result in batch_results.items():
                    if 'performance' in result:
                        perf = result['performance']
                        if not perf.error_message:
                            print(f"  {format_name.upper()}:")
                            print(f"    Avg Inference Time: {perf.avg_inference_time_ms:.2f} ms")
                            print(f"    Throughput: {perf.throughput_fps:.1f} FPS")
                            print(f"    Memory Usage: {perf.peak_memory_mb:.1f} MB")
                            
                            # 检查T4性能目标
                            if perf.avg_inference_time_ms < 50.0:
                                print("    ✅ Meets T4 target (<50ms)")
                            else:
                                print("    ⚠️  Exceeds T4 target (>50ms)")
                        else:
                            print(f"  {format_name.upper()}: Error - {perf.error_message}")
            
            # 显示建议
            recommendations = benchmark_results.get('recommendations', [])
            if recommendations:
                print("\n=== Recommendations ===")
                for rec in recommendations:
                    print(f"  {rec}")
        
        else:
            print(f"❌ PyTorch export failed: {pt_result.error_message}")
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def demonstrate_comprehensive_benchmark():
    """演示全面的性能基准测试"""
    print("\n=== Comprehensive Performance Benchmark Demo ===")
    
    # 创建配置
    config = {
        'model': {
            'name': 'mock_speedvqa_comprehensive',
            'vision': {'feature_dim': 1024},
            'text': {'feature_dim': 384, 'max_length': 128},
            'fusion': {'hidden_dim': 1408},
            'classifier': {'num_classes': 2}
        },
        'data': {'image': {'size': [224, 224]}},
        'inference': {'device': 'cpu'},
        'export': {
            'validation': {'enabled': True, 'tolerance': 1e-3},
            'benchmark': {
                'enabled': True,
                'warmup_iterations': 10,
                'test_iterations': 50
            },
            'benchmark_report_dir': './benchmark_reports'
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
        print(f"Export directory: {temp_dir}")
        
        # 导出多种格式
        base_path = Path(temp_dir) / 'comprehensive_model'
        
        # 导出PyTorch
        pt_path = base_path.with_suffix('.pt')
        pt_result = exporter.export_pytorch(model, str(pt_path))
        
        model_paths = {}
        if pt_result.success:
            model_paths['pytorch'] = str(pt_path)
            print(f"✓ PyTorch exported: {pt_result.model_size_mb:.2f} MB")
        
        # 尝试导出ONNX（如果可用）
        try:
            onnx_path = base_path.with_suffix('.onnx')
            onnx_result = exporter.export_onnx(model, str(onnx_path))
            if onnx_result.success:
                model_paths['onnx'] = str(onnx_path)
                print(f"✓ ONNX exported: {onnx_result.model_size_mb:.2f} MB")
        except Exception as e:
            print(f"⚠️  ONNX export skipped: {str(e)}")
        
        if model_paths:
            print(f"\nRunning comprehensive benchmark on {len(model_paths)} formats...")
            
            # 运行全面基准测试
            benchmark_results = exporter.benchmark_exported_models(
                model_paths,
                num_iterations=50,
                batch_sizes=[1, 2, 4, 8],
                warmup_iterations=10,
                reference_format='pytorch'
            )
            
            # 显示性能比较
            perf_comparison = benchmark_results.get('performance_comparison', {})
            
            if perf_comparison.get('overall_ranking'):
                print("\n=== Overall Performance Ranking ===")
                for i, (format_name, score) in enumerate(perf_comparison['overall_ranking']):
                    print(f"  {i+1}. {format_name.upper()} (score: {score:.2f})")
            
            # 显示批次大小分析
            batch_analysis = perf_comparison.get('batch_size_analysis', {})
            if batch_analysis:
                print("\n=== Batch Size Analysis ===")
                for batch_size, analysis in batch_analysis.items():
                    print(f"\n  Batch Size {batch_size}:")
                    print(f"    Speed Ranking: {' > '.join(analysis['speed_ranking'])}")
                    print(f"    Memory Ranking: {' > '.join(analysis['memory_ranking'])}")
                    
                    # 显示具体指标
                    for format_name, metrics in analysis['metrics'].items():
                        print(f"    {format_name.upper()}: "
                              f"{metrics['avg_time_ms']:.1f}ms, "
                              f"{metrics['throughput_fps']:.1f}FPS, "
                              f"{metrics['memory_mb']:.1f}MB")
            
            # 显示一致性结果
            consistency_results = benchmark_results.get('consistency_results', {})
            if consistency_results:
                print("\n=== Consistency Analysis ===")
                for batch_key, batch_consistency in consistency_results.items():
                    batch_size = batch_key.split('_')[1]
                    print(f"\n  Batch Size {batch_size}:")
                    
                    for format_name, consistency in batch_consistency.items():
                        if hasattr(consistency, 'consistency_score'):
                            score = consistency.consistency_score
                            match_rate = consistency.prediction_match_rate
                            print(f"    {format_name.upper()} vs PyTorch: "
                                  f"consistency={score:.3f}, "
                                  f"prediction_match={match_rate:.3f}")
                            
                            if score < 0.95:
                                print("      ⚠️  Low consistency detected!")
            
            # 显示内存分析
            print("\n=== Memory Usage Analysis ===")
            for batch_key, batch_results in benchmark_results.get('detailed_results', {}).items():
                batch_size = batch_key.split('_')[1]
                print(f"\n  Batch Size {batch_size}:")
                
                for format_name, result in batch_results.items():
                    if 'memory' in result:
                        memory = result['memory']
                        print(f"    {format_name.upper()}: "
                              f"peak={memory.peak_memory_mb:.1f}MB, "
                              f"avg={memory.avg_memory_mb:.1f}MB")
                        
                        if memory.gpu_peak_memory_mb is not None:
                            print(f"      GPU: peak={memory.gpu_peak_memory_mb:.1f}MB")
            
            print("\n✓ Comprehensive benchmark completed!")
            print("  Detailed report saved to: ./benchmark_reports/")
        
        else:
            print("❌ No models were successfully exported")
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def demonstrate_t4_performance_validation():
    """演示T4性能目标验证"""
    print("\n=== T4 Performance Target Validation ===")
    
    # 创建针对T4优化的配置
    config = {
        'model': {
            'name': 'speedvqa_t4_optimized',
            'vision': {'feature_dim': 512},  # 更小的特征维度
            'text': {'feature_dim': 256, 'max_length': 64},  # 更短的序列
            'fusion': {'hidden_dim': 768},
            'classifier': {'num_classes': 2}
        },
        'data': {'image': {'size': [224, 224]}},
        'inference': {'device': 'cpu'},  # 在CPU上模拟
        'export': {
            'validation': {'enabled': True, 'tolerance': 1e-3},
            'benchmark': {
                'enabled': True,
                'warmup_iterations': 20,
                'test_iterations': 100  # 更多迭代以获得准确结果
            }
        }
    }
    
    # 创建优化的模型
    model = MockSpeedVQAModel(config['model'])
    model.eval()
    
    # 创建导出器
    exporter = ModelExporter(config)
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    try:
        # 导出模型
        pt_path = Path(temp_dir) / 't4_optimized_model.pt'
        pt_result = exporter.export_pytorch(model, str(pt_path))
        
        if pt_result.success:
            print(f"✓ T4 optimized model exported: {pt_result.model_size_mb:.2f} MB")
            
            # 运行T4性能验证
            model_paths = {'pytorch': str(pt_path)}
            
            print("\nValidating T4 performance targets...")
            benchmark_results = exporter.benchmark_exported_models(
                model_paths,
                num_iterations=100,
                batch_sizes=[1, 2, 4, 8, 16],  # 测试多种批次大小
                warmup_iterations=20
            )
            
            # T4性能目标分析
            target_time_ms = 50.0
            print(f"\n=== T4 Performance Target Analysis (< {target_time_ms}ms) ===")
            
            meets_target = {}
            for batch_key, batch_results in benchmark_results.get('detailed_results', {}).items():
                batch_size = int(batch_key.split('_')[1])
                
                for format_name, result in batch_results.items():
                    if 'performance' in result and not result['performance'].error_message:
                        perf = result['performance']
                        avg_time = perf.avg_inference_time_ms
                        min_time = perf.min_inference_time_ms
                        max_time = perf.max_inference_time_ms
                        
                        meets_target[batch_size] = avg_time < target_time_ms
                        
                        status = "✅" if meets_target[batch_size] else "❌"
                        print(f"  Batch {batch_size:2d}: {status} "
                              f"avg={avg_time:5.1f}ms, "
                              f"min={min_time:5.1f}ms, "
                              f"max={max_time:5.1f}ms, "
                              f"fps={perf.throughput_fps:6.1f}")
            
            # 推荐最佳批次大小
            optimal_batch_sizes = [bs for bs, meets in meets_target.items() if meets]
            if optimal_batch_sizes:
                max_optimal_batch = max(optimal_batch_sizes)
                print(f"\n🎯 Recommended batch size for T4: {max_optimal_batch}")
                print(f"   (Largest batch size meeting <{target_time_ms}ms target)")
            else:
                print(f"\n⚠️  No batch size meets T4 target (<{target_time_ms}ms)")
                print("   Consider model optimization or hardware upgrade")
            
            # 内存效率分析
            print("\n=== Memory Efficiency Analysis ===")
            for batch_key, batch_results in benchmark_results.get('detailed_results', {}).items():
                batch_size = int(batch_key.split('_')[1])
                
                for format_name, result in batch_results.items():
                    if 'performance' in result and 'memory' in result:
                        perf = result['performance']
                        memory = result['memory']
                        
                        if not perf.error_message:
                            # 计算效率指标
                            fps_per_mb = perf.throughput_fps / max(memory.peak_memory_mb, 1.0)
                            samples_per_mb = batch_size / max(memory.peak_memory_mb, 1.0)
                            
                            print(f"  Batch {batch_size:2d}: "
                                  f"memory={memory.peak_memory_mb:5.1f}MB, "
                                  f"efficiency={fps_per_mb:5.2f}FPS/MB, "
                                  f"density={samples_per_mb:5.2f}samples/MB")
        
        else:
            print(f"❌ Model export failed: {pt_result.error_message}")
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def demonstrate_memory_profiling():
    """演示内存使用分析"""
    print("\n=== Memory Usage Profiling Demo ===")
    
    # 创建配置
    config = {
        'model': {
            'name': 'memory_profile_model',
            'vision': {'feature_dim': 1024},
            'text': {'feature_dim': 768, 'max_length': 128},
            'fusion': {'hidden_dim': 1792},
            'classifier': {'num_classes': 2}
        },
        'data': {'image': {'size': [224, 224]}},
        'inference': {'device': 'cpu'},
        'export': {
            'validation': {'enabled': False},  # 跳过验证以专注内存分析
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
        # 导出模型
        pt_path = Path(temp_dir) / 'memory_profile_model.pt'
        pt_result = exporter.export_pytorch(model, str(pt_path))
        
        if pt_result.success:
            print("✓ Model exported for memory profiling")
            
            # 运行内存分析
            model_paths = {'pytorch': str(pt_path)}
            
            print("\nRunning memory usage analysis...")
            benchmark_results = exporter.benchmark_exported_models(
                model_paths,
                num_iterations=10,  # 较少迭代，专注内存分析
                batch_sizes=[1, 4, 8, 16, 32],
                warmup_iterations=3
            )
            
            # 分析内存使用模式
            print("\n=== Memory Usage Patterns ===")
            
            memory_data = []
            for batch_key, batch_results in benchmark_results.get('detailed_results', {}).items():
                batch_size = int(batch_key.split('_')[1])
                
                for format_name, result in batch_results.items():
                    if 'memory' in result:
                        memory = result['memory']
                        memory_data.append({
                            'batch_size': batch_size,
                            'peak_memory': memory.peak_memory_mb,
                            'avg_memory': memory.avg_memory_mb
                        })
                        
                        print(f"  Batch {batch_size:2d}: "
                              f"peak={memory.peak_memory_mb:6.1f}MB, "
                              f"avg={memory.avg_memory_mb:6.1f}MB")
                        
                        # 分析内存时间线
                        if memory.memory_timeline:
                            timeline = memory.memory_timeline
                            print(f"    Timeline: {len(timeline)} measurements")
                            
                            # 计算内存增长
                            if len(timeline) >= 2:
                                initial_mem = timeline[0][1]
                                final_mem = timeline[-1][1]
                                growth = final_mem - initial_mem
                                print(f"    Memory growth: {growth:+.1f}MB")
            
            # 内存缩放分析
            if len(memory_data) >= 2:
                print("\n=== Memory Scaling Analysis ===")
                
                # 计算内存与批次大小的关系
                batch_sizes = [d['batch_size'] for d in memory_data]

                # 简单线性拟合
                if len(batch_sizes) >= 2:
                    # 计算每个样本的内存开销
                    memory_per_sample = []
                    for i in range(1, len(memory_data)):
                        prev_data = memory_data[i-1]
                        curr_data = memory_data[i]
                        
                        batch_diff = curr_data['batch_size'] - prev_data['batch_size']
                        memory_diff = curr_data['peak_memory'] - prev_data['peak_memory']
                        
                        if batch_diff > 0:
                            per_sample = memory_diff / batch_diff
                            memory_per_sample.append(per_sample)
                    
                    if memory_per_sample:
                        avg_per_sample = np.mean(memory_per_sample)
                        print(f"  Average memory per sample: {avg_per_sample:.2f} MB")
                        
                        # 预测更大批次的内存使用
                        base_memory = memory_data[0]['peak_memory'] - memory_data[0]['batch_size'] * avg_per_sample
                        
                        print(f"  Base model memory: {base_memory:.1f} MB")
                        print(f"  Predicted memory for batch 64: {base_memory + 64 * avg_per_sample:.1f} MB")
                        print(f"  Predicted memory for batch 128: {base_memory + 128 * avg_per_sample:.1f} MB")
        
        else:
            print(f"❌ Model export failed: {pt_result.error_message}")
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """主函数"""
    print("🚀 SpeedVQA Performance Benchmark Demo")
    print("=" * 60)
    
    try:
        # 演示各种基准测试功能
        demonstrate_basic_performance_benchmark()
        demonstrate_comprehensive_benchmark()
        demonstrate_t4_performance_validation()
        demonstrate_memory_profiling()
        
        print("\n" + "=" * 60)
        print("🎉 All performance benchmark demos completed successfully!")
        print("\n📊 Key Features Demonstrated:")
        print("  ✓ Basic performance benchmarking")
        print("  ✓ Multi-format comparison (PyTorch, ONNX)")
        print("  ✓ T4 performance target validation")
        print("  ✓ Memory usage profiling and analysis")
        print("  ✓ Batch size optimization")
        print("  ✓ Consistency verification")
        print("  ✓ Automated report generation")
        
        print("\n📁 Check ./benchmark_reports/ for detailed reports and visualizations")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()