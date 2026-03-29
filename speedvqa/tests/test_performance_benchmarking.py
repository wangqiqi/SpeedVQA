"""
性能基准测试功能测试

测试ModelExporter的增强性能基准测试功能，包括：
- 推理速度测试
- 内存使用监控
- 结果一致性验证
- 性能比较分析
- 报告生成
"""

import tempfile
import shutil
from pathlib import Path
import pytest
import torch
import numpy as np
from unittest.mock import patch

from speedvqa.export.exporter import (
    ModelExporter,
    PerformanceBenchmarkResult,
    MemoryUsageResult,
    ConsistencyResult,
)
from speedvqa.models.speedvqa import SpeedVQAModel


class TestPerformanceBenchmarking:
    """性能基准测试功能测试"""
    
    @pytest.fixture
    def config(self):
        """测试配置（与真实 SpeedVQAModel / export 检查点格式一致）"""
        return {
            'model': {
                'name': 'speedvqa',
                'vision': {
                    'backbone': 'mobilenet_v3_small',
                    'pretrained': False,
                    'feature_dim': 512,
                    'dropout': 0.1,
                },
                'text': {
                    'encoder': 'distilbert-base-uncased',
                    'max_length': 64,
                    'feature_dim': 384,
                    'freeze_encoder': True,
                },
                'fusion': {
                    'method': 'concat',
                    'hidden_dim': 896,
                    'dropout': 0.3,
                },
                'classifier': {
                    'hidden_dims': [256, 128],
                    'num_classes': 2,
                    'dropout': 0.2,
                },
            },
            'data': {'image': {'size': [224, 224]}},
            'inference': {'device': 'cpu'},
            'export': {
                'validation': {'enabled': True, 'tolerance': 1e-3},
                'benchmark': {
                    'enabled': True,
                    'warmup_iterations': 2,
                    'test_iterations': 5,
                },
            },
        }

    @pytest.fixture
    def model(self, config):
        """测试模型"""
        m = SpeedVQAModel(config['model'])
        m.eval()
        return m
    
    @pytest.fixture
    def exporter(self, config):
        """模型导出器"""
        return ModelExporter(config)
    
    @pytest.fixture
    def temp_dir(self):
        """临时目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_create_test_inputs(self, exporter):
        """测试创建测试输入数据"""
        batch_size = 4
        test_inputs = exporter._create_test_inputs(batch_size)
        
        # 验证输入格式
        assert 'image' in test_inputs
        assert 'input_ids' in test_inputs
        assert 'attention_mask' in test_inputs
        
        # 验证形状
        assert test_inputs['image'].shape == (batch_size, 3, 224, 224)
        assert test_inputs['input_ids'].shape == (batch_size, 64)  # max_length from config
        assert test_inputs['attention_mask'].shape == (batch_size, 64)
        
        # 验证数据类型
        assert test_inputs['image'].dtype == torch.float32
        assert test_inputs['input_ids'].dtype == torch.long
        assert test_inputs['attention_mask'].dtype == torch.float32
    
    def test_benchmark_pytorch_detailed(self, exporter, model, temp_dir):
        """测试详细的PyTorch性能基准测试"""
        # 导出模型
        save_path = Path(temp_dir) / 'test_model.pt'
        export_result = exporter.export_pytorch(model, str(save_path))
        assert export_result.success
        
        # 创建测试输入
        test_inputs = exporter._create_test_inputs(2)
        
        # 运行性能基准测试
        benchmark_result = exporter._benchmark_pytorch_detailed(
            str(save_path), test_inputs, num_iterations=5, warmup_iterations=2
        )
        
        # 验证结果
        assert isinstance(benchmark_result, PerformanceBenchmarkResult)
        assert benchmark_result.format_name == 'pytorch'
        assert benchmark_result.avg_inference_time_ms > 0
        assert benchmark_result.min_inference_time_ms > 0
        assert benchmark_result.max_inference_time_ms >= benchmark_result.min_inference_time_ms
        assert benchmark_result.throughput_fps > 0
        assert benchmark_result.total_time_s > 0
        assert benchmark_result.num_iterations == 5
        assert benchmark_result.batch_size == 2
        assert benchmark_result.warmup_iterations == 2
        assert benchmark_result.consistency_score == 1.0  # 自己与自己比较
        assert benchmark_result.error_message is None
    
    def test_run_memory_benchmark(self, exporter, model, temp_dir):
        """测试内存使用基准测试"""
        # 导出模型
        save_path = Path(temp_dir) / 'memory_test_model.pt'
        export_result = exporter.export_pytorch(model, str(save_path))
        assert export_result.success
        
        # 创建测试输入
        test_inputs = exporter._create_test_inputs(2)
        
        # 运行内存基准测试
        memory_result = exporter._run_memory_benchmark(
            'pytorch', str(save_path), test_inputs
        )
        
        # 验证结果
        assert isinstance(memory_result, MemoryUsageResult)
        assert memory_result.peak_memory_mb >= 0
        assert memory_result.avg_memory_mb >= 0
        assert len(memory_result.memory_timeline) > 0
        
        # 验证内存时间线格式
        for timestamp, memory_mb in memory_result.memory_timeline:
            assert isinstance(timestamp, float)
            assert isinstance(memory_mb, float)
            assert timestamp >= 0
            assert memory_mb >= 0
    
    def test_get_inference_outputs(self, exporter, model, temp_dir):
        """测试获取推理输出"""
        # 导出模型
        save_path = Path(temp_dir) / 'inference_test_model.pt'
        export_result = exporter.export_pytorch(model, str(save_path))
        assert export_result.success
        
        # 创建测试输入
        test_inputs = exporter._create_test_inputs(2)
        
        # 获取推理输出
        outputs = exporter._get_inference_outputs(
            'pytorch', str(save_path), test_inputs
        )
        
        # 验证输出
        assert outputs is not None
        assert isinstance(outputs, torch.Tensor)
        assert outputs.shape == (2, 2)  # batch_size=2, num_classes=2
        assert torch.all(torch.isfinite(outputs))
    
    def test_calculate_consistency(self, exporter):
        """测试结果一致性计算"""
        # 创建模拟输出
        batch_size = 4
        num_classes = 2
        
        # 参考输出
        reference_outputs = torch.randn(batch_size, num_classes)
        
        # 完全一致的输出
        identical_outputs = reference_outputs.clone()
        consistency = exporter._calculate_consistency(
            reference_outputs, identical_outputs, 'pytorch', 'pytorch_copy'
        )
        
        assert isinstance(consistency, ConsistencyResult)
        assert consistency.reference_format == 'pytorch'
        assert consistency.compared_format == 'pytorch_copy'
        assert consistency.max_difference < 1e-6
        assert consistency.mean_difference < 1e-6
        assert consistency.consistency_score > 0.99
        assert consistency.prediction_match_rate == 1.0
        assert consistency.num_samples == batch_size
        assert consistency.error_message is None
        
        # 略有差异的输出
        noisy_outputs = reference_outputs + torch.randn_like(reference_outputs) * 0.01
        consistency_noisy = exporter._calculate_consistency(
            reference_outputs, noisy_outputs, 'pytorch', 'onnx'
        )
        
        assert consistency_noisy.consistency_score < 1.0
        assert consistency_noisy.consistency_score > 0.8  # 应该仍然相对一致
        assert consistency_noisy.max_difference > 0
        
        # 形状不匹配的输出
        wrong_shape_outputs = torch.randn(batch_size, num_classes + 1)
        consistency_wrong = exporter._calculate_consistency(
            reference_outputs, wrong_shape_outputs, 'pytorch', 'wrong'
        )
        
        assert consistency_wrong.consistency_score == 0.0
        assert consistency_wrong.error_message is not None
        assert 'Shape mismatch' in consistency_wrong.error_message
    
    def test_comprehensive_benchmark(self, exporter, model, temp_dir):
        """测试全面的性能基准测试"""
        # 导出模型
        save_path = Path(temp_dir) / 'comprehensive_test_model.pt'
        export_result = exporter.export_pytorch(model, str(save_path))
        assert export_result.success
        
        # 运行全面基准测试
        model_paths = {'pytorch': str(save_path)}
        benchmark_results = exporter.benchmark_exported_models(
            model_paths,
            num_iterations=5,
            batch_sizes=[1, 2],
            warmup_iterations=2,
            reference_format='pytorch'
        )
        
        # 验证结果结构
        assert 'summary' in benchmark_results
        assert 'detailed_results' in benchmark_results
        assert 'consistency_results' in benchmark_results
        assert 'performance_comparison' in benchmark_results
        assert 'recommendations' in benchmark_results
        
        # 验证摘要信息
        summary = benchmark_results['summary']
        assert summary['total_formats'] == 1
        assert summary['batch_sizes_tested'] == [1, 2]
        assert summary['iterations_per_test'] == 5
        assert summary['warmup_iterations'] == 2
        assert summary['reference_format'] == 'pytorch'
        
        # 验证详细结果
        detailed_results = benchmark_results['detailed_results']
        assert 'batch_1' in detailed_results
        assert 'batch_2' in detailed_results
        
        for batch_key, batch_results in detailed_results.items():
            assert 'pytorch' in batch_results
            pytorch_result = batch_results['pytorch']
            
            assert 'performance' in pytorch_result
            assert 'memory' in pytorch_result
            assert 'outputs' in pytorch_result
            
            # 验证性能结果
            perf = pytorch_result['performance']
            assert isinstance(perf, PerformanceBenchmarkResult)
            assert perf.avg_inference_time_ms > 0
            assert perf.throughput_fps > 0
            
            # 验证内存结果
            memory = pytorch_result['memory']
            assert isinstance(memory, MemoryUsageResult)
            assert memory.peak_memory_mb >= 0
        
        # 验证建议
        recommendations = benchmark_results['recommendations']
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
    
    def test_performance_comparison_generation(self, exporter):
        """测试性能比较分析生成"""
        # 创建模拟的详细结果
        detailed_results = {
            'batch_1': {
                'pytorch': {
                    'performance': PerformanceBenchmarkResult(
                        format_name='pytorch',
                        avg_inference_time_ms=10.0,
                        min_inference_time_ms=8.0,
                        max_inference_time_ms=12.0,
                        std_inference_time_ms=1.0,
                        throughput_fps=100.0,
                        peak_memory_mb=50.0,
                        avg_memory_mb=45.0,
                        total_time_s=0.05,
                        num_iterations=5,
                        batch_size=1,
                        warmup_iterations=2,
                        consistency_score=1.0
                    )
                },
                'onnx': {
                    'performance': PerformanceBenchmarkResult(
                        format_name='onnx',
                        avg_inference_time_ms=8.0,
                        min_inference_time_ms=7.0,
                        max_inference_time_ms=9.0,
                        std_inference_time_ms=0.5,
                        throughput_fps=125.0,
                        peak_memory_mb=40.0,
                        avg_memory_mb=38.0,
                        total_time_s=0.04,
                        num_iterations=5,
                        batch_size=1,
                        warmup_iterations=2,
                        consistency_score=0.95
                    )
                }
            }
        }
        
        # 生成性能比较
        comparison = exporter._generate_performance_comparison(detailed_results)
        
        # 验证比较结果
        assert 'batch_size_analysis' in comparison
        assert 'overall_ranking' in comparison
        
        batch_analysis = comparison['batch_size_analysis']
        assert 1 in batch_analysis
        
        batch_1_analysis = batch_analysis[1]
        assert 'speed_ranking' in batch_1_analysis
        assert 'memory_ranking' in batch_1_analysis
        assert 'throughput_ranking' in batch_1_analysis
        assert 'metrics' in batch_1_analysis
        
        # ONNX应该在速度和吞吐量上排名更高
        assert batch_1_analysis['speed_ranking'][0] == 'onnx'  # 更快
        assert batch_1_analysis['throughput_ranking'][0] == 'onnx'  # 更高吞吐量
        assert batch_1_analysis['memory_ranking'][0] == 'onnx'  # 更少内存
    
    def test_recommendations_generation(self, exporter):
        """测试性能优化建议生成"""
        # 创建模拟的基准测试结果
        benchmark_results = {
            'performance_comparison': {
                'overall_ranking': [('onnx', 2.5), ('pytorch', 1.5)]
            },
            'detailed_results': {
                'batch_1': {
                    'pytorch': {
                        'performance': PerformanceBenchmarkResult(
                            format_name='pytorch',
                            avg_inference_time_ms=30.0,  # 满足T4目标
                            min_inference_time_ms=25.0,
                            max_inference_time_ms=35.0,
                            std_inference_time_ms=2.0,
                            throughput_fps=33.3,
                            peak_memory_mb=100.0,
                            avg_memory_mb=95.0,
                            total_time_s=0.15,
                            num_iterations=5,
                            batch_size=1,
                            warmup_iterations=2,
                            consistency_score=1.0
                        )
                    }
                },
                'batch_4': {
                    'pytorch': {
                        'performance': PerformanceBenchmarkResult(
                            format_name='pytorch',
                            avg_inference_time_ms=80.0,  # 超过T4目标
                            min_inference_time_ms=75.0,
                            max_inference_time_ms=85.0,
                            std_inference_time_ms=3.0,
                            throughput_fps=50.0,
                            peak_memory_mb=400.0,
                            avg_memory_mb=380.0,
                            total_time_s=0.32,
                            num_iterations=5,
                            batch_size=4,
                            warmup_iterations=2,
                            consistency_score=1.0
                        )
                    }
                }
            },
            'consistency_results': {}
        }
        
        # 生成建议
        recommendations = exporter._generate_recommendations(benchmark_results)
        
        # 验证建议
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # 应该包含最佳格式建议
        best_format_rec = [rec for rec in recommendations if 'Overall best performing format' in rec]
        assert len(best_format_rec) > 0
        assert 'ONNX' in best_format_rec[0]
        
        # 应该包含T4性能分析
        t4_recs = [rec for rec in recommendations if 'T4 target' in rec]
        assert len(t4_recs) >= 1
        
        # 应该有满足目标和不满足目标的建议
        meets_target = [rec for rec in t4_recs if 'meets T4 target' in rec]
        exceeds_target = [rec for rec in t4_recs if 'No format meets T4 target' in rec]
        
        # 批次1应该满足目标，批次4应该不满足
        assert len(meets_target) >= 1 or len(exceeds_target) >= 1
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_benchmark_report_generation(self, mock_close, mock_savefig, exporter, model, temp_dir):
        """测试基准测试报告生成"""
        # 设置报告目录
        report_dir = Path(temp_dir) / 'reports'
        exporter.config['export']['benchmark_report_dir'] = str(report_dir)
        
        # 导出模型
        save_path = Path(temp_dir) / 'report_test_model.pt'
        export_result = exporter.export_pytorch(model, str(save_path))
        assert export_result.success
        
        # 运行基准测试（会自动生成报告）
        model_paths = {'pytorch': str(save_path)}
        exporter.benchmark_exported_models(
            model_paths,
            num_iterations=3,
            batch_sizes=[1, 2],
            warmup_iterations=1,
        )
        
        # 验证报告目录被创建
        assert report_dir.exists()
        
        # 验证JSON报告文件被创建
        json_files = list(report_dir.glob('performance_benchmark_*.json'))
        assert len(json_files) > 0
        
        # 验证JSON报告内容
        import json
        with open(json_files[0], 'r') as f:
            report_data = json.load(f)
        
        assert 'summary' in report_data
        assert 'detailed_results' in report_data
        assert 'recommendations' in report_data
        
        # 验证图表生成被调用
        assert mock_savefig.called
        assert mock_close.called
    
    def test_serialization_helper(self, exporter):
        """测试序列化辅助函数"""
        # 创建包含各种类型的测试对象
        test_obj = {
            'string': 'test',
            'int': 42,
            'float': 3.14,
            'list': [1, 2, 3],
            'numpy_int': np.int64(100),
            'numpy_float': np.float32(2.71),
            'numpy_array': np.array([1, 2, 3]),
            'nested': {
                'inner_list': [np.float64(1.5), 'text'],
                'inner_dict': {'key': np.int32(999)}
            },
            'benchmark_result': PerformanceBenchmarkResult(
                format_name='test',
                avg_inference_time_ms=10.0,
                min_inference_time_ms=8.0,
                max_inference_time_ms=12.0,
                std_inference_time_ms=1.0,
                throughput_fps=100.0,
                peak_memory_mb=50.0,
                avg_memory_mb=45.0,
                total_time_s=0.05,
                num_iterations=5,
                batch_size=1,
                warmup_iterations=2,
                consistency_score=1.0
            )
        }
        
        # 序列化
        serialized = exporter._make_serializable(test_obj)
        
        # 验证序列化结果
        assert isinstance(serialized, dict)
        assert serialized['string'] == 'test'
        assert serialized['int'] == 42
        assert isinstance(serialized['numpy_int'], float)
        assert isinstance(serialized['numpy_float'], float)
        assert isinstance(serialized['numpy_array'], list)
        assert isinstance(serialized['nested']['inner_list'], list)
        assert isinstance(serialized['benchmark_result'], dict)
        
        # 验证可以JSON序列化
        import json
        json_str = json.dumps(serialized)
        assert isinstance(json_str, str)
        
        # 验证可以反序列化
        deserialized = json.loads(json_str)
        assert isinstance(deserialized, dict)
    
    def test_error_handling_in_benchmark(self, exporter, temp_dir):
        """测试基准测试中的错误处理"""
        # 测试不存在的模型文件
        non_existent_paths = {'pytorch': '/non/existent/path.pt'}
        
        benchmark_results = exporter.benchmark_exported_models(
            non_existent_paths,
            num_iterations=5,
            batch_sizes=[1],
            warmup_iterations=2
        )
        
        # 应该返回错误信息而不是崩溃
        assert 'error' in benchmark_results
        assert benchmark_results['error'] == 'No valid model files found'
        
        # 测试无效的模型文件
        invalid_model_path = Path(temp_dir) / 'invalid_model.pt'
        invalid_model_path.write_text('invalid content')
        
        invalid_paths = {'pytorch': str(invalid_model_path)}
        
        benchmark_results = exporter.benchmark_exported_models(
            invalid_paths,
            num_iterations=5,
            batch_sizes=[1],
            warmup_iterations=2
        )
        
        # 应该在详细结果中包含错误信息
        assert 'detailed_results' in benchmark_results
        
        for batch_results in benchmark_results['detailed_results'].values():
            if 'pytorch' in batch_results:
                pytorch_result = batch_results['pytorch']
                # 应该有错误信息
                assert 'error' in pytorch_result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])