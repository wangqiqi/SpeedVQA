"""
ModelExporter单元测试

测试PyTorch、ONNX、TensorRT格式的模型导出功能。
"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import os

# 禁用Hugging Face模型下载
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from speedvqa.export.exporter import ModelExporter, ExportResult, ValidationResult, export_model
from speedvqa.models.speedvqa import SpeedVQAModel


class TestModelExporter:
    """ModelExporter测试类"""
    
    @pytest.fixture
    def config(self):
        """测试配置"""
        return {
            'model': {
                'name': 'speedvqa',
                'vision': {
                    'backbone': 'mobilenet_v3_small',
                    'pretrained': False,  # 测试时不使用预训练权重
                    'feature_dim': 512,
                    'dropout': 0.1
                },
                'text': {
                    'encoder': 'distilbert-base-uncased',
                    'max_length': 64,  # 减小以加快测试
                    'feature_dim': 384,
                    'freeze_encoder': True
                },
                'fusion': {
                    'method': 'concat',
                    'hidden_dim': 896,  # 512 + 384
                    'dropout': 0.3
                },
                'classifier': {
                    'hidden_dims': [256, 128],
                    'num_classes': 2,
                    'dropout': 0.2
                }
            },
            'data': {
                'image': {'size': [224, 224]},
            },
            'inference': {
                'device': 'cpu'  # 测试时使用CPU
            },
            'export': {
                'validation': {
                    'enabled': True,
                    'tolerance': 1e-3,
                    'num_samples': 2
                },
                'benchmark': {
                    'enabled': False  # 测试时禁用基准测试
                }
            }
        }
    
    @pytest.fixture
    def model(self, config):
        """测试模型"""
        model = SpeedVQAModel(config['model'])
        model.eval()
        return model
    
    @pytest.fixture
    def exporter(self, config):
        """ModelExporter实例"""
        return ModelExporter(config)
    
    @pytest.fixture
    def temp_dir(self):
        """临时目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_exporter_initialization(self, config):
        """测试导出器初始化"""
        exporter = ModelExporter(config)
        
        assert exporter.config == config
        assert exporter.device == torch.device('cpu')
        assert exporter.validation_config['enabled'] is True
        assert exporter.validation_config['tolerance'] == 1e-3
    
    def test_pytorch_export_success(self, exporter, model, temp_dir):
        """测试PyTorch格式导出成功"""
        save_path = Path(temp_dir) / 'test_model.pt'
        
        result = exporter.export_pytorch(model, str(save_path))
        
        assert result.success is True
        assert result.format == 'pytorch'
        assert result.model_size_mb > 0
        assert result.export_time_s > 0
        assert Path(result.export_path).exists()
        
        # 验证保存的内容
        checkpoint = torch.load(save_path, map_location='cpu')
        assert 'model_state_dict' in checkpoint
        assert 'model_config' in checkpoint
        assert 'model_architecture' in checkpoint
        assert checkpoint['model_architecture'] == 'SpeedVQA'
    
    def test_pytorch_export_with_optimizer(self, exporter, model, temp_dir):
        """测试包含优化器状态的PyTorch导出"""
        save_path = Path(temp_dir) / 'test_model_with_opt.pt'
        
        # 创建虚拟优化器状态
        optimizer_state = {
            'state': {},
            'param_groups': [{'lr': 0.001, 'weight_decay': 0.0005}]
        }
        
        result = exporter.export_pytorch(
            model, str(save_path), 
            include_optimizer=True, 
            optimizer_state=optimizer_state
        )
        
        assert result.success is True
        
        # 验证优化器状态被保存
        checkpoint = torch.load(save_path, map_location='cpu')
        assert 'optimizer_state_dict' in checkpoint
        assert checkpoint['optimizer_state_dict'] == optimizer_state
    
    def test_pytorch_export_validation(self, exporter, model, temp_dir):
        """测试PyTorch导出验证"""
        save_path = Path(temp_dir) / 'test_model_validation.pt'
        
        result = exporter.export_pytorch(model, str(save_path))
        
        assert result.success is True
        assert result.validation_result is not None
        
        validation = ValidationResult(**result.validation_result)
        assert validation.success is True
        assert validation.inference_time_ms > 0
        assert validation.numerical_accuracy > 0.99  # 应该几乎完全一致
    
    @patch('speedvqa.export.exporter.ONNX_AVAILABLE', True)
    @patch('onnx.load')
    @patch('onnx.checker.check_model')
    @patch('torch.onnx.export')
    def test_onnx_export_success(self, mock_export, mock_check, mock_load, 
                                exporter, model, temp_dir):
        """测试ONNX格式导出成功"""
        save_path = Path(temp_dir) / 'test_model.onnx'
        
        # 模拟ONNX导出成功
        mock_export.return_value = None
        mock_load.return_value = MagicMock()
        mock_check.return_value = None
        
        # 创建虚拟文件以模拟导出结果
        save_path.touch()
        save_path.write_bytes(b'fake_onnx_data' * 1000)  # 创建一些数据
        
        result = exporter.export_onnx(model, str(save_path))
        
        assert result.success is True
        assert result.format == 'onnx'
        assert result.model_size_mb > 0
        assert mock_export.called
        assert mock_check.called
    
    @patch('speedvqa.export.exporter.ONNX_AVAILABLE', False)
    def test_onnx_export_unavailable(self, exporter, model, temp_dir):
        """测试ONNX不可用时的处理"""
        save_path = Path(temp_dir) / 'test_model.onnx'
        
        result = exporter.export_onnx(model, str(save_path))
        
        assert result.success is False
        assert result.format == 'onnx'
        assert 'ONNX not available' in result.error_message
    
    @patch('speedvqa.export.exporter.TENSORRT_AVAILABLE', True)
    @patch('tensorrt.Builder')
    @patch('tensorrt.OnnxParser')
    def test_tensorrt_export_success(self, mock_parser, mock_builder, 
                                   exporter, temp_dir):
        """测试TensorRT格式导出成功"""
        onnx_path = Path(temp_dir) / 'test_model.onnx'
        trt_path = Path(temp_dir) / 'test_model.engine'
        
        # 创建虚拟ONNX文件
        onnx_path.write_bytes(b'fake_onnx_model')
        
        # 模拟TensorRT组件
        mock_engine = MagicMock()
        mock_engine.serialize.return_value = b'fake_trt_engine' * 1000
        
        mock_config = MagicMock()
        mock_network = MagicMock()
        mock_network.num_inputs = 3
        
        mock_builder_instance = MagicMock()
        mock_builder_instance.create_builder_config.return_value = mock_config
        mock_builder_instance.create_network.return_value = mock_network
        mock_builder_instance.build_engine.return_value = mock_engine
        mock_builder_instance.create_optimization_profile.return_value = MagicMock()
        
        mock_builder.return_value = mock_builder_instance
        
        mock_parser_instance = MagicMock()
        mock_parser_instance.parse.return_value = True
        mock_parser_instance.num_errors = 0
        mock_parser.return_value = mock_parser_instance
        
        result = exporter.export_tensorrt(str(onnx_path), str(trt_path))
        
        assert result.success is True
        assert result.format == 'tensorrt'
        assert result.model_size_mb > 0
        assert trt_path.exists()
    
    @patch('speedvqa.export.exporter.TENSORRT_AVAILABLE', False)
    def test_tensorrt_export_unavailable(self, exporter, temp_dir):
        """测试TensorRT不可用时的处理"""
        onnx_path = Path(temp_dir) / 'test_model.onnx'
        trt_path = Path(temp_dir) / 'test_model.engine'
        
        result = exporter.export_tensorrt(str(onnx_path), str(trt_path))
        
        assert result.success is False
        assert result.format == 'tensorrt'
        assert 'TensorRT not available' in result.error_message
    
    def test_tensorrt_export_missing_onnx(self, exporter, temp_dir):
        """测试ONNX文件不存在时的TensorRT导出"""
        onnx_path = Path(temp_dir) / 'nonexistent.onnx'
        trt_path = Path(temp_dir) / 'test_model.engine'
        
        with patch('speedvqa.export.exporter.TENSORRT_AVAILABLE', True):
            result = exporter.export_tensorrt(str(onnx_path), str(trt_path))
        
        assert result.success is False
        assert 'not found' in result.error_message
    
    def test_export_all_formats(self, exporter, model, temp_dir):
        """测试导出所有格式"""
        base_path = Path(temp_dir) / 'test_model'
        
        with patch('speedvqa.export.exporter.ONNX_AVAILABLE', False), \
             patch('speedvqa.export.exporter.TENSORRT_AVAILABLE', False):
            
            results = exporter.export_all_formats(model, str(base_path), ['pytorch'])
        
        assert 'pytorch' in results
        assert results['pytorch'].success is True
        assert Path(results['pytorch'].export_path).exists()
    
    def test_create_dummy_inputs(self, exporter):
        """测试创建虚拟输入数据"""
        input_shapes = {
            'image': (2, 3, 224, 224),
            'input_ids': (2, 128),
            'attention_mask': (2, 128)
        }
        
        dummy_inputs = exporter._create_dummy_inputs(input_shapes)
        
        assert 'image' in dummy_inputs
        assert 'input_ids' in dummy_inputs
        assert 'attention_mask' in dummy_inputs
        
        assert dummy_inputs['image'].shape == (2, 3, 224, 224)
        assert dummy_inputs['input_ids'].shape == (2, 128)
        assert dummy_inputs['attention_mask'].shape == (2, 128)
        
        # 检查数据类型和范围
        assert dummy_inputs['image'].dtype == torch.float32
        assert dummy_inputs['input_ids'].dtype == torch.long
        assert dummy_inputs['attention_mask'].dtype == torch.float32
        
        assert torch.all(dummy_inputs['input_ids'] >= 0)
        assert torch.all(dummy_inputs['input_ids'] < 30522)  # DistilBERT词汇表大小
        assert torch.all(dummy_inputs['attention_mask'] == 1)
    
    def test_validate_pytorch_export_success(self, exporter, model, temp_dir):
        """测试PyTorch导出验证成功"""
        save_path = Path(temp_dir) / 'test_model.pt'
        
        # 先导出模型
        export_result = exporter.export_pytorch(model, str(save_path))
        assert export_result.success
        
        # 手动验证
        validation_result = exporter._validate_pytorch_export(model, str(save_path))
        
        assert validation_result.success is True
        assert validation_result.inference_time_ms > 0
        assert validation_result.numerical_accuracy > 0.99
        assert validation_result.output_shape == (2, 2)  # batch_size=2, num_classes=2
    
    def test_validate_pytorch_export_failure(self, exporter, model, temp_dir):
        """测试PyTorch导出验证失败"""
        nonexistent_path = Path(temp_dir) / 'nonexistent.pt'
        
        validation_result = exporter._validate_pytorch_export(model, str(nonexistent_path))
        
        assert validation_result.success is False
        assert validation_result.error_message is not None
    
    @patch('speedvqa.export.exporter.ONNX_AVAILABLE', True)
    @patch('speedvqa.export.exporter.ort.InferenceSession')
    def test_validate_onnx_export(self, mock_session, exporter, model):
        """测试ONNX导出验证"""
        # 模拟ONNX Runtime会话
        mock_session_instance = MagicMock()
        mock_session_instance.get_inputs.return_value = [
            MagicMock(name='image'),
            MagicMock(name='input_ids'),
            MagicMock(name='attention_mask')
        ]
        mock_session.return_value = mock_session_instance

        # 创建测试输入
        test_inputs = {
            'image': torch.randn(2, 3, 224, 224),
            'input_ids': torch.randint(0, 1000, (2, 64)),
            'attention_mask': torch.ones(2, 64)
        }

        with torch.no_grad():
            ref_logits = model(test_inputs)['logits'].cpu().numpy()
        mock_session_instance.run.return_value = [ref_logits]

        validation_result = exporter._validate_onnx_export(
            model, 'fake_path.onnx', test_inputs
        )
        
        assert validation_result.success is True
        assert validation_result.inference_time_ms > 0
    
    def test_benchmark_pytorch_model(self, exporter, model, temp_dir):
        """测试PyTorch模型性能基准测试"""
        save_path = Path(temp_dir) / 'test_model.pt'
        
        # 导出模型
        export_result = exporter.export_pytorch(model, str(save_path))
        assert export_result.success
        
        # 创建测试输入
        test_inputs = {
            'image': torch.randn(1, 3, 224, 224),
            'input_ids': torch.randint(0, 1000, (1, 64)),
            'attention_mask': torch.ones(1, 64)
        }
        
        # 运行基准测试
        benchmark_result = exporter._benchmark_pytorch(str(save_path), test_inputs, 10)
        
        assert 'avg_inference_time_ms' in benchmark_result
        assert 'throughput_fps' in benchmark_result
        assert 'total_time_s' in benchmark_result
        
        assert benchmark_result['avg_inference_time_ms'] > 0
        assert benchmark_result['throughput_fps'] > 0
        assert benchmark_result['total_time_s'] > 0
    
    def test_export_model_function(self, model, temp_dir):
        """测试便捷导出函数"""
        config = {
            'model': model.config,
            'data': {'image': {'size': [224, 224]}},
            'inference': {'device': 'cpu'},
            'export': {'validation': {'enabled': True, 'tolerance': 1e-3}}
        }
        
        results = export_model(
            model=model,
            output_dir=temp_dir,
            model_name='test_speedvqa',
            config=config,
            formats=['pytorch']
        )
        
        assert 'pytorch' in results
        assert results['pytorch'].success is True
        
        expected_path = Path(temp_dir) / 'test_speedvqa.pt'
        assert expected_path.exists()


class TestExportResultDataClass:
    """ExportResult数据类测试"""
    
    def test_export_result_creation(self):
        """测试ExportResult创建"""
        result = ExportResult(
            success=True,
            export_path='/path/to/model.pt',
            format='pytorch',
            model_size_mb=15.5,
            export_time_s=2.3
        )
        
        assert result.success is True
        assert result.export_path == '/path/to/model.pt'
        assert result.format == 'pytorch'
        assert result.model_size_mb == 15.5
        assert result.export_time_s == 2.3
        assert result.error_message is None
        assert result.validation_result is None
    
    def test_export_result_with_error(self):
        """测试带错误的ExportResult"""
        result = ExportResult(
            success=False,
            export_path='/path/to/model.pt',
            format='onnx',
            model_size_mb=0.0,
            export_time_s=0.0,
            error_message='Export failed'
        )
        
        assert result.success is False
        assert result.error_message == 'Export failed'


class TestValidationResultDataClass:
    """ValidationResult数据类测试"""
    
    def test_validation_result_success(self):
        """测试成功的ValidationResult"""
        result = ValidationResult(
            success=True,
            inference_time_ms=25.5,
            output_shape=(1, 2),
            output_dtype='float32',
            numerical_accuracy=0.9999
        )
        
        assert result.success is True
        assert result.inference_time_ms == 25.5
        assert result.output_shape == (1, 2)
        assert result.output_dtype == 'float32'
        assert result.numerical_accuracy == 0.9999
        assert result.error_message is None
    
    def test_validation_result_failure(self):
        """测试失败的ValidationResult"""
        result = ValidationResult(
            success=False,
            inference_time_ms=0.0,
            output_shape=(0,),
            output_dtype='unknown',
            numerical_accuracy=0.0,
            error_message='Validation failed'
        )
        
        assert result.success is False
        assert result.error_message == 'Validation failed'


if __name__ == '__main__':
    pytest.main([__file__])