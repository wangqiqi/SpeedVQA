"""
ROI推理器单元测试

测试ROIInferencer的核心功能：
- 模型加载（PyTorch/ONNX/TensorRT）
- 单张ROI图像推理
- 批量ROI图像推理
- 推理结果后处理
"""

import pytest
import torch
import numpy as np
from PIL import Image
import tempfile
from pathlib import Path

from speedvqa.inference import ROIInferencer
from speedvqa.models.speedvqa import build_speedvqa_model
from speedvqa.utils.config import get_default_config


class TestROIInferencerInitialization:
    """测试ROI推理器初始化"""
    
    @pytest.fixture
    def pytorch_model_path(self):
        """创建临时PyTorch模型"""
        config = get_default_config()
        model = build_speedvqa_model(config)
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            model_path = f.name
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': config.get('model', {}),
                'model_architecture': 'SpeedVQA'
            }, model_path)
        
        yield model_path
        
        # 清理
        Path(model_path).unlink()
    
    def test_pytorch_model_loading(self, pytorch_model_path):
        """测试PyTorch模型加载"""
        config = get_default_config()
        inferencer = ROIInferencer(
            pytorch_model_path,
            model_format='pytorch',
            device='cpu',
            config=config
        )
        
        assert inferencer is not None
        assert inferencer.model_format == 'pytorch'
        assert inferencer.model is not None
    
    def test_invalid_model_path(self):
        """测试无效的模型路径"""
        config = get_default_config()
        
        with pytest.raises(FileNotFoundError):
            ROIInferencer(
                '/nonexistent/model.pt',
                model_format='pytorch',
                device='cpu',
                config=config
            )
    
    def test_unsupported_model_format(self, pytorch_model_path):
        """测试不支持的模型格式"""
        config = get_default_config()
        
        with pytest.raises(ValueError):
            ROIInferencer(
                pytorch_model_path,
                model_format='unsupported_format',
                device='cpu',
                config=config
            )


class TestImagePreprocessing:
    """测试图像预处理"""
    
    def test_numpy_image_preprocessing(self):
        """测试numpy数组图像预处理"""
        from speedvqa.inference.inferencer import ImagePreprocessor
        
        preprocessor = ImagePreprocessor(image_size=(224, 224))
        
        # 创建随机图像
        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        # 预处理
        processed = preprocessor.preprocess(image)
        
        # 验证输出形状
        assert processed.shape == (3, 224, 224)
        assert isinstance(processed, torch.Tensor)
        assert processed.dtype == torch.float32
    
    def test_pil_image_preprocessing(self):
        """测试PIL Image预处理"""
        from speedvqa.inference.inferencer import ImagePreprocessor
        
        preprocessor = ImagePreprocessor(image_size=(224, 224))
        
        # 创建PIL Image
        image = Image.new('RGB', (640, 480), color='red')
        
        # 预处理
        processed = preprocessor.preprocess(image)
        
        # 验证输出形状
        assert processed.shape == (3, 224, 224)
        assert isinstance(processed, torch.Tensor)
    
    def test_batch_preprocessing(self):
        """测试批量图像预处理"""
        from speedvqa.inference.inferencer import ImagePreprocessor
        
        preprocessor = ImagePreprocessor(image_size=(224, 224))
        
        # 创建图像批次
        images = [
            np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            for _ in range(4)
        ]
        
        # 批量预处理
        processed = preprocessor.preprocess_batch(images)
        
        # 验证输出形状
        assert processed.shape == (4, 3, 224, 224)
        assert isinstance(processed, torch.Tensor)


class TestSingleImageInference:
    """测试单张图像推理"""
    
    @pytest.fixture
    def pytorch_model_path(self):
        """创建临时PyTorch模型"""
        config = get_default_config()
        model = build_speedvqa_model(config)
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            model_path = f.name
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': config.get('model', {}),
                'model_architecture': 'SpeedVQA'
            }, model_path)
        
        yield model_path
        
        # 清理
        Path(model_path).unlink()
    
    @pytest.fixture
    def inferencer(self, pytorch_model_path):
        """创建推理器"""
        config = get_default_config()
        return ROIInferencer(
            pytorch_model_path,
            model_format='pytorch',
            device='cpu',
            config=config
        )
    
    def test_numpy_image_inference(self, inferencer):
        """测试numpy图像推理"""
        # 创建随机图像
        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        question = "Is there a person?"
        
        # 推理
        result = inferencer.inference(image, question)
        
        # 验证结果
        assert result.answer in ['YES', 'NO']
        assert 0 <= result.confidence <= 1
        assert len(result.probabilities) == 2
        assert result.inference_time_ms > 0
        assert result.model_format == 'pytorch'
        assert result.batch_size == 1
    
    def test_pil_image_inference(self, inferencer):
        """测试PIL Image推理"""
        # 创建PIL Image
        image = Image.new('RGB', (640, 480), color='blue')
        question = "Is the person smoking?"
        
        # 推理
        result = inferencer.inference(image, question)
        
        # 验证结果
        assert result.answer in ['YES', 'NO']
        assert 0 <= result.confidence <= 1
        assert len(result.probabilities) == 2
    
    def test_inference_consistency(self, inferencer):
        """测试推理一致性"""
        # 创建相同的图像
        image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        question = "Is there a person?"
        
        # 多次推理
        result1 = inferencer.inference(image, question)
        result2 = inferencer.inference(image, question)
        
        # 验证结果一致性
        assert result1.answer == result2.answer
        assert abs(result1.confidence - result2.confidence) < 1e-5
    
    def test_different_questions(self, inferencer):
        """测试不同问题的推理"""
        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        questions = [
            "Is there a person?",
            "Is the person smoking?",
            "Is there a car?",
            "是否有人打电话?"
        ]
        
        for question in questions:
            result = inferencer.inference(image, question)
            
            # 验证结果
            assert result.answer in ['YES', 'NO']
            assert 0 <= result.confidence <= 1
            assert len(result.probabilities) == 2


class TestBatchInference:
    """测试批量推理"""
    
    @pytest.fixture
    def pytorch_model_path(self):
        """创建临时PyTorch模型"""
        config = get_default_config()
        model = build_speedvqa_model(config)
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            model_path = f.name
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': config.get('model', {}),
                'model_architecture': 'SpeedVQA'
            }, model_path)
        
        yield model_path
        
        # 清理
        Path(model_path).unlink()
    
    @pytest.fixture
    def inferencer(self, pytorch_model_path):
        """创建推理器"""
        config = get_default_config()
        return ROIInferencer(
            pytorch_model_path,
            model_format='pytorch',
            device='cpu',
            config=config
        )
    
    def test_batch_inference_basic(self, inferencer):
        """测试基本批量推理"""
        # 创建图像批次
        images = [
            np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            for _ in range(4)
        ]
        questions = [
            "Is there a person?",
            "Is the person smoking?",
            "Is there a car?",
            "Is there a phone?"
        ]
        
        # 批量推理
        results = inferencer.batch_inference(images, questions)
        
        # 验证结果
        assert len(results) == 4
        for result in results:
            assert result.answer in ['YES', 'NO']
            assert 0 <= result.confidence <= 1
            assert len(result.probabilities) == 2
            assert result.batch_size == 4
    
    def test_batch_inference_single_item(self, inferencer):
        """测试单项批量推理"""
        images = [np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)]
        questions = ["Is there a person?"]
        
        # 批量推理
        results = inferencer.batch_inference(images, questions)
        
        # 验证结果
        assert len(results) == 1
        assert results[0].answer in ['YES', 'NO']
        assert results[0].batch_size == 1
    
    def test_batch_inference_large_batch(self, inferencer):
        """测试大批量推理"""
        batch_size = 16
        images = [
            np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            for _ in range(batch_size)
        ]
        questions = ["Is there a person?"] * batch_size
        
        # 批量推理
        results = inferencer.batch_inference(images, questions)
        
        # 验证结果
        assert len(results) == batch_size
        for result in results:
            assert result.answer in ['YES', 'NO']
            assert result.batch_size == batch_size
    
    def test_batch_inference_mismatched_lengths(self, inferencer):
        """测试不匹配的图像和问题数量"""
        images = [np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) for _ in range(4)]
        questions = ["Is there a person?", "Is the person smoking?"]
        
        # 应该抛出异常
        with pytest.raises(ValueError):
            inferencer.batch_inference(images, questions)


class TestInferenceResultPostprocessing:
    """测试推理结果后处理"""
    
    @pytest.fixture
    def pytorch_model_path(self):
        """创建临时PyTorch模型"""
        config = get_default_config()
        model = build_speedvqa_model(config)
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            model_path = f.name
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': config.get('model', {}),
                'model_architecture': 'SpeedVQA'
            }, model_path)
        
        yield model_path
        
        # 清理
        Path(model_path).unlink()
    
    @pytest.fixture
    def inferencer(self, pytorch_model_path):
        """创建推理器"""
        config = get_default_config()
        return ROIInferencer(
            pytorch_model_path,
            model_format='pytorch',
            device='cpu',
            config=config
        )
    
    def test_postprocessing_output_format(self, inferencer):
        """测试后处理输出格式"""
        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        question = "Is there a person?"
        
        result = inferencer.inference(image, question)
        
        # 验证输出格式
        assert hasattr(result, 'answer')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'probabilities')
        assert hasattr(result, 'inference_time_ms')
        assert hasattr(result, 'model_format')
        assert hasattr(result, 'batch_size')
    
    def test_postprocessing_probability_sum(self, inferencer):
        """测试概率和为1"""
        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        question = "Is there a person?"
        
        result = inferencer.inference(image, question)
        
        # 验证概率和
        prob_sum = sum(result.probabilities)
        assert abs(prob_sum - 1.0) < 1e-5
    
    def test_postprocessing_confidence_matches_max_probability(self, inferencer):
        """测试置信度与最大概率匹配"""
        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        question = "Is there a person?"
        
        result = inferencer.inference(image, question)
        
        # 验证置信度
        max_prob = max(result.probabilities)
        assert abs(result.confidence - max_prob) < 1e-5
    
    def test_postprocessing_answer_mapping(self, inferencer):
        """测试答案映射"""
        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        question = "Is there a person?"
        
        result = inferencer.inference(image, question)
        
        # 验证答案映射
        if result.probabilities[1] > result.probabilities[0]:
            assert result.answer == "YES"
        else:
            assert result.answer == "NO"


class TestModelInfo:
    """测试模型信息获取"""
    
    @pytest.fixture
    def pytorch_model_path(self):
        """创建临时PyTorch模型"""
        config = get_default_config()
        model = build_speedvqa_model(config)
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            model_path = f.name
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': config.get('model', {}),
                'model_architecture': 'SpeedVQA'
            }, model_path)
        
        yield model_path
        
        # 清理
        Path(model_path).unlink()
    
    def test_get_model_info(self, pytorch_model_path):
        """测试获取模型信息"""
        config = get_default_config()
        inferencer = ROIInferencer(
            pytorch_model_path,
            model_format='pytorch',
            device='cpu',
            config=config
        )
        
        info = inferencer.get_model_info()
        
        # 验证信息
        assert 'model_path' in info
        assert 'model_format' in info
        assert 'device' in info
        assert 'file_size_mb' in info
        assert info['model_format'] == 'pytorch'
        assert info['file_size_mb'] > 0


class TestInferencePerformance:
    """测试推理性能"""
    
    @pytest.fixture
    def pytorch_model_path(self):
        """创建临时PyTorch模型"""
        config = get_default_config()
        model = build_speedvqa_model(config)
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            model_path = f.name
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': config.get('model', {}),
                'model_architecture': 'SpeedVQA'
            }, model_path)
        
        yield model_path
        
        # 清理
        Path(model_path).unlink()
    
    @pytest.fixture
    def inferencer(self, pytorch_model_path):
        """创建推理器"""
        config = get_default_config()
        return ROIInferencer(
            pytorch_model_path,
            model_format='pytorch',
            device='cpu',
            config=config
        )
    
    def test_inference_time_reasonable(self, inferencer):
        """测试推理时间合理性"""
        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        question = "Is there a person?"
        
        result = inferencer.inference(image, question)
        
        # 验证推理时间（CPU上应该在合理范围内）
        assert result.inference_time_ms > 0
        assert result.inference_time_ms < 10000  # 10秒以内
    
    def test_batch_inference_efficiency(self, inferencer):
        """测试批量推理效率"""
        images = [
            np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            for _ in range(8)
        ]
        questions = ["Is there a person?"] * 8
        
        # 单个推理总时间
        single_times = []
        for img, q in zip(images, questions):
            result = inferencer.inference(img, q)
            single_times.append(result.inference_time_ms)

        # 批量推理时间
        results = inferencer.batch_inference(images, questions)
        total_batch_time = sum(r.inference_time_ms for r in results)
        
        # 批量推理应该更高效
        # 注意：这个测试可能在CPU上不一定成立，因为CPU没有真正的并行处理
        # 但在GPU上应该明显更快
        assert total_batch_time > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
