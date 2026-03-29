"""
ROI推理器简化单元测试

测试ROIInferencer的核心功能，不依赖完整的模型加载
"""

import pytest
import torch
import numpy as np
from PIL import Image

from speedvqa.inference.inferencer import ImagePreprocessor, InferenceResult


class TestImagePreprocessor:
    """测试图像预处理器"""
    
    def test_preprocessor_initialization(self):
        """测试预处理器初始化"""
        preprocessor = ImagePreprocessor(image_size=(224, 224))
        
        assert preprocessor.image_size == (224, 224)
        assert preprocessor.mean is not None
        assert preprocessor.std is not None
    
    def test_numpy_image_preprocessing(self):
        """测试numpy数组图像预处理"""
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
    
    def test_preprocessing_normalization(self):
        """测试预处理标准化"""
        preprocessor = ImagePreprocessor(image_size=(224, 224))
        
        # 创建全白图像
        image = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        # 预处理
        processed = preprocessor.preprocess(image)
        
        # 验证标准化
        # 白色像素 (1.0) 标准化后应该是 (1.0 - mean) / std
        # 对于ImageNet标准化，应该是正值
        assert processed.max() > 0
    
    def test_preprocessing_different_sizes(self):
        """测试不同大小的图像预处理"""
        preprocessor = ImagePreprocessor(image_size=(224, 224))
        
        sizes = [(480, 640), (1080, 1920), (100, 100)]
        
        for h, w in sizes:
            image = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            processed = preprocessor.preprocess(image)
            
            # 所有图像都应该被调整到目标大小
            assert processed.shape == (3, 224, 224)


class TestInferenceResult:
    """测试推理结果数据结构"""
    
    def test_inference_result_creation(self):
        """测试推理结果创建"""
        result = InferenceResult(
            answer='YES',
            confidence=0.95,
            probabilities=[0.05, 0.95],
            inference_time_ms=25.5,
            model_format='pytorch'
        )
        
        assert result.answer == 'YES'
        assert result.confidence == 0.95
        assert len(result.probabilities) == 2
        assert result.inference_time_ms == 25.5
        assert result.model_format == 'pytorch'
        assert result.batch_size == 1
    
    def test_inference_result_fields(self):
        """测试推理结果字段"""
        result = InferenceResult(
            answer='NO',
            confidence=0.72,
            probabilities=[0.72, 0.28],
            inference_time_ms=30.0,
            model_format='onnx',
            batch_size=4
        )
        
        # 验证所有字段
        assert hasattr(result, 'answer')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'probabilities')
        assert hasattr(result, 'inference_time_ms')
        assert hasattr(result, 'model_format')
        assert hasattr(result, 'batch_size')
        
        assert result.batch_size == 4


class TestImagePreprocessorEdgeCases:
    """测试图像预处理的边界情况"""
    
    def test_small_image_preprocessing(self):
        """测试小图像预处理"""
        preprocessor = ImagePreprocessor(image_size=(224, 224))
        
        # 创建很小的图像
        image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        
        # 预处理
        processed = preprocessor.preprocess(image)
        
        # 应该被放大到目标大小
        assert processed.shape == (3, 224, 224)
    
    def test_large_image_preprocessing(self):
        """测试大图像预处理"""
        preprocessor = ImagePreprocessor(image_size=(224, 224))
        
        # 创建很大的图像
        image = np.random.randint(0, 256, (4096, 4096, 3), dtype=np.uint8)
        
        # 预处理
        processed = preprocessor.preprocess(image)
        
        # 应该被缩小到目标大小
        assert processed.shape == (3, 224, 224)
    
    def test_non_square_image_preprocessing(self):
        """测试非正方形图像预处理"""
        preprocessor = ImagePreprocessor(image_size=(224, 224))
        
        # 创建非正方形图像
        image = np.random.randint(0, 256, (100, 800, 3), dtype=np.uint8)
        
        # 预处理
        processed = preprocessor.preprocess(image)
        
        # 应该被调整到目标大小
        assert processed.shape == (3, 224, 224)
    
    def test_grayscale_image_preprocessing(self):
        """测试灰度图像预处理"""
        preprocessor = ImagePreprocessor(image_size=(224, 224))
        
        # 创建灰度图像
        image = Image.new('L', (640, 480), color=128)
        
        # 转换为RGB
        image = image.convert('RGB')
        
        # 预处理
        processed = preprocessor.preprocess(image)
        
        # 应该成功处理
        assert processed.shape == (3, 224, 224)


class TestImagePreprocessorConsistency:
    """测试图像预处理的一致性"""
    
    def test_preprocessing_consistency(self):
        """测试预处理一致性"""
        preprocessor = ImagePreprocessor(image_size=(224, 224))
        
        # 创建相同的图像
        image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        
        # 多次预处理
        processed1 = preprocessor.preprocess(image)
        processed2 = preprocessor.preprocess(image)
        
        # 应该完全相同
        assert torch.allclose(processed1, processed2)
    
    def test_batch_vs_individual_preprocessing(self):
        """测试批量预处理与单个预处理的一致性"""
        preprocessor = ImagePreprocessor(image_size=(224, 224))
        
        # 创建图像
        images = [
            np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            for _ in range(4)
        ]
        
        # 单个预处理
        individual_results = [preprocessor.preprocess(img) for img in images]
        individual_batch = torch.stack(individual_results)
        
        # 批量预处理
        batch_result = preprocessor.preprocess_batch(images)
        
        # 应该相同
        assert torch.allclose(individual_batch, batch_result)


class TestInferenceResultValidation:
    """测试推理结果验证"""
    
    def test_valid_answer_values(self):
        """测试有效的答案值"""
        valid_answers = ['YES', 'NO']
        
        for answer in valid_answers:
            result = InferenceResult(
                answer=answer,
                confidence=0.5,
                probabilities=[0.5, 0.5],
                inference_time_ms=10.0,
                model_format='pytorch'
            )
            assert result.answer in valid_answers
    
    def test_confidence_range(self):
        """测试置信度范围"""
        confidences = [0.0, 0.5, 1.0]
        
        for conf in confidences:
            result = InferenceResult(
                answer='YES',
                confidence=conf,
                probabilities=[1-conf, conf],
                inference_time_ms=10.0,
                model_format='pytorch'
            )
            assert 0 <= result.confidence <= 1
    
    def test_probability_sum(self):
        """测试概率和"""
        result = InferenceResult(
            answer='YES',
            confidence=0.8,
            probabilities=[0.2, 0.8],
            inference_time_ms=10.0,
            model_format='pytorch'
        )
        
        # 概率和应该接近1
        prob_sum = sum(result.probabilities)
        assert abs(prob_sum - 1.0) < 1e-5
    
    def test_inference_time_positive(self):
        """测试推理时间为正"""
        result = InferenceResult(
            answer='YES',
            confidence=0.5,
            probabilities=[0.5, 0.5],
            inference_time_ms=25.5,
            model_format='pytorch'
        )
        
        assert result.inference_time_ms > 0


class TestImagePreprocessorMemory:
    """测试图像预处理的内存效率"""
    
    def test_preprocessing_memory_efficiency(self):
        """测试预处理内存效率"""
        preprocessor = ImagePreprocessor(image_size=(224, 224))
        
        # 创建大量图像
        images = [
            np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            for _ in range(100)
        ]
        
        # 批量预处理
        processed = preprocessor.preprocess_batch(images)
        
        # 验证输出
        assert processed.shape == (100, 3, 224, 224)
        
        # 验证内存大小合理
        # 100 * 3 * 224 * 224 * 4 bytes (float32) = ~150MB
        memory_mb = processed.element_size() * processed.nelement() / (1024 * 1024)
        assert memory_mb < 200  # 应该小于200MB


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
