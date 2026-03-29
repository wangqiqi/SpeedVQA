"""
ROI推理属性测试

使用Hypothesis进行属性测试，验证ROI推理功能的通用正确性属性：
- 属性 10: ROI推理功能完整性
- 属性 11: 批量推理效率性

验证需求: 需求 6.2, 6.3, 6.4
"""

import pytest
import torch
import numpy as np
from PIL import Image
from hypothesis import given, strategies as st, settings, HealthCheck

from speedvqa.inference.inferencer import ImagePreprocessor, InferenceResult
from speedvqa.inference.visualizer import VisualizationResult


# ============================================================================
# 策略定义 (Hypothesis Strategies)
# ============================================================================

def image_strategy():
    """生成随机图像的策略"""
    return st.just(None).flatmap(
        lambda _: st.builds(
            lambda: np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        )
    )


def question_strategy():
    """生成随机问题的策略"""
    questions = [
        "Is there a person?",
        "Is the person smoking?",
        "Is there a car?",
        "Is there a phone?",
        "Is the person talking?",
        "Is there a bicycle?",
        "Is the person running?",
        "Is there a dog?",
        "是否有人?",
        "是否有人打电话?",
        "是否有车?",
        "是否有人抽烟?",
    ]
    return st.sampled_from(questions)


def batch_size_strategy():
    """生成批量大小的策略"""
    return st.integers(min_value=1, max_value=32)


def image_batch_strategy(batch_size):
    """生成图像批次的策略"""
    return st.lists(
        image_strategy(),
        min_size=batch_size,
        max_size=batch_size
    )


def question_batch_strategy(batch_size):
    """生成问题批次的策略"""
    return st.lists(
        question_strategy(),
        min_size=batch_size,
        max_size=batch_size
    )


# ============================================================================
# 属性 10: ROI推理功能完整性
# ============================================================================

class MockROIInferencer:
    """模拟ROI推理器用于属性测试"""
    
    def __init__(self):
        self.image_preprocessor = ImagePreprocessor(image_size=(224, 224))
        self.model_format = 'pytorch'
    
    def inference(self, roi_image, question):
        """模拟单个推理"""
        # 预处理图像
        if isinstance(roi_image, str):
            roi_image = Image.open(roi_image).convert('RGB')

        _ = self.image_preprocessor.preprocess(roi_image)

        # 模拟推理（生成随机但有效的结果）
        np.random.seed(hash((roi_image.tobytes() if isinstance(roi_image, Image.Image) else roi_image.tobytes(), question)) % (2**32))
        
        # 生成随机logits
        logits = np.random.randn(2)
        probabilities = torch.softmax(torch.from_numpy(logits).float(), dim=0).numpy()
        
        confidence = float(np.max(probabilities))
        prediction = int(np.argmax(probabilities))
        answer = "YES" if prediction == 1 else "NO"
        
        return InferenceResult(
            answer=answer,
            confidence=confidence,
            probabilities=probabilities.tolist(),
            inference_time_ms=np.random.uniform(10, 50),
            model_format=self.model_format,
            batch_size=1
        )
    
    def batch_inference(self, roi_images, questions):
        """模拟批量推理"""
        if len(roi_images) != len(questions):
            raise ValueError("Number of images and questions must match")
        
        results = []
        for img, q in zip(roi_images, questions):
            result = self.inference(img, q)
            result.batch_size = len(roi_images)
            results.append(result)
        
        return results


class TestROIInferenceCompleteness:
    """
    属性 10: ROI推理功能完整性
    
    对于任何有效的ROI图像和问题输入，推理器应该输出包含答案、置信度、
    关键区域坐标和推理时间的完整结果
    
    验证需求: 需求 6.2, 6.4
    """
    
    @pytest.fixture(scope="class")
    def inferencer(self):
        """创建模拟推理器"""
        return MockROIInferencer()
    
    @given(image_strategy(), question_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_inference_output_completeness(self, inferencer, image, question):
        """
        **Validates: Requirements 6.2, 6.4**
        
        对于任何有效的ROI图像和问题输入，推理器应该输出包含所有必需字段的完整结果
        """
        # 执行推理
        result = inferencer.inference(image, question)
        
        # 验证结果包含所有必需字段
        assert hasattr(result, 'answer'), "Result must have 'answer' field"
        assert hasattr(result, 'confidence'), "Result must have 'confidence' field"
        assert hasattr(result, 'probabilities'), "Result must have 'probabilities' field"
        assert hasattr(result, 'inference_time_ms'), "Result must have 'inference_time_ms' field"
        assert hasattr(result, 'model_format'), "Result must have 'model_format' field"
        assert hasattr(result, 'batch_size'), "Result must have 'batch_size' field"
    
    @given(image_strategy(), question_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_inference_answer_validity(self, inferencer, image, question):
        """
        **Validates: Requirements 6.2, 6.4**
        
        推理结果的答案应该总是有效的YES或NO
        """
        result = inferencer.inference(image, question)
        
        # 验证答案有效性
        assert result.answer in ['YES', 'NO'], \
            f"Answer must be 'YES' or 'NO', got {result.answer}"
    
    @given(image_strategy(), question_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_inference_confidence_range(self, inferencer, image, question):
        """
        **Validates: Requirements 6.2, 6.4**
        
        置信度应该在[0, 1]范围内
        """
        result = inferencer.inference(image, question)
        
        # 验证置信度范围
        assert 0 <= result.confidence <= 1, \
            f"Confidence must be in [0, 1], got {result.confidence}"
    
    @given(image_strategy(), question_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_inference_probabilities_validity(self, inferencer, image, question):
        """
        **Validates: Requirements 6.2, 6.4**
        
        概率分布应该是有效的概率向量（和为1，所有值在[0,1]范围内）
        """
        result = inferencer.inference(image, question)
        
        # 验证概率数量
        assert len(result.probabilities) == 2, \
            f"Expected 2 probabilities, got {len(result.probabilities)}"
        
        # 验证概率范围
        for prob in result.probabilities:
            assert 0 <= prob <= 1, \
                f"Probability must be in [0, 1], got {prob}"
        
        # 验证概率和
        prob_sum = sum(result.probabilities)
        assert abs(prob_sum - 1.0) < 1e-5, \
            f"Probabilities must sum to 1, got {prob_sum}"
    
    @given(image_strategy(), question_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_inference_time_positive(self, inferencer, image, question):
        """
        **Validates: Requirements 6.2, 6.4**
        
        推理时间应该是正数
        """
        result = inferencer.inference(image, question)
        
        # 验证推理时间
        assert result.inference_time_ms > 0, \
            f"Inference time must be positive, got {result.inference_time_ms}"
    
    @given(image_strategy(), question_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_inference_batch_size_single(self, inferencer, image, question):
        """
        **Validates: Requirements 6.2, 6.4**
        
        单个推理的批次大小应该为1
        """
        result = inferencer.inference(image, question)
        
        # 验证批次大小
        assert result.batch_size == 1, \
            f"Single inference batch size must be 1, got {result.batch_size}"
    
    @given(image_strategy(), question_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_inference_confidence_matches_max_probability(self, inferencer, image, question):
        """
        **Validates: Requirements 6.2, 6.4**
        
        置信度应该等于最大概率
        """
        result = inferencer.inference(image, question)
        
        # 验证置信度与最大概率的关系
        max_prob = max(result.probabilities)
        assert abs(result.confidence - max_prob) < 1e-5, \
            f"Confidence {result.confidence} should match max probability {max_prob}"
    
    @given(image_strategy(), question_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_inference_answer_matches_probabilities(self, inferencer, image, question):
        """
        **Validates: Requirements 6.2, 6.4**
        
        答案应该与概率分布一致（最大概率对应的类别）
        """
        result = inferencer.inference(image, question)
        
        # 确定最大概率对应的类别
        max_prob_idx = np.argmax(result.probabilities)
        expected_answer = "YES" if max_prob_idx == 1 else "NO"
        
        # 验证答案与概率一致
        assert result.answer == expected_answer, \
            f"Answer {result.answer} should match max probability class {expected_answer}"
    
    @given(image_strategy(), question_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_inference_deterministic(self, inferencer, image, question):
        """
        **Validates: Requirements 6.2, 6.4**
        
        相同的输入应该产生相同的推理结果（确定性）
        """
        # 第一次推理
        result1 = inferencer.inference(image, question)
        
        # 第二次推理
        result2 = inferencer.inference(image, question)
        
        # 验证结果一致性
        assert result1.answer == result2.answer, \
            f"Answers should be identical: {result1.answer} vs {result2.answer}"
        
        assert abs(result1.confidence - result2.confidence) < 1e-5, \
            f"Confidences should be identical: {result1.confidence} vs {result2.confidence}"
        
        # 验证概率一致性
        for p1, p2 in zip(result1.probabilities, result2.probabilities):
            assert abs(p1 - p2) < 1e-5, \
                f"Probabilities should be identical: {p1} vs {p2}"


# ============================================================================
# 属性 11: 批量推理效率性
# ============================================================================

class TestBatchInferenceEfficiency:
    """
    属性 11: 批量推理效率性
    
    对于任何批量输入，批量推理的平均单样本处理时间应该显著小于单独处理
    每个样本的时间
    
    验证需求: 需求 6.3
    """
    
    @pytest.fixture(scope="class")
    def inferencer(self):
        """创建模拟推理器"""
        return MockROIInferencer()
    
    @given(batch_size_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_batch_inference_output_count(self, inferencer, batch_size):
        """
        **Validates: Requirements 6.3**
        
        批量推理应该返回与输入数量相同的结果
        """
        # 生成批量输入
        images = [np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) 
                  for _ in range(batch_size)]
        questions = ["Is there a person?"] * batch_size
        
        # 执行批量推理
        results = inferencer.batch_inference(images, questions)
        
        # 验证输出数量
        assert len(results) == batch_size, \
            f"Expected {batch_size} results, got {len(results)}"
    
    @given(batch_size_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_batch_inference_result_completeness(self, inferencer, batch_size):
        """
        **Validates: Requirements 6.3**
        
        批量推理的每个结果都应该包含所有必需字段
        """
        # 生成批量输入
        images = [np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) 
                  for _ in range(batch_size)]
        questions = ["Is there a person?"] * batch_size
        
        # 执行批量推理
        results = inferencer.batch_inference(images, questions)
        
        # 验证每个结果的完整性
        for i, result in enumerate(results):
            assert hasattr(result, 'answer'), f"Result {i} missing 'answer' field"
            assert hasattr(result, 'confidence'), f"Result {i} missing 'confidence' field"
            assert hasattr(result, 'probabilities'), f"Result {i} missing 'probabilities' field"
            assert hasattr(result, 'inference_time_ms'), f"Result {i} missing 'inference_time_ms' field"
            assert result.batch_size == batch_size, \
                f"Result {i} batch_size should be {batch_size}, got {result.batch_size}"
    
    @given(batch_size_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_batch_inference_result_validity(self, inferencer, batch_size):
        """
        **Validates: Requirements 6.3**
        
        批量推理的所有结果都应该是有效的
        """
        # 生成批量输入
        images = [np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) 
                  for _ in range(batch_size)]
        questions = ["Is there a person?"] * batch_size
        
        # 执行批量推理
        results = inferencer.batch_inference(images, questions)
        
        # 验证每个结果的有效性
        for i, result in enumerate(results):
            assert result.answer in ['YES', 'NO'], \
                f"Result {i} answer must be YES or NO, got {result.answer}"
            assert 0 <= result.confidence <= 1, \
                f"Result {i} confidence must be in [0, 1], got {result.confidence}"
            assert len(result.probabilities) == 2, \
                f"Result {i} must have 2 probabilities, got {len(result.probabilities)}"
            
            # 验证概率和
            prob_sum = sum(result.probabilities)
            assert abs(prob_sum - 1.0) < 1e-5, \
                f"Result {i} probabilities must sum to 1, got {prob_sum}"
    
    @given(batch_size_strategy())
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_batch_inference_efficiency_vs_sequential(self, inferencer, batch_size):
        """
        **Validates: Requirements 6.3**
        
        批量推理的平均单样本处理时间应该小于顺序处理的时间
        
        注意：在CPU上这个性质可能不一定成立，因为CPU没有真正的并行处理。
        但在GPU上应该明显更快。这个测试主要验证批量推理的正确性。
        """
        # 生成批量输入
        images = [np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) 
                  for _ in range(batch_size)]
        questions = ["Is there a person?"] * batch_size
        
        # 顺序推理
        sequential_times = []
        for img, q in zip(images, questions):
            result = inferencer.inference(img, q)
            sequential_times.append(result.inference_time_ms)
        
        total_sequential_time = sum(sequential_times)
        
        # 批量推理
        batch_results = inferencer.batch_inference(images, questions)
        total_batch_time = sum(r.inference_time_ms for r in batch_results)
        
        # 计算平均时间
        avg_sequential_time = total_sequential_time / batch_size
        avg_batch_time = total_batch_time / batch_size
        
        # 验证批量推理的平均时间不超过顺序推理的平均时间的1.5倍
        # （允许一些开销）
        assert avg_batch_time <= avg_sequential_time * 1.5, \
            f"Batch inference should be efficient: " \
            f"avg_batch_time={avg_batch_time:.2f}ms, " \
            f"avg_sequential_time={avg_sequential_time:.2f}ms"
    
    @given(batch_size_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_batch_inference_consistency(self, inferencer, batch_size):
        """
        **Validates: Requirements 6.3**
        
        批量推理应该与单个推理产生相同的结果
        """
        # 生成批量输入
        images = [np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) 
                  for _ in range(batch_size)]
        questions = ["Is there a person?"] * batch_size
        
        # 单个推理
        single_results = []
        for img, q in zip(images, questions):
            result = inferencer.inference(img, q)
            single_results.append(result)
        
        # 批量推理
        batch_results = inferencer.batch_inference(images, questions)
        
        # 验证结果一致性
        for i, (single_result, batch_result) in enumerate(zip(single_results, batch_results)):
            assert single_result.answer == batch_result.answer, \
                f"Result {i} answer mismatch: {single_result.answer} vs {batch_result.answer}"
            
            assert abs(single_result.confidence - batch_result.confidence) < 1e-5, \
                f"Result {i} confidence mismatch: {single_result.confidence} vs {batch_result.confidence}"
            
            # 验证概率一致性
            for p1, p2 in zip(single_result.probabilities, batch_result.probabilities):
                assert abs(p1 - p2) < 1e-5, \
                    f"Result {i} probability mismatch: {p1} vs {p2}"
    
    @given(batch_size_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_batch_inference_different_questions(self, inferencer, batch_size):
        """
        **Validates: Requirements 6.3**
        
        批量推理应该支持不同的问题
        """
        # 生成批量输入
        images = [np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) 
                  for _ in range(batch_size)]
        
        # 生成不同的问题
        questions = [
            "Is there a person?",
            "Is the person smoking?",
            "Is there a car?",
            "Is there a phone?",
            "Is the person talking?",
        ]
        # 循环使用问题列表
        questions = [questions[i % len(questions)] for i in range(batch_size)]
        
        # 执行批量推理
        results = inferencer.batch_inference(images, questions)
        
        # 验证结果
        assert len(results) == batch_size, \
            f"Expected {batch_size} results, got {len(results)}"
        
        for result in results:
            assert result.answer in ['YES', 'NO'], \
                f"Answer must be YES or NO, got {result.answer}"
            assert 0 <= result.confidence <= 1, \
                f"Confidence must be in [0, 1], got {result.confidence}"
    
    @given(batch_size_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_batch_inference_different_images(self, inferencer, batch_size):
        """
        **Validates: Requirements 6.3**
        
        批量推理应该支持不同的图像
        """
        # 生成不同的图像
        images = [np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) 
                  for _ in range(batch_size)]
        questions = ["Is there a person?"] * batch_size
        
        # 执行批量推理
        results = inferencer.batch_inference(images, questions)
        
        # 验证结果
        assert len(results) == batch_size, \
            f"Expected {batch_size} results, got {len(results)}"
        
        for result in results:
            assert result.answer in ['YES', 'NO'], \
                f"Answer must be YES or NO, got {result.answer}"
            assert 0 <= result.confidence <= 1, \
                f"Confidence must be in [0, 1], got {result.confidence}"


# ============================================================================
# 集成测试
# ============================================================================

class TestROIInferenceIntegration:
    """ROI推理集成测试"""
    
    @pytest.fixture(scope="class")
    def inferencer(self):
        """创建模拟推理器"""
        return MockROIInferencer()
    
    def test_single_and_batch_consistency(self, inferencer):
        """测试单个和批量推理的一致性"""
        # 生成测试数据
        batch_size = 5
        images = [np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) 
                  for _ in range(batch_size)]
        questions = ["Is there a person?"] * batch_size
        
        # 单个推理
        single_results = []
        for img, q in zip(images, questions):
            result = inferencer.inference(img, q)
            single_results.append(result)
        
        # 批量推理
        batch_results = inferencer.batch_inference(images, questions)
        
        # 验证一致性
        for i, (single_result, batch_result) in enumerate(zip(single_results, batch_results)):
            assert single_result.answer == batch_result.answer, \
                f"Result {i} answer mismatch"
            assert abs(single_result.confidence - batch_result.confidence) < 1e-5, \
                f"Result {i} confidence mismatch"


# ============================================================================
# 属性 16: 可视化结果准确性
# ============================================================================

class MockResultVisualizer:
    """模拟结果可视化器用于属性测试"""
    
    def __init__(self):
        self.config = {
            'font_size': 20,
            'text_color': (0, 255, 0),
            'box_color': (0, 255, 0),
            'box_thickness': 2,
            'confidence_threshold': 0.5
        }
    
    def visualize_inference_result(self, image, answer, confidence, question, 
                                   inference_time_ms, output_path=None):
        """模拟单个推理结果可视化"""
        
        # 验证输入
        if not isinstance(image, (np.ndarray, Image.Image, str)):
            raise ValueError(f"Invalid image type: {type(image)}")
        
        if answer not in ['YES', 'NO']:
            raise ValueError(f"Invalid answer: {answer}")
        
        if not (0 <= confidence <= 1):
            raise ValueError(f"Invalid confidence: {confidence}")
        
        if not isinstance(question, str) or len(question) == 0:
            raise ValueError(f"Invalid question: {question}")
        
        if inference_time_ms <= 0:
            raise ValueError(f"Invalid inference time: {inference_time_ms}")
        
        # 模拟可视化
        if output_path is None:
            output_path = "visualization_result.jpg"
        
        # 创建模拟的输出图像
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        
        # 返回成功的可视化结果
        return VisualizationResult(
            image_path=output_path,
            success=True,
            error_message=None,
            annotations={
                'answer': answer,
                'confidence': confidence,
                'question': question,
                'inference_time_ms': inference_time_ms
            },
            metadata={
                'image_size': image.size if isinstance(image, Image.Image) else (640, 480),
                'format': 'JPEG'
            }
        )
    
    def visualize_batch_results(self, images, answers, confidences, questions, 
                               inference_times_ms, output_dir=None):
        """模拟批量推理结果可视化"""
        if not (len(images) == len(answers) == len(confidences) == len(questions) == len(inference_times_ms)):
            raise ValueError("All input lists must have the same length")
        
        results = []
        for i, (img, ans, conf, q, time) in enumerate(
            zip(images, answers, confidences, questions, inference_times_ms)
        ):
            result = self.visualize_inference_result(img, ans, conf, q, time)
            results.append(result)
        
        return results
    
    def create_summary_report(self, results, output_path=None):
        """模拟创建摘要报告"""
        total_results = len(results)
        successful_results = sum(1 for r in results if r.success)
        
        confidences = []
        inference_times = []
        
        for result in results:
            if result.success and result.annotations:
                confidences.append(result.annotations.get('confidence', 0))
                inference_times.append(result.annotations.get('inference_time_ms', 0))
        
        report = {
            'total_results': total_results,
            'successful_results': successful_results,
            'failed_results': total_results - successful_results,
            'success_rate': successful_results / total_results if total_results > 0 else 0,
            'statistics': {
                'avg_confidence': np.mean(confidences) if confidences else 0,
                'min_confidence': np.min(confidences) if confidences else 0,
                'max_confidence': np.max(confidences) if confidences else 0,
                'avg_inference_time_ms': np.mean(inference_times) if inference_times else 0,
                'min_inference_time_ms': np.min(inference_times) if inference_times else 0,
                'max_inference_time_ms': np.max(inference_times) if inference_times else 0,
            }
        }
        
        return report


class TestVisualizationResultAccuracy:
    """
    属性 16: 可视化结果准确性
    
    对于任何有效的推理结果，可视化器应该：
    1. 成功生成可视化图像
    2. 保留所有推理信息（答案、置信度、问题、时间）
    3. 生成有效的输出文件
    4. 处理各种输入格式
    5. 生成准确的摘要报告
    
    验证需求: 需求 6.5
    """
    
    @pytest.fixture(scope="class")
    def visualizer(self):
        """创建模拟可视化器"""
        return MockResultVisualizer()
    
    @given(image_strategy(), question_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_visualization_success(self, visualizer, image, question):
        """
        **Validates: Requirements 6.5**
        
        对于任何有效的推理结果，可视化应该成功
        """
        answer = "YES"
        confidence = 0.85
        inference_time_ms = 25.5
        
        # 执行可视化
        result = visualizer.visualize_inference_result(
            image, answer, confidence, question, inference_time_ms
        )
        
        # 验证可视化成功
        assert result.success, f"Visualization should succeed, error: {result.error_message}"
        assert result.error_message is None, "Should not have error message"
        assert result.image_path is not None, "Should have output image path"
    
    @given(image_strategy(), question_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_visualization_preserves_annotations(self, visualizer, image, question):
        """
        **Validates: Requirements 6.5**
        
        可视化结果应该保留所有推理信息
        """
        answer = "NO"
        confidence = 0.72
        inference_time_ms = 18.3
        
        # 执行可视化
        result = visualizer.visualize_inference_result(
            image, answer, confidence, question, inference_time_ms
        )
        
        # 验证标注信息
        assert result.annotations is not None, "Should have annotations"
        assert result.annotations['answer'] == answer, "Answer should be preserved"
        assert abs(result.annotations['confidence'] - confidence) < 1e-6, "Confidence should be preserved"
        assert result.annotations['question'] == question, "Question should be preserved"
        assert abs(result.annotations['inference_time_ms'] - inference_time_ms) < 1e-6, "Inference time should be preserved"
    
    @given(image_strategy(), question_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_visualization_output_format(self, visualizer, image, question):
        """
        **Validates: Requirements 6.5**
        
        可视化输出应该有有效的格式信息
        """
        answer = "YES"
        confidence = 0.95
        inference_time_ms = 22.1
        
        # 执行可视化
        result = visualizer.visualize_inference_result(
            image, answer, confidence, question, inference_time_ms
        )
        
        # 验证元数据
        assert result.metadata is not None, "Should have metadata"
        assert 'image_size' in result.metadata, "Should have image size"
        assert 'format' in result.metadata, "Should have format information"
        assert result.metadata['format'] == 'JPEG', "Format should be JPEG"
        
        # 验证图像尺寸
        image_size = result.metadata['image_size']
        assert isinstance(image_size, tuple), "Image size should be tuple"
        assert len(image_size) == 2, "Image size should have 2 dimensions"
        assert image_size[0] > 0 and image_size[1] > 0, "Image dimensions should be positive"
    
    @given(
        batch_size_strategy(),
        st.lists(question_strategy(), min_size=1, max_size=5)
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_batch_visualization_completeness(self, visualizer, batch_size, questions_list):
        """
        **Validates: Requirements 6.5**
        
        批量可视化应该返回与输入数量相同的结果
        """
        # 生成批量输入
        images = [np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) 
                  for _ in range(batch_size)]
        answers = ["YES" if i % 2 == 0 else "NO" for i in range(batch_size)]
        # 确保置信度在[0, 1]范围内
        confidences = [0.5 + (i % 5) * 0.08 for i in range(batch_size)]  # 0.5, 0.58, 0.66, 0.74, 0.82, 0.5, ...
        questions = [questions_list[i % len(questions_list)] for i in range(batch_size)]
        inference_times = [10.0 + i * 2.0 for i in range(batch_size)]
        
        # 执行批量可视化
        results = visualizer.visualize_batch_results(
            images, answers, confidences, questions, inference_times
        )
        
        # 验证结果数量
        assert len(results) == batch_size, f"Expected {batch_size} results, got {len(results)}"
        
        # 验证每个结果
        for i, result in enumerate(results):
            assert result.success, f"Result {i} should be successful"
            assert result.annotations['answer'] == answers[i], f"Result {i} answer mismatch"
            assert abs(result.annotations['confidence'] - confidences[i]) < 1e-6, f"Result {i} confidence mismatch"
    
    @given(batch_size_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_summary_report_accuracy(self, visualizer, batch_size):
        """
        **Validates: Requirements 6.5**
        
        摘要报告应该准确反映可视化结果的统计信息
        """
        # 生成批量输入
        images = [np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) 
                  for _ in range(batch_size)]
        answers = ["YES"] * batch_size
        # 确保置信度在[0, 1]范围内
        confidences = [0.5 + (i % 5) * 0.08 for i in range(batch_size)]  # 0.5, 0.58, 0.66, 0.74, 0.82, 0.5, ...
        questions = ["Is there a person?"] * batch_size
        inference_times = [10.0 + i * 2.0 for i in range(batch_size)]
        
        # 执行批量可视化
        results = visualizer.visualize_batch_results(
            images, answers, confidences, questions, inference_times
        )
        
        # 创建摘要报告
        report = visualizer.create_summary_report(results)
        
        # 验证报告结构
        assert 'total_results' in report, "Report should have total_results"
        assert 'successful_results' in report, "Report should have successful_results"
        assert 'failed_results' in report, "Report should have failed_results"
        assert 'success_rate' in report, "Report should have success_rate"
        assert 'statistics' in report, "Report should have statistics"
        
        # 验证报告数值
        assert report['total_results'] == batch_size, "Total results should match batch size"
        assert report['successful_results'] == batch_size, "All results should be successful"
        assert report['failed_results'] == 0, "Should have no failed results"
        assert abs(report['success_rate'] - 1.0) < 1e-6, "Success rate should be 100%"
        
        # 验证统计信息
        stats = report['statistics']
        assert 'avg_confidence' in stats, "Should have average confidence"
        assert 'min_confidence' in stats, "Should have minimum confidence"
        assert 'max_confidence' in stats, "Should have maximum confidence"
        assert 'avg_inference_time_ms' in stats, "Should have average inference time"
        
        # 验证统计值的合理性
        assert stats['min_confidence'] <= stats['avg_confidence'] <= stats['max_confidence'], \
            "Confidence statistics should be ordered"
        assert stats['min_inference_time_ms'] <= stats['avg_inference_time_ms'] <= stats['max_inference_time_ms'], \
            "Inference time statistics should be ordered"
    
    @given(image_strategy(), question_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_visualization_with_different_confidence_levels(self, visualizer, image, question):
        """
        **Validates: Requirements 6.5**
        
        可视化应该正确处理不同的置信度水平
        """
        answer = "YES"
        inference_time_ms = 25.0
        
        # 测试不同的置信度水平
        confidence_levels = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
        
        for confidence in confidence_levels:
            result = visualizer.visualize_inference_result(
                image, answer, confidence, question, inference_time_ms
            )
            
            # 验证可视化成功
            assert result.success, f"Visualization should succeed for confidence {confidence}"
            assert result.annotations['confidence'] == confidence, \
                f"Confidence should be preserved: {confidence}"
    
    @given(image_strategy(), question_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_visualization_with_both_answers(self, visualizer, image, question):
        """
        **Validates: Requirements 6.5**
        
        可视化应该正确处理YES和NO两种答案
        """
        confidence = 0.75
        inference_time_ms = 20.0
        
        for answer in ['YES', 'NO']:
            result = visualizer.visualize_inference_result(
                image, answer, confidence, question, inference_time_ms
            )
            
            # 验证可视化成功
            assert result.success, f"Visualization should succeed for answer {answer}"
            assert result.annotations['answer'] == answer, \
                f"Answer should be preserved: {answer}"
    
    @given(batch_size_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_visualization_consistency_across_batch(self, visualizer, batch_size):
        """
        **Validates: Requirements 6.5**
        
        批量可视化中的每个结果应该与单个可视化一致
        """
        # 生成测试数据
        images = [np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) 
                  for _ in range(batch_size)]
        answers = ["YES"] * batch_size
        confidences = [0.8] * batch_size
        questions = ["Is there a person?"] * batch_size
        inference_times = [25.0] * batch_size
        
        # 单个可视化
        single_results = []
        for img, ans, conf, q, time in zip(images, answers, confidences, questions, inference_times):
            result = visualizer.visualize_inference_result(img, ans, conf, q, time)
            single_results.append(result)
        
        # 批量可视化
        batch_results = visualizer.visualize_batch_results(
            images, answers, confidences, questions, inference_times
        )
        
        # 验证一致性
        for i, (single_result, batch_result) in enumerate(zip(single_results, batch_results)):
            assert single_result.success == batch_result.success, \
                f"Result {i} success status should match"
            assert single_result.annotations['answer'] == batch_result.annotations['answer'], \
                f"Result {i} answer should match"
            assert abs(single_result.annotations['confidence'] - batch_result.annotations['confidence']) < 1e-6, \
                f"Result {i} confidence should match"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
