"""
属性测试 5: 模型性能评估准确性
验证需求: 需求 2.3

使用Hypothesis进行属性测试，验证性能评估系统的准确性和一致性。
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from hypothesis import given, strategies as st, settings, assume
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, invariant

from speedvqa.utils.metrics import MetricsCalculator, PerformanceMonitor, calculate_metrics


class TestPerformanceEvaluationProperties:
    """属性测试 5: 模型性能评估准确性"""
    
    def setup_method(self):
        """测试设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.save_dir = Path(self.temp_dir) / 'performance_test'
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def teardown_method(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @given(
        data=st.data()
    )
    @settings(max_examples=20, deadline=5000)
    def test_binary_classification_metrics_consistency(self, data):
        """
        属性 5: 模型性能评估准确性 - 二分类指标一致性
        验证需求: 需求 2.3
        
        测试属性:
        1. 指标值应该在合理范围内 [0, 1]
        2. 完美预测应该得到最高分数
        3. 随机预测应该得到中等分数
        4. 相同输入应该产生相同输出
        """
        # 生成相同长度的预测和目标
        size = data.draw(st.integers(min_value=10, max_value=100))
        predictions = data.draw(st.lists(st.integers(min_value=0, max_value=1), min_size=size, max_size=size))
        targets = data.draw(st.lists(st.integers(min_value=0, max_value=1), min_size=size, max_size=size))
        
        # 确保有多个类别
        assume(len(set(targets)) > 1)
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # 计算指标
        metrics = calculate_metrics(predictions, targets, num_classes=2, class_names=['NO', 'YES'])
        
        # 属性1: 指标值范围检查
        assert 0 <= metrics['accuracy'] <= 1, "准确率应该在[0,1]范围内"
        assert 0 <= metrics['precision'] <= 1, "精确率应该在[0,1]范围内"
        assert 0 <= metrics['recall'] <= 1, "召回率应该在[0,1]范围内"
        assert 0 <= metrics['f1_score'] <= 1, "F1分数应该在[0,1]范围内"
        
        # 属性2: 完美预测测试
        perfect_predictions = targets.copy()
        perfect_metrics = calculate_metrics(perfect_predictions, targets, num_classes=2)
        
        assert perfect_metrics['accuracy'] == 1.0, "完美预测的准确率应该是1.0"
        assert perfect_metrics['precision'] == 1.0, "完美预测的精确率应该是1.0"
        assert perfect_metrics['recall'] == 1.0, "完美预测的召回率应该是1.0"
        assert perfect_metrics['f1_score'] == 1.0, "完美预测的F1分数应该是1.0"
        
        # 属性3: 一致性测试（相同输入产生相同输出）
        metrics2 = calculate_metrics(predictions, targets, num_classes=2, class_names=['NO', 'YES'])
        
        assert metrics['accuracy'] == metrics2['accuracy'], "相同输入应该产生相同的准确率"
        assert metrics['precision'] == metrics2['precision'], "相同输入应该产生相同的精确率"
        assert metrics['recall'] == metrics2['recall'], "相同输入应该产生相同的召回率"
        assert metrics['f1_score'] == metrics2['f1_score'], "相同输入应该产生相同的F1分数"
        
        # 属性4: 指标关系检查
        # F1分数应该是精确率和召回率的调和平均数（当两者都不为0时）
        if metrics['precision'] > 0 and metrics['recall'] > 0:
            expected_f1 = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
            assert abs(metrics['f1_score'] - expected_f1) < 1e-6, "F1分数应该是精确率和召回率的调和平均数"
    
    @given(
        num_classes=st.integers(min_value=2, max_value=5),
        num_samples=st.integers(min_value=20, max_value=100)
    )
    @settings(max_examples=10, deadline=8000)
    def test_multiclass_metrics_consistency(self, num_classes, num_samples):
        """
        属性测试: 多分类指标一致性
        验证多分类场景下的指标计算正确性
        """
        # 生成随机多分类数据
        np.random.seed(42)  # 固定种子确保可重现
        predictions = np.random.randint(0, num_classes, num_samples)
        targets = np.random.randint(0, num_classes, num_samples)
        
        class_names = [f'Class_{i}' for i in range(num_classes)]
        
        # 计算指标
        metrics = calculate_metrics(
            predictions, targets, 
            probabilities=None, 
            num_classes=num_classes, 
            class_names=class_names
        )
        
        # 验证基本属性
        assert 0 <= metrics['accuracy'] <= 1, "多分类准确率应该在[0,1]范围内"
        assert 0 <= metrics['precision'] <= 1, "多分类精确率应该在[0,1]范围内"
        assert 0 <= metrics['recall'] <= 1, "多分类召回率应该在[0,1]范围内"
        assert 0 <= metrics['f1_score'] <= 1, "多分类F1分数应该在[0,1]范围内"
        
        # 验证样本数量
        assert metrics['num_samples'] == num_samples, "样本数量应该正确"
        
        # 验证每类别指标存在（仅对多分类）
        if num_classes > 2:
            for class_name in class_names:
                assert f'precision_{class_name}' in metrics, f"应该包含{class_name}的精确率"
                assert f'recall_{class_name}' in metrics, f"应该包含{class_name}的召回率"
                assert f'f1_{class_name}' in metrics, f"应该包含{class_name}的F1分数"
    
    @given(
        batch_sizes=st.lists(
            st.integers(min_value=1, max_value=20),
            min_size=3, max_size=10
        )
    )
    @settings(max_examples=8, deadline=6000)
    def test_metrics_calculator_accumulation(self, batch_sizes):
        """
        属性测试: 指标计算器累积一致性
        验证分批更新和一次性计算的结果一致性
        """
        calculator = MetricsCalculator(num_classes=2, class_names=['NO', 'YES'])
        
        all_predictions = []
        all_targets = []
        
        # 分批更新
        for batch_size in batch_sizes:
            batch_predictions = np.random.randint(0, 2, batch_size)
            batch_targets = np.random.randint(0, 2, batch_size)
            
            calculator.update(batch_predictions, batch_targets)
            
            all_predictions.extend(batch_predictions)
            all_targets.extend(batch_targets)
        
        # 计算累积指标
        accumulated_metrics = calculator.compute_basic_metrics()
        
        # 一次性计算指标
        direct_metrics = calculate_metrics(
            np.array(all_predictions), 
            np.array(all_targets), 
            num_classes=2, 
            class_names=['NO', 'YES']
        )
        
        # 验证一致性
        tolerance = 1e-6
        assert abs(accumulated_metrics['accuracy'] - direct_metrics['accuracy']) < tolerance, "累积准确率应该与直接计算一致"
        assert abs(accumulated_metrics['precision'] - direct_metrics['precision']) < tolerance, "累积精确率应该与直接计算一致"
        assert abs(accumulated_metrics['recall'] - direct_metrics['recall']) < tolerance, "累积召回率应该与直接计算一致"
        assert abs(accumulated_metrics['f1_score'] - direct_metrics['f1_score']) < tolerance, "累积F1分数应该与直接计算一致"
    
    @given(
        num_epochs=st.integers(min_value=3, max_value=10),
        metrics_per_epoch=st.integers(min_value=2, max_value=5)
    )
    @settings(max_examples=8, deadline=8000)
    def test_performance_monitor_consistency(self, num_epochs, metrics_per_epoch):
        """
        属性测试: 性能监控器一致性
        验证性能监控器的记录和统计功能
        """
        monitor = PerformanceMonitor(str(self.save_dir), 'test_experiment')
        
        # 模拟训练过程
        for epoch in range(num_epochs):
            # 训练指标
            train_loss = max(0.1, 1.0 - epoch * 0.1)  # 确保损失为正
            train_accuracy = min(0.95, 0.5 + epoch * 0.08)  # 确保准确率不超过1
            train_metrics = {'loss': train_loss, 'accuracy': train_accuracy}
            
            monitor.log_train_metrics(epoch, train_metrics, 0.001 * (0.9 ** epoch))
            
            # 验证指标
            val_loss = max(0.05, 0.9 - epoch * 0.08)
            val_accuracy = min(0.98, 0.6 + epoch * 0.07)
            val_precision = min(0.97, 0.55 + epoch * 0.06)
            val_recall = min(0.96, 0.58 + epoch * 0.05)
            val_f1 = min(0.95, 0.56 + epoch * 0.04)
            
            val_metrics = {
                'loss': val_loss,
                'accuracy': val_accuracy,
                'precision': val_precision,
                'recall': val_recall,
                'f1_score': val_f1
            }
            
            monitor.log_val_metrics(epoch, val_metrics)
        
        # 验证历史记录长度
        assert len(monitor.train_history['epoch']) == num_epochs, "训练历史长度应该等于epoch数"
        assert len(monitor.val_history['epoch']) == num_epochs, "验证历史长度应该等于epoch数"
        
        # 验证最佳指标记录
        assert 'accuracy' in monitor.best_metrics, "应该记录最佳准确率"
        assert 'loss' in monitor.best_metrics, "应该记录最佳损失"
        
        # 验证最佳指标的合理性
        best_val_accuracy = monitor.best_metrics['accuracy']['value']
        best_val_loss = monitor.best_metrics['loss']['value']
        
        assert 0 <= best_val_accuracy <= 1, "最佳准确率应该在[0,1]范围内"
        assert best_val_loss >= 0, "最佳损失应该非负"
        
        # 验证最佳指标确实是历史中的最优值
        all_val_accuracies = monitor.val_history['accuracy']
        all_val_losses = monitor.val_history['loss']
        
        assert best_val_accuracy == max(all_val_accuracies), "最佳准确率应该是历史最大值"
        assert best_val_loss == min(all_val_losses), "最佳损失应该是历史最小值"
        
        # 验证摘要信息
        summary = monitor.get_summary()
        assert summary['total_epochs'] == monitor.current_epoch, "摘要中的总epoch数应该正确"
        assert 'best_metrics' in summary, "摘要应该包含最佳指标"
    
    def test_confusion_matrix_properties(self):
        """
        测试混淆矩阵的属性
        """
        # 测试完美预测的混淆矩阵
        perfect_predictions = np.array([0, 0, 1, 1, 0, 1])
        perfect_targets = np.array([0, 0, 1, 1, 0, 1])
        
        calculator = MetricsCalculator(num_classes=2)
        calculator.update(perfect_predictions, perfect_targets)
        
        cm = calculator.compute_confusion_matrix()
        
        # 完美预测的混淆矩阵应该只有对角线元素
        assert cm[0, 1] == 0, "完美预测不应该有假正例"
        assert cm[1, 0] == 0, "完美预测不应该有假负例"
        assert cm[0, 0] + cm[1, 1] == len(perfect_predictions), "对角线元素之和应该等于总样本数"
    
    def test_edge_cases(self):
        """
        测试边界情况
        """
        calculator = MetricsCalculator(num_classes=2)
        
        # 测试空数据
        empty_metrics = calculator.compute_basic_metrics()
        assert empty_metrics == {}, "空数据应该返回空指标字典"
        
        # 测试单类别数据
        single_class_predictions = np.array([1, 1, 1, 1])
        single_class_targets = np.array([1, 1, 1, 1])
        
        calculator.reset()
        calculator.update(single_class_predictions, single_class_targets)
        
        single_class_metrics = calculator.compute_basic_metrics()
        assert single_class_metrics['accuracy'] == 1.0, "单类别完美预测准确率应该是1.0"
    
    @given(
        noise_level=st.floats(min_value=0.0, max_value=0.5)
    )
    @settings(max_examples=5, deadline=4000)
    def test_metrics_robustness_to_noise(self, noise_level):
        """
        属性测试: 指标对噪声的鲁棒性
        验证在不同噪声水平下指标的稳定性
        """
        np.random.seed(42)
        n_samples = 100
        
        # 生成基础数据
        base_targets = np.random.randint(0, 2, n_samples)
        
        # 添加噪声：随机翻转一定比例的预测
        predictions = base_targets.copy()
        n_flip = int(n_samples * noise_level)
        flip_indices = np.random.choice(n_samples, n_flip, replace=False)
        predictions[flip_indices] = 1 - predictions[flip_indices]
        
        # 计算指标
        metrics = calculate_metrics(predictions, base_targets, num_classes=2)
        
        # 验证指标的合理性
        expected_accuracy = 1.0 - noise_level
        tolerance = 0.1  # 允许一定的随机性
        
        assert abs(metrics['accuracy'] - expected_accuracy) <= tolerance, f"噪声水平{noise_level}下的准确率应该接近{expected_accuracy}"
        
        # 验证指标仍在合理范围内
        assert 0 <= metrics['accuracy'] <= 1, "即使有噪声，准确率也应该在[0,1]范围内"
        assert 0 <= metrics['precision'] <= 1, "即使有噪声，精确率也应该在[0,1]范围内"
        assert 0 <= metrics['recall'] <= 1, "即使有噪声，召回率也应该在[0,1]范围内"
        assert 0 <= metrics['f1_score'] <= 1, "即使有噪声，F1分数也应该在[0,1]范围内"


class PerformanceEvaluationStateMachine(RuleBasedStateMachine):
    """
    基于状态的属性测试：性能评估状态一致性
    """
    
    def __init__(self):
        super().__init__()
        self.calculator = None
        self.total_samples = 0
        self.total_correct = 0
    
    @initialize()
    def setup(self):
        """初始化测试环境"""
        self.calculator = MetricsCalculator(num_classes=2, class_names=['NO', 'YES'])
        self.total_samples = 0
        self.total_correct = 0
    
    @rule(
        batch_size=st.integers(min_value=1, max_value=10),
        accuracy_rate=st.floats(min_value=0.0, max_value=1.0)
    )
    def add_batch_predictions(self, batch_size, accuracy_rate):
        """添加批次预测"""
        if self.calculator:
            # 生成具有指定准确率的预测
            targets = np.random.randint(0, 2, batch_size)
            predictions = targets.copy()
            
            # 随机翻转一些预测以达到指定准确率
            n_wrong = int(batch_size * (1 - accuracy_rate))
            if n_wrong > 0:
                wrong_indices = np.random.choice(batch_size, n_wrong, replace=False)
                predictions[wrong_indices] = 1 - predictions[wrong_indices]
            
            self.calculator.update(predictions, targets)
            
            # 更新统计
            self.total_samples += batch_size
            self.total_correct += np.sum(predictions == targets)
    
    @invariant()
    def metrics_consistency(self):
        """不变量：指标一致性"""
        if self.calculator and self.total_samples > 0:
            metrics = self.calculator.compute_basic_metrics()
            
            # 验证样本数量一致性
            assert metrics['num_samples'] == self.total_samples, "样本数量应该一致"
            
            # 验证准确率一致性
            expected_accuracy = self.total_correct / self.total_samples
            actual_accuracy = metrics['accuracy']
            
            # 允许小的浮点误差
            assert abs(actual_accuracy - expected_accuracy) < 1e-10, "准确率计算应该一致"
            
            # 验证指标范围
            assert 0 <= actual_accuracy <= 1, "准确率应该在[0,1]范围内"
            assert 0 <= metrics['precision'] <= 1, "精确率应该在[0,1]范围内"
            assert 0 <= metrics['recall'] <= 1, "召回率应该在[0,1]范围内"
            assert 0 <= metrics['f1_score'] <= 1, "F1分数应该在[0,1]范围内"


# 运行状态机测试
TestPerformanceEvaluationState = PerformanceEvaluationStateMachine.TestCase


if __name__ == '__main__':
    # 运行属性测试
    pytest.main([__file__, '-v'])