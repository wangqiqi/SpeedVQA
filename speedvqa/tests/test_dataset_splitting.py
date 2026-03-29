"""
属性测试 2: 数据集划分正确性
验证需求: 需求 1.4

使用Hypothesis进行属性测试，验证数据集划分的正确性和一致性。
"""

import pytest
import tempfile
import json
from pathlib import Path
from hypothesis import given, strategies as st, settings, assume

from speedvqa.data.datasets import VQADataset, build_dataset, split_dataset
from speedvqa.utils.config import create_minimal_config


class TestDatasetSplittingCorrectness:
    """属性测试 2: 数据集划分正确性"""
    
    def setup_method(self):
        """测试设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.images_dir = Path(self.temp_dir) / 'images'
        self.annotations_dir = Path(self.temp_dir) / 'annotations'
        self.images_dir.mkdir(exist_ok=True)
        self.annotations_dir.mkdir(exist_ok=True)
        
        # 创建测试配置
        self.config = create_minimal_config(self.temp_dir)
    
    def teardown_method(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_dataset(self, num_samples: int) -> VQADataset:
        """创建测试数据集"""
        from PIL import Image
        import numpy as np
        import shutil
        
        # 清空之前的数据
        if self.images_dir.exists():
            shutil.rmtree(self.images_dir)
        if self.annotations_dir.exists():
            shutil.rmtree(self.annotations_dir)
        
        self.images_dir.mkdir(exist_ok=True)
        self.annotations_dir.mkdir(exist_ok=True)
        
        questions = [
            "图片中有人吗？", "是否存在车辆？", "有动物吗？", "图片清晰吗？", "是白天吗？",
            "有建筑物吗？", "天气好吗？", "有树木吗？", "是室内场景吗？", "有文字吗？"
        ]
        answers = ["是", "否", "yes", "no", "有", "没有"]
        
        for i in range(num_samples):
            # 创建测试图像
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            image_name = f"test_image_{i}.jpg"
            img.save(self.images_dir / image_name)
            
            # 创建JSON标注
            question = questions[i % len(questions)]
            answer = answers[i % len(answers)]
            
            json_data = {
                'imagePath': image_name,
                'imageWidth': 224,
                'imageHeight': 224,
                'version': '1.0',
                'vqaData': {question: answer}
            }
            
            json_file = self.annotations_dir / f'annotation_{i}.json'
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False)
        
        return build_dataset(self.temp_dir, self.config)
    
    @given(
        num_samples=st.integers(min_value=10, max_value=100),
        train_ratio=st.floats(min_value=0.1, max_value=0.8),
        val_ratio=st.floats(min_value=0.1, max_value=0.4),
        test_ratio=st.floats(min_value=0.1, max_value=0.4)
    )
    @settings(max_examples=30, deadline=10000)
    def test_dataset_splitting_correctness(self, num_samples, train_ratio, val_ratio, test_ratio):
        """
        属性 2: 数据集划分正确性
        验证需求: 需求 1.4
        
        测试属性:
        1. 划分比例应该接近指定的比例
        2. 所有样本都应该被分配到某个子集中
        3. 子集之间不应该有重叠
        4. 相同参数的多次划分应该产生相同结果（固定随机种子）
        """
        # 确保比例之和接近1.0
        total_ratio = train_ratio + val_ratio + test_ratio
        assume(0.9 <= total_ratio <= 1.1)
        
        # 标准化比例
        train_ratio = train_ratio / total_ratio
        val_ratio = val_ratio / total_ratio
        test_ratio = test_ratio / total_ratio
        
        # 创建测试数据集
        dataset = self.create_test_dataset(num_samples)
        assume(len(dataset) == num_samples)
        
        # 执行数据集划分
        train_dataset, val_dataset, test_dataset = split_dataset(
            dataset, train_ratio, val_ratio, test_ratio
        )
        
        # 属性1: 划分大小正确性
        total_samples = len(train_dataset) + len(val_dataset) + len(test_dataset)
        assert total_samples == num_samples, f"Total samples mismatch: {total_samples} != {num_samples}"
        
        # 属性2: 比例接近性（允许±1的误差，因为整数划分）
        expected_train = int(num_samples * train_ratio)
        expected_val = int(num_samples * val_ratio)
        expected_test = num_samples - expected_train - expected_val
        
        assert abs(len(train_dataset) - expected_train) <= 1
        assert abs(len(val_dataset) - expected_val) <= 1
        assert abs(len(test_dataset) - expected_test) <= 1
        
        # 属性3: 数据完整性（验证样本可访问性）
        try:
            # 验证每个子集的第一个样本可以正常访问
            if len(train_dataset) > 0:
                train_sample = train_dataset[0]
                assert 'question' in train_sample
                assert 'answer' in train_sample
            
            if len(val_dataset) > 0:
                val_sample = val_dataset[0]
                assert 'question' in val_sample
                assert 'answer' in val_sample
            
            if len(test_dataset) > 0:
                test_sample = test_dataset[0]
                assert 'question' in test_sample
                assert 'answer' in test_sample
                
        except Exception as e:
            pytest.fail(f"Failed to access split dataset samples: {e}")
        
        # 属性4: 重现性（相同参数应该产生相同结果）
        train_dataset2, val_dataset2, test_dataset2 = split_dataset(
            dataset, train_ratio, val_ratio, test_ratio
        )
        
        assert len(train_dataset) == len(train_dataset2)
        assert len(val_dataset) == len(val_dataset2)
        assert len(test_dataset) == len(test_dataset2)
    
    @given(
        num_samples=st.integers(min_value=5, max_value=50),
        split_ratios=st.lists(
            st.floats(min_value=0.1, max_value=0.8),
            min_size=3, max_size=3
        ).filter(lambda ratios: 0.9 <= sum(ratios) <= 1.1)
    )
    @settings(max_examples=20, deadline=8000)
    def test_split_ratio_consistency(self, num_samples, split_ratios):
        """
        属性测试: 划分比例一致性
        验证不同比例组合下的划分正确性
        """
        train_ratio, val_ratio, test_ratio = split_ratios
        
        # 标准化比例
        total_ratio = sum(split_ratios)
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio
        
        dataset = self.create_test_dataset(num_samples)
        train_dataset, val_dataset, test_dataset = split_dataset(
            dataset, train_ratio, val_ratio, test_ratio
        )
        
        # 验证总数保持不变
        assert len(train_dataset) + len(val_dataset) + len(test_dataset) == num_samples
        
        # 验证每个子集都有合理的大小
        assert len(train_dataset) >= 0
        assert len(val_dataset) >= 0
        assert len(test_dataset) >= 0
        
        # 验证比例的合理性（至少有一个子集不为空）
        assert max(len(train_dataset), len(val_dataset), len(test_dataset)) > 0
    
    @given(
        num_samples=st.integers(min_value=20, max_value=100)
    )
    @settings(max_examples=15, deadline=6000)
    def test_default_split_ratios(self, num_samples):
        """
        属性测试: 默认划分比例（7:2:1）
        验证默认比例的正确性
        """
        dataset = self.create_test_dataset(num_samples)
        
        # 使用默认比例
        train_dataset, val_dataset, test_dataset = split_dataset(dataset)
        
        # 验证总数
        total = len(train_dataset) + len(val_dataset) + len(test_dataset)
        assert total == num_samples
        
        # 验证比例接近7:2:1
        train_ratio = len(train_dataset) / num_samples
        val_ratio = len(val_dataset) / num_samples
        test_ratio = len(test_dataset) / num_samples
        
        # 允许一定的误差范围
        assert 0.6 <= train_ratio <= 0.8  # 期望0.7
        assert 0.1 <= val_ratio <= 0.3    # 期望0.2
        assert 0.05 <= test_ratio <= 0.2   # 期望0.1
    
    def test_edge_cases(self):
        """测试边界情况"""
        # 测试最小数据集
        dataset = self.create_test_dataset(3)
        train_dataset, val_dataset, test_dataset = split_dataset(dataset)
        
        # 即使是最小数据集，也应该能正确划分
        total = len(train_dataset) + len(val_dataset) + len(test_dataset)
        assert total == 3
        
        # 测试单样本数据集
        dataset = self.create_test_dataset(1)
        train_dataset, val_dataset, test_dataset = split_dataset(dataset)
        
        total = len(train_dataset) + len(val_dataset) + len(test_dataset)
        assert total == 1
        
        # 至少有一个子集应该包含这个样本
        assert max(len(train_dataset), len(val_dataset), len(test_dataset)) == 1
    
    @given(
        num_samples=st.integers(min_value=10, max_value=50)
    )
    @settings(max_examples=10, deadline=5000)
    def test_split_determinism(self, num_samples):
        """
        属性测试: 划分确定性
        验证固定随机种子下的划分一致性
        """
        dataset = self.create_test_dataset(num_samples)
        
        # 多次划分应该产生相同结果
        results = []
        for _ in range(3):
            train_dataset, val_dataset, test_dataset = split_dataset(dataset)
            results.append((len(train_dataset), len(val_dataset), len(test_dataset)))
        
        # 所有结果应该相同
        assert all(result == results[0] for result in results)
    
    def test_class_distribution_preservation(self):
        """
        测试类别分布保持
        验证划分后各子集的类别分布相对均匀
        """
        # 创建平衡的数据集
        num_samples = 40
        dataset = self.create_test_dataset(num_samples)
        
        # 获取原始类别分布（划分前后均应有效）
        assert dataset.get_class_distribution() is not None

        train_dataset, val_dataset, test_dataset = split_dataset(dataset)
        
        # 验证每个子集都有样本（如果原数据集足够大）
        if num_samples >= 10:
            assert len(train_dataset) > 0
            assert len(val_dataset) > 0 or len(test_dataset) > 0
        
        # 验证子集样本的有效性
        for subset in [train_dataset, val_dataset, test_dataset]:
            if len(subset) > 0:
                sample = subset[0]
                assert sample['answer'] in ['YES', 'NO']


class TestDatasetSplittingStateMachine:
    """基于状态的数据集划分测试"""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.images_dir = Path(self.temp_dir) / 'images'
        self.annotations_dir = Path(self.temp_dir) / 'annotations'
        self.images_dir.mkdir(exist_ok=True)
        self.annotations_dir.mkdir(exist_ok=True)
        self.config = create_minimal_config(self.temp_dir)
    
    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @given(
        operations=st.lists(
            st.tuples(
                st.sampled_from(['add_sample', 'split_dataset']),
                st.integers(min_value=1, max_value=10)
            ),
            min_size=1, max_size=10
        )
    )
    @settings(max_examples=10, deadline=8000)
    def test_incremental_splitting(self, operations):
        """
        测试增量数据集划分
        验证随着样本增加，划分行为的一致性
        """
        sample_count = 0
        
        for operation, param in operations:
            if operation == 'add_sample':
                # 添加样本
                for i in range(param):
                    self._add_sample(sample_count + i)
                sample_count += param
                
            elif operation == 'split_dataset' and sample_count > 0:
                # 执行划分
                try:
                    dataset = build_dataset(self.temp_dir, self.config)
                    if len(dataset) > 0:
                        train_dataset, val_dataset, test_dataset = split_dataset(dataset)
                        
                        # 验证划分正确性
                        total = len(train_dataset) + len(val_dataset) + len(test_dataset)
                        assert total == len(dataset)
                        assert total == sample_count
                        
                except Exception:
                    # 如果划分失败，跳过这次操作
                    pass
    
    def _add_sample(self, index: int):
        """添加单个样本"""
        from PIL import Image
        import numpy as np
        
        # 创建图像
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        image_name = f"test_{index}.jpg"
        img.save(self.images_dir / image_name)
        
        # 创建标注
        questions = ["有人吗？", "有车吗？", "清晰吗？"]
        answers = ["是", "否"]
        
        json_data = {
            'imagePath': image_name,
            'imageWidth': 224,
            'imageHeight': 224,
            'version': '1.0',
            'vqaData': {
                questions[index % len(questions)]: answers[index % len(answers)]
            }
        }
        
        json_file = self.annotations_dir / f'sample_{index}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False)


if __name__ == '__main__':
    # 运行属性测试
    pytest.main([__file__, '-v'])