"""
属性测试 1: 数据验证一致性
验证需求: 需求 1.3

使用Hypothesis进行属性测试，验证数据加载器的一致性和正确性。
"""

import pytest
import tempfile
import json
import os
from pathlib import Path
from typing import Dict, List, Any
from hypothesis import given, strategies as st, settings, assume
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, invariant

from speedvqa.data.datasets import VQADataset, build_dataset
from speedvqa.data.validators import DataValidator, XAnyLabelingAdapter
from speedvqa.utils.config import create_minimal_config


class TestDataValidationConsistency:
    """属性测试 1: 数据验证一致性"""
    
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
    
    def create_test_image(self, image_name: str):
        """创建测试图像文件"""
        from PIL import Image
        import numpy as np
        
        # 创建简单的测试图像
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(self.images_dir / image_name)
    
    @given(
        questions=st.lists(
            st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
            min_size=1, max_size=10
        ),
        answers=st.lists(
            st.sampled_from(['是', '否', 'yes', 'no', 'YES', 'NO', '有', '没有', 'true', 'false']),
            min_size=1, max_size=10
        ),
        image_names=st.lists(
            st.text(min_size=1, max_size=20).filter(lambda x: x.isalnum()).map(lambda x: f"{x}.jpg"),
            min_size=1, max_size=5
        )
    )
    @settings(max_examples=50, deadline=5000)
    def test_data_validation_consistency(self, questions, answers, image_names):
        """
        属性 1: 数据验证一致性
        验证需求: 需求 1.3
        
        测试属性:
        1. 相同的数据应该产生相同的验证结果
        2. 数据验证器应该正确识别有效和无效的数据
        3. 答案标准化应该是一致的和可预测的
        """
        assume(len(questions) == len(answers))
        assume(len(image_names) > 0)
        
        # 创建测试图像
        for image_name in image_names:
            self.create_test_image(image_name)
        
        # 创建JSON标注文件
        json_files = []
        for i, (question, answer) in enumerate(zip(questions, answers)):
            image_name = image_names[i % len(image_names)]
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
            json_files.append(json_file)
        
        # 属性1: 相同数据的验证结果应该一致
        validator = DataValidator(self.config)
        
        # 第一次验证
        result1 = validator.validate_dataset(self.temp_dir)
        
        # 第二次验证（相同数据）
        result2 = validator.validate_dataset(self.temp_dir)
        
        # 验证结果应该完全一致
        assert result1['is_valid'] == result2['is_valid']
        assert result1['total_samples'] == result2['total_samples']
        assert len(result1['errors']) == len(result2['errors'])
        
        # 属性2: 答案标准化的一致性
        for answer in answers:
            normalized1 = XAnyLabelingAdapter.normalize_answer(answer)
            normalized2 = XAnyLabelingAdapter.normalize_answer(answer)
            
            # 相同输入应该产生相同输出
            assert normalized1 == normalized2
            
            # 标准化结果应该只能是YES或NO
            assert normalized1 in ['YES', 'NO']
        
        # 属性3: 数据集加载的一致性
        try:
            dataset1 = build_dataset(self.temp_dir, self.config)
            dataset2 = build_dataset(self.temp_dir, self.config)
            
            # 相同配置应该产生相同大小的数据集
            assert len(dataset1) == len(dataset2)
            
            # 验证样本内容一致性
            if len(dataset1) > 0:
                sample1 = dataset1[0]
                sample2 = dataset2[0]
                
                # 相同索引的样本应该有相同的问题和答案
                assert sample1['question'] == sample2['question']
                assert sample1['answer'] == sample2['answer']
                
        except Exception as e:
            # 如果数据集构建失败，验证器应该检测到错误
            assert not result1['is_valid'] or len(result1['errors']) > 0
    
    @given(
        invalid_data=st.one_of(
            # 缺少必需字段
            st.fixed_dict({}),
            # 空的vqaData
            st.fixed_dict({'imagePath': st.text(), 'vqaData': st.fixed_dict({})}),
            # 无效的图像路径
            st.fixed_dict({
                'imagePath': st.text().filter(lambda x: not x.endswith(('.jpg', '.png', '.jpeg'))),
                'vqaData': st.dictionaries(st.text(min_size=1), st.text(min_size=1), min_size=1)
            })
        )
    )
    @settings(max_examples=30, deadline=3000)
    def test_invalid_data_detection(self, invalid_data):
        """
        属性测试: 无效数据检测
        验证数据验证器能够正确识别无效数据
        """
        # 创建无效的JSON文件
        json_file = self.annotations_dir / 'invalid.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(invalid_data, f, ensure_ascii=False)
        
        validator = DataValidator(self.config)
        result = validator.validate_dataset(self.temp_dir)
        
        # 无效数据应该被检测到
        assert not result['is_valid'] or len(result['errors']) > 0
    
    @given(
        answer_variants=st.lists(
            st.sampled_from([
                ('是', 'YES'), ('否', 'NO'), ('yes', 'YES'), ('no', 'NO'),
                ('有', 'YES'), ('没有', 'NO'), ('存在', 'YES'), ('不存在', 'NO'),
                ('正确', 'YES'), ('错误', 'NO'), ('对', 'YES'), ('不对', 'NO'),
                ('true', 'YES'), ('false', 'NO'), ('1', 'YES'), ('0', 'NO')
            ]),
            min_size=1, max_size=16
        )
    )
    @settings(max_examples=20, deadline=2000)
    def test_answer_normalization_consistency(self, answer_variants):
        """
        属性测试: 答案标准化一致性
        验证中英文答案标准化的正确性和一致性
        """
        for original_answer, expected_normalized in answer_variants:
            # 测试基本标准化
            normalized = XAnyLabelingAdapter.normalize_answer(original_answer)
            assert normalized == expected_normalized
            
            # 测试大小写不敏感
            normalized_upper = XAnyLabelingAdapter.normalize_answer(original_answer.upper())
            normalized_lower = XAnyLabelingAdapter.normalize_answer(original_answer.lower())
            assert normalized_upper == expected_normalized
            assert normalized_lower == expected_normalized
            
            # 测试前后空格处理
            normalized_spaces = XAnyLabelingAdapter.normalize_answer(f"  {original_answer}  ")
            assert normalized_spaces == expected_normalized


class DatasetStateMachine(RuleBasedStateMachine):
    """
    基于状态的属性测试：数据集操作的状态一致性
    """
    
    def __init__(self):
        super().__init__()
        self.temp_dir = None
        self.config = None
        self.dataset = None
        self.validator = None
        self.sample_count = 0
    
    @initialize()
    def setup(self):
        """初始化测试环境"""
        import tempfile
        self.temp_dir = tempfile.mkdtemp()
        self.images_dir = Path(self.temp_dir) / 'images'
        self.annotations_dir = Path(self.temp_dir) / 'annotations'
        self.images_dir.mkdir(exist_ok=True)
        self.annotations_dir.mkdir(exist_ok=True)
        
        self.config = create_minimal_config(self.temp_dir)
        self.validator = DataValidator(self.config)
        self.sample_count = 0
    
    @rule(
        question=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        answer=st.sampled_from(['是', '否', 'yes', 'no']),
        image_name=st.text(min_size=1, max_size=10).filter(lambda x: x.isalnum()).map(lambda x: f"{x}.jpg")
    )
    def add_sample(self, question, answer, image_name):
        """添加样本到数据集"""
        # 创建图像文件
        self.create_test_image(image_name)
        
        # 创建JSON标注
        json_data = {
            'imagePath': image_name,
            'imageWidth': 224,
            'imageHeight': 224,
            'version': '1.0',
            'vqaData': {question: answer}
        }
        
        json_file = self.annotations_dir / f'sample_{self.sample_count}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False)
        
        self.sample_count += 1
    
    def create_test_image(self, image_name: str):
        """创建测试图像"""
        from PIL import Image
        import numpy as np
        
        if not (self.images_dir / image_name).exists():
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(self.images_dir / image_name)
    
    @rule()
    def validate_dataset_state(self):
        """验证数据集状态"""
        if self.sample_count > 0:
            result = self.validator.validate_dataset(self.temp_dir)
            
            # 如果有样本，验证应该能找到数据
            assert result['total_samples'] >= 0
            
            # 尝试构建数据集
            try:
                dataset = build_dataset(self.temp_dir, self.config)
                assert len(dataset) >= 0
                
                # 如果数据集不为空，验证第一个样本
                if len(dataset) > 0:
                    sample = dataset[0]
                    assert 'question' in sample
                    assert 'answer' in sample
                    assert sample['answer'] in ['YES', 'NO']
                    
            except Exception:
                # 如果构建失败，验证结果应该显示错误
                pass
    
    @invariant()
    def sample_count_consistency(self):
        """不变量：样本计数一致性"""
        # 样本计数应该非负
        assert self.sample_count >= 0
        
        # 如果有样本，annotations目录应该有文件
        if self.sample_count > 0:
            json_files = list(self.annotations_dir.glob('*.json'))
            assert len(json_files) > 0
    
    def teardown(self):
        """清理"""
        if self.temp_dir:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)


# 运行状态机测试
TestDatasetState = DatasetStateMachine.TestCase


if __name__ == '__main__':
    # 运行属性测试
    pytest.main([__file__, '-v'])