#!/usr/bin/env python3
"""
SpeedVQA测试运行器

运行属性测试和单元测试，验证数据处理组件的正确性。
"""

import sys
import os
sys.path.append('.')

from hypothesis import given, strategies as st, settings
from speedvqa.data.validators import XAnyLabelingAdapter, DataValidator
from speedvqa.utils.config import create_minimal_config
import tempfile
import json
from pathlib import Path


def test_answer_normalization():
    """测试答案标准化功能"""
    print("=== 测试答案标准化功能 ===")
    
    test_cases = [
        ('是', 'YES'), ('否', 'NO'), ('yes', 'YES'), ('no', 'NO'),
        ('有', 'YES'), ('没有', 'NO'), ('true', 'YES'), ('false', 'NO'),
        ('存在', 'YES'), ('不存在', 'NO'), ('正确', 'YES'), ('错误', 'NO'),
        ('对', 'YES'), ('不对', 'NO'), ('1', 'YES'), ('0', 'NO')
    ]
    
    all_passed = True
    for original, expected in test_cases:
        result = XAnyLabelingAdapter.normalize_answer(original)
        status = '✓' if result == expected else '✗'
        if result != expected:
            all_passed = False
        print(f'{status} {original} -> {result} (expected: {expected})')
    
    print(f"答案标准化测试: {'通过' if all_passed else '失败'}")
    return all_passed


@given(
    answers=st.lists(
        st.sampled_from(['是', '否', 'yes', 'no', 'YES', 'NO', '有', '没有', 'true', 'false']),
        min_size=1, max_size=10
    )
)
@settings(max_examples=20, deadline=2000)
def property_test_answer_consistency(answers):
    """属性测试：答案标准化一致性"""
    for answer in answers:
        normalized1 = XAnyLabelingAdapter.normalize_answer(answer)
        normalized2 = XAnyLabelingAdapter.normalize_answer(answer)
        
        # 相同输入应该产生相同输出
        assert normalized1 == normalized2
        
        # 标准化结果应该只能是YES或NO
        assert normalized1 in ['YES', 'NO']


def test_data_validator():
    """测试数据验证器"""
    print("\n=== 测试数据验证器 ===")
    
    # 创建临时测试环境
    temp_dir = tempfile.mkdtemp()
    images_dir = Path(temp_dir) / 'images'
    annotations_dir = Path(temp_dir) / 'annotations'
    images_dir.mkdir(exist_ok=True)
    annotations_dir.mkdir(exist_ok=True)
    
    try:
        # 创建测试图像文件（空文件用于测试）
        test_image = images_dir / 'test.jpg'
        test_image.touch()
        
        # 创建测试JSON标注
        json_data = {
            'imagePath': 'test.jpg',
            'imageWidth': 224,
            'imageHeight': 224,
            'version': '1.0',
            'vqaData': {
                '图片中有人吗？': '是',
                'Is there a car?': 'no'
            }
        }
        
        json_file = annotations_dir / 'test.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False)
        
        # 测试数据验证
        config = create_minimal_config(temp_dir)
        validator = DataValidator(config)
        result = validator.validate_dataset(temp_dir)
        
        print(f"✓ 数据验证器创建成功")
        print(f"✓ 发现样本数: {result.get('total_samples', 0)}")
        print(f"✓ 验证结果: {'有效' if result.get('is_valid', False) else '无效'}")
        
        if result.get('errors'):
            print(f"  错误: {result['errors']}")
        
        return True
        
    except Exception as e:
        print(f"✗ 数据验证器测试失败: {e}")
        return False
    
    finally:
        # 清理临时文件
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_all_tests():
    """运行所有测试"""
    print("SpeedVQA 数据处理组件测试")
    print("=" * 50)
    
    results = []
    
    # 基本功能测试
    results.append(test_answer_normalization())
    results.append(test_data_validator())
    
    # 属性测试
    print("\n=== 运行属性测试 ===")
    try:
        property_test_answer_consistency()
        print("✓ 答案标准化一致性属性测试通过")
        results.append(True)
    except Exception as e:
        print(f"✗ 属性测试失败: {e}")
        results.append(False)
    
    # 总结
    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！数据处理组件工作正常。")
        return True
    else:
        print("❌ 部分测试失败，请检查实现。")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)