#!/usr/bin/env python3
"""
SpeedVQA测试运行器

运行属性测试和单元测试，验证数据处理组件的正确性。
"""

import json
import os
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.chdir(_REPO_ROOT)

from hypothesis import given, settings, strategies as st

from speedvqa.data.validators import DataValidator, XAnyLabelingAdapter
from speedvqa.utils.config import create_minimal_config


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

        assert normalized1 == normalized2
        assert normalized1 in ['YES', 'NO']


def test_data_validator():
    """测试数据验证器"""
    print("\n=== 测试数据验证器 ===")

    temp_dir = tempfile.mkdtemp()
    images_dir = Path(temp_dir) / 'images'
    annotations_dir = Path(temp_dir) / 'annotations'
    images_dir.mkdir(exist_ok=True)
    annotations_dir.mkdir(exist_ok=True)

    try:
        test_image = images_dir / 'test.jpg'
        test_image.touch()

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

        config = create_minimal_config(temp_dir)
        validator = DataValidator(config)
        result = validator.validate_dataset(temp_dir)

        print("✓ 数据验证器创建成功")
        print(f"✓ 发现样本数: {result.get('total_samples', 0)}")
        print(f"✓ 验证结果: {'有效' if result.get('is_valid', False) else '无效'}")

        if result.get('errors'):
            print(f"  错误: {result['errors']}")

        return True

    except Exception as e:
        print(f"✗ 数据验证器测试失败: {e}")
        return False

    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_all_tests():
    """运行所有测试"""
    print("SpeedVQA 数据处理组件测试")
    print("=" * 50)

    results = []

    results.append(test_answer_normalization())
    results.append(test_data_validator())

    print("\n=== 运行属性测试 ===")
    try:
        property_test_answer_consistency()
        print("✓ 答案标准化一致性属性测试通过")
        results.append(True)
    except Exception as e:
        print(f"✗ 属性测试失败: {e}")
        results.append(False)

    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"测试结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有测试通过！数据处理组件工作正常。")
        return True
    print("❌ 部分测试失败，请检查实现。")
    return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
