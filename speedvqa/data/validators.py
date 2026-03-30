"""
SpeedVQA数据验证器

提供数据质量检查、标注一致性验证等功能。
完全支持X-AnyLabeling格式的验证。
"""

import os
import json
import glob
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np


class DataValidator:
    """数据验证器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.validation_errors = []
    
    def validate_dataset(self, data_path: str) -> Dict[str, Any]:
        """验证整个数据集"""
        data_path = Path(data_path)
        validation_result = {
            'is_valid': True,
            'total_samples': 0,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # 检查目录结构
        structure_result = self._validate_directory_structure(data_path)
        validation_result.update(structure_result)
        
        # 检查数据格式
        format_result = self._validate_data_formats(data_path)
        validation_result.update(format_result)
        
        # 检查数据质量
        quality_result = self._validate_data_quality(data_path)
        validation_result.update(quality_result)
        
        # 汇总结果
        validation_result['is_valid'] = len(validation_result['errors']) == 0
        
        return validation_result
    
    def _validate_directory_structure(self, data_path: Path) -> Dict[str, Any]:
        """验证目录结构"""
        result = {'structure_errors': []}
        
        # 检查必需的目录
        required_dirs = ['images']
        for dir_name in required_dirs:
            dir_path = data_path / dir_name
            if not dir_path.exists():
                result['structure_errors'].append(f"Missing required directory: {dir_name}")
        
        # 检查可选的目录
        optional_dirs = ['annotations']
        for dir_name in optional_dirs:
            dir_path = data_path / dir_name
            if dir_path.exists():
                result[f'has_{dir_name}'] = True
        
        return result
    
    def _validate_data_formats(self, data_path: Path) -> Dict[str, Any]:
        """验证数据格式"""
        result = {
            'format_errors': [],
            'has_json_annotations': False,
            'has_jsonl': False,
            'has_questions_txt': False,
            'total_samples': 0
        }
        
        # 检查JSON标注文件
        annotations_dir = data_path / 'annotations'
        if annotations_dir.exists():
            json_files = list(annotations_dir.glob('*.json'))
            if json_files:
                result['has_json_annotations'] = True
                json_errors = self._validate_json_files(json_files)
                result['format_errors'].extend(json_errors)
                result['total_samples'] += len(json_files)
        
        # 检查JSONL文件
        jsonl_file = data_path / 'vqa_labels.jsonl'
        if jsonl_file.exists():
            result['has_jsonl'] = True
            jsonl_errors, jsonl_count = self._validate_jsonl_file(jsonl_file)
            result['format_errors'].extend(jsonl_errors)
            result['total_samples'] += jsonl_count
        
        # 检查questions.txt
        questions_file = data_path / 'questions.txt'
        if questions_file.exists():
            result['has_questions_txt'] = True
            txt_errors, txt_count = self._validate_txt_file(questions_file)
            result['format_errors'].extend(txt_errors)
            result['total_samples'] += txt_count
        
        # 检查是否有任何有效的数据源
        if not any([result['has_json_annotations'], result['has_jsonl'], result['has_questions_txt']]):
            result['format_errors'].append("No valid data source found (JSON, JSONL, or TXT)")
        
        return result
    
    def _validate_json_files(self, json_files: List[Path]) -> List[str]:
        """验证JSON标注文件"""
        errors = []
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 检查必需字段
                required_fields = ['imagePath']
                for field in required_fields:
                    if field not in data:
                        errors.append(f"{json_file}: Missing required field '{field}'")
                
                # 检查vqaData字段
                if 'vqaData' not in data or not data['vqaData']:
                    errors.append(f"{json_file}: Missing or empty 'vqaData' field")
                else:
                    # 验证问答对格式
                    for question, answer in data['vqaData'].items():
                        if not question.strip():
                            errors.append(f"{json_file}: Empty question found")
                        
                        if isinstance(answer, list):
                            if not answer:
                                errors.append(f"{json_file}: Empty answer list for question '{question}'")
                        elif not str(answer).strip():
                            errors.append(f"{json_file}: Empty answer for question '{question}'")
                
                # 检查图像文件是否存在
                image_path = json_file.parent.parent / 'images' / data.get('imagePath', '')
                if not image_path.exists():
                    errors.append(f"{json_file}: Referenced image not found: {data.get('imagePath', '')}")
            
            except json.JSONDecodeError as e:
                errors.append(f"{json_file}: Invalid JSON format: {e}")
            except Exception as e:
                errors.append(f"{json_file}: Validation error: {e}")
        
        return errors
    
    def _validate_jsonl_file(self, jsonl_file: Path) -> Tuple[List[str], int]:
        """验证JSONL文件"""
        errors = []
        sample_count = 0
        
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        sample_count += 1
                        
                        # 检查必需字段
                        if 'image' not in data:
                            errors.append(f"{jsonl_file}:{line_num}: Missing 'image' field")
                            continue
                        
                        # 检查图像文件是否存在
                        image_path = jsonl_file.parent / 'images' / data['image']
                        if not image_path.exists():
                            errors.append(f"{jsonl_file}:{line_num}: Referenced image not found: {data['image']}")
                        
                        # 检查是否有问答对（除了image, width, height字段）
                        qa_fields = [k for k in data.keys() if k not in ['image', 'width', 'height']]
                        if not qa_fields:
                            errors.append(f"{jsonl_file}:{line_num}: No question-answer pairs found")
                        
                        # 验证问答对
                        for question in qa_fields:
                            if not question.strip():
                                errors.append(f"{jsonl_file}:{line_num}: Empty question found")
                            
                            answer = data[question]
                            if isinstance(answer, list):
                                if not answer:
                                    errors.append(f"{jsonl_file}:{line_num}: Empty answer list for question '{question}'")
                            elif not str(answer).strip():
                                errors.append(f"{jsonl_file}:{line_num}: Empty answer for question '{question}'")
                    
                    except json.JSONDecodeError as e:
                        errors.append(f"{jsonl_file}:{line_num}: Invalid JSON: {e}")
        
        except Exception as e:
            errors.append(f"{jsonl_file}: File reading error: {e}")
        
        return errors, sample_count
    
    def _validate_txt_file(self, txt_file: Path) -> Tuple[List[str], int]:
        """验证TXT文件"""
        errors = []
        sample_count = 0
        
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split(',')
                    if len(parts) < 7:
                        errors.append(f"{txt_file}:{line_num}: Invalid format, expected at least 7 fields")
                        continue
                    
                    sample_count += 1
                    
                    # 验证图像文件
                    image_name = parts[0]
                    image_path = txt_file.parent / 'images' / image_name
                    if not image_path.exists():
                        errors.append(f"{txt_file}:{line_num}: Referenced image not found: {image_name}")
                    
                    # 验证bbox坐标
                    try:
                        bbox = [int(parts[i]) for i in range(1, 5)]
                        if any(coord < 0 for coord in bbox):
                            errors.append(f"{txt_file}:{line_num}: Invalid bbox coordinates (negative values)")
                        if bbox[2] <= 0 or bbox[3] <= 0:
                            errors.append(f"{txt_file}:{line_num}: Invalid bbox size (width or height <= 0)")
                    except ValueError:
                        errors.append(f"{txt_file}:{line_num}: Invalid bbox coordinates (not integers)")
                    
                    # 验证问题和答案
                    question = parts[5].strip()
                    answer = parts[6].strip()
                    
                    if not question:
                        errors.append(f"{txt_file}:{line_num}: Empty question")
                    if not answer:
                        errors.append(f"{txt_file}:{line_num}: Empty answer")
        
        except Exception as e:
            errors.append(f"{txt_file}: File reading error: {e}")
        
        return errors, sample_count
    
    def _validate_data_quality(self, data_path: Path) -> Dict[str, Any]:
        """验证数据质量"""
        result = {
            'quality_warnings': [],
            'statistics': {}
        }
        
        # 这里可以添加更多的数据质量检查
        # 例如：重复样本检测、答案分布检查等
        
        return result
    
    def validate_annotation(self, annotation: Dict[str, Any]) -> bool:
        """验证单个标注"""
        self.validation_errors = []
        
        # 检查必需字段
        required_fields = ['question', 'answer']
        for field in required_fields:
            if field not in annotation or not annotation[field]:
                self.validation_errors.append(f"Missing or empty field: {field}")
        
        # 检查问题格式
        if 'question' in annotation:
            question = annotation['question']
            if not isinstance(question, str) or not question.strip():
                self.validation_errors.append("Question must be a non-empty string")
        
        # 检查答案格式
        if 'answer' in annotation:
            answer = annotation['answer']
            if answer not in ['YES', 'NO']:
                self.validation_errors.append(f"Answer must be 'YES' or 'NO', got: {answer}")
        
        # 检查bbox格式（如果存在）
        if 'bbox' in annotation:
            bbox = annotation['bbox']
            if not isinstance(bbox, list) or len(bbox) != 4:
                self.validation_errors.append("bbox must be a list of 4 numbers")
            else:
                try:
                    x, y, w, h = bbox
                    if w <= 0 or h <= 0:
                        self.validation_errors.append("bbox width and height must be positive")
                    if x < 0 or y < 0:
                        self.validation_errors.append("bbox coordinates must be non-negative")
                except (ValueError, TypeError):
                    self.validation_errors.append("bbox must contain numeric values")
        
        return len(self.validation_errors) == 0
    
    def get_validation_errors(self, annotation: Dict[str, Any]) -> List[str]:
        """获取验证错误信息"""
        self.validate_annotation(annotation)
        return self.validation_errors.copy()


class XAnyLabelingAdapter:
    """X-AnyLabeling标注格式适配器"""
    
    @staticmethod
    def normalize_answer(answer: str) -> str:
        """标准化中英文答案"""
        answer = answer.strip().lower()
        
        # 中文肯定答案
        positive_cn = ['是', '有', '存在', '正确', '对']
        # 英文肯定答案
        positive_en = ['yes', 'y', 'true', '1']
        
        # 中文否定答案 - 注意顺序，长的词放前面避免部分匹配
        negative_cn = ['没有', '不存在', '不是', '不对', '否', '错误']
        # 英文否定答案
        negative_en = ['no', 'n', 'false', '0']
        
        # 先检查否定答案（包含更长的词组）
        for word in negative_cn + negative_en:
            if word in answer:
                return 'NO'
        
        # 再检查肯定答案
        for word in positive_cn + positive_en:
            if word in answer:
                return 'YES'
        
        # 默认为NO
        return 'NO'
    
    @staticmethod
    def validate_data_path(data_path: str) -> Dict[str, Any]:
        """验证X-AnyLabeling数据路径"""
        data_path = Path(data_path)
        validation_result = {
            'is_valid': False,
            'has_images': False,
            'has_json_annotations': False,
            'has_jsonl': False,
            'has_questions_txt': False,
            'total_samples': 0,
            'errors': []
        }
        
        # 检查images目录
        images_dir = data_path / 'images'
        if images_dir.exists() and any(images_dir.iterdir()):
            validation_result['has_images'] = True
        else:
            validation_result['errors'].append("Missing or empty 'images' directory")
        
        # 检查JSONL文件
        jsonl_file = data_path / 'vqa_labels.jsonl'
        if jsonl_file.exists():
            validation_result['has_jsonl'] = True
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                validation_result['total_samples'] += sum(1 for _ in f)
        
        # 检查JSON标注文件
        annotations_dir = data_path / 'annotations'
        if annotations_dir.exists():
            json_files = list(annotations_dir.glob('*.json'))
            validation_result['has_json_annotations'] = len(json_files) > 0
            validation_result['total_samples'] += len(json_files)
        
        # 检查questions.txt
        questions_file = data_path / 'questions.txt'
        validation_result['has_questions_txt'] = questions_file.exists()
        
        # 检查是否有任何数据源
        has_data_source = any([
            validation_result['has_jsonl'],
            validation_result['has_json_annotations'],
            validation_result['has_questions_txt']
        ])
        
        if not has_data_source:
            validation_result['errors'].append("No valid data source found")
        
        validation_result['is_valid'] = (
            validation_result['has_images'] and 
            has_data_source and 
            len(validation_result['errors']) == 0
        )
        
        return validation_result
    
    @staticmethod
    def convert_to_unified_format(data_path: str, output_path: str):
        """将所有格式转换为统一的JSONL格式"""
        from .datasets import VQADataset
        
        # 创建临时配置
        temp_config = {
            'model': {'text': {'encoder': 'distilbert-base-uncased', 'max_length': 128}},
            'data': {'answer_mapping': {
                'positive': ['是', 'yes', 'y', '有', '存在', '正确', '对', 'true', '1'],
                'negative': ['否', 'no', 'n', '不是', '没有', '不存在', '错误', '不对', 'false', '0'],
                'default': 'NO'
            }}
        }
        
        dataset = VQADataset(data_path, temp_config)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in dataset.samples:
                unified_sample = {
                    'image': os.path.basename(sample['image_path']),
                    'question': sample['question'],
                    'answer': sample['answer'],
                    'source': sample.get('source', 'unknown'),
                    'metadata': sample.get('metadata', {})
                }
                f.write(json.dumps(unified_sample, ensure_ascii=False) + '\n')
        
        print(f"Converted {len(dataset.samples)} samples to {output_path}")