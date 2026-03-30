"""
SpeedVQA数据集加载器

完全兼容X-AnyLabeling标注格式，支持JSON和JSONL两种格式。
自动标准化中英文答案，提供数据验证和质量检查功能。
"""

import os
import json
import glob
import random
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from transformers import AutoTokenizer
import numpy as np

from .loaders import load_vqa_official_if_enabled


class VQADataset(Dataset):
    """VQA数据集加载器（完全兼容X-AnyLabeling）"""
    
    def __init__(self, data_path: str, config: Dict, split: str = 'train'):
        self.data_path = Path(data_path)
        self.config = config
        self.split = split
        self.samples = self._load_samples()
        self.transform = self._build_transforms()
        
        # 初始化文本分词器
        text_config = config.get('model', {}).get('text', {})
        encoder_name = text_config.get('encoder', 'distilbert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.max_length = text_config.get('max_length', 128)
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_samples(self) -> List[Dict]:
        """加载训练样本（支持X-AnyLabeling的多种格式）"""
        samples = []
        
        # 格式1: 支持单个JSON文件（X-AnyLabeling单张图片标注）
        annotations_dir = self.data_path / 'annotations'
        if annotations_dir.exists():
            json_files = list(annotations_dir.glob('*.json'))
            for json_file in json_files:
                samples.extend(self._load_from_single_json(json_file))
        
        # 格式2: 支持JSONL文件（X-AnyLabeling批量标注）
        jsonl_file = self.data_path / 'vqa_labels.jsonl'
        if jsonl_file.exists():
            samples.extend(self._load_from_jsonl(jsonl_file))
        
        # 格式3: 支持简单的questions.txt格式（向后兼容）
        questions_file = self.data_path / 'questions.txt'
        if questions_file.exists():
            samples.extend(self._load_from_txt(questions_file))
        
        # 格式4: 官方 VQA 发布（visualqa.org）Questions + Annotations JSON + images/
        vo = self.config.get('data', {}).get('vqa_official', {})
        samples.extend(
            load_vqa_official_if_enabled(
                self.data_path,
                self._normalize_answer,
                vo.get('enabled', 'auto'),
                vo.get('questions_json'),
                vo.get('annotations_json'),
            )
        )
        
        samples = self._maybe_subsample(samples)
        
        if not samples:
            raise ValueError(f"No valid data found in {self.data_path}")
        
        print(f"Data loading summary:")
        print(f"  - Total samples: {len(samples)}")
        print(f"  - Data sources: {self._get_data_sources(samples)}")
        
        return samples
    
    def _maybe_subsample(self, samples: List[Dict]) -> List[Dict]:
        """快速试验：data.max_samples + 可选 data.subsample_seed（先打乱再截断，可复现）。"""
        dc = self.config.get("data", {})
        raw = dc.get("max_samples")
        if raw is None:
            return samples
        try:
            max_n = int(raw)
        except (TypeError, ValueError):
            return samples
        if max_n <= 0 or len(samples) <= max_n:
            return samples
        sub_seed = dc.get("subsample_seed")
        if sub_seed is not None:
            rng = random.Random(int(sub_seed))
            order = list(range(len(samples)))
            rng.shuffle(order)
            picked = [samples[i] for i in order[:max_n]]
            print(f"Subsampled {len(samples)} -> {len(picked)} (max_samples={max_n}, subsample_seed={sub_seed})")
            return picked
        print(f"Subsampled {len(samples)} -> {max_n} (max_samples, head slice)")
        return samples[:max_n]
    
    def _load_from_single_json(self, json_file: Path) -> List[Dict]:
        """从X-AnyLabeling单个JSON文件加载数据"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"Warning: Failed to load {json_file}: {e}")
            return []
        
        samples = []
        image_path = self.data_path / 'images' / data.get('imagePath', '')
        
        # 检查图像文件是否存在
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            return []
        
        # 解析vqaData字段
        vqa_data = data.get('vqaData', {})
        if not vqa_data:
            print(f"Warning: No vqaData found in {json_file}")
            return []
        
        for question, answer in vqa_data.items():
            # 处理答案格式（可能是字符串或列表）
            if isinstance(answer, list):
                answer_text = answer[0] if answer else "否"
            else:
                answer_text = str(answer)
            
            # 标准化答案格式
            normalized_answer = self._normalize_answer(answer_text)
            
            samples.append({
                'image_path': str(image_path),
                'question': question.strip(),
                'answer': normalized_answer,
                'source': 'x_anylabeling_json',
                'metadata': {
                    'image_width': data.get('imageWidth', 0),
                    'image_height': data.get('imageHeight', 0),
                    'version': data.get('version', 'unknown'),
                    'json_file': str(json_file)
                }
            })
        
        return samples
    
    def _load_from_jsonl(self, jsonl_file: Path) -> List[Dict]:
        """从X-AnyLabeling JSONL文件加载数据"""
        samples = []
        
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON at line {line_num}: {e}")
                        continue
                    
                    image_name = data.get('image', '')
                    if not image_name:
                        continue
                    
                    image_path = self.data_path / 'images' / image_name
                    if not image_path.exists():
                        print(f"Warning: Image not found: {image_path}")
                        continue
                    
                    # 遍历所有问答对（除了image, width, height字段）
                    for key, value in data.items():
                        if key not in ['image', 'width', 'height']:
                            question = key
                            
                            # 处理答案格式
                            if isinstance(value, list):
                                answer_text = value[0] if value else "否"
                            else:
                                answer_text = str(value)
                            
                            # 标准化答案格式
                            normalized_answer = self._normalize_answer(answer_text)
                            
                            samples.append({
                                'image_path': str(image_path),
                                'question': question.strip(),
                                'answer': normalized_answer,
                                'source': 'x_anylabeling_jsonl',
                                'metadata': {
                                    'image_width': data.get('width', 0),
                                    'image_height': data.get('height', 0),
                                    'line_number': line_num
                                }
                            })
        
        except (FileNotFoundError, UnicodeDecodeError) as e:
            print(f"Warning: Failed to load {jsonl_file}: {e}")
        
        return samples
    
    def _load_from_txt(self, txt_file: Path) -> List[Dict]:
        """从questions.txt加载数据（向后兼容）"""
        samples = []
        
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split(',')
                    if len(parts) >= 7:
                        image_name = parts[0]
                        bbox = [int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])]
                        question = parts[5]
                        answer = parts[6]
                        
                        image_path = self.data_path / 'images' / image_name
                        if image_path.exists():
                            samples.append({
                                'image_path': str(image_path),
                                'bbox': bbox,
                                'question': question.strip(),
                                'answer': self._normalize_answer(answer),
                                'source': 'questions_txt',
                                'metadata': {'line_number': line_num}
                            })
        
        except (FileNotFoundError, UnicodeDecodeError) as e:
            print(f"Warning: Failed to load {txt_file}: {e}")
        
        return samples
    
    def _normalize_answer(self, answer: str) -> str:
        """标准化答案格式（支持中英文）"""
        answer = answer.strip().lower()
        
        # 获取配置中的答案映射
        answer_config = self.config.get('data', {}).get('answer_mapping', {})
        positive_words = answer_config.get('positive', ['是', 'yes', 'y', '有', '存在', '正确', '对', 'true', '1'])
        negative_words = answer_config.get('negative', ['否', 'no', 'n', '不是', '没有', '不存在', '错误', '不对', 'false', '0'])
        default_answer = answer_config.get('default', 'NO')
        
        # 检查肯定答案
        if any(word in answer for word in positive_words):
            return 'YES'
        
        # 检查否定答案
        if any(word in answer for word in negative_words):
            return 'NO'
        
        # 默认答案
        return default_answer
    
    def _build_transforms(self):
        """构建图像预处理变换"""
        image_config = self.config.get('data', {}).get('image', {})
        size = image_config.get('size', [224, 224])
        mean = image_config.get('mean', [0.485, 0.456, 0.406])
        std = image_config.get('std', [0.229, 0.224, 0.225])
        
        transform_list = [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
        
        # 训练时添加数据增强
        if self.split == 'train':
            aug_config = self.config.get('data', {}).get('augmentation', {})
            if aug_config.get('enabled', True):
                augmentations = []
                
                if aug_config.get('random_flip', True):
                    augmentations.append(transforms.RandomHorizontalFlip(p=0.5))
                
                if aug_config.get('color_jitter', {}):
                    cj_config = aug_config['color_jitter']
                    augmentations.append(transforms.ColorJitter(
                        brightness=cj_config.get('brightness', 0.2),
                        contrast=cj_config.get('contrast', 0.2),
                        saturation=cj_config.get('saturation', 0.2),
                        hue=cj_config.get('hue', 0.1)
                    ))
                
                if aug_config.get('random_rotation', 0) > 0:
                    augmentations.append(
                        transforms.RandomRotation(aug_config['random_rotation'])
                    )
                
                # 在Resize之后、ToTensor之前插入增强
                transform_list = [transforms.Resize(size)] + augmentations + transform_list[1:]
        
        return transforms.Compose(transform_list)
    
    def _get_data_sources(self, samples: List[Dict]) -> Dict[str, int]:
        """获取数据源统计"""
        sources = {}
        for sample in samples:
            source = sample.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1
        return sources
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个训练样本"""
        sample = self.samples[idx]
        
        # 加载和预处理图像
        try:
            image = Image.open(sample['image_path']).convert('RGB')
            
            # 如果有bbox信息，裁剪ROI区域
            if 'bbox' in sample:
                x, y, w, h = sample['bbox']
                image = image.crop((x, y, x+w, y+h))
            
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {sample['image_path']}: {e}")
            # 创建一个默认的黑色图像
            image = torch.zeros(3, 224, 224)
        
        # 预处理文本
        try:
            text_inputs = self.tokenizer(
                sample['question'],
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = text_inputs['input_ids'].squeeze()
            attention_mask = text_inputs['attention_mask'].squeeze()
        except Exception as e:
            # 如果tokenizer失败，使用默认值
            print(f"Warning: Tokenizer error for question '{sample['question']}': {e}")
            input_ids = torch.zeros(self.max_length, dtype=torch.long)
            attention_mask = torch.zeros(self.max_length, dtype=torch.long)
        
        # 标签编码
        label = 1 if sample['answer'] == 'YES' else 0
        
        return {
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long),
            'question': sample['question'],  # 保留原始问题用于调试
            'answer': sample['answer'],      # 保留原始答案用于调试
            'metadata': sample.get('metadata', {})
        }
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def get_class_distribution(self) -> Dict[str, int]:
        """获取类别分布统计"""
        distribution = {'YES': 0, 'NO': 0}
        for sample in self.samples:
            distribution[sample['answer']] += 1
        return distribution
    
    def get_question_statistics(self) -> Dict[str, int]:
        """获取问题统计"""
        question_counts = {}
        for sample in self.samples:
            question = sample['question']
            question_counts[question] = question_counts.get(question, 0) + 1
        return question_counts


def build_dataset(data_path: str, config: Dict, split: str = 'train') -> VQADataset:
    """构建数据集（YOLO风格的简单接口）"""
    dataset = VQADataset(data_path, config, split)
    
    # 打印数据集统计信息
    print(f"\n=== {split.upper()} Dataset Statistics ===")
    print(f"Total samples: {len(dataset)}")
    
    class_dist = dataset.get_class_distribution()
    print(f"Class distribution: {class_dist}")
    
    question_stats = dataset.get_question_statistics()
    print(f"Unique questions: {len(question_stats)}")
    for question, count in sorted(question_stats.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  '{question}': {count} samples")
    
    return dataset


def split_dataset(dataset: VQADataset, train_ratio: float = 0.7, 
                 val_ratio: float = 0.2, test_ratio: float = 0.1) -> Tuple[Dataset, Dataset, Dataset]:
    """划分数据集为训练/验证/测试集"""
    
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    # 使用随机划分
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # 固定随机种子确保可重现
    )
    
    print(f"\nDataset split:")
    print(f"Train: {len(train_dataset)} samples ({len(train_dataset)/total_size*100:.1f}%)")
    print(f"Val: {len(val_dataset)} samples ({len(val_dataset)/total_size*100:.1f}%)")
    print(f"Test: {len(test_dataset)} samples ({len(test_dataset)/total_size*100:.1f}%)")
    
    return train_dataset, val_dataset, test_dataset


def create_dataloader(dataset: Dataset, config: Dict, split: str = 'train') -> DataLoader:
    """创建数据加载器"""
    if split == 'train':
        dataloader_config = config.get('data', {}).get('dataloader', {})
        return DataLoader(
            dataset,
            batch_size=dataloader_config.get('batch_size', 32),
            shuffle=dataloader_config.get('shuffle', True),
            num_workers=dataloader_config.get('num_workers', 4),
            pin_memory=dataloader_config.get('pin_memory', True),
            drop_last=dataloader_config.get('drop_last', True)
        )
    else:
        val_config = config.get('val', {})
        return DataLoader(
            dataset,
            batch_size=val_config.get('batch_size', 64),
            shuffle=False,
            num_workers=val_config.get('num_workers', 4),
            pin_memory=True
        )