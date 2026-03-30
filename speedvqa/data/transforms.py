"""
SpeedVQA数据预处理变换

提供图像预处理、数据增强等功能。
"""

from typing import Dict, Any, List
import torch
from torchvision import transforms
from PIL import Image
import numpy as np


def build_transforms(config: Dict[str, Any], split: str = 'train') -> transforms.Compose:
    """构建图像预处理变换"""
    image_config = config.get('data', {}).get('image', {})
    size = image_config.get('size', [224, 224])
    mean = image_config.get('mean', [0.485, 0.456, 0.406])
    std = image_config.get('std', [0.229, 0.224, 0.225])
    interpolation = image_config.get('interpolation', 'bilinear')
    
    # 插值方法映射
    interpolation_map = {
        'bilinear': transforms.InterpolationMode.BILINEAR,
        'bicubic': transforms.InterpolationMode.BICUBIC,
        'nearest': transforms.InterpolationMode.NEAREST
    }
    interp_mode = interpolation_map.get(interpolation, transforms.InterpolationMode.BILINEAR)
    
    transform_list = []
    
    # 基础变换
    transform_list.extend([
        transforms.Resize(size, interpolation=interp_mode),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # 训练时添加数据增强
    if split == 'train':
        aug_config = config.get('data', {}).get('augmentation', {})
        if aug_config.get('enabled', True):
            augmentations = []
            
            # 随机水平翻转
            if aug_config.get('random_flip', True):
                augmentations.append(transforms.RandomHorizontalFlip(p=0.5))
            
            # 颜色抖动
            if aug_config.get('color_jitter', {}):
                cj_config = aug_config['color_jitter']
                augmentations.append(transforms.ColorJitter(
                    brightness=cj_config.get('brightness', 0.2),
                    contrast=cj_config.get('contrast', 0.2),
                    saturation=cj_config.get('saturation', 0.2),
                    hue=cj_config.get('hue', 0.1)
                ))
            
            # 随机旋转
            if aug_config.get('random_rotation', 0) > 0:
                augmentations.append(
                    transforms.RandomRotation(aug_config['random_rotation'])
                )
            
            # 随机裁剪
            if aug_config.get('random_crop', False):
                augmentations.append(
                    transforms.RandomResizedCrop(
                        size, 
                        scale=(0.8, 1.0),
                        ratio=(0.75, 1.33)
                    )
                )
            
            # 在Resize之后、ToTensor之前插入增强
            if augmentations:
                if aug_config.get('random_crop', False):
                    # 如果使用随机裁剪，替换Resize
                    transform_list = augmentations + transform_list[1:]
                else:
                    # 否则在Resize之后添加增强
                    transform_list = [transform_list[0]] + augmentations + transform_list[1:]
    
    return transforms.Compose(transform_list)


class ImagePreprocessor:
    """图像预处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.transform = build_transforms(config, 'inference')
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """预处理单张图像"""
        if isinstance(image, np.ndarray):
            # 转换numpy数组为PIL图像
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        return self.transform(image)
    
    def preprocess_batch(self, images: List[np.ndarray]) -> torch.Tensor:
        """批量预处理图像"""
        processed_images = []
        for image in images:
            processed_images.append(self.preprocess(image))
        
        return torch.stack(processed_images)


def denormalize_image(tensor: torch.Tensor, mean: List[float] = None, 
                     std: List[float] = None) -> torch.Tensor:
    """反标准化图像张量"""
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    
    return tensor * std + mean


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """将张量转换为PIL图像"""
    # 反标准化
    tensor = denormalize_image(tensor)
    
    # 限制到[0, 1]范围
    tensor = torch.clamp(tensor, 0, 1)
    
    # 转换为PIL图像
    tensor = (tensor * 255).byte()
    if tensor.dim() == 3:
        tensor = tensor.permute(1, 2, 0)
    
    return Image.fromarray(tensor.numpy())


class CustomTransforms:
    """自定义变换"""
    
    @staticmethod
    def random_erasing(p: float = 0.1, scale: tuple = (0.02, 0.33), 
                      ratio: tuple = (0.3, 3.3), value: str = 'random'):
        """随机擦除变换"""
        return transforms.RandomErasing(
            p=p, scale=scale, ratio=ratio, value=value
        )
    
    @staticmethod
    def gaussian_blur(kernel_size: int = 3, sigma: tuple = (0.1, 2.0)):
        """高斯模糊变换"""
        return transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    
    @staticmethod
    def random_grayscale(p: float = 0.1):
        """随机灰度变换"""
        return transforms.RandomGrayscale(p=p)


def create_test_transforms(size: tuple = (224, 224)) -> transforms.Compose:
    """创建测试时的变换（无增强）"""
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def create_inference_transforms(config: Dict[str, Any]) -> transforms.Compose:
    """创建推理时的变换"""
    return build_transforms(config, 'inference')