"""
ROI推理器 (ROIInferencer)

支持PyTorch/ONNX/TensorRT模型加载和推理，包括：
- 单张ROI图像推理
- 批量ROI图像推理
- 推理结果后处理
"""

import os
import time
import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from PIL import Image
import psutil

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

from transformers import AutoTokenizer


@dataclass
class InferenceResult:
    """推理结果数据结构"""
    answer: str  # YES/NO
    confidence: float  # 置信度 [0, 1]
    probabilities: List[float]  # 类别概率分布
    inference_time_ms: float  # 推理时间(毫秒)
    model_format: str  # 模型格式
    batch_size: int = 1  # 批次大小


class ImagePreprocessor:
    """图像预处理器"""
    
    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        """
        初始化图像预处理器
        
        Args:
            image_size: 目标图像大小 (height, width)
        """
        self.image_size = image_size
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
    
    def preprocess(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """
        预处理单张图像
        
        Args:
            image: 输入图像 (numpy array或PIL Image)
            
        Returns:
            torch.Tensor: 预处理后的图像张量 [3, H, W]
        """
        # 转换为PIL Image
        if isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                image = Image.fromarray(image)
            else:
                # 假设是浮点数，范围[0, 1]
                image = Image.fromarray((image * 255).astype(np.uint8))
        
        # 调整大小
        image = image.resize(self.image_size, Image.BILINEAR)
        
        # 转换为numpy数组
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # 标准化
        image_array = (image_array - self.mean.astype(np.float32)) / self.std.astype(np.float32)
        
        # 转换为张量 [C, H, W]，确保是float32
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()
        
        return image_tensor
    
    def preprocess_batch(self, images: List[Union[np.ndarray, Image.Image]]) -> torch.Tensor:
        """
        预处理图像批次
        
        Args:
            images: 输入图像列表
            
        Returns:
            torch.Tensor: 预处理后的图像张量 [B, 3, H, W]
        """
        processed_images = []
        for image in images:
            processed_images.append(self.preprocess(image))
        
        return torch.stack(processed_images, dim=0)


class ROIInferencer:
    """
    ROI推理器
    
    支持PyTorch/ONNX/TensorRT模型加载和推理
    """
    
    def __init__(self, model_path: str, model_format: str = 'pytorch',
                 device: str = 'cuda', config: Optional[Dict[str, Any]] = None):
        """
        初始化ROI推理器
        
        Args:
            model_path: 模型文件路径
            model_format: 模型格式 ('pytorch', 'onnx', 'tensorrt')
            device: 推理设备 ('cuda', 'cpu')
            config: 配置字典
        """
        self.model_path = model_path
        self.model_format = model_format
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config = config or {}
        
        # 设置日志
        self.logger = self._setup_logger()
        
        # 验证模型文件存在
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # 加载模型
        self.model = self._load_model(model_path, model_format)
        
        # 初始化预处理器
        image_size = self.config.get('image_size', (224, 224))
        self.image_preprocessor = ImagePreprocessor(image_size)
        
        # 初始化文本分词器
        text_encoder = self.config.get('text_encoder', 'distilbert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder)
        
        # 推理配置
        self.max_text_length = self.config.get('max_text_length', 128)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        
        self.logger.info(f"ROIInferencer initialized with {model_format} model")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('ROIInferencer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_model(self, model_path: str, model_format: str) -> Any:
        """
        加载不同格式的模型
        
        Args:
            model_path: 模型文件路径
            model_format: 模型格式
            
        Returns:
            加载的模型
        """
        if model_format == 'pytorch':
            return self._load_pytorch_model(model_path)
        elif model_format == 'onnx':
            return self._load_onnx_model(model_path)
        elif model_format == 'tensorrt':
            return self._load_tensorrt_model(model_path)
        else:
            raise ValueError(f"Unsupported model format: {model_format}")
    
    def _load_pytorch_model(self, model_path: str) -> torch.nn.Module:
        """加载PyTorch模型"""
        self.logger.info(f"Loading PyTorch model from {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 导入模型类
        from ..models.speedvqa import SpeedVQAModel
        
        # 重建模型
        model_config = checkpoint.get('model_config', {})
        model = SpeedVQAModel(model_config)
        
        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        self.logger.info(f"PyTorch model loaded successfully")
        return model
    
    def _load_onnx_model(self, model_path: str) -> ort.InferenceSession:
        """加载ONNX模型"""
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX Runtime not available. Please install onnxruntime.")
        
        self.logger.info(f"Loading ONNX model from {model_path}")
        
        # 选择执行提供程序
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        if self.device.type == 'cpu':
            providers = ['CPUExecutionProvider']
        
        session = ort.InferenceSession(model_path, providers=providers)
        
        self.logger.info(f"ONNX model loaded successfully")
        return session
    
    def _load_tensorrt_model(self, model_path: str) -> Any:
        """加载TensorRT模型"""
        if not TENSORRT_AVAILABLE:
            raise RuntimeError("TensorRT not available. Please install tensorrt and pycuda.")
        
        self.logger.info(f"Loading TensorRT model from {model_path}")
        
        # 加载TensorRT引擎
        trt_logger = trt.Logger(trt.Logger.WARNING)
        
        with open(model_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(trt_logger)
        engine = runtime.deserialize_cuda_engine(engine_data)
        
        self.logger.info(f"TensorRT model loaded successfully")
        return engine
    
    def inference(self, roi_image: Union[np.ndarray, Image.Image, str],
                 question: str) -> InferenceResult:
        """
        单张ROI图像推理
        
        Args:
            roi_image: ROI图像 (numpy array、PIL Image或图像路径)
            question: 问题文本
            
        Returns:
            InferenceResult: 推理结果
        """
        start_time = time.time()
        
        # 加载图像
        if isinstance(roi_image, str):
            roi_image = Image.open(roi_image).convert('RGB')
        
        # 预处理图像
        image_tensor = self.image_preprocessor.preprocess(roi_image)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)  # [1, 3, H, W]
        
        # 预处理文本
        text_inputs = self.tokenizer(
            question,
            max_length=self.max_text_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = text_inputs['input_ids'].to(self.device)
        attention_mask = text_inputs['attention_mask'].to(self.device)
        
        # 模型推理
        if self.model_format == 'pytorch':
            logits = self._pytorch_inference(image_tensor, input_ids, attention_mask)
        elif self.model_format == 'onnx':
            logits = self._onnx_inference(image_tensor, input_ids, attention_mask)
        elif self.model_format == 'tensorrt':
            logits = self._tensorrt_inference(image_tensor, input_ids, attention_mask)
        
        # 后处理
        result = self._postprocess_output(logits, time.time() - start_time)
        result.batch_size = 1
        
        return result
    
    def batch_inference(self, roi_images: List[Union[np.ndarray, Image.Image, str]],
                       questions: List[str]) -> List[InferenceResult]:
        """
        批量ROI图像推理
        
        Args:
            roi_images: ROI图像列表
            questions: 问题文本列表
            
        Returns:
            List[InferenceResult]: 推理结果列表
        """
        if len(roi_images) != len(questions):
            raise ValueError("Number of images and questions must match")
        
        batch_size = len(roi_images)
        start_time = time.time()
        
        # 加载和预处理图像
        processed_images = []
        for roi_image in roi_images:
            if isinstance(roi_image, str):
                roi_image = Image.open(roi_image).convert('RGB')
            processed_images.append(self.image_preprocessor.preprocess(roi_image))
        
        image_tensor = torch.stack(processed_images, dim=0).to(self.device)  # [B, 3, H, W]
        
        # 预处理文本
        text_inputs = self.tokenizer(
            questions,
            max_length=self.max_text_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = text_inputs['input_ids'].to(self.device)
        attention_mask = text_inputs['attention_mask'].to(self.device)
        
        # 批量推理
        if self.model_format == 'pytorch':
            logits = self._pytorch_inference(image_tensor, input_ids, attention_mask)
        elif self.model_format == 'onnx':
            logits = self._onnx_inference(image_tensor, input_ids, attention_mask)
        elif self.model_format == 'tensorrt':
            logits = self._tensorrt_inference(image_tensor, input_ids, attention_mask)
        
        # 后处理
        total_time = time.time() - start_time
        results = []
        
        for i in range(batch_size):
            result = self._postprocess_output(logits[i:i+1], total_time / batch_size)
            result.batch_size = batch_size
            results.append(result)
        
        return results
    
    def _pytorch_inference(self, image_tensor: torch.Tensor,
                          input_ids: torch.Tensor,
                          attention_mask: torch.Tensor) -> torch.Tensor:
        """PyTorch模型推理"""
        with torch.no_grad():
            batch = {
                'image': image_tensor,
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
            outputs = self.model(batch)
            logits = outputs['logits']
        
        return logits
    
    def _onnx_inference(self, image_tensor: torch.Tensor,
                       input_ids: torch.Tensor,
                       attention_mask: torch.Tensor) -> torch.Tensor:
        """ONNX模型推理"""
        # 准备ONNX输入
        ort_inputs = {
            'image': image_tensor.cpu().numpy(),
            'input_ids': input_ids.cpu().numpy(),
            'attention_mask': attention_mask.cpu().numpy()
        }
        
        # 运行推理
        ort_outputs = self.model.run(None, ort_inputs)
        
        # 转换为张量
        logits = torch.from_numpy(ort_outputs[0]).to(self.device)
        
        return logits
    
    def _tensorrt_inference(self, image_tensor: torch.Tensor,
                           input_ids: torch.Tensor,
                           attention_mask: torch.Tensor) -> torch.Tensor:
        """TensorRT模型推理"""
        # 创建执行上下文
        context = self.model.create_execution_context()
        
        # 准备输入输出缓冲区
        batch_size = image_tensor.shape[0]
        
        # 分配GPU内存
        image_gpu = cuda.mem_alloc(image_tensor.cpu().numpy().nbytes)
        input_ids_gpu = cuda.mem_alloc(input_ids.cpu().numpy().nbytes)
        attention_mask_gpu = cuda.mem_alloc(attention_mask.cpu().numpy().nbytes)
        
        # 输出缓冲区
        output_shape = (batch_size, 2)  # 假设输出是[batch_size, num_classes]
        output_gpu = cuda.mem_alloc(np.prod(output_shape) * 4)  # 4字节用于float32
        
        # 复制输入到GPU
        cuda.memcpy_htod(image_gpu, image_tensor.cpu().numpy())
        cuda.memcpy_htod(input_ids_gpu, input_ids.cpu().numpy())
        cuda.memcpy_htod(attention_mask_gpu, attention_mask.cpu().numpy())
        
        # 运行推理
        context.execute_v2([
            int(image_gpu),
            int(input_ids_gpu),
            int(attention_mask_gpu),
            int(output_gpu)
        ])
        
        # 复制输出到CPU
        output_cpu = np.empty(output_shape, dtype=np.float32)
        cuda.memcpy_dtoh(output_cpu, output_gpu)
        
        # 释放GPU内存
        image_gpu.free()
        input_ids_gpu.free()
        attention_mask_gpu.free()
        output_gpu.free()
        
        # 转换为张量
        logits = torch.from_numpy(output_cpu).to(self.device)
        
        return logits
    
    def _postprocess_output(self, logits: torch.Tensor, inference_time: float) -> InferenceResult:
        """
        后处理推理输出
        
        Args:
            logits: 模型输出的logits [1, num_classes]
            inference_time: 推理时间(秒)
            
        Returns:
            InferenceResult: 后处理后的结果
        """
        # 计算概率
        probabilities = F.softmax(logits, dim=1)
        
        # 获取预测结果
        confidence, prediction = torch.max(probabilities, dim=1)
        
        # 转换为Python类型
        confidence = confidence.item()
        prediction = prediction.item()
        probabilities_list = probabilities.squeeze().cpu().numpy().tolist()
        
        # 转换为YES/NO答案
        answer = "YES" if prediction == 1 else "NO"
        
        # 创建结果对象
        result = InferenceResult(
            answer=answer,
            confidence=confidence,
            probabilities=probabilities_list,
            inference_time_ms=inference_time * 1000,
            model_format=self.model_format
        )
        
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = {
            'model_path': str(self.model_path),
            'model_format': self.model_format,
            'device': str(self.device),
            'file_size_mb': Path(self.model_path).stat().st_size / (1024 * 1024)
        }
        
        # 获取PyTorch模型的详细信息
        if self.model_format == 'pytorch' and hasattr(self.model, 'get_model_info'):
            info.update(self.model.get_model_info())
        
        return info


def build_roi_inferencer(model_path: str, model_format: str = 'pytorch',
                        device: str = 'cuda',
                        config: Optional[Dict[str, Any]] = None) -> ROIInferencer:
    """
    构建ROI推理器（YOLO风格的简单接口）
    
    Args:
        model_path: 模型文件路径
        model_format: 模型格式 ('pytorch', 'onnx', 'tensorrt')
        device: 推理设备 ('cuda', 'cpu')
        config: 配置字典
        
    Returns:
        ROIInferencer: ROI推理器实例
    """
    inferencer = ROIInferencer(model_path, model_format, device, config)
    
    # 打印推理器信息
    info = inferencer.get_model_info()
    print(f"\n=== ROI Inferencer Info ===")
    print(f"Model Path: {info['model_path']}")
    print(f"Model Format: {info['model_format']}")
    print(f"Device: {info['device']}")
    print(f"File Size: {info['file_size_mb']:.2f} MB")
    
    return inferencer


if __name__ == '__main__':
    # 测试ROI推理器
    import sys
    
    # 示例用法
    print("ROI Inferencer Module")
    print("=" * 50)
    print("\nUsage Example:")
    print("  from speedvqa.inference import ROIInferencer")
    print("  inferencer = ROIInferencer('model.pt', model_format='pytorch')")
    print("  result = inferencer.inference('roi_image.jpg', 'Is there a person?')")
    print("  print(result.answer, result.confidence)")
