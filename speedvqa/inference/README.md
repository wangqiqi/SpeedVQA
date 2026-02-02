# ROI推理器 (ROIInferencer)

ROI推理器是SpeedVQA系统的核心推理模块，支持多种模型格式的加载和推理。

## 功能特性

### 支持的模型格式
- **PyTorch (.pt)**: 原生PyTorch模型格式
- **ONNX (.onnx)**: 跨平台推理格式
- **TensorRT (.engine)**: NVIDIA GPU优化格式

### 核心功能
- ✅ 单张ROI图像推理
- ✅ 批量ROI图像推理
- ✅ 推理结果后处理
- ✅ 多种输入格式支持（numpy array、PIL Image、图像路径）
- ✅ 自动图像预处理和标准化
- ✅ 推理时间统计

## 快速开始

### 基本使用

```python
from speedvqa.inference import ROIInferencer
from speedvqa.utils.config import get_default_config

# 初始化推理器
config = get_default_config()
inferencer = ROIInferencer(
    model_path='model.pt',
    model_format='pytorch',
    device='cuda',
    config=config
)

# 单张图像推理
import numpy as np
roi_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
question = "Is there a person?"

result = inferencer.inference(roi_image, question)
print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Inference time: {result.inference_time_ms:.2f}ms")
```

### 批量推理

```python
# 批量推理
roi_images = [
    np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    for _ in range(8)
]
questions = [
    "Is there a person?",
    "Is the person smoking?",
    "Is there a car?",
    "Is there a phone?",
    "Is there a person?",
    "Is the person smoking?",
    "Is there a car?",
    "Is there a phone?"
]

results = inferencer.batch_inference(roi_images, questions)
for i, result in enumerate(results):
    print(f"Image {i}: {result.answer} (confidence: {result.confidence:.2%})")
```

## API文档

### ROIInferencer 类

#### 初始化

```python
ROIInferencer(
    model_path: str,
    model_format: str = 'pytorch',
    device: str = 'cuda',
    config: Optional[Dict[str, Any]] = None
)
```

**参数:**
- `model_path`: 模型文件路径
- `model_format`: 模型格式 ('pytorch', 'onnx', 'tensorrt')
- `device`: 推理设备 ('cuda', 'cpu')
- `config`: 配置字典

#### 单张图像推理

```python
inference(
    roi_image: Union[np.ndarray, Image.Image, str],
    question: str
) -> InferenceResult
```

**参数:**
- `roi_image`: ROI图像 (numpy array、PIL Image或图像路径)
- `question`: 问题文本

**返回:**
- `InferenceResult`: 推理结果对象

#### 批量推理

```python
batch_inference(
    roi_images: List[Union[np.ndarray, Image.Image, str]],
    questions: List[str]
) -> List[InferenceResult]
```

**参数:**
- `roi_images`: ROI图像列表
- `questions`: 问题文本列表

**返回:**
- `List[InferenceResult]`: 推理结果列表

#### 获取模型信息

```python
get_model_info() -> Dict[str, Any]
```

**返回:**
- 模型信息字典，包含模型路径、格式、设备、文件大小等

### InferenceResult 数据结构

```python
@dataclass
class InferenceResult:
    answer: str                    # YES/NO
    confidence: float              # 置信度 [0, 1]
    probabilities: List[float]     # 类别概率分布
    inference_time_ms: float       # 推理时间(毫秒)
    model_format: str              # 模型格式
    batch_size: int = 1            # 批次大小
```

## 输入格式

### 图像输入

支持多种图像输入格式：

```python
# 1. numpy array (uint8, 0-255)
image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

# 2. PIL Image
from PIL import Image
image = Image.open('roi_image.jpg')

# 3. 图像文件路径
image = 'roi_image.jpg'
```

### 问题输入

支持中英文问题：

```python
# 英文问题
question = "Is there a person?"

# 中文问题
question = "是否有人打电话?"
```

## 输出格式

推理结果包含以下信息：

```python
result = inferencer.inference(image, question)

# 答案 (YES/NO)
print(result.answer)  # 'YES' 或 'NO'

# 置信度 (0-1)
print(result.confidence)  # 0.95

# 概率分布
print(result.probabilities)  # [0.05, 0.95]

# 推理时间 (毫秒)
print(result.inference_time_ms)  # 25.5

# 模型格式
print(result.model_format)  # 'pytorch'

# 批次大小
print(result.batch_size)  # 1
```

## 性能优化

### 批量推理优化

对于多个图像的推理，使用批量推理比单个推理更高效：

```python
# 低效：单个推理
for image, question in zip(images, questions):
    result = inferencer.inference(image, question)

# 高效：批量推理
results = inferencer.batch_inference(images, questions)
```

### 设备选择

- **GPU推理**: 使用 `device='cuda'` 获得最佳性能
- **CPU推理**: 使用 `device='cpu'` 用于没有GPU的环境

### 模型格式选择

- **PyTorch**: 最灵活，支持所有功能
- **ONNX**: 跨平台兼容性好
- **TensorRT**: GPU上性能最优（需要NVIDIA GPU）

## 配置选项

### 图像配置

```python
config = {
    'image_size': (224, 224),  # 目标图像大小
}
```

### 文本配置

```python
config = {
    'text_encoder': 'distilbert-base-uncased',  # 文本编码器
    'max_text_length': 128,                      # 最大文本长度
}
```

### 推理配置

```python
config = {
    'confidence_threshold': 0.5,  # 置信度阈值
}
```

## 错误处理

### 模型加载错误

```python
try:
    inferencer = ROIInferencer('model.pt', model_format='pytorch')
except FileNotFoundError:
    print("Model file not found")
except ValueError:
    print("Unsupported model format")
```

### 推理错误

```python
try:
    result = inferencer.inference(image, question)
except Exception as e:
    print(f"Inference failed: {e}")
```

## 示例代码

### 完整示例

```python
import numpy as np
from PIL import Image
from speedvqa.inference import ROIInferencer
from speedvqa.utils.config import get_default_config

# 初始化推理器
config = get_default_config()
inferencer = ROIInferencer(
    model_path='weights/speedvqa.pt',
    model_format='pytorch',
    device='cuda',
    config=config
)

# 加载图像
image = Image.open('roi_image.jpg')

# 定义问题
questions = [
    "Is there a person?",
    "Is the person smoking?",
    "Is there a car?"
]

# 推理
for question in questions:
    result = inferencer.inference(image, question)
    print(f"Q: {question}")
    print(f"A: {result.answer} (confidence: {result.confidence:.2%})")
    print(f"Time: {result.inference_time_ms:.2f}ms\n")

# 获取模型信息
info = inferencer.get_model_info()
print(f"Model: {info['model_path']}")
print(f"Format: {info['model_format']}")
print(f"Device: {info['device']}")
```

### 批量处理示例

```python
import os
from pathlib import Path
from speedvqa.inference import ROIInferencer
from speedvqa.utils.config import get_default_config

# 初始化推理器
config = get_default_config()
inferencer = ROIInferencer(
    model_path='weights/speedvqa.pt',
    model_format='pytorch',
    device='cuda',
    config=config
)

# 批量处理图像
image_dir = 'roi_images'
images = []
questions = []

for image_file in Path(image_dir).glob('*.jpg'):
    images.append(str(image_file))
    questions.append("Is there a person?")

# 批量推理
results = inferencer.batch_inference(images, questions)

# 处理结果
for image_file, result in zip(Path(image_dir).glob('*.jpg'), results):
    print(f"{image_file.name}: {result.answer} ({result.confidence:.2%})")
```

## 常见问题

### Q: 如何选择合适的模型格式？

A: 
- 开发和调试：使用 PyTorch 格式
- 跨平台部署：使用 ONNX 格式
- GPU 生产环境：使用 TensorRT 格式

### Q: 推理时间太长怎么办？

A:
1. 使用 GPU 推理（`device='cuda'`）
2. 使用 TensorRT 格式优化模型
3. 使用批量推理而不是单个推理
4. 减小图像大小

### Q: 如何处理不同大小的图像？

A: 推理器会自动将所有图像调整到 224x224 大小，无需手动处理。

### Q: 支持哪些问题类型？

A: 支持任何 YES/NO 问题，包括中英文混合问题。

## 性能基准

在 NVIDIA T4 GPU 上的性能指标：

| 模型格式 | 单张推理 | 批量推理(B=16) | 内存占用 |
|---------|--------|--------------|--------|
| PyTorch | ~30ms  | ~25ms/img    | ~2GB   |
| ONNX    | ~25ms  | ~20ms/img    | ~1.5GB |
| TensorRT| ~15ms  | ~12ms/img    | ~1GB   |

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
