# SpeedVQA ModelExporter

SpeedVQA模型导出器，支持多种格式的模型导出，包含功能验证和性能基准测试。

## 功能特性

### 支持的导出格式

1. **PyTorch (.pt)** - 完整的PyTorch模型保存
   - 包含模型状态字典、配置信息、元数据
   - 支持可选的优化器状态保存
   - 完整的模型重建和加载验证

2. **ONNX (.onnx)** - 跨平台推理格式
   - 支持动态批次大小
   - 可配置的opset版本
   - 自动模型验证和检查

3. **TensorRT (.engine)** - NVIDIA GPU优化格式
   - FP16/FP32精度支持
   - 针对T4显卡优化
   - 动态形状配置

### 核心功能

- **功能验证**: 自动验证导出模型的功能一致性
- **性能基准测试**: 测试不同格式的推理性能
- **错误处理**: 完整的错误分类和处理
- **配置驱动**: 通过YAML配置控制导出行为
- **日志记录**: 详细的导出过程日志

## 快速开始

### 基本使用

```python
from speedvqa.export.exporter import ModelExporter, export_model
from speedvqa.models.speedvqa import SpeedVQAModel
from speedvqa.utils.config import get_default_config

# 加载配置和模型
config = get_default_config()
model = SpeedVQAModel(config['model'])

# 方法1: 使用便捷函数导出所有格式
results = export_model(
    model=model,
    output_dir='./exports',
    model_name='speedvqa_v1',
    config=config,
    formats=['pytorch', 'onnx', 'tensorrt']
)

# 方法2: 使用ModelExporter类进行精细控制
exporter = ModelExporter(config)

# 导出PyTorch格式
pt_result = exporter.export_pytorch(model, 'model.pt')
print(f"PyTorch export: {pt_result.success}, {pt_result.model_size_mb:.2f} MB")

# 导出ONNX格式
onnx_result = exporter.export_onnx(model, 'model.onnx')
print(f"ONNX export: {onnx_result.success}, {onnx_result.model_size_mb:.2f} MB")

# 导出TensorRT格式
trt_result = exporter.export_tensorrt('model.onnx', 'model.engine')
print(f"TensorRT export: {trt_result.success}, {trt_result.model_size_mb:.2f} MB")
```

### 配置选项

```yaml
export:
  # 验证配置
  validation:
    enabled: true
    tolerance: 1e-4  # 数值精度容差
    num_samples: 10  # 验证样本数量
  
  # 基准测试配置
  benchmark:
    enabled: true
    warmup_iterations: 10
    test_iterations: 100
```

## API 参考

### ModelExporter类

#### 构造函数
```python
ModelExporter(config: Dict[str, Any])
```

#### 主要方法

##### export_pytorch()
```python
export_pytorch(
    model: SpeedVQAModel, 
    save_path: str,
    include_optimizer: bool = False,
    optimizer_state: Optional[Dict] = None
) -> ExportResult
```

导出PyTorch格式模型。

**参数:**
- `model`: 训练好的SpeedVQA模型
- `save_path`: 保存路径
- `include_optimizer`: 是否包含优化器状态
- `optimizer_state`: 优化器状态字典

**返回:** `ExportResult` 对象

##### export_onnx()
```python
export_onnx(
    model: SpeedVQAModel,
    save_path: str,
    input_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
    opset_version: int = 11,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
) -> ExportResult
```

导出ONNX格式模型。

**参数:**
- `model`: 训练好的SpeedVQA模型
- `save_path`: 保存路径
- `input_shapes`: 输入形状字典
- `opset_version`: ONNX opset版本
- `dynamic_axes`: 动态轴配置

**返回:** `ExportResult` 对象

##### export_tensorrt()
```python
export_tensorrt(
    onnx_path: str,
    save_path: str,
    max_batch_size: int = 16,
    precision: str = 'fp16',
    workspace_size: int = 1 << 30
) -> ExportResult
```

导出TensorRT格式模型。

**参数:**
- `onnx_path`: ONNX模型路径
- `save_path`: TensorRT引擎保存路径
- `max_batch_size`: 最大批次大小
- `precision`: 精度模式 ('fp32', 'fp16', 'int8')
- `workspace_size`: 工作空间大小（字节）

**返回:** `ExportResult` 对象

##### export_all_formats()
```python
export_all_formats(
    model: SpeedVQAModel,
    base_path: str,
    formats: Optional[List[str]] = None
) -> Dict[str, ExportResult]
```

导出所有支持的格式。

**参数:**
- `model`: 训练好的SpeedVQA模型
- `base_path`: 基础保存路径（不含扩展名）
- `formats`: 要导出的格式列表

**返回:** 格式名到 `ExportResult` 的映射

##### benchmark_exported_models()
```python
benchmark_exported_models(
    model_paths: Dict[str, str],
    num_iterations: int = 100
) -> Dict[str, Dict[str, float]]
```

对导出的模型进行性能基准测试。

**参数:**
- `model_paths`: 格式到模型路径的映射
- `num_iterations`: 测试迭代次数

**返回:** 各格式的性能指标

### 数据结构

#### ExportResult
```python
@dataclass
class ExportResult:
    success: bool                           # 导出是否成功
    export_path: str                        # 导出文件路径
    format: str                            # 导出格式
    model_size_mb: float                   # 模型文件大小(MB)
    export_time_s: float                   # 导出耗时(秒)
    error_message: Optional[str] = None    # 错误信息
    validation_result: Optional[Dict] = None # 验证结果
```

#### ValidationResult
```python
@dataclass
class ValidationResult:
    success: bool                          # 验证是否成功
    inference_time_ms: float              # 推理时间(毫秒)
    output_shape: Tuple[int, ...]         # 输出形状
    output_dtype: str                     # 输出数据类型
    numerical_accuracy: float             # 数值精度
    error_message: Optional[str] = None   # 错误信息
```

## 使用示例

### 示例1: 基本导出
```python
from speedvqa.export.exporter import export_model

# 导出所有格式
results = export_model(
    model=trained_model,
    output_dir='./models',
    model_name='speedvqa_final',
    config=config
)

# 检查结果
for format_name, result in results.items():
    if result.success:
        print(f"✓ {format_name}: {result.model_size_mb:.1f}MB")
    else:
        print(f"✗ {format_name}: {result.error_message}")
```

### 示例2: 自定义ONNX导出
```python
from speedvqa.export.exporter import ModelExporter

exporter = ModelExporter(config)

# 自定义输入形状和动态轴
result = exporter.export_onnx(
    model=model,
    save_path='custom_model.onnx',
    input_shapes={
        'image': (1, 3, 224, 224),
        'input_ids': (1, 128),
        'attention_mask': (1, 128)
    },
    dynamic_axes={
        'image': {0: 'batch_size'},
        'input_ids': {0: 'batch_size'},
        'attention_mask': {0: 'batch_size'},
        'logits': {0: 'batch_size'}
    },
    opset_version=12
)
```

### 示例3: 性能基准测试
```python
# 导出多种格式
exporter = ModelExporter(config)
pt_result = exporter.export_pytorch(model, 'model.pt')
onnx_result = exporter.export_onnx(model, 'model.onnx')

# 运行基准测试
if pt_result.success and onnx_result.success:
    benchmark_results = exporter.benchmark_exported_models({
        'pytorch': 'model.pt',
        'onnx': 'model.onnx'
    }, num_iterations=200)
    
    for format_name, metrics in benchmark_results.items():
        print(f"{format_name}:")
        print(f"  平均推理时间: {metrics['avg_inference_time_ms']:.2f} ms")
        print(f"  吞吐量: {metrics['throughput_fps']:.1f} FPS")
```

## 依赖要求

### 必需依赖
- PyTorch >= 1.8.0
- NumPy
- pathlib

### 可选依赖
- **ONNX导出**: `onnx`, `onnxruntime`
- **TensorRT导出**: `tensorrt`, `pycuda`

### 安装可选依赖
```bash
# ONNX支持
pip install onnx onnxruntime

# TensorRT支持 (需要NVIDIA GPU)
pip install tensorrt pycuda
```

## 性能优化建议

### T4显卡优化
1. **使用TensorRT**: 在T4显卡上可获得50%+的性能提升
2. **FP16精度**: 启用半精度可显著提升速度
3. **批处理**: 使用适当的批次大小提高吞吐量

### 内存优化
1. **工作空间大小**: 根据GPU内存调整TensorRT工作空间
2. **动态形状**: 合理设置输入形状范围
3. **模型剪枝**: 考虑模型压缩技术

## 故障排除

### 常见问题

#### 1. ONNX导出失败
```
错误: ONNX export failed: unsupported operation
解决: 检查模型中是否使用了ONNX不支持的操作
```

#### 2. TensorRT构建失败
```
错误: Failed to build TensorRT engine
解决: 
- 检查CUDA和TensorRT版本兼容性
- 增加工作空间大小
- 检查输入形状配置
```

#### 3. 验证失败
```
错误: Numerical difference too large
解决:
- 增加验证容差
- 检查模型是否包含随机操作
- 确保模型处于eval模式
```

### 调试技巧

1. **启用详细日志**:
```python
import logging
logging.getLogger('ModelExporter').setLevel(logging.DEBUG)
```

2. **检查导出文件**:
```python
# 检查PyTorch文件
checkpoint = torch.load('model.pt')
print(checkpoint.keys())

# 检查ONNX文件
import onnx
model = onnx.load('model.onnx')
onnx.checker.check_model(model)
```

3. **性能分析**:
```python
# 使用基准测试功能
results = exporter.benchmark_exported_models(model_paths, num_iterations=10)
```

## 贡献指南

欢迎提交Issue和Pull Request来改进ModelExporter！

### 开发环境设置
```bash
git clone <repository>
cd speedvqa
pip install -e .
pip install -r requirements-dev.txt
```

### 运行测试
```bash
python -m pytest tests/test_model_export.py -v
```

### 添加新格式支持
1. 在 `ModelExporter` 类中添加新的导出方法
2. 实现相应的验证和基准测试方法
3. 添加单元测试
4. 更新文档

## 许可证

本项目采用MIT许可证，详见LICENSE文件。