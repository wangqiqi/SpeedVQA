# SpeedVQA 输出目录结构

本项目使用统一的 `runs/` 目录来管理所有训练、验证、推理和优化的输出文件。

## 目录结构

```
runs/
├── train/              # 训练输出
│   ├── {experiment_name}/
│   │   ├── checkpoints/
│   │   ├── logs/
│   │   └── tensorboard/
│
├── val/                # 验证输出
│   ├── predictions/
│   └── metrics/
│
├── inference/          # 推理输出
│   ├── results/
│   └── visualizations/
│
├── export/             # 模型导出
│   ├── pytorch/
│   ├── onnx/
│   └── tensorrt/
│
└── hyperopt/           # 超参数优化
    ├── {timestamp}/
    │   ├── trials/
    │   └── summary.json
```

## 配置说明

### 训练输出

在 `speedvqa/configs/default.yaml` 中配置：

```yaml
train:
  save_dir: './runs/train'
  experiment_name: 'speedvqa_exp1'
```

训练过程中会自动创建：
- `latest_checkpoint.pth` - 最新检查点
- `best_checkpoint.pth` - 最佳检查点
- `checkpoint_epoch_N.pth` - 周期性检查点
- `training.log` - 训练日志

### 验证输出

```yaml
val:
  save_dir: './runs/val'
  save_predictions: true
```

### 超参数优化

超参数优化会自动在 `runs/hyperopt/{timestamp}/` 下创建目录，包含：
- `trial_*.json` - 每次试验的结果
- `optimization_summary.json` - 优化总结
- `optuna_results.json` 或 `skopt_results.json` - 优化器特定结果

### 模型导出

建议使用以下路径导出模型：

```python
# PyTorch
exporter.export_pytorch(model, 'runs/export/pytorch/model.pt')

# ONNX
exporter.export_onnx(model, 'runs/export/onnx/model.onnx')

# TensorRT
exporter.export_tensorrt(model, 'runs/export/tensorrt/model.engine')
```

### 推理输出

建议使用以下路径保存推理结果：

```python
# 保存推理结果
visualizer.save_results('runs/inference/results/')

# 保存可视化
visualizer.save_visualizations('runs/inference/visualizations/')
```

## 版本控制

`runs/` 目录已添加到 `.gitignore`，不会被版本控制系统跟踪。这确保：
- 大型模型文件不会被提交到仓库
- 每个开发者可以有自己的实验输出
- 保持仓库干净和轻量

## 清理

如需清理所有输出：

```bash
rm -rf runs/
```

如需清理特定类型的输出：

```bash
rm -rf runs/train/      # 清理训练输出
rm -rf runs/hyperopt/   # 清理超参数优化结果
```

## 最佳实践

1. **使用时间戳命名实验**：在实验名称中包含时间戳，便于追踪
   ```yaml
   experiment_name: 'speedvqa_20260203_215000'
   ```

2. **定期备份重要检查点**：将最佳模型复制到安全位置
   ```bash
   cp runs/train/best_checkpoint.pth backups/model_v1.pth
   ```

3. **记录实验配置**：每次实验都保存完整的配置文件
   ```bash
   cp speedvqa/configs/default.yaml runs/train/{experiment_name}/config.yaml
   ```

4. **使用符号链接**：如果 `runs/` 目录需要在不同位置（如更大的磁盘）
   ```bash
   ln -s /path/to/large/disk/runs ./runs
   ```
