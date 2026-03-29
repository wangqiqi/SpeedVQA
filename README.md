# SpeedVQA

面向 **ROI 图像 + 自然语言** 的轻量级 **二分类视觉问答（VQA）** 工程方案：配置驱动，支持训练、多格式导出（含 TensorRT 路径）与低延迟推理。

**版本**：见 `speedvqa/__init__.py` 中的 `__version__`。

## 文档

| 资源 | 说明 |
|------|------|
| [项目介绍](docs/PROJECT_OVERVIEW.md) | 定位、架构、目录与工作流 |
| [文档索引](docs/README.md) | `runs/` 约定及模块 README 链接 |
| [推理模块](speedvqa/inference/README.md) | `ROIInferencer` API |
| [导出模块](speedvqa/export/README.md) | `ModelExporter` 与格式说明 |

## 快速开始

```bash
cd /path/to/SpeedVQA
pip install -r requirements.txt
```

配置见 `speedvqa/configs/default.yaml`；示例在 `speedvqa/examples/`；一键流程：仓库根 `scripts/onekey_*.sh`，或 `python -m speedvqa.scripts.onekey_train` 等。测试：在仓库根执行 `pytest`（见根目录 `pyproject.toml` 与 `speedvqa/pytest.ini`），或：

```bash
python speedvqa/tests/run_tests.py
python speedvqa/tests/test_model_factory.py
python speedvqa/tests/test_model_basic.py
```

## 许可

本仓库采用 **项目自定义许可**，见根目录 **[LICENSE](LICENSE)**。要点如下（**不构成法律意见**，以 `LICENSE` 全文为准）：

- **研究与学习**：非商业场景下可免费使用（学术、教学、个人学习等）。  
- **商用**：须取得权利人 **书面许可** 并 **支付许可费用**；未经授权不得商用。  
- **Fork / 修改**：衍生作品须 **公开完整源代码**，并采用 **OSI 认可的开源许可证** 分发。

商业合作或授权咨询：请通过本仓库维护者在平台上提供的联系方式取得书面许可。

## 致谢

依赖的第三方库受其各自许可证约束，详见各依赖声明与 `LICENSE` 中「第三方组件」条款。
