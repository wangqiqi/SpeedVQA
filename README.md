# SpeedVQA

面向 **ROI 图像 + 自然语言** 的轻量级 **二分类视觉问答（VQA）** 工程方案：配置驱动，支持训练、多格式导出（含 TensorRT 路径）与低延迟推理。

**版本**：见 `speedvqa/__init__.py` 中的 `__version__`。

**布局**：可运行模块、默认配置、示例与测试均在 **`speedvqa/`** 目录下；仓库根保留 `scripts/*.sh`、`docs/`、`pyproject.toml` 等脚手架。开发时可执行 **`pip install -e .`** 以可编辑方式安装包。

## 文档

| 资源 | 说明 |
|------|------|
| [使用指南](docs/使用指南.md) | 分主题阅读顺序与模块链接 |
| [01 设计](docs/01_设计.md) | 定位、特性、目录结构、工作流、扩展骨干 |
| [02 开发](docs/02_开发.md) | 环境、推荐导入、测试与质量 |
| [03 使用](docs/03_使用.md) | 配置、训练、预测与示例入口 |
| [04 部署](docs/04_部署.md) | 导出、推理、`runs/` 约定与清理 |
| [推理模块](speedvqa/inference/README.md) | `ROIInferencer` API |
| [导出模块](speedvqa/export/README.md) | `ModelExporter` 与格式说明 |

## 快速开始

```bash
cd /path/to/SpeedVQA
pip install -r requirements.txt
```

配置见 `speedvqa/configs/default.yaml`；示例在 `speedvqa/examples/`；一键流程：仓库根 `scripts/onekey_*.sh`（含 **`onekey_clean.sh`** 清理缓存与 `runs/` 等），或 `python -m speedvqa.cli.onekey_train` 等；**`pip install -e .`** 后也可使用 **`speedvqa-train`** / **`speedvqa-predict`** / **`speedvqa-export`**。测试：在仓库根执行 `pytest`（见根目录 `pyproject.toml` 与 `speedvqa/pytest.ini`），或：

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
