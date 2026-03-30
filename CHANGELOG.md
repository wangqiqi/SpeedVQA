# Changelog

## Tag 规范

Git **tag** 采用 **`N.MMDD.ABCD`** 三段点分数字形态（非 SemVer）：

| 段 | 占位 | 含义 |
|----|------|------|
| **N** | 单段数字 | 主序号（如主版本/批次），由项目约定递增或固定。 |
| **MMDD** | 四位 | **`MM`**（01–12）+ **`DD`**（01–31），与发布日对齐。 |
| **ABCD** | 四位 | **`HHMM`**，24 小时制的小时与分钟（如 `0930`、`1430`）。 |

**示例**：`1.0329.1430` 表示 `N=1`、**3 月 29 日**（`0329`）、时间 **14:30**。

> 与历史条目中的语义化别名 tag（如 `chore-speedvqa-*`）可并存；新发布若采用本规范，建议在 CHANGELOG 对应条目标注 **`tag N.MMDD.ABCD`**（将各段换为实际数字）。

---

## [2026-03-30] — tag `1.0330.2110`

### 新增

- **融合（Phase A）**：**`MultiModalFusion`** 支持 **`film`**（文本 FiLM 调制视觉后再拼接 MLP）与 **`cross_attn`**（文本 Q、视觉 KV 的单层跨模态注意力）；**`model.fusion.method`** 默认仍为 **`concat`**。

### 变更

- **`.gitignore`**：将 **`data/`**、**`datasets/`**、**`annotations/`**、**`images/`**、**`models/`** 改为仅匹配仓库根（**`/data/`** 等），误忽略的 **`speedvqa/models/`**、**`speedvqa/data/`** 现纳入版本控制。
- **ONNX 导出**：**`SpeedVQAOnnxWrapper`** 包装 **`SpeedVQAModel`**，**`torch.onnx.export`** 使用 **`(image, input_ids, attention_mask)`** 三参数；导出在 **CPU** 上追踪；**`_validate_onnx_export`** 将输入张量对齐到模型参数设备后再比 logits。
- **配置 / 工厂**：**`default.yaml`** 融合注释；**`ModelFactory.SUPPORTED_FUSION_METHODS`** 增补 **`film`**、**`cross_attn`**。
- **文档与计划**：**`docs/01_设计.md`** 模型行补充融合与 ONNX 说明；**`docs/02_使用.md`** Phase 0 ONNX 小节补充 Wrapper 与 I/O 名；**`plan.md`** Phase A 任务状态与基线表 ONNX 说明更新。
- **测试**：**`test_speedvqa_model.py`** 覆盖 **`film`** / **`cross_attn`** 及集成用例。
---

## [2026-03-30] — tag `1.0330.2104`

### 变更

- **计划 Phase 0**：**`plan.md`** 更新 — P0-1～P0-3 标为完成（冒烟），新增「基线表（Phase 0 冒烟）」与已知 ONNX 导出问题说明；**Phase A** 标为下一阶段。
- **文档**：**`docs/02_使用.md`** 新增「**Phase 0 基线与实验协议**」（锁定 seed / fusion / 训练与导出命令、**加权 F1** 与 **best 权重按 val_loss** 的实现说明、冒烟流程、ONNX 与延迟登记约定）。
- **配置**：**`speedvqa/configs/phase0_smoke.yaml`** — 继承 `default` 的短训冒烟配置（独立 **`train.save_dir`**）。
- **示例**：**`speedvqa/examples/phase0_smoke_dataset.py`** — 生成 **`datasets/phase0_smoke`** 最小 PNG + **`vqa_labels.jsonl`**。
- **默认配置**：**`default.yaml`** 中 **`optimizer.eps`**、**`scheduler.warmup_lr`** / **`min_lr`** 改为 **`1.0e-8` / `1.0e-6`**，避免 PyYAML 将 **`1e-8`** 解析为字符串导致 **AdamW** 初始化失败。

---

## [2026-03-30] — tag `1.0330.2049`

### 新增

- **文档**：**`docs/03_思考思路.md`** — 合并 **LiteMind** 仓库说明（轻量 LLM/VLM 学习路径与外链）与 **VisionJudge 2.0** 全功能参考内容，统一标题层级并对齐 fenced 代码块内注释；与 **`01_设计.md`**、**`02_使用.md`** 并列，便于与本项目 ROI 二分类 VQA 对照阅读。

---

## [2026-03-29] — tag `1.0329.2410`

### 变更

- **产物路径**：默认导出 **`runs/exports/`**、基准报告 **`runs/benchmark_reports/`**；**`artifact_paths`**、**`speedvqa/configs/default.yaml`**、**`onekey_export`**、**`export/README`**、**`trainer`** 日志与 **`onekey_clean.sh`**（**`--help`** / 注释含 **`runs/exports`**）已对齐；仓库根 **`exports/`** 仍兼容旧布局。
- **测试**：**`test_artifact_paths`** 增补 export 与兼容路径用例；**`test_model_export`** 修正 **`speedvqa.export.exporter.ort.InferenceSession`** mock、基准调用 **`_benchmark_pytorch_detailed`**；**`test_performance_benchmarking`** 改用 **`SpeedVQAModel`** 与导出检查点格式一致；**`test_t4_performance`** 在 CUDA 下输入张量与模型同设备；**Ruff** 清理 **`test_performance_benchmarking`** 未使用导入。
- **计划**：**`plan.md`** 增补可排期执行阶段、验收指标、风险与周历等说明。

---

## [2026-03-29] — tag `1.0329.2234`

### 变更

- **导出基准报告**：默认 **`export.benchmark_report_dir`** 由仓库根 **`./benchmark_reports`** 改为 **`./runs/benchmark_reports`**，与 **`runs/`** 产物约定一致；**`performance_benchmark_example.py`**、**`docs/02_使用.md`** 目录说明与 **`onekey_clean.sh`**（不再单独删根目录 **`benchmark_reports`**，随 **`runs/`** 一并清理）已同步。

---

## [2026-03-29] — tag `1.0329.2231`

### 新增

- **CI**：**`.github/workflows/ci.yml`** — 推送/PR 至 `main`/`master` 时运行 **Ruff**（Python 3.11）与 **pytest**（3.10 / 3.12；`HYPOTHESIS_PROFILE=dev`，`HF_DATASETS_OFFLINE` / `TRANSFORMERS_OFFLINE`）。
- **`requirements-ci.txt`**：CI 安装用依赖（省略 TensorRT、wandb 等环境相关包）。
- **`.pre-commit-config.yaml`**：可选本地 **Ruff** 钩子（`--fix`）。

### 变更

- **`pyproject.toml`**：**`[project.optional-dependencies].dev`**（Ruff）；**`[tool.ruff]`** / **`[tool.ruff.lint]`**（E4、E7、E9、F；示例与部分测试脚本 **E402** 按文件忽略）。
- **`README.md`**：「**开发与 CI**」— `pip install -e ".[dev]"`、`ruff check`、pre-commit、Actions 与 **`requirements-ci.txt`** 说明。
- **`requirements.txt`**：补充 **`psutil`**（与导出/监控等用法一致）。
- **`plan.md`**：**算法设计评审意见（2026-03-29）** 记录。
- **源码与测试**：按 Ruff 清理**未使用导入**及若干小问题（**`speedvqa/engine`**、**`examples`**、**`export`**、**`inference`**、**`monitoring`**、**`optimization`**、**`benchmark`**、**`utils`**、**`speedvqa/tests/*`** 等）。

---

## [2026-03-29] — tag `1.0329.2325`

### 新增

- **文档**：**`docs/02_使用.md`** 增加「**参数调优与最佳实践**」：按目标速查表、调参顺序、分模块（vision/text/fusion/classifier/loss/data/train/inference/hardware）建议与实验记录习惯。

---

## [2026-03-29] — tag `1.0329.2310`

### 变更

- **文档**：**`docs/02_使用.md`** 扩充为与实现对齐的实操说明：数据三种格式与 JSON/JSONL 示例、答案归一化与标签、`default.yaml` 节点表、训练/预测/导出 CLI 参数表与示例命令、导出产物路径、`ROIInferencer` 代码示例、部署检查清单、常见问题、`examples/` 脚本索引。

---

## [2026-03-29] — tag `1.0329.2245`

### 变更

- **文档**：将 **`docs/03_使用.md`** 重命名为 **`docs/02_使用.md`**（与 **`01_设计.md`** 编号衔接）；**`README.md`**、**`01_设计.md`**、**`speedvqa/engine/trainer.py`** 内链接已更新。

---

## [2026-03-29] — tag `1.0329.2235`

### 变更

- **文档**：`docs/` 仅保留 **`01_设计.md`**、**`03_使用.md`**；删除 **`02_开发.md`**、**`04_部署.md`**、**`使用指南.md`**。将环境、数据、训练、导出、部署、`runs/`、测试与导入合并入 **`03_使用.md`（全栈流程）**；**`README.md`**、**`01_设计.md`**、**`speedvqa/engine/trainer.py`** 中的文档链接同步更新。

---

## [2026-03-29] — tag `1.0329.2202`

### 破坏性变更

- **`python -m speedvqa.scripts.*`** 已移除；请改用 **`python -m speedvqa.cli.onekey_train`**（及 **`onekey_predict`**、**`onekey_export`**），或 **`pip install`** 后的 **`speedvqa-train`**、**`speedvqa-predict`**、**`speedvqa-export`**。

### 新增

- **`pyproject.toml`**：**`[project.scripts]`** 注册上述三个控制台命令。

### 变更

- **`speedvqa/scripts/`** 整包迁至 **`speedvqa/cli/`**（仅一处 Python 实现，避免与仓库根 **`scripts/*.sh`** 混淆）。
- 仓库根 **`scripts/onekey_{train,predict,export}.sh`**：内部改为 **`python -m speedvqa.cli.onekey_*`**。
- **`README.md`**、**`speedvqa/__init__.py`**、**`docs/01_设计.md`**、**`docs/02_开发.md`**、**`docs/03_使用.md`**：同步 CLI 路径与 entry points 说明。

---

## [2026-03-29] — tag `1.0329.2158`

### 变更

- **文档**：将 **`docs/PROJECT_OVERVIEW.md`**、**`docs/RUNS_DIRECTORY.md`** 分别并入 **`docs/01_设计.md`**、**`docs/02_开发.md`**、**`docs/04_部署.md`**；删除上述两文件及 **`docs/README.md`**，其索引并入根目录 **`README.md`**；**`docs/使用指南.md`** 与 **`speedvqa/engine/trainer.py`** 中的文档链接同步改为分册路径。

---

## [2026-03-29] — tag `1.0329.2150`

### 变更

- **`.gitignore`**：补充 **`exports/`**、**`cache/`**（与默认配置一致），以及 **`.mypy_cache/`**、**`.ruff_cache/`**、**`.tox/`**、**`coverage.xml`**，避免误提交产物与工具缓存。
- **`README.md`**：一键脚本说明中增加 **`onekey_clean.sh`**（清理 **`runs/`** 等可再生目录）。

---

## [2026-03-29] — tag `1.0329.2148`

### 变更

- **`scripts/onekey_clean.sh`**：一并清理 **`visualizations/`**、**`plots/`**、**`inference_outputs/`**、**`predictions/`**、**`model_exports/`**、**`exported_models/`**；删除仓库根目录下的 **`*.png` / `*.jpg` / `*.jpeg` / `*.gif`**（多为 benchmark/导出图表）；删除根目录 **`results_*`** 子目录。已执行清理命令（当前工作区若无对应路径则无输出）。

---

## [2026-03-29] — tag `1.0329.2141`

### 新增

- **`speedvqa/utils/artifact_paths.py`**：解析训练 **`save_dir`** 与 **`torch.save` 写入路径**；在 SpeedVQA 仓库内时，强制落在 **`runs/`**、**`exports/`** 或 **`cache/`** 下（否则改写到 **`runs/train/<experiment_name>/`** 或 **`exports/<experiment_name>/`**），避免在仓库根或源码树散落无扩展名检查点。
- **`speedvqa/tests/test_artifact_paths.py`**：覆盖上述解析与改向逻辑。

### 变更

- **`speedvqa/engine/trainer.py`**：使用 **`resolve_train_save_dir`**；若发生改向则打日志说明。
- **`speedvqa/utils/training_logger.py`**：**`save_checkpoint`** 经 **`resolve_torch_write_path(..., artifact_kind='train')`**。
- **`speedvqa/export/exporter.py`**：PyTorch 导出路径经 **`resolve_torch_write_path(..., artifact_kind='export')`**。
- **`speedvqa/configs/default.yaml`**、**`docs/RUNS_DIRECTORY.md`**：补充产物目录约定与 **`artifact_paths`** 说明。
- **`scripts/onekey_clean.sh`**：增加对仓库根 **无扩展名且为 Zip** 文件的清理（多为误 **`torch.save`** 产物）。

---

## [2026-03-29] — tag `1.29.2139`

### 文档

- **`CHANGELOG.md`**：新增 **Tag 规范**（`N.MMDD.ABCD`：第二段为 **`MMDD`**，`ABCD` 为 **`HHMM`**）。

---

## [2026-03-29]

### 新增

- 根目录 **`scripts/onekey_clean.sh`**：清理 **`runs/`**、**`exports/`**、**`cache/`**、**`.pytest_cache`** / **`.hypothesis`**、**`build`** / **`dist`**、**`*.egg-info`**、`speedvqa` 与 **`scripts/`** 下 **`__pycache__`** 等；支持 **`--dry-run`**；不删除 **`.venv` / `venv`**。

### 变更

- **`.gitignore`**：补充 **`exports/`**、**`cache/`**、**`.mypy_cache/`**、**`.ruff_cache/`**、**`.tox/`**、**`coverage.xml`**。
- **`README.md`**、**`docs/PROJECT_OVERVIEW.md`**、**`docs/RUNS_DIRECTORY.md`**：说明 **`onekey_clean.sh`** 用法。

---

## [2026-03-29] — tag `docs-speedvqa-api-packaging`

### 变更

- **`pyproject.toml`**：`setuptools.packages.find` 使用 `include = ["speedvqa*"]`，并 **`exclude`** **`speedvqa.tests*`**、**`speedvqa.examples*`**，使发布 wheel 更精简（开发仍用源码树跑测试与示例）。
- **`speedvqa/__init__.py`**：模块文档串补充 **顶层 `__all__` 分工**、**训练/导出/推理/一键脚本** 的推荐导入与示例。
- **`docs/PROJECT_OVERVIEW.md`**：新增 **推荐导入** 表与 **扩展新骨干/编码器** 四步说明，并与 wheel 排除策略对齐。

---

## [2026-03-29] — tag `chore-speedvqa-monolithic-tree`

### 变更

- **单包树约定**：不设 **`src/`**，业务与资源均在仓库根下 **`speedvqa/`**（含 **`tests/`**、**`examples/`**、**`configs/`**、**`scripts/`** 等），与核心库同属一棵包树，便于统一维护。
- 恢复 **`speedvqa/pytest.ini`**；**`pyproject.toml`** 增加 **`[project]`** 与 **`[tool.setuptools.packages.find]`**（`where = "."`，`include = ["speedvqa"]`），支持 **`pip install -e .`**。

---

## [2026-03-29] — tag `chore-speedvqa-onekey-scripts`

### 变更

- 一键训练/推理/导出的 Python 实现迁至 **`speedvqa/scripts/onekey_train.py`**、**`onekey_predict.py`**、**`onekey_export.py`**；根目录 **`scripts/onekey_*.sh`** 改为执行 **`python -m speedvqa.scripts.onekey_*`**（须在仓库根目录运行）。

---

## [2026-03-29] — tag `chore-speedvqa-package-layout`

### 变更

- 将 **`configs/`**、**`examples/`**、**`tests/`** 迁入 **`speedvqa/`**；**`conftest.py`** 迁至 **`speedvqa/tests/conftest.py`**；**`pytest.ini`** 迁至 **`speedvqa/pytest.ini`**（在 `speedvqa/` 下可用 `pytest -c pytest.ini`）。
- 仓库根目录新增 **`pyproject.toml`** 中 **`[tool.pytest.ini_options]`**，保证在仓库根执行 **`pytest`** 时收集 **`speedvqa/tests`**；删除根目录 **`pytest.ini`**、**`conftest.py`**。
- **`speedvqa.utils.config`**：默认配置路径改为包内 **`speedvqa/configs/default.yaml`**，并提供 **`resolve_config_file()`**，仍支持传入旧式相对路径 `configs/default.yaml`（在包内解析）。

### 新增

- 根目录 `scripts/`：`onekey_train.sh` / `onekey_predict.sh` / `onekey_export.sh`（调用包内 `speedvqa.scripts.onekey_*`，见上条变更）。
- 根目录 `README.md` 与自定义 `LICENSE`：非商业研究与学习免费使用；**商用须书面授权并付费**；**Fork/修改的衍生作品须以 OSI 认可的开源许可证公开完整源代码**。

### 变更（此前条目，路径已随本次迁入 `speedvqa/` 更新）

- 自测脚本位于 **`speedvqa/tests/`**（`run_tests.py`、`test_model_basic.py`、`test_model_factory.py`）。
- **`speedvqa/tests/conftest.py`**：`collect_ignore` 排除上述两脚本式模块，避免与 `pytest` 重复收集。
- `ROIInferencer` 加载 PyTorch 检查点时识别 **`config['model']`**（与 `ConfigurableTrainer` 一致）及旧字段 **`model_config`**。

### 文档

- 新增 `docs/PROJECT_OVERVIEW.md`（SpeedVQA 项目介绍）与 `docs/README.md`（文档索引）。

---

## [2026-03-29] — tag `chore-project-scaffolding`

### 新增

- 建立 `archive/` 目录，用于后续归档文档与文件（建议命名：`YYYYMMDD_HHMMSS_功能_模块说明.md`）。
- 新增根目录 `plan.md`，用于记录较大任务与计划。
- 建立 `.cursor/`：`rules/`、`skills/` 占位及 `AGENTS.md`，便于后续维护 Cursor 规则与技能。

---

## [2026-03-29] — tag `chore-ignore-hypothesis-kiro`

### 变更

- 将 `.hypothesis/`、`.kiro/` 从版本库中移除并写入 `.gitignore`，不再跟踪测试缓存与 Kiro 规格目录。
