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
