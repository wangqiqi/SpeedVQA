# Changelog

## [2026-03-29] — tag `chore-speedvqa-package-layout`

### 变更

- 将 **`configs/`**、**`examples/`**、**`tests/`** 迁入 **`speedvqa/`**；**`conftest.py`** 迁至 **`speedvqa/tests/conftest.py`**；**`pytest.ini`** 迁至 **`speedvqa/pytest.ini`**（在 `speedvqa/` 下可用 `pytest -c pytest.ini`）。
- 仓库根目录新增 **`pyproject.toml`** 中 **`[tool.pytest.ini_options]`**，保证在仓库根执行 **`pytest`** 时收集 **`speedvqa/tests`**；删除根目录 **`pytest.ini`**、**`conftest.py`**。
- **`speedvqa.utils.config`**：默认配置路径改为包内 **`speedvqa/configs/default.yaml`**，并提供 **`resolve_config_file()`**，仍支持传入旧式相对路径 `configs/default.yaml`（在包内解析）。

### 新增

- 根目录 `scripts/`：`onekey_train.sh` / `onekey_predict.sh` / `onekey_export.sh` 及 `onekey_*.py` 入口（训练、单图推理、检查点导出）。
- 根目录 `README.md` 与自定义 `LICENSE`：非商业研究与学习免费使用；**商用须书面授权并付费**；**Fork/修改的衍生作品须以 OSI 认可的开源许可证公开完整源代码**。

### 变更（此前条目，路径已随本次迁入 `speedvqa/` 更新）

- 自测脚本位于 **`speedvqa/tests/`**（`run_tests.py`、`test_model_basic.py`、`test_model_factory.py`）；删除 **`speedvqa/scripts/`** 下重复副本，仅保留占位 `__init__.py`。
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
