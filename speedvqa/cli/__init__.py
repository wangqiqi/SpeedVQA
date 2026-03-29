"""
命令行入口（一键训练 / 推理 / 导出）。

在**仓库根目录**执行，例如：

    python -m speedvqa.cli.onekey_train --help
    python -m speedvqa.cli.onekey_predict --help
    python -m speedvqa.cli.onekey_export --help

安装本包后也可使用控制台命令：**`speedvqa-train`**、**`speedvqa-predict`**、**`speedvqa-export`**（见根目录 **`pyproject.toml`** 的 **`[project.scripts]`**）。

亦可使用仓库根目录 **`scripts/onekey_*.sh`**，其内部调用上述模块。
"""
