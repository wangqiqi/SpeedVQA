"""
训练检查点、导出文件等产物路径约束。

仓库内相对路径若会落在 `runs/`（含 `runs/exports/`）、旧版根目录 `exports/`、`cache/` 之外，
则自动改写到上述目录下，避免在仓库根目录或源码树里误生成无扩展名检查点等文件。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

_LOG = logging.getLogger(__name__)

ArtifactKind = Literal["train", "export"]


def find_speedvqa_repo_root(start: Optional[Path] = None) -> Optional[Path]:
    """定位含本项目的仓库根（存在 pyproject.toml 且 name = speedvqa）。"""
    cur = (start or Path.cwd()).resolve()
    for d in [cur, *cur.parents]:
        pp = d / "pyproject.toml"
        if not pp.is_file():
            continue
        try:
            text = pp.read_text(encoding="utf-8")
        except OSError:
            continue
        if 'name = "speedvqa"' in text or "name = 'speedvqa'" in text:
            return d
    return None


def sanitize_path_component(name: str) -> str:
    """单层目录名：去掉路径分隔符与空名，防止穿越。"""
    s = re.sub(r'[/\\:\0]+', "_", str(name).strip())
    s = s.strip(".") or "default_exp"
    return s[:120]


def _is_under(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def resolve_train_save_dir(
    save_dir: str,
    experiment_name: str,
    *,
    cwd: Optional[Path] = None,
) -> Tuple[Path, bool]:
    """
    解析训练保存目录。相对路径在仓库内且不在 runs/、根目录 exports/、cache/ 下时，改写到 runs/train/<experiment>/。

    Returns:
        (绝对路径, 是否发生了改向)
    """
    cwd = Path(cwd or Path.cwd()).resolve()
    root = find_speedvqa_repo_root(cwd)
    raw = Path(save_dir).expanduser()
    exp_safe = sanitize_path_component(experiment_name)

    if raw.is_absolute():
        return raw.resolve(), False

    resolved = (cwd / raw).resolve()

    if root is None:
        return resolved, False

    runs_root = (root / "runs").resolve()
    legacy_exports = (root / "exports").resolve()
    cache_root = (root / "cache").resolve()

    for base in (runs_root, legacy_exports, cache_root):
        if _is_under(resolved, base):
            return resolved, False

    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        return resolved, False

    target = runs_root / "train" / exp_safe
    _LOG.warning(
        "train.save_dir=%r 会落在仓库内但不在 runs/、exports/、cache/ 下，已改写到 %s",
        save_dir,
        target,
    )
    return target, True


def resolve_torch_write_path(
    file_path: Union[str, Path],
    *,
    experiment_name: str = "default_exp",
    cwd: Optional[Path] = None,
    artifact_kind: ArtifactKind = "train",
) -> Path:
    """
    解析 torch.save / 导出文件路径：禁止在仓库根或源码树散落的「单文件无目录」落点。

    - 绝对路径且不在本仓库内：原样返回。
    - 本仓库内但父目录不在 runs/、根目录 exports/、cache/：改写到
      ``runs/train/<experiment>/`` 或 ``runs/exports/<experiment>/``（由 artifact_kind 决定）。
    """
    cwd = Path(cwd or Path.cwd()).resolve()
    root = find_speedvqa_repo_root(cwd)
    path = Path(file_path).expanduser()
    exp_safe = sanitize_path_component(experiment_name)

    if path.is_absolute():
        full = path.resolve()
    else:
        full = (cwd / path).resolve()

    if root is None:
        return full

    runs_root = (root / "runs").resolve()
    legacy_exports = (root / "exports").resolve()
    cache_root = (root / "cache").resolve()

    parent = full.parent

    for base in (runs_root, legacy_exports, cache_root):
        if _is_under(parent, base):
            return full

    try:
        parent.relative_to(root.resolve())
    except ValueError:
        return full

    if artifact_kind == "export":
        new_parent = runs_root / "exports" / exp_safe
    else:
        new_parent = runs_root / "train" / exp_safe

    redirected = new_parent / full.name
    _LOG.warning(
        "写入路径 %s 位于仓库内但不在 runs/、exports/、cache/ 下，已改写到 %s",
        full,
        redirected,
    )
    return redirected
