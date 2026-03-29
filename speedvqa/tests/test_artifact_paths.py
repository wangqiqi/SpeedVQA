"""artifact_paths：训练/导出产物目录约束。"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path


from speedvqa.utils.artifact_paths import (
    find_speedvqa_repo_root,
    resolve_train_save_dir,
    resolve_torch_write_path,
    sanitize_path_component,
)


def _write_fake_repo(root):
    (root / "pyproject.toml").write_text(
        '[project]\nname = "speedvqa"\nversion = "0.0.0"\n',
        encoding="utf-8",
    )


def test_sanitize_path_component():
    assert sanitize_path_component("  a/b  ") == "a_b"
    assert sanitize_path_component("..") == "default_exp"


def test_find_speedvqa_repo_root(tmp_path, monkeypatch):
    _write_fake_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert find_speedvqa_repo_root() == tmp_path.resolve()


def test_resolve_train_save_dir_keeps_runs_train(tmp_path, monkeypatch):
    _write_fake_repo(tmp_path)
    (tmp_path / "runs" / "train").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    p, coerced = resolve_train_save_dir("./runs/train", "exp1")
    assert not coerced
    assert p == tmp_path / "runs" / "train"


def test_resolve_train_save_dir_coerces_dot(tmp_path, monkeypatch):
    _write_fake_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    p, coerced = resolve_train_save_dir(".", "my_exp")
    assert coerced
    assert p == tmp_path / "runs" / "train" / "my_exp"


def test_resolve_train_save_dir_coerces_stray_folder(tmp_path, monkeypatch):
    _write_fake_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    p, coerced = resolve_train_save_dir("junk_ckpt", "e2")
    assert coerced
    assert p == tmp_path / "runs" / "train" / "e2"


def test_resolve_torch_write_path_train(tmp_path, monkeypatch):
    _write_fake_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    out = resolve_torch_write_path(
        "eccrljpojs",
        experiment_name="e1",
        artifact_kind="train",
    )
    assert out == tmp_path / "runs" / "train" / "e1" / "eccrljpojs"


def test_resolve_torch_write_path_export(tmp_path, monkeypatch):
    _write_fake_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    out = resolve_torch_write_path(
        "model.pt",
        experiment_name="ex",
        artifact_kind="export",
    )
    assert out == tmp_path / "exports" / "ex" / "model.pt"


def test_resolve_torch_write_path_absolute_outside_repo(tmp_path, monkeypatch):
    _write_fake_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    td = Path(tempfile.mkdtemp(prefix="speedvqa_art_"))
    try:
        outside = td / "m.pt"
        outside.write_bytes(b"x")
        out = resolve_torch_write_path(outside, experiment_name="e", artifact_kind="train")
        assert out == outside.resolve()
    finally:
        shutil.rmtree(td, ignore_errors=True)
