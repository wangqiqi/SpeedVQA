#!/usr/bin/env python3
"""从仓库根目录读取配置并启动 ConfigurableTrainer 训练。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    from speedvqa.utils.config import get_builtin_default_config_path

    parser = argparse.ArgumentParser(description="SpeedVQA one-key training")
    parser.add_argument(
        "--config",
        default=get_builtin_default_config_path(),
        help="YAML 配置路径（默认包内 speedvqa/configs/default.yaml）",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="覆盖 data.dataset_path",
    )
    args = parser.parse_args()

    extra: dict = {}
    if args.data:
        extra["data"] = {"dataset_path": args.data}

    from speedvqa.data.datasets import build_dataset, create_dataloader, split_dataset
    from speedvqa.engine.trainer import ConfigurableTrainer
    from speedvqa.models.speedvqa import build_speedvqa_model
    from speedvqa.utils.config import load_config

    config = load_config(args.config, **extra)
    data_path = config["data"]["dataset_path"]

    full_ds = build_dataset(data_path, config, split="train")
    split_cfg = config.get("data", {}).get("split", {})
    train_ds, val_ds, _test_ds = split_dataset(
        full_ds,
        train_ratio=split_cfg.get("train_ratio", 0.7),
        val_ratio=split_cfg.get("val_ratio", 0.2),
        test_ratio=split_cfg.get("test_ratio", 0.1),
    )
    train_loader = create_dataloader(train_ds, config, split="train")
    val_loader = create_dataloader(val_ds, config, split="val")

    model = build_speedvqa_model(config)
    trainer = ConfigurableTrainer(config)
    trainer.train(model, train_loader, val_loader)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
