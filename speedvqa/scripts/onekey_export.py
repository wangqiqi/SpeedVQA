#!/usr/bin/env python3
"""从训练检查点构建模型并导出为多格式（默认 pytorch + onnx）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="SpeedVQA one-key export")
    parser.add_argument("--checkpoint", required=True, help="训练产生的 .pth 检查点")
    parser.add_argument(
        "--config",
        default=None,
        help="YAML；默认使用检查点内嵌 config",
    )
    parser.add_argument(
        "--output",
        default="exports",
        help="导出目录（相对仓库根目录）",
    )
    parser.add_argument(
        "--name",
        default="speedvqa_export",
        help="导出文件基名",
    )
    parser.add_argument(
        "--formats",
        default="pytorch,onnx",
        help="逗号分隔：pytorch,onnx,tensorrt",
    )
    args = parser.parse_args()

    import torch

    from speedvqa.export.exporter import export_model
    from speedvqa.models.speedvqa import build_speedvqa_model
    from speedvqa.utils.config import load_config

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if args.config:
        config = load_config(args.config)
    else:
        config = ckpt.get("config")
        if not config:
            raise ValueError("检查点中无 config，请传入 --config")

    model = build_speedvqa_model(config)
    model.load_state_dict(ckpt["model_state_dict"])

    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    export_model(
        model=model,
        output_dir=args.output,
        model_name=args.name,
        config=config,
        formats=formats,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
