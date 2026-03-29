#!/usr/bin/env python3
"""加载检查点或导出的权重，对单张 ROI 图做一次推理。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _infer_config_from_checkpoint(ckpt_path: Path) -> dict:
    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu")
    full = ckpt.get("config") or {}
    model_cfg = full.get("model") or {}
    data_cfg = full.get("data", {})
    inf_cfg = full.get("inference", {})
    image_size = tuple(data_cfg.get("image", {}).get("size", [224, 224]))
    text_enc = model_cfg.get("text", {}).get("encoder", "distilbert-base-uncased")
    max_len = model_cfg.get("text", {}).get("max_length", 128)
    return {
        "image_size": image_size,
        "text_encoder": text_enc,
        "max_text_length": max_len,
        "confidence_threshold": inf_cfg.get("postprocess", {}).get("confidence_threshold", 0.5),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="SpeedVQA one-key predict")
    parser.add_argument("--model", required=True, help="模型或检查点 .pth 路径")
    parser.add_argument("--image", required=True, help="ROI 图像路径")
    parser.add_argument("--question", required=True, help="问句")
    parser.add_argument(
        "--format",
        default="pytorch",
        choices=["pytorch", "onnx", "tensorrt"],
        help="模型格式",
    )
    parser.add_argument("--device", default="cuda", help="cuda 或 cpu")
    parser.add_argument(
        "--config",
        default=None,
        help="可选 YAML；若不指定且为 pytorch，则从检查点 config 推断 tokenizer 等",
    )
    args = parser.parse_args()

    from speedvqa.inference.inferencer import ROIInferencer
    from speedvqa.utils.config import load_config

    infer_cfg: dict = {}
    if args.config:
        full = load_config(args.config)
        d = full.get("data", {})
        inf = full.get("inference", {})
        infer_cfg = {
            "image_size": tuple(d.get("image", {}).get("size", [224, 224])),
            "text_encoder": full.get("model", {}).get("text", {}).get(
                "encoder", "distilbert-base-uncased"
            ),
            "max_text_length": full.get("model", {}).get("text", {}).get("max_length", 128),
            "confidence_threshold": inf.get("postprocess", {}).get("confidence_threshold", 0.5),
        }
    elif args.format == "pytorch":
        infer_cfg = _infer_config_from_checkpoint(Path(args.model))

    inferencer = ROIInferencer(
        args.model,
        model_format=args.format,
        device=args.device,
        config=infer_cfg,
    )
    result = inferencer.inference(args.image, args.question)
    print(f"answer={result.answer} confidence={result.confidence:.4f} time_ms={result.inference_time_ms:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
