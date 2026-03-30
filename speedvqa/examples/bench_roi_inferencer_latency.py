"""
单样本 ROIInferencer 延迟压测（batch=1）：打印 p50 / p95（毫秒）。

示例：
  python -m speedvqa.examples.bench_roi_inferencer_latency \\
    --checkpoint runs/train/phase0_baseline_smoke/best_checkpoint.pth \\
    --device cuda --runs 100
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default=None, help="默认用检查点内嵌 config")
    p.add_argument("--device", default="cuda")
    p.add_argument("--runs", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    args = p.parse_args()

    import torch
    from speedvqa.inference.inferencer import ROIInferencer
    from speedvqa.utils.config import load_config

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = load_config(args.config) if args.config else ckpt.get("config")
    if not config:
        raise SystemExit("检查点无 config，请传 --config")

    inf = ROIInferencer(
        model_path=args.checkpoint,
        model_format="pytorch",
        device=args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu",
        config=config,
    )

    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    question = "Is there a person?"

    for _ in range(args.warmup):
        inf.inference(img, question)

    times_ms = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        inf.inference(img, question)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.sort(np.array(times_ms))
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    print(f"runs={args.runs} device={str(inf.device)} checkpoint={args.checkpoint}")
    print(f"p50_ms={p50:.3f} p95_ms={p95:.3f} mean_ms={arr.mean():.3f}")


if __name__ == "__main__":
    main()
