#!/usr/bin/env python3
"""生成 Phase 0 冒烟用小数据集（纯色素图 + vqa_labels.jsonl），便于无真实数据时登记流程与短训。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser(description="Create minimal VQA dataset for Phase 0 smoke")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("datasets/phase0_smoke"),
        help="Dataset root (will create images/ and vqa_labels.jsonl)",
    )
    parser.add_argument("--num-samples", type=int, default=24, help="Number of PNG + JSONL lines")
    args = parser.parse_args()
    root: Path = args.out
    img_dir = root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    question = "Is the sample marked even-index?"
    lines: list[str] = []
    for i in range(args.num_samples):
        name = f"smoke_{i:03d}.png"
        # 偶数下标视为正例（与问题语义一致）
        rgb = (200, 60, 60) if i % 2 == 0 else (60, 60, 200)
        Image.new("RGB", (224, 224), rgb).save(img_dir / name)
        answer = "yes" if i % 2 == 0 else "no"
        lines.append(json.dumps({"image": name, question: answer}, ensure_ascii=False))

    jsonl_path = root / "vqa_labels.jsonl"
    jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.num_samples} images under {img_dir}")
    print(f"Wrote {jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
