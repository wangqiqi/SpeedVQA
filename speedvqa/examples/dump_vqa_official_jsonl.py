"""
将官方 VQA（questions + annotations）导出为 vqa_labels.jsonl，供外部工具或二次分发。

用法：
  python -m speedvqa.examples.dump_vqa_official_jsonl \\
    --root ./datasets/vqa_abstract_binary_2017/val2017 \\
    --out ./datasets/vqa_abstract_binary_2017/val2017/vqa_labels.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from speedvqa.data.loaders.vqa_official import load_vqa_official_samples


def main() -> None:
    p = argparse.ArgumentParser(description="Export official VQA JSON to vqa_labels.jsonl")
    p.add_argument("--root", type=Path, required=True, help="数据集根目录（含 images/ 与 JSON）")
    p.add_argument("--out", type=Path, required=True, help="输出 jsonl 路径")
    p.add_argument("-q", "--questions", type=Path, default=None)
    p.add_argument("-a", "--annotations", type=Path, default=None)
    args = p.parse_args()

    def passthrough(ans: str) -> str:
        return (ans or "").strip()

    samples = load_vqa_official_samples(
        args.root,
        passthrough,
        args.questions,
        args.annotations,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for s in samples:
            img_name = Path(s["image_path"]).name
            row = {"image": img_name, s["question"]: s["answer"]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(samples)} lines to {args.out}")


if __name__ == "__main__":
    main()
