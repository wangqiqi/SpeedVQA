"""
官方 VQA 发布格式（visualqa.org）适配：Questions JSON + Annotations JSON + images/。

当前支持 data_type == abstract_v002（Balanced Binary Abstract 等）的 PNG 命名规则；
其它 data_type 可后续扩展 resolve 函数，不必改 VQADataset 主流程。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


def _abstract_v002_image_basename(data_subtype: str, image_id: int) -> Optional[str]:
    """abstract 场景图：zip 内文件名为 train2015/val2015/test2015，与 data_subtype 的 *2017 等并存。"""
    s = (data_subtype or "").lower()
    if "train" in s:
        split_seg = "train2015"
    elif "val" in s:
        split_seg = "val2015"
    elif "test" in s:
        split_seg = "test2015"
    else:
        return None
    return f"abstract_v002_{split_seg}_{image_id:012d}.png"


def _mscoco_image_basename(data_subtype: str, image_id: int) -> Optional[str]:
    """MSCOCO 上 VQA 常用文件名：COCO_<subtype>_<image_id:012d>.jpg"""
    s = (data_subtype or "").lower()
    if not s:
        return None
    return f"COCO_{s}_{image_id:012d}.jpg"


def resolve_vqa_image_basename(data_type: str, data_subtype: str, image_id: int) -> Optional[str]:
    if data_type == "abstract_v002":
        return _abstract_v002_image_basename(data_subtype, image_id)
    if data_type == "mscoco":
        return _mscoco_image_basename(data_subtype, image_id)
    return None


def _is_questions_json(data: Dict[str, Any]) -> bool:
    return isinstance(data, dict) and isinstance(data.get("questions"), list) and len(data["questions"]) > 0


def _is_annotations_json(data: Dict[str, Any]) -> bool:
    anns = data.get("annotations") if isinstance(data, dict) else None
    if not isinstance(anns, list) or not anns:
        return False
    z = anns[0]
    return isinstance(z, dict) and "question_id" in z and "multiple_choice_answer" in z


def find_official_vqa_pairs(root: Path) -> List[Tuple[Path, Path]]:
    """在数据集根目录查找 (questions.json, annotations.json) 对，按 data_type + data_subtype 配对。"""
    root = root.resolve()
    q_candidates = sorted(root.glob("OpenEnded_*_questions.json"))
    if not q_candidates:
        q_candidates = sorted(
            p for p in root.glob("*_questions.json") if p.is_file() and "composition" not in p.name.lower()
        )
    a_candidates = sorted(p for p in root.glob("*_annotations.json") if p.is_file())

    pairs: List[Tuple[Path, Path]] = []
    used_annotation: set = set()
    for qp in q_candidates:
        try:
            with open(qp, "r", encoding="utf-8") as f:
                qd = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if not _is_questions_json(qd):
            continue
        q_type = qd.get("data_type")
        q_sub = qd.get("data_subtype")
        for ap in a_candidates:
            if ap in used_annotation:
                continue
            try:
                with open(ap, "r", encoding="utf-8") as f:
                    ad = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            if not _is_annotations_json(ad):
                continue
            if ad.get("data_type") == q_type and ad.get("data_subtype") == q_sub:
                pairs.append((qp, ap))
                used_annotation.add(ap)
                break
    return pairs


def load_vqa_official_samples(
    root: Path,
    normalize_answer: Callable[[str], str],
    questions_path: Optional[Path] = None,
    annotations_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    将官方 questions + annotations 转为与 X-AnyLabeling 路径一致的 sample 列表。
    """
    root = root.resolve()
    images_dir = root / "images"
    if questions_path and annotations_path:
        pairs = [(Path(questions_path).resolve(), Path(annotations_path).resolve())]
    else:
        pairs = find_official_vqa_pairs(root)

    samples: List[Dict[str, Any]] = []
    for qp, ap in pairs:
        with open(qp, "r", encoding="utf-8") as f:
            qd = json.load(f)
        with open(ap, "r", encoding="utf-8") as f:
            ad = json.load(f)

        data_type = ad.get("data_type", "")
        data_subtype = ad.get("data_subtype", "")
        by_qid = {int(q["question_id"]): q for q in qd["questions"]}

        for ann in ad["annotations"]:
            qid = int(ann["question_id"])
            qrec = by_qid.get(qid)
            if qrec is None:
                continue
            image_id = int(ann.get("image_id", qrec.get("image_id", -1)))
            basename = resolve_vqa_image_basename(data_type, data_subtype, image_id)
            if not basename:
                continue
            image_path = images_dir / basename
            if not image_path.exists():
                continue

            raw_answer = str(ann.get("multiple_choice_answer", ""))
            norm = normalize_answer(raw_answer)
            question_text = str(qrec.get("question", "")).strip()

            samples.append(
                {
                    "image_path": str(image_path),
                    "question": question_text,
                    "answer": norm,
                    "source": "vqa_official_json",
                    "metadata": {
                        "question_id": qid,
                        "image_id": image_id,
                        "answer_type": ann.get("answer_type"),
                        "multiple_choice_answer": raw_answer,
                        "data_type": data_type,
                        "data_subtype": data_subtype,
                        "questions_file": str(qp),
                        "annotations_file": str(ap),
                    },
                }
            )
    return samples


def _resolve_under_root(root: Path, p: Optional[str]) -> Optional[Path]:
    if not p:
        return None
    path = Path(p)
    if not path.is_absolute():
        path = root / path
    return path


def load_vqa_official_if_enabled(
    root: Path,
    normalize_answer: Callable[[str], str],
    enabled: Any,
    questions_path: Optional[str] = None,
    annotations_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    enabled: 'auto' | True | False （YAML 可能为布尔）
    questions_json / annotations_json 若为相对路径，相对于数据集根目录 data_path。
    """
    root = root.resolve()
    mode = enabled
    qpath = _resolve_under_root(root, questions_path)
    apath = _resolve_under_root(root, annotations_path)

    if mode is True or mode == "true":
        if qpath and apath:
            samples = load_vqa_official_samples(root, normalize_answer, qpath, apath)
        else:
            samples = load_vqa_official_samples(root, normalize_answer)
        if not samples:
            raise ValueError(
                f"vqa_official enabled but no samples loaded from {root}. "
                "Check OpenEnded_*_questions.json, *_annotations.json, and images/."
            )
        return samples

    if mode is False or mode == "false":
        return []

    # auto
    if qpath and apath and qpath.is_file() and apath.is_file():
        return load_vqa_official_samples(root, normalize_answer, qpath, apath)
    if not find_official_vqa_pairs(root):
        return []
    return load_vqa_official_samples(root, normalize_answer)
