"""
Pytest：Hypothesis 与收集策略（位于 `speedvqa/tests/`）。
"""

import os
from pathlib import Path

from hypothesis import HealthCheck, settings

# 独立脚本式测试（请用 python speedvqa/tests/<name>.py 运行），避免 pytest 重复收集
collect_ignore = [
    str(Path(__file__).resolve().parent / "test_model_basic.py"),
    str(Path(__file__).resolve().parent / "test_model_factory.py"),
]

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

settings.register_profile(
    "default",
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    deadline=None,
)
settings.register_profile("ci", max_examples=1000)
settings.register_profile("dev", max_examples=10)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "default"))
