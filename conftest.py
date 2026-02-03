"""
Pytest配置文件

配置Hypothesis和其他测试框架的行为
"""

import os
from pathlib import Path
from hypothesis import settings, HealthCheck

# 禁用Hugging Face模型下载
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# 配置Hypothesis
# 设置Hypothesis数据库位置到.hypothesis目录，避免在根目录创建文件
settings.register_profile("default", max_examples=100)
settings.register_profile("ci", max_examples=1000)
settings.register_profile("dev", max_examples=10)

# 根据环境选择配置
profile = os.getenv("HYPOTHESIS_PROFILE", "default")
settings.load_profile(profile)

# 全局Hypothesis设置
settings.register_profile(
    "default",
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    deadline=None,
)

settings.load_profile("default")
