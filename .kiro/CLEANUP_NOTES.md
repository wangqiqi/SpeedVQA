# 项目清理记录

## 问题描述

项目根目录出现了大量乱码文件，这些文件名包含无效的UTF-8字符，例如：
- `è`
- `°×fÄb𦊠`
- `񭋙ÌBW𺑜\:哤è`
- 等等

## 根本原因

这些乱码文件通常由以下原因产生：

1. **Hypothesis属性测试框架** - 在某些情况下，Hypothesis可能会在根目录创建临时文件
2. **文件系统编码问题** - 某些进程在写入文件名时出现了编码错误
3. **测试框架副作用** - 某些测试可能在根目录创建了文件而没有正确清理

## 解决方案

### 1. 已清理的文件
所有乱码文件已从根目录删除。

### 2. 新增配置文件

#### `pytest.ini`
- 配置pytest的行为
- 定义测试发现规则
- 设置Hypothesis配置文件

#### `conftest.py`
- 全局pytest配置
- Hypothesis配置和初始化
- 禁用Hugging Face模型下载（加快测试）

### 3. .gitignore更新
- 添加规则防止乱码文件被提交
- 确保`.hypothesis/`目录被忽略

## 预防措施

为了防止这个问题再次发生：

1. **使用临时目录** - 所有测试应该使用`tempfile.mkdtemp()`创建临时文件
2. **正确清理** - 使用`shutil.rmtree()`清理临时目录
3. **避免根目录写入** - 不要在根目录创建任何文件，除非必要

## 测试运行建议

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_model_export.py

# 运行属性测试（Hypothesis）
pytest tests/test_roi_inference_properties.py -v

# 使用特定的Hypothesis配置
HYPOTHESIS_PROFILE=ci pytest
```

## 相关文件

- `.hypothesis/` - Hypothesis测试数据库（已在.gitignore中）
- `pytest.ini` - Pytest配置
- `conftest.py` - 全局测试配置
- `.gitignore` - Git忽略规则
