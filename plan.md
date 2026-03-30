# 计划（Plan）

用于记录较大规模任务（例如超过 5 个 todo 的改动）、方案讨论与评审结论；执行完成后在 `CHANGELOG.md` 中记一笔，并视需要把过程材料归档到 `archive/`。

## 命名与归档

- 归档文件放在 `archive/`，建议文件名：`YYYYMMDD_HHMMSS_功能_模块说明.md`。

## 状态字段（排期用）

| 字段 | 含义 |
|------|------|
| **负责人** | 可填姓名或 `@` |
| **目标完成** | `YYYY-MM-DD` 或「第 N 迭代」 |
| **状态** | `待开始` / `进行中` / `阻塞` / `已完成` / `已取消` |

---

## 当前进行

### Phase A — 轻量跨模态融合（收尾中）

| 字段 | 内容 |
|------|------|
| **进展** | 冒烟集上 **A-5** **`concat` vs `film`** 已跑并记入下表；**A-6** 在 **`phase0_baseline_smoke`** 检查点上 **`onekey_export` → ONNX** 成功，**ORT 校验 `numerical_accuracy` 通过**（见基线表）。**`_validate_pytorch_export`** 已与模型参数设备对齐。 |
| **待办** | 在**业务真实数据集**上重复 A-5/A-6；**`film` 检查点**上补跑一次 ONNX（与 concat 等价路径）；**p50/p95** 仍待登记。 |
| **状态** | `进行中`（冒烟范围已闭合） |

---

## 待办 / 讨论中

### 算法设计评审摘要（2026-03-29）

- **工程适配度** 约 7.5～8 / 10；**学术新颖性** 约 5.5～6 / 10。  
- **短板**：多模态交互偏浅（默认 concat；现有 `attention` 为两 token 自注意力，无真正跨模态注意力）。  
- **改进方向（本计划已排期）**：轻量跨模态融合 → 可选 CLIP 式对齐 → 整图/空间建模（backlog）。

---

### 算法改进 — 可排期执行计划（2026-03-29）

#### 总目标与范围

在 **不推翻现有训练/导出/推理链路** 的前提下，按优先级提升 **多模态交互与可解释的上限**；每条线均需 **可量化验收** 与 **导出路径验证**（至少 PyTorch + ONNX；TensorRT 按环境选测）。

**不在本计划首期范围**：更换任务为开放域生成式 VQA；大规模换骨干（可作为独立立项）。

#### 基线与验收（全局约定）

1. **基线登记（Phase 0 产出）**  
   - 记录：`default.yaml` 当前融合方式、骨干、随机种子、数据划分版本。  
   - 指标：**验证集** Accuracy、Macro-F1（或业务约定主指标）、混淆矩阵；**推理** 在约定硬件上的 **p50 / p95 延迟**（batch=1，与 `ROIInferencer` 一致）。  
   - 产物路径：写入 `runs/<实验名>/` 与本文档「基线表」行。

2. **阶段验收默认门槛**（可按业务收紧；**开工前由负责人填具体数字**）  

   | 项 | 默认规则 |
   |---|---|
   | 精度 | 验证集主指标 **不低于基线 −0.5 个百分点**（若多次运行方差大，取三次均值） |
   | 延迟 | 相对基线 p50 **增幅 ≤ 15%**；p95 **增幅 ≤ 25%**（否则需记录原因并走「回滚/配置开关」） |
   | 导出 | 新融合/新头须 **ONNX 导出成功** 且与 PyTorch 在固定样本上 **logits 误差在约定阈值内**（由测试或脚本定义） |
   | 回归 | 现有 **`pytest`** 与关键 **`speedvqa/tests/test_model_*`** 全绿；新增用例覆盖新 `fusion.method` |

3. **回滚策略**  
   - 配置层保留 `fusion.method: concat` 为默认；新方法仅作可选。  
   - 若导出失败或延迟超标：**不合并主分支** 或 **功能开关默认关闭**，直至满足上表。

---

#### Phase 0 — 基线与实验纪律（建议 0.5～1 人日）

| 任务 ID | 内容 | 交付物 | 状态 |
|---------|------|--------|------|
| P0-1 | 固定评估命令与数据集划分版本（含 seed） | `docs/02_使用.md`「**Phase 0 基线与实验协议**」 | **已完成** |
| P0-2 | 跑通一次完整 eval / 短训，记录基线指标与 `runs/` 目录名 | 见下「基线表（Phase 0 冒烟）」+ `runs/train/phase0_baseline_smoke/training.log` | **已完成**（冒烟集） |
| P0-3 | 记录当前 ONNX 导出命令与单样本数值对齐方式 | **`docs/02_使用.md`** + 基线表；**ONNX** 已在冒烟 ckpt 上跑通（见「Phase A 冒烟消融」） | **已完成** |

**建议排期**：迭代第 1 周第 1～2 个工作日。  
**依赖**：可用验证集与推理测试环境（CPU 可仅测导出；延迟在目标 GPU 上测）。

---

#### Phase A — 轻量跨模态融合（建议 3～6 人日）

**目标**：在 **单向量视觉特征 + 单向量文本特征** 约束下，引入显式跨模态交互，优于纯 concat / 双 token 自注意力；参数量与算力可控。

**方案优先级（实现时二选一为主，另一可作消融）**  

1. **A1（推荐首选）**：**FiLM / 门控调制** — 由文本产生对视觉特征的 scale/shift 或门控（小 MLP），再进入原有 MLP 头；实现面小，导出友好。  
2. **A2**：**单层跨模态注意力** — 例如以文本为 query、视觉为 key/value（特征各先投影到 `hidden_dim`，序列长度=1）；或与现有 `MultiModalFusion` 并列的新 `fusion.method`（如 `cross_attn` / `film`，命名在实现时统一）。

**任务清单**

| 任务 ID | 内容 | 涉及路径（预期） | 状态 |
|---------|------|------------------|------|
| A-1 | 设计张量形状与配置键，更新 `default.yaml` 示例与注释 | `speedvqa/configs/default.yaml` | **已完成** |
| A-2 | 实现新融合模块并接入 `SpeedVQAModel` | `speedvqa/models/speedvqa.py`（含 **`SpeedVQAOnnxWrapper`**） | **已完成** |
| A-3 | `factory` / 配置校验与 `SUPPORTED_*` | `speedvqa/models/factory.py` | **已完成** |
| A-4 | 单元测试：前向形状、与 concat 切换、配置非法报错 | `speedvqa/tests/test_speedvqa_model.py` | **已完成** |
| A-5 | 训练消融：新融合 vs concat，填基线对比表 | `runs/` + 下表「Phase A 冒烟消融」 | **已完成**（冒烟集；真实数据待补） |
| A-6 | ONNX 导出与数值对齐验证 | `speedvqa/export/exporter.py`、`onekey_export` | **已完成**（冒烟 **`concat`** ckpt；**`film`** ckpt 待同法复测） |
| A-7 | 更新 `docs/01_设计.md` 融合小节 | `docs/01_设计.md` | **已完成** |

**验收**：满足上文「阶段验收默认门槛」；默认 **不** 将新方法改为全局默认，直至消融达标后由评审改 `default.yaml`。

**建议排期**：Phase 0 结束后 **第 1 周后半～第 2 周**。

**风险**：小数据上过拟合；延迟略升。缓解：dropout、早停、可选冻结文本编码器。

---

#### Phase B — 可选：CLIP 式对齐 + 主任务头（建议 5～10 人日，独立里程碑）

**目标**：在双编码器之上增加 **对齐约束**（如 image-text 对比项或投影空间一致性），提升分布外鲁棒性；**主损失仍为二分类 CE**。

**任务清单（粗粒度）**

| 任务 ID | 内容 | 状态 |
|---------|------|------|
| B-1 | 文献与实现选型：对称对比 vs 非对称、是否复用现有投影维 | 待开始 |
| B-2 | 损失与权重调度（`loss` 配置或 `train` 节点） | 待开始 |
| B-3 | 训练循环与日志字段（对齐 loss / CE 分项） | `speedvqa/engine/trainer.py` 等 | 待开始 |
| B-4 | 验收：Phase A 同门槛 + 明确 OOD 或小验证子集协议（若有） | 待开始 |

**依赖**：Phase A 或至少 Phase 0 基线稳定。  
**风险**：训练不稳、调参成本高；**建议仅在 A 达标后立项**。

---

#### Phase C — Backlog：整图上下文 / 空间建模

**触发条件**：产品明确需要 **ROI + 整图** 或 **多区域推理**。

**预研任务（不定人日）**：空间特征保留（去 GAP 前 feature map）、轻量 ROI-全局融合、对 TensorRT 的影响评估。

**状态**：长期 backlog；**不进入当前 Sprint  unless 产品签字**。

---

#### 建议周历（可按人力压缩/拉长）

| 周次 | 内容 |
|------|------|
| W1 | Phase 0；启动 A-1～A-3 |
| W2 | A-2～A-6，中间对照基线 |
| W3 | A-7、合并决策；若启动 Phase B，则 B-1～B-2 |
| W4+ | Phase B 深度开发与实验（若立项） |

---

#### 基线表（由 Phase 0 填写）

**业务真实数据**行仍需在自有数据集上跑一次并替换下表「冒烟」行。

##### 基线表（Phase 0 冒烟 — 2026-03-30）

| 项 | 值 |
|----|-----|
| 日期 | 2026-03-30 |
| Git commit | 与 tag **`1.0330.2120`** 指向的提交一致（`git show 1.0330.2120`） |
| 配置 | `speedvqa/configs/phase0_smoke.yaml`（`defaults: [default]` + 短训 overrides） |
| `fusion.method` | `concat` |
| `data.split.random_seed` | `42`（同 `default.yaml`） |
| 数据 | `python -m speedvqa.examples.phase0_smoke_dataset --out ./datasets/phase0_smoke`（24 条合成样本，**仅验证链路**） |
| 验证集 Acc / F1（末 epoch） | Acc **1.0** / weighted F1 **1.0**（val **4** 条，极低方差，不可替代真实基线） |
| 推理 p50 / p95（硬件） | **未登记**；请在目标 GPU 上用 `ROIInferencer`、batch=1 补测并更新本表 |
| `runs/` 实验目录 | `runs/train/phase0_baseline_smoke/`（`best_checkpoint.pth`、`training.log`） |
| ONNX 对齐备注 | **`concat`** 冒烟 ckpt：**`onekey_export`** 生成 **`runs/exports/phase0_onnx_verify.onnx`**，校验 **`Validation: ✓ … accuracy=1.0`**（相对 PyTorch logits 在默认 tolerance 内）。**`film`** 结构同路径，建议在 **`phase0_baseline_smoke_film/best_checkpoint.pth`** 上再导一次归档。 |

##### Phase A 冒烟消融（`concat` vs `film`，2026-03-30）

数据：**`datasets/phase0_smoke`**（24 条），**`data.split.random_seed=42`**，**3 epoch**，配置 **`phase0_smoke.yaml`** / **`phase0_smoke_film.yaml`**。验证集仅 **4** 条，指标**无推断意义**，仅用于流程与相对对照。

| 融合 | 配置 | `runs/` | 末 epoch Val（Loss / Acc / wF1） |
|------|------|---------|----------------------------------|
| **concat** | `phase0_smoke.yaml` | `phase0_baseline_smoke/` | 0.6803 / 0.50 / 0.333 |
| **film** | `phase0_smoke_film.yaml` | `phase0_baseline_smoke_film/` | 0.7132 / 0.50 / 0.333 |

**说明**：本冒烟上 **`film` 末轮 val_loss 略高于 `concat`**，不具统计显著性；**默认生产仍保持 `concat`**，待真实数据评审后再切换。

---

## 已完成（近期）

### Phase 0 — 基线与实验纪律（冒烟，2026-03-30）

- **`docs/02_使用.md`**：新增「**Phase 0 基线与实验协议**」（锁定配置键、真实数据与冒烟命令、ONNX 说明、p50/p95 约定）。
- **`speedvqa/configs/phase0_smoke.yaml`**、**`speedvqa/examples/phase0_smoke_dataset.py`**：可复现短训与最小数据集。
- **`default.yaml`**：`optimizer.eps`、`scheduler.warmup_lr` / `min_lr` 改为 **`1.0e-8` / `1.0e-6`**，避免 PyYAML 将 `1e-8` 解析为字符串导致 `AdamW` 报错。
- **基线表**：见上文；CHANGELOG 同期条目请对照 Git tag 规范更新。

### Phase A — 融合与导出（迭代一，2026-03-30）

- **`MultiModalFusion`**：新增 **`film`**、 **`cross_attn`**；默认仍为 **`concat`**。
- **`ModelExporter.export_onnx`**：**`SpeedVQAOnnxWrapper`**、CPU 导出、ORT 校验时设备对齐。
- **文档/测试**：**`docs/01_设计.md`**、**`test_speedvqa_model.py`** 覆盖新融合。
- **A-5/A-6（冒烟）**：**`phase0_smoke_film.yaml`**；**`concat`** ckpt **ONNX** 导出与 ORT 数值校验通过；**`_validate_pytorch_export`** 设备对齐修复。

（阶段完成后将摘要移入此处，并引用 `CHANGELOG.md` 对应条目。）
