# 计划（Plan）

较大任务、评审结论与基线表；执行完成后更新 **`CHANGELOG.md`**，过程材料可放入 **`archive/`**（命名建议：`YYYYMMDD_HHMMSS_功能_模块说明.md`）。

## 状态字段

| 字段 | 含义 |
|------|------|
| **负责人** | 可填姓名或 `@` |
| **目标完成** | `YYYY-MM-DD` 或「第 N 迭代」 |
| **状态** | `待开始` / `进行中` / `阻塞` / `已完成` / `已取消` |

---

## 归档索引

- **Phase 0 + Phase A（冒烟闭环）**全文（旧基线表、A-1～A-7 冒烟交付说明）已迁至：  
  **`archive/20260330_233000_plan_Phase0与PhaseA冒烟完成归档.md`**

---

## 当前进行

### Phase A — 真实数据消融与导出收尾

| 字段 | 内容 |
|------|------|
| **进展** | **val2017** 上 **1 epoch** **`concat` vs `film`** 已跑；**`film` 冒烟 ckpt** 已 **`onekey_export` → ONNX**（`runs/exports/phase0_film_onnx_verify.onnx`）；**真实数据 1 epoch ckpt** 已分别导出 ONNX（`abstract_val2017_concat_1ep` / `abstract_val2017_film_1ep`）。**ROIInferencer** batch=1、**CUDA** 上 **p50/p95** 已用 `python -m speedvqa.examples.bench_roi_inferencer_latency` 登记（见下表）。 |
| **待办** | 提高 **epoch 数 / 调参** 形成可汇报基线；**train2017** 全量或业务自有数据；若需与 val 互斥划分，需独立配置 **train/val 目录** 而非同一目录随机切分。 |
| **状态** | `进行中` |

### Phase B — CLIP 式对齐 + 主任务头

| 任务 ID | 内容 | 状态 |
|---------|------|------|
| B-1～B-4 | 文献选型、损失与调度、训练循环与日志、验收 | **待开始** |

**依赖**：Phase A 真实数据基线稳定后再立项。

### Phase C — Backlog（整图 / 空间建模）

**触发条件**：产品明确 ROI + 整图或多区域推理。**状态**：长期 backlog。

---

## 算法设计评审摘要（2026-03-29）

- **工程适配度** 约 7.5～8 / 10；**学术新颖性** 约 5.5～6 / 10。  
- **改进方向**：轻量跨模态融合（已交付 film/cross_attn）→ 可选 CLIP 对齐 → 整图/空间建模（backlog）。

---

## 基线表（抽象 val2017，Phase A 真实数据短训 — 2026-03-30）

**数据**：`datasets/vqa_abstract_binary_2017/val2017`（**11 328** 条，**`vqa_official:auto`**），**`data.split.random_seed=42`**，**70/20/10** 随机划分（注意：训练与验证均来自同一标注集，非官方互斥 split）。**1 epoch** 快速对比，非收敛基线。

| 融合 | 配置 | `runs/train/` | Val（Loss / Acc / wF1） |
|------|------|---------------|-------------------------|
| **concat** | `abstract_binary_val2017_A5_short.yaml` | `abstract_binary_val2017_concat_A5/` | 0.6932 / **0.5033** / 0.3631 |
| **film** | `abstract_binary_val2017_film_A5_short.yaml` | `abstract_binary_val2017_film_A5/` | 0.6934 / **0.5064** / 0.3405 |

**ONNX**：`runs/exports/abstract_val2017_concat_1ep.onnx`、`abstract_val2017_film_1ep.onnx`（导出校验 **accuracy=1.0** 为导出探头一致性，非验证集精度）。

## 推理延迟（ROIInferencer，batch=1，2026-03-30）

**硬件**：NVIDIA GeForce RTX 4090，**CUDA**。脚本：`python -m speedvqa.examples.bench_roi_inferencer_latency --device cuda --runs 50 --warmup 10`。

| 检查点 | p50_ms | p95_ms |
|--------|--------|--------|
| `runs/train/phase0_baseline_smoke/best_checkpoint.pth`（**concat**） | 4.230 | 4.709 |
| `runs/train/phase0_baseline_smoke_film/best_checkpoint.pth`（**film**） | 4.124 | 4.568 |

**说明**：随机 ROI 图像 + 固定英文问题；用于相对比较 smoke 模型上 **film 相对 concat** 的延迟档位。

---

## 建议周历（参考）

| 周次 | 内容 |
|------|------|
| W1 | Phase A：加 epoch、可选 train2017；评审是否立项 Phase B |
| W2+ | Phase B（若立项） |
