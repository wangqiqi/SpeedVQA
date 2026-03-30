# Workflows（可重复流程）

在此存放 **步骤化** 的流程说明（发版、数据集准备、CI 本地复现等），文件名建议语义化，例如：

- `release-changelog-tag.md` — 与 `CHANGELOG.md` + `git tag` 对齐的检查清单
- `smoke-train-export.md` — 短训 + 导出冒烟

新建 workflow 后，在 **`.cursor/AGENTS.md`** 的「资源地图」表中增加一行，便于 Agent 发现。
