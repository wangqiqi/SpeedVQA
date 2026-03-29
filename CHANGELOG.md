# Changelog

## [2026-03-29] — tag `chore-project-scaffolding`

### 新增

- 建立 `archive/` 目录，用于后续归档文档与文件（建议命名：`YYYYMMDD_HHMMSS_功能_模块说明.md`）。
- 新增根目录 `plan.md`，用于记录较大任务与计划。
- 建立 `.cursor/`：`rules/`、`skills/` 占位及 `AGENTS.md`，便于后续维护 Cursor 规则与技能。

---

## [2026-03-29] — tag `chore-ignore-hypothesis-kiro`

### 变更

- 将 `.hypothesis/`、`.kiro/` 从版本库中移除并写入 `.gitignore`，不再跟踪测试缓存与 Kiro 规格目录。
