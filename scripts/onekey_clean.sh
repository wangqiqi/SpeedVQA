#!/usr/bin/env bash
# 清理仓库内可再生的缓存与实验/导出产物（不删除虚拟环境 .venv/venv）。
# 默认导出在 runs/exports/，删除 runs/ 即一并清理；仓库根 exports/ 仅用于清除旧版遗留目录。
# 用法：./scripts/onekey_clean.sh [--dry-run|-n]

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run|-n) DRY_RUN=1 ;;
    -h|--help)
      cat <<'EOF'
清理 runs/（含 runs/exports/ 等）、根目录遗留 exports/、cache、基准/图表输出、测试缓存、Python 缓存等（见脚本内列表）。

  ./scripts/onekey_clean.sh           # 执行删除
  ./scripts/onekey_clean.sh --dry-run # 仅打印将删除的路径
EOF
      exit 0
      ;;
  esac
done

rm_path() {
  local p="$1"
  [[ -e "$p" ]] || return 0
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] rm -rf $p"
  else
    rm -rf "$p"
    echo "removed: $p"
  fi
}

shopt -s nullglob

# —— 仓库根目录下的约定目录 ——
# runs/：训练、验证、runs/exports/（onekey_export 默认）、runs/benchmark_reports/ 等
# exports/：旧版根目录导出（现默认已迁至 runs/exports/，此项仅清遗留）
for name in runs exports cache test_runs test_outputs test_data \
  .pytest_cache .hypothesis build dist htmlcov .mypy_cache .ruff_cache .tox \
  mlruns tb_logs performance_reports \
  visualizations plots inference_outputs predictions \
  model_exports exported_models; do
  rm_path "$ROOT/$name"
done

# 仓库根目录下的图表/动图（多为 exporter / benchmark 生成，.gitignore 已忽略）
for f in "$ROOT"/*.png "$ROOT"/*.jpg "$ROOT"/*.jpeg "$ROOT"/*.gif; do
  [[ -f "$f" ]] || continue
  rm_path "$f"
done

# 根目录 results_* 目录（与 .gitignore results_*/ 一致）
for d in "$ROOT"/results_*; do
  [[ -d "$d" ]] && rm_path "$d"
done

for d in "$ROOT"/hyperopt_results_*; do
  [[ -d "$d" ]] && rm_path "$d"
done

# 根目录无扩展名且 file 判定为 Zip 的文件（多为误执行 torch.save 于仓库根产生的无扩展名检查点）
while IFS= read -r -d '' f; do
  [[ -f "$f" ]] || continue
  if file -b -- "$f" 2>/dev/null | grep -qi "zip archive"; then
    rm_path "$f"
  fi
done < <(find "$ROOT" -maxdepth 1 -type f ! -name '*.*' -print0 2>/dev/null || true)

for d in "$ROOT"/*.egg-info; do
  [[ -d "$d" ]] && rm_path "$d"
done

# 根目录常见单文件
for f in "$ROOT"/.coverage "$ROOT"/coverage.xml; do
  rm_path "$f"
done

# —— 包与脚本树内的 Python / pytest 缓存（不遍历 .git）——
CLEAN_FIND_ROOTS=("$ROOT/speedvqa" "$ROOT/scripts")
for extra in "$ROOT/examples" "$ROOT/tests"; do
  [[ -d "$extra" ]] && CLEAN_FIND_ROOTS+=("$extra")
done
for base in "${CLEAN_FIND_ROOTS[@]}"; do
  [[ -d "$base" ]] || continue
  while IFS= read -r -d '' dir; do
    rm_path "$dir"
  done < <(find "$base" -type d \( -name '__pycache__' -o -name '.pytest_cache' \) -print0 2>/dev/null || true)
done

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[dry-run] 完成（未实际删除）"
else
  echo "onekey_clean: 完成"
fi
