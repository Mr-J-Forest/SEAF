#!/usr/bin/env bash
set -euo pipefail
#==========================================
# sync_to_server.sh
# 将 SEAF 本地代码同步到远程服务器。
# 排除规则自动从 .gitignore 读取，保持一致。
#
# 用法: ./sync_to_server.sh
#==========================================

SERVER="${SEAF_SERVER:-${TSC_SERVER:-root@connect.westd.seetacloud.com}}"
PORT="${SEAF_SERVER_PORT:-${TSC_SERVER_PORT:-39323}}"
REMOTE_DIR="${SEAF_REMOTE_DIR:-${TSC_REMOTE_DIR:-/root/TSC-Fusion}}"
SSH_BIN="${SEAF_SSH_BIN:-${TSC_SSH_BIN:-ssh}}"

echo "=== 同步代码到服务器 $SERVER:$REMOTE_DIR ==="
echo "由 Git 枚举已跟踪和未忽略的工作区文件..."

# 不把 .gitignore 模式直接转交给 tar。目录规则（例如 .venv/）在不同 tar
# 实现和 ./ 前缀下可能匹配不一致。让 Git 负责 ignore 语义，再将实际存在的
# 文件逐个交给 tar，可从根本上避免上传 Data、虚拟环境、缓存和训练输出。
FILES=()
while IFS= read -r -d '' path; do
  if [ "$path" = "sync_to_server.sh" ]; then
    continue
  fi
  if [ -e "$path" ] || [ -L "$path" ]; then
    FILES+=("$path")
  fi
done < <(git ls-files -co --exclude-standard --deduplicate -z)

if [ "${#FILES[@]}" -eq 0 ]; then
  echo "错误：没有找到可同步文件" >&2
  exit 1
fi

echo "将同步 ${#FILES[@]} 个文件（不包含被 .gitignore 忽略的内容）"

GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || true)"
if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
  GIT_DIRTY=true
else
  GIT_DIRTY=false
fi
SOURCE_HASH="$({
  for path in "${FILES[@]}"; do
    printf '%s\0' "$path"
    git hash-object -- "$path"
  done
} | git hash-object --stdin)"
TRAINING_FILES=(
  train.py config.py seaf_model.py model_factory.py recent_baseline_models.py
  paper_reimplementation_models.py
  data_loader.py metrics_utils.py font_config.py requirements.txt
  scripts/run_experiment_queue.py scripts/aggregate_results.py
)
while IFS= read -r path; do
  TRAINING_FILES+=("$path")
done < <(find configs -type f -name '*.json' -print | LC_ALL=C sort)
TRAINING_SOURCE_HASH="$({
  for path in "${TRAINING_FILES[@]}"; do
    if [ -e "$path" ]; then
      printf '%s\0' "$path"
      git hash-object -- "$path"
    fi
  done
} | git hash-object --stdin)"
SYNCED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# 通过 tar + ssh 管道同步
tar czf - "${FILES[@]}" | "$SSH_BIN" -p "$PORT" \
    -o StrictHostKeyChecking=no \
    -o ConnectTimeout=15 \
    "$SERVER" "cd $REMOTE_DIR && tar xzf -" 2>&1

# 远端没有 .git 目录；单独保存可追溯的本地源码状态供每次训练归档。
"$SSH_BIN" -p "$PORT" \
  -o StrictHostKeyChecking=no \
  -o ConnectTimeout=15 \
  "$SERVER" \
  "cd $REMOTE_DIR && printf '%s\n' '{\"git_commit\":\"$GIT_COMMIT\",\"git_dirty\":$GIT_DIRTY,\"source_hash\":\"$SOURCE_HASH\",\"training_source_hash\":\"$TRAINING_SOURCE_HASH\",\"synced_at\":\"$SYNCED_AT\"}' > source_state.json"

echo "=== 同步完成 ==="
