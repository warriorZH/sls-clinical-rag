#!/usr/bin/env bash
# 停止补充分析（之后运行 bash run_all.sh 从断点继续）
source "$(cd "$(dirname "$0")" && pwd)/tools/env.sh"
pkill -f "run_all.sh --fg" 2>/dev/null || true
for i in $(seq 1 60); do pgrep -f "run_all.sh --fg" >/dev/null 2>&1 || break; sleep 2; done
pkill -f "$HERE/sup/" 2>/dev/null || true
[ -x "$PY" ] && "$PY" "$HERE/sup/serve.py" stop 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
pkill -f "^VLLM::" 2>/dev/null || true
rm -f "$STATE/FAILED"; touch "$STATE/STOPPED"
echo "已停止。之后运行 bash run_all.sh 会从断点继续。"
