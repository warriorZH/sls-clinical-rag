#!/usr/bin/env bash
# 一键运行补充分析：  bash run_all.sh
# 自动转到后台运行；关闭终端不影响。随时用 bash status.sh 查看进度。中断后再次运行会从断点继续。
#
# 顺序（“先验证、后长跑”）：
#   1. 装环境 → 预检显卡（下载大模型之前）→ 下载小模型 → 用小模型实际启动 vLLM 预检 → 下载其余模型 → 准备 S4 数据
#   2. 全部试跑：三个服务的接口检查与官方评测试跑、S4 生成试跑；评审单独服务的接口检查与 S3/S4 重评试跑；
#      S4 微调与打分的冒烟测试（含用时估计）
#   3. 正式运行：S4 微调、打分、选证据 → S1/S2 官方评测、S4 生成答案、S4 选择题 → S3 重评、S4 答案评审 → 汇总打包
# 需要 vLLM 服务的步骤：服务中断或卡死时（退出码 3）自动重启服务、从断点继续，最多 3 次；其他错误立即停下。
source "$(cd "$(dirname "$0")" && pwd)/tools/env.sh"
LOG="$SUP_WORK/logs/run_all.log"
if [ "${1:-}" != "--fg" ]; then
  if pgrep -f "run_all.sh --fg" >/dev/null 2>&1; then
    echo "补充分析已经在后台运行。查看进度：bash status.sh"; exit 0
  fi
  rm -f "$STATE/FAILED" "$STATE/STOPPED"
  setsid nohup bash "$HERE/run_all.sh" --fg >> "$LOG" 2>&1 < /dev/null &
  sleep 2
  echo "已在后台启动（日志：$LOG）。"
  echo "查看进度：bash status.sh        停止：bash stop.sh"
  exit 0
fi

set -euo pipefail
cd "$HERE"
on_exit() {
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "$(date "+%m-%d %H:%M:%S") 出错退出（步骤：$(cat "$STATE/CURRENT" 2>/dev/null)，返回码 $rc）" > "$STATE/FAILED"
    echo "===== $(date '+%Y-%m-%d %H:%M:%S') 出错停止（步骤：$(cat "$STATE/CURRENT" 2>/dev/null)）"
    [ -x "$PY" ] && "$PY" "$HERE/sup/serve.py" stop >/dev/null 2>&1 || true
  fi
}
trap on_exit EXIT
echo "===== $(date '+%Y-%m-%d %H:%M:%S') 开始（工作目录 $SUP_WORK）"

stage() {   # stage <步骤名> 命令...：不需要 vLLM 服务的步骤
  local name=$1; shift
  if [ -f "$STATE/$name.done" ]; then echo "[跳过，已完成] $name"; return 0; fi
  echo "[开始] $name  $(date '+%m-%d %H:%M:%S')"
  echo "$name" > "$STATE/CURRENT"
  "$@"
  touch "$STATE/$name.done"
  echo "[完成] $name  $(date '+%m-%d %H:%M:%S')"
}

srv_stage() {   # srv_stage <all|judge> <步骤名> 命令...：需要 vLLM 服务的步骤
  local mode=$1 name=$2; shift 2
  if [ -f "$STATE/$name.done" ]; then echo "[跳过，已完成] $name"; return 0; fi
  local try rc
  for try in 1 2 3 4; do
    echo "$name（启动服务）" > "$STATE/CURRENT"
    "$PY" "$HERE/sup/serve.py" start "$mode"
    echo "$name" > "$STATE/CURRENT"
    echo "[开始] $name（第 $try 次）  $(date '+%m-%d %H:%M:%S')"
    rc=0; "$@" || rc=$?
    if [ "$rc" -eq 0 ]; then
      touch "$STATE/$name.done"; echo "[完成] $name  $(date '+%m-%d %H:%M:%S')"; return 0
    fi
    if [ "$rc" -ne 3 ]; then echo "[失败] $name（返回码 $rc）"; return "$rc"; fi
    echo "[服务中断] $name：停止并重启服务后从断点继续  $(date '+%m-%d %H:%M:%S')"
    "$PY" "$HERE/sup/serve.py" stop || true
  done
  echo "[失败] $name：服务反复中断"; return 1
}

cfg() { python3 -c "import json;c=json.load(open('$HERE/config/sup.json'));print(int(bool($1)))"; }
REJUDGE=$(cfg "c['rejudge']['enabled']")
S4=$(cfg "c.get('s4',{}).get('enabled',False)")

# ---------------------------------------------------------------- 1. 环境、预检、模型
stage setup bash "$HERE/setup.sh"
"$PY" "$HERE/tools/check_freeze.py"
stage preflight_gpu "$PY" "$HERE/sup/preflight.py" gpu
source "$HERE/tools/env.sh"                         # 载入预检得出的 NCCL 设置
stage models_small "$PY" "$HERE/sup/download_models.py" --only gen
stage preflight_vllm "$PY" "$HERE/sup/preflight.py" vllm
stage models "$PY" "$HERE/sup/download_models.py"
if [ "$S4" = "1" ]; then stage s4_data "$PY" "$HERE/sup/s4_prepare_data.py"; fi

# ---------------------------------------------------------------- 2. 试跑（全部在长时间运行之前）
srv_stage all preflight_api_all "$PY" "$HERE/sup/preflight.py" api all
srv_stage all smoke_official "$PY" "$HERE/sup/task_official.py" --smoke
if [ "$S4" = "1" ]; then srv_stage all s4_smoke_gen "$PY" "$HERE/sup/s4_generate.py" --smoke; fi
if [ "$REJUDGE" = "1" ] || [ "$S4" = "1" ]; then
  srv_stage judge preflight_api_judge "$PY" "$HERE/sup/preflight.py" api judge
fi
if [ "$REJUDGE" = "1" ]; then srv_stage judge smoke_rejudge "$PY" "$HERE/sup/task_rejudge.py" --smoke; fi
if [ "$S4" = "1" ]; then
  srv_stage judge s4_smoke_rejudge "$PY" "$HERE/sup/task_rejudge.py" --s4 --smoke
  # S4 的微调与打分要独占全部显卡：先停掉 vLLM 服务
  if [ ! -f "$STATE/s4_select.done" ]; then echo "s4（停止服务）" > "$STATE/CURRENT"; "$PY" "$HERE/sup/serve.py" stop || true; fi
  stage s4_smoke "$PY" "$HERE/sup/s4_orchestrate.py" smoke
  echo "===== $(date '+%Y-%m-%d %H:%M:%S') 全部试跑通过，开始正式运行"

# ---------------------------------------------------------------- 3. 正式运行
  stage s4_train "$PY" "$HERE/sup/s4_orchestrate.py" train
  stage s4_score "$PY" "$HERE/sup/s4_orchestrate.py" score
  stage s4_select "$PY" "$HERE/sup/s4_select.py"
fi
srv_stage all official "$PY" "$HERE/sup/task_official.py"
if [ "$S4" = "1" ]; then
  srv_stage all s4_generate "$PY" "$HERE/sup/s4_generate.py"
  srv_stage all s4_mcq "$PY" "$HERE/sup/s4_mcq.py"
fi
if [ "$REJUDGE" = "1" ]; then srv_stage judge rejudge "$PY" "$HERE/sup/task_rejudge.py"; fi
if [ "$S4" = "1" ]; then srv_stage judge s4_rejudge "$PY" "$HERE/sup/task_rejudge.py" --s4; fi
echo "summary" > "$STATE/CURRENT"
"$PY" "$HERE/sup/serve.py" stop || true
"$PY" "$HERE/sup/summarize_sup.py"

TS=$(date '+%Y%m%d_%H%M')
OUT="$SUP_WORK/results_sup_$TS.tar.gz"
cp "$HERE/config/sup.json" "$HERE/config/prompts_sup.json" "$HERE/FREEZE_MANIFEST_sup.json" "$SUP_WORK/results/" 2>/dev/null || true
cp "$SUP_MODELS/registry.json" "$SUP_WORK/results/models_registry.json" 2>/dev/null || true
cp "$STATE"/preflight_*.json "$STATE"/vllm_variants.json "$SUP_WORK/results/" 2>/dev/null || true
S4PART=""
[ -d "$SUP_WORK/s4" ] && S4PART="s4"
tar czf "$OUT" --exclude='s4/train/*/adapter' --exclude='s4/stage4' --exclude='*.npy.tmp' -C "$SUP_WORK" results logs $S4PART
cp "$OUT" "$HERE/" 2>/dev/null || true
echo "done" > "$STATE/CURRENT"
touch "$STATE/ALL_DONE"
echo "===== $(date '+%Y-%m-%d %H:%M:%S') 全部完成。结果包：$OUT"
echo "请下载这个文件发给 Claude，然后关机（平台按开机时长计费）。"
