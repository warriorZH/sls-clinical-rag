#!/usr/bin/env bash
# 服务器端辅助脚本（eval_v1）。由本机的 run_eval.ps1 通过 ssh 调用，也可以在服务器上手动运行：
#   bash eval_v1/server_run.sh probe            选出装有 torch+transformers 的 Python，检查 GPU 与 Hugging Face 连通性
#   bash eval_v1/server_run.sh discover         查找检查点，写出 discover_report.md 与 models.json
#   bash eval_v1/server_run.sh inspect          收集 CE 检查点（best_checkpoint.pt）与训练、打分源代码信息
#   bash eval_v1/server_run.sh start calib      后台运行校准集冒烟测试
#   bash eval_v1/server_run.sh start sealed     后台运行封存集一次性打分
#   bash eval_v1/server_run.sh status calib     查看后台任务状态
#   bash eval_v1/server_run.sh summary calib    汇总 RUN_MANIFEST.json；有方法失败时退出码为 1
cd "$(dirname "$0")/.." || exit 2
EV=eval_v1
LOGS=$EV/logs
mkdir -p "$LOGS"
PYFILE=$EV/.python_path
HFFILE=$EV/.hf_endpoint

probe_python() {
  local cands=() base e p
  for base in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "$HOME/mambaforge" "$HOME/.conda" \
              /opt/conda /opt/miniconda3 /opt/anaconda3 /usr/local/miniconda3 /usr/local/anaconda3; do
    [ -x "$base/bin/python" ] && cands+=("$base/bin/python")
    for e in "$base"/envs/*; do [ -x "$e/bin/python" ] && cands+=("$e/bin/python"); done
  done
  for p in python3 python; do command -v "$p" >/dev/null 2>&1 && cands+=("$(command -v "$p")"); done
  local best="" bestscore=-1 out score
  for p in "${cands[@]}"; do
    out=$(timeout 120 "$p" - <<'PYEOF' 2>/dev/null
import json
info = {'ok': False}
try:
    import torch, transformers
    info.update(ok=True, torch=torch.__version__, transformers=transformers.__version__,
                cuda=bool(torch.cuda.is_available()))
    v = tuple(int(x) for x in transformers.__version__.split('.')[:2])
    info['transformers_ge_4_51'] = v >= (4, 51)
    try:
        import sentencepiece  # noqa: F401
        info['sentencepiece'] = True
    except Exception:
        info['sentencepiece'] = False
    try:
        import huggingface_hub  # noqa: F401
        info['huggingface_hub'] = True
    except Exception:
        info['huggingface_hub'] = False
except Exception as e:
    info['error'] = type(e).__name__
score = -1
if info['ok']:
    score = 1 + 4 * info['cuda'] + 2 * info['transformers_ge_4_51'] + info['sentencepiece'] + info['huggingface_hub']
print(json.dumps(info))
print('SCORE', score)
PYEOF
)
    score=$(printf '%s\n' "$out" | awk '/^SCORE/{print $2}')
    printf 'PYTHON %s %s\n' "$p" "$(printf '%s\n' "$out" | head -n 1)"
    [ -z "$score" ] && score=-1
    if [ "$score" -gt "$bestscore" ]; then bestscore=$score; best=$p; fi
  done
  if [ "$bestscore" -lt 1 ]; then
    echo "ERROR 没有找到同时装有 torch 和 transformers 的 Python。"
    return 1
  fi
  echo "$best" > "$PYFILE"
  echo "CHOSEN_PYTHON $best (score $bestscore)"
}

probe_hf() {
  if [ -n "${HF_ENDPOINT:-}" ]; then echo "$HF_ENDPOINT" > "$HFFILE"; echo "HF_ENDPOINT 已设置为 $HF_ENDPOINT"; return; fi
  if curl -sS -m 10 -o /dev/null https://huggingface.co/api/models/BAAI/bge-reranker-v2-m3 2>/dev/null; then
    echo "https://huggingface.co" > "$HFFILE"; echo "HF 可直连"
  elif curl -sS -m 10 -o /dev/null https://hf-mirror.com/api/models/BAAI/bge-reranker-v2-m3 2>/dev/null; then
    echo "https://hf-mirror.com" > "$HFFILE"; echo "HF 不可直连，改用 https://hf-mirror.com"
  else
    : > "$HFFILE"; echo "HF 与镜像都不可达：公开模型须已在本机缓存或手动放到服务器上"
  fi
}

job_running() {
  # 只有 pid 存活、且确实是本脚本启动的同名任务、且尚未写出退出码时，才算正在运行（防止 pid 被系统复用）
  local job=$1 pid
  [ -f "$LOGS/$job.exit" ] && return 1
  [ -f "$LOGS/$job.pid" ] || return 1
  pid=$(cat "$LOGS/$job.pid")
  kill -0 "$pid" 2>/dev/null || return 1
  tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "server_run.sh" || return 1
  tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -qw "$job" || return 1
  return 0
}

pyexec() {
  local py
  py=$(cat "$PYFILE" 2>/dev/null)
  [ -z "$py" ] && { echo "ERROR 请先运行 probe"; return 1; }
  local hf
  hf=$(cat "$HFFILE" 2>/dev/null)
  if [ -n "$hf" ] && [ "$hf" != "https://huggingface.co" ]; then export HF_ENDPOINT="$hf"; fi
  export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
  "$py" "$@"
}

cmd=${1:-}
case "$cmd" in
  probe)
    echo "HOST $(hostname)  USER $(whoami)  PWD $(pwd)"
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
    df -h . | tail -n 1
    probe_python || exit 1
    probe_hf
    ;;
  discover)
    pyexec $EV/server_score.py discover "${@:2}"
    ;;
  check_calib)
    pyexec $EV/server_score.py check-calib
    ;;
  inspect)
    pyexec $EV/server_inspect.py "${@:2}"
    ;;
  start)
    job=${2:?用法: start calib|sealed}
    case "$job" in calib|sealed) ;; *) echo "ERROR 未知任务 $job"; exit 2;; esac
    if job_running "$job"; then
      echo "ALREADY_RUNNING $(cat "$LOGS/$job.pid")"; exit 3
    fi
    if [ "$job" = calib ] && [ -d "$EV/scores/calib" ]; then
      mv "$EV/scores/calib" "$EV/scores/calib_prev_$(date +%Y%m%d_%H%M%S)"   # 校准集每次从头记录
    fi
    rm -f "$LOGS/$job.exit"
    extra=("${@:3}")
    nohup bash -c 'cd "$1" && bash eval_v1/server_run.sh _run "$2" "${@:3}"; echo $? > "eval_v1/logs/$2.exit"' _ \
      "$(pwd)" "$job" "${extra[@]}" > "$LOGS/$job.out" 2>&1 < /dev/null &
    echo $! > "$LOGS/$job.pid"
    echo "STARTED $job pid $(cat "$LOGS/$job.pid")"
    ;;
  _run)
    job=$2
    pyexec $EV/server_score.py score --target "$job" "${@:3}"
    ;;
  status)
    job=${2:?用法: status calib|sealed}
    if [ -f "$LOGS/$job.exit" ]; then echo "DONE exit=$(cat "$LOGS/$job.exit")"
    elif job_running "$job"; then echo "RUNNING pid $(cat "$LOGS/$job.pid")"
    else echo "NOT_RUNNING"; fi
    [ -f "$LOGS/$job.out" ] && grep -v "Token indices sequence length\|PreTrainedTokenizerFast\|warnings.warn\|UserWarning" "$LOGS/$job.out" | tail -n "${3:-12}"
    ;;
  summary)
    job=${2:?用法: summary calib|sealed}
    pyexec - "$EV/scores/$job/RUN_MANIFEST.json" "$EV/models.json" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1], encoding='utf-8'))
cfg = json.load(open(sys.argv[2], encoding='utf-8'))
enabled = [x['name'] for x in cfg['models'] if x.get('enabled', True)]
bad = []
for name in enabled:
    r = m['runs'].get(name)
    status = r.get('status') if r else 'missing'
    info = (r or {}).get('info', {})
    print('{:<26} {:<8} trunc={} dtype={} {}'.format(name, status, info.get('n_truncated'), info.get('dtype_used'),
                                                     (r or {}).get('error', '')))
    if status != 'ok':
        bad.append(name)
print('SUMMARY ok={} bad={}'.format(len(enabled) - len(bad), ','.join(bad) or '-'))
sys.exit(1 if bad else 0)
PYEOF
    ;;
  *)
    sed -n '2,10p' "$0"; exit 2
    ;;
esac
