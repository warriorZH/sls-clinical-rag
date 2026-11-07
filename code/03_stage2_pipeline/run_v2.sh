#!/usr/bin/env bash
# eval_v2 服务器端入口。由本机的“一键启动实验.bat”等通过 ssh 调用，也可以在服务器上手动运行：
#   bash eval_v2/run_v2.sh start     在后台启动（或续跑）流水线
#   bash eval_v2/run_v2.sh status    打印进度表和 GPU 占用
#   bash eval_v2/run_v2.sh stop      请求停止（正在运行的作业会被结束，下次启动时续跑）
#   bash eval_v2/run_v2.sh pack      生成汇总并打包结果（不含模型权重）
#   bash eval_v2/run_v2.sh plan      列出全部作业
#   bash eval_v2/run_v2.sh prepare v2.3   （启动器调用）runs/ 不是该方案版本的就改名保存，从干净的状态开始
#   bash eval_v2/run_v2.sh sysinfo   服务器资源与 ssh 服务情况（启动器只记进本机的记录文件，不在窗口里显示）
cd "$(dirname "$0")" || exit 2
mkdir -p runs/logs results
PY=""
[ -f ../eval_v1/.python_path ] && PY=$(cat ../eval_v1/.python_path)
if [ -z "$PY" ] || [ ! -x "$PY" ]; then PY=$(command -v python3); fi
export PYTHONPATH="$(pwd):$(pwd)/pylib${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONIOENCODING=utf-8

running() {
  [ -f runs/pipeline.pid ] || return 1
  local pid
  pid=$(cat runs/pipeline.pid 2>/dev/null)
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null || return 1
  tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "pipeline.py" || return 1
  return 0
}

gib() { awk -v b="$1" 'BEGIN { if (b == "max" || (b ~ /^[0-9]+$/ && b + 0 >= 1e18)) printf "不限"; else if (b ~ /^[0-9]+$/) printf "%.1f GiB", b / 1073741824; else printf "%s", b }'; }

sysinfo() {
  # 服务器资源：内存（含容器上限与内存不足的次数）、负载、本用户的进程数、磁盘、占内存最多的进程
  echo "服务器资源（$(date '+%F %T')）："
  echo "  负载：$(cut -d' ' -f1-3 /proc/loadavg)    CPU 核数：$(nproc 2>/dev/null)"
  awk '/^(MemTotal|MemAvailable|SwapTotal|SwapFree):/ {printf "  %s %.1f GiB\n", $1, $2 / 1048576}' /proc/meminfo
  if [ -f /sys/fs/cgroup/memory.max ]; then
    echo "  容器内存上限：$(gib "$(cat /sys/fs/cgroup/memory.max)")，已用：$(gib "$(cat /sys/fs/cgroup/memory.current 2>/dev/null)")"
    grep -E '^(oom|oom_kill) ' /sys/fs/cgroup/memory.events 2>/dev/null | sed 's/^/  内存不足事件 /'
  elif [ -f /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
    echo "  容器内存上限：$(gib "$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)")，已用：$(gib "$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null)")，最高：$(gib "$(cat /sys/fs/cgroup/memory/memory.max_usage_in_bytes 2>/dev/null)")，达到上限次数：$(cat /sys/fs/cgroup/memory/memory.failcnt 2>/dev/null)"
    grep -E '^oom_kill ' /sys/fs/cgroup/memory/memory.oom_control 2>/dev/null | sed 's/^/  内存不足事件 /'
  fi
  awk '/^oom_kill / {print "  内存不足时内核强制结束进程的次数（开机以来）：" $2}' /proc/vmstat 2>/dev/null
  for r in memory io; do
    [ -r "/proc/pressure/$r" ] && awk -v r="$r" '/^some/ {gsub(/avg[0-9]+=/, ""); printf "  %s 压力（最近 10/60/300 秒里任务在等%s的时间比例）：%s%% %s%% %s%%\n", r, (r == "memory" ? "内存" : "磁盘"), $2, $3, $4}' "/proc/pressure/$r"
  done
  echo "  本用户进程与线程：$(ps -L -u "$(id -u)" --no-headers 2>/dev/null | wc -l)（上限 $(ulimit -u)）"
  echo "  磁盘：$(df -h . 2>/dev/null | awk 'NR==2 {print $4 " 可用（已用 " $5 "）"}')"
  echo "  占内存最多的进程："
  ps -eo rss=,etime=,args= --sort=-rss 2>/dev/null | head -5 | awk '{a=""; for (i=3; i<=NF && i<=7; i++) a=a" "$i; printf "    %.1f GiB  已运行 %s %s\n", $1 / 1048576, $2, a}'
  dmesg 2>/dev/null | grep -i -E 'killed process|out of memory' | tail -3 | sed 's/^/  内核：/'
  sshinfo
}

sshinfo() {
  # ssh 服务的情况：用来判断本机偶尔连不上（服务器接受连接后马上断开）的原因
  echo "  ssh：$(ssh -V 2>&1 | head -1)；已建立的 ssh 连接 $(ss -tn state established '( sport = :22 or sport = :<PORT> )' 2>/dev/null | tail -n +2 | wc -l) 个"
  cfg=$(grep -h -i -E '^[[:space:]]*(MaxStartups|PerSourcePenalties|PerSourceMaxStartups|MaxSessions|LoginGraceTime|UseDNS)[[:space:]]' /etc/ssh/sshd_config /etc/ssh/sshd_config.d/*.conf 2>/dev/null | tr -s ' \t' ' ' | paste -sd ';' -)
  [ -n "$cfg" ] && echo "  sshd 设置：$cfg"
  ip="${SSH_CLIENT%% *}"
  pat="MaxStartups|penalty|fatal|error: |Cannot allocate|Resource temporarily|Out of memory|${ip:-NO-CLIENT-IP}"
  skip='Accepted |session opened|session closed|Disconnected from user|Received disconnect from|pam_unix\(sshd:session\)'
  shown=''
  for f in /var/log/auth.log /var/log/secure; do
    if [ -r "$f" ]; then
      echo "  ssh 服务日志（$f）：其他来源的失败登录 $(grep -a -c -E 'sshd.*(Invalid user|Failed password|authentication failure)' "$f" 2>/dev/null) 次；与本机有关或出错的最近记录："
      grep -a -E "sshd(-session)?\[[0-9]+\]: .*($pat)" "$f" 2>/dev/null | grep -a -v -E "$skip" | tail -8 | cut -c1-220 | sed 's/^/    /'
      shown=1
      break
    fi
  done
  if [ -z "$shown" ] && command -v journalctl >/dev/null 2>&1; then
    all=$(journalctl -q --no-pager -t sshd -t sshd-session --since '-6h' 2>/dev/null)
    if [ -n "$all" ]; then
      echo "  ssh 服务日志（journal，最近 6 小时）：其他来源的失败登录 $(echo "$all" | grep -a -c -E 'Invalid user|Failed password|authentication failure') 次；与本机有关或出错的最近记录："
      echo "$all" | grep -a -E "$pat" | grep -a -v -E "$skip" | tail -8 | cut -c1-220 | sed 's/^/    /'
      shown=1
    fi
  fi
  [ -z "$shown" ] && echo "  （没有权限读取 ssh 服务日志）"
}

case "${1:-status}" in
  start)
    if running; then
      now=$(cat pipeline.py conditions.py common.py | sha256sum | cut -c1-16)
      if [ "$(cat runs/pipeline.code 2>/dev/null)" = "$now" ]; then
        echo "ALREADY_RUNNING $(cat runs/pipeline.pid)"
        exit 0
      fi
      # 调度程序的代码更新了：只换掉调度程序本身，正在运行的作业不受影响，由新的调度程序接管
      kill "$(cat runs/pipeline.pid)" 2>/dev/null
      for i in $(seq 1 30); do running || break; sleep 1; done
      if running; then echo "START_FAILED 旧的调度程序没有退出"; exit 1; fi
      echo "RESTARTED 调度程序已换成新版本，正在运行的作业继续运行"
    fi
    chmod 600 secrets/judge_keys.env 2>/dev/null
    nohup "$PY" pipeline.py run >> runs/logs/pipeline.out 2>&1 < /dev/null &
    echo $! > runs/pipeline.pid
    sleep 5
    if running; then
      echo "STARTED $(cat runs/pipeline.pid)"
    elif grep -q '"orchestrator": "finished"' runs/pipeline_state.json 2>/dev/null; then
      echo "ALL_DONE 全部作业已经完成，没有需要运行的作业。"
    else
      echo "START_FAILED 流水线没有启动成功，最近的输出："
      tail -n 20 runs/logs/pipeline.out
      exit 1
    fi
    ;;
  status)
    "$PY" pipeline.py status
    if command -v nvidia-smi >/dev/null 2>&1; then
      echo "GPU 占用："
      nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
    fi
    ;;
  sysinfo)
    sysinfo
    ;;
  stop)
    if running; then touch runs/STOP; echo "STOP_REQUESTED"; else echo "NOT_RUNNING"; fi
    ;;
  pack)
    timeout 600 "$PY" summarize.py --quick >/dev/null 2>&1
    "$PY" pipeline.py status > /dev/null 2>&1
    sysinfo > runs/sysinfo.txt 2>&1
    rm -f results_pack.tar.gz
    tar -czf results_pack.tar.gz --exclude='runs/e2/*/model' --exclude='*.npy' --exclude='runs/inspect/code' \
      --exclude='*.tmp' status.txt results runs/env.json runs/pipeline_state.json runs/done runs/e0 runs/scores \
      runs/selections runs/gen runs/judge runs/logs runs/models runs/freeze_nonoracle.json runs/e2 runs/inspect runs/external runs/sysinfo.txt \
      config REGISTRATION_v2.json FREEZE_MANIFEST_v2.json 2>/dev/null
    echo "PACKED $(du -h results_pack.tar.gz | cut -f1)"
    ;;
  prepare)
    # 由启动器在每次上传代码后调用：runs/ 不是当前方案版本留下的（例如按旧版本启动过、中途出错），
    # 就整体改名保存，从干净的状态开始；版本相同时什么也不做（断点续跑）。
    want="${2:-}"
    if [ -z "$want" ]; then echo "PREPARE_FAILED 缺少方案版本"; exit 2; fi
    have=$(cat runs/PLAN_VERSION 2>/dev/null)
    if [ "$have" != "$want" ]; then
      if running; then
        touch runs/STOP
        for i in $(seq 1 45); do running || break; sleep 2; done
      fi
      if running; then echo "PREPARE_FAILED 旧的流水线仍在运行"; exit 1; fi
      if [ -e runs/pipeline_state.json ] || [ -d runs/done ] || [ -d runs/models ]; then
        old="runs_before_${want}_$(date +%Y%m%d_%H%M%S)"
        mv runs "$old" && echo "MOVED runs -> $old"
      fi
      rm -rf results results_pack.tar.gz status.txt
      mkdir -p runs/logs results
      echo "$want" > runs/PLAN_VERSION
    fi
    echo "PREPARED $(cat runs/PLAN_VERSION)"
    ;;
  plan)
    "$PY" pipeline.py plan
    ;;
  *)
    sed -n '2,10p' "$0"
    exit 2
    ;;
esac
