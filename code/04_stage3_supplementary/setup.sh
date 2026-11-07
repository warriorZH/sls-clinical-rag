#!/usr/bin/env bash
# 环境准备：检查显卡与驱动，建虚拟环境，按 requirements_cloud.txt 安装固定版本，记录环境信息。
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/tools/env.sh"
cd "$HERE"
echo "工作目录：$SUP_WORK"

command -v nvidia-smi >/dev/null || { echo "错误：找不到 nvidia-smi，这台机器没有可用的 NVIDIA 显卡。"; exit 1; }
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
echo "显卡驱动主版本：$DRV"
if [ "$DRV" -lt 570 ]; then
  echo "错误：显卡驱动 $DRV 太旧。本环境用 CUDA 12.8 版 PyTorch，需要驱动 ≥ 570（平台上标注“最高支持 CUDA ≥ 12.8”的机器）。请换一台机器。"
  exit 1
fi
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

AVAIL=$(df -BG --output=avail "$SUP_WORK" | tail -1 | tr -dc '0-9')
echo "工作目录所在磁盘剩余：${AVAIL} GB"
if [ "$AVAIL" -lt 220 ]; then
  echo "错误：磁盘剩余不足 220 GB（模型约 113 GB、环境约 15 GB、编译缓存与微调结果若干）。请把数据盘扩到 300 GB 后再运行。"
  exit 1
fi

# Triton / torch.compile 运行时要用 C 编译器；缺少时尝试安装
if ! command -v gcc >/dev/null 2>&1 && ! command -v cc >/dev/null 2>&1; then
  echo "没有 C 编译器，尝试安装 gcc……"
  (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq gcc) >/dev/null 2>&1 || true
  command -v gcc >/dev/null 2>&1 || { echo "错误：缺少 gcc 且无法自动安装。请换用平台的 PyTorch/CUDA 开发镜像。"; exit 1; }
fi

PYBIN=""
for c in python3.12 python3.11 python3.10 python3.13 python3; do
  if command -v $c >/dev/null 2>&1; then
    v=$($c -c 'import sys; print("%d%02d" % sys.version_info[:2])')
    if [ "$v" -ge 310 ] && [ "$v" -le 313 ]; then PYBIN=$c; break; fi
  fi
done
[ -n "$PYBIN" ] || { echo "错误：需要 Python 3.10–3.13。"; exit 1; }
echo "使用 $($PYBIN --version)"

# Triton 编译内核时还要 Python 头文件（Python.h）；系统 Python 缺少时尝试安装 python3.x-dev，不行就改用 conda
has_h() { $1 -c 'import sysconfig,os,sys; sys.exit(0 if os.path.exists(os.path.join(sysconfig.get_paths()["include"],"Python.h")) else 1)'; }
if ! has_h $PYBIN; then
  PV=$($PYBIN -c 'import sys; print("%d.%d" % sys.version_info[:2])')
  (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python${PV}-dev) >/dev/null 2>&1 || true
fi
if [ ! -x "$PY" ]; then
  if has_h $PYBIN && $PYBIN -m venv "$VENV" 2>/dev/null; then :
  elif command -v conda >/dev/null; then rm -rf "$VENV"; conda create -y -p "$VENV" python=3.12
  else echo "错误：无法建立带 Python 头文件的环境（缺少 python3-dev/python3-venv，也没有 conda）。请换用平台的 PyTorch/CUDA 镜像。"; exit 1; fi
fi
has_h "$PY" || { echo "错误：Python 环境缺少头文件 Python.h（Triton 需要）。"; exit 1; }

PIP_ARGS=()
if [ -n "${PIP_INDEX_URL:-}" ]; then
  echo "使用指定的 pip 源：$PIP_INDEX_URL"
elif "$PY" -m pip config list 2>/dev/null | grep -q index-url || [ -f /etc/pip.conf ] || [ -f "$HOME/.pip/pip.conf" ] || [ -f "$HOME/.config/pip/pip.conf" ]; then
  echo "使用平台已配置的 pip 源"
elif curl -s -m 8 -o /dev/null https://pypi.org/simple/pip/; then
  echo "使用 PyPI 官方源"
else
  PIP_ARGS=(-i https://pypi.tuna.tsinghua.edu.cn/simple)
  echo "PyPI 不可达，改用清华镜像"
fi
"$PY" -m pip install -q -U pip "${PIP_ARGS[@]}"
"$PY" -m pip install -r "$HERE/requirements_cloud.txt" "${PIP_ARGS[@]}"

"$PY" - <<'PYEOF'
import json, platform, subprocess, sys, os
import torch, vllm, transformers, openai
info = {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
        'cuda_available': torch.cuda.is_available(), 'gpu_count': torch.cuda.device_count(),
        'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        'vllm': vllm.__version__, 'transformers': transformers.__version__, 'openai': openai.__version__,
        'nvidia_smi': subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout}
assert info['cuda_available'] and info['gpu_count'] >= 2, 'PyTorch 看不到至少 2 张显卡'
info['capability'] = [list(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]
info['arch_list'] = torch.cuda.get_arch_list()
out = os.path.join(os.environ['SUP_WORK'], 'logs', 'env.json')
json.dump(info, open(out, 'w'), ensure_ascii=False, indent=1)
print('环境检查通过：torch {} (CUDA {})，vllm {}，{} 张显卡'.format(info['torch'], info['torch_cuda'], info['vllm'], info['gpu_count']))
PYEOF
"$PY" -m pip freeze > "$SUP_WORK/logs/pip_freeze.txt"
echo "环境准备完成"
