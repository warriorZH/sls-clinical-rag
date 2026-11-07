#!/usr/bin/env bash
# 公用环境变量：工作目录（放模型、虚拟环境、缓存、结果）、模型目录、Python。
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -z "${SUP_WORK:-}" ]; then
  if [ -d /root/autodl-tmp ]; then SUP_WORK=/root/autodl-tmp/eval_v2_sup_work      # AutoDL 的数据盘
  elif [ -d /mnt/data ] && [ -w /mnt/data ]; then SUP_WORK=/mnt/data/eval_v2_sup_work
  else SUP_WORK="$HOME/eval_v2_sup_work"; fi
fi
export SUP_WORK
export SUP_MODELS="${SUP_MODELS:-$SUP_WORK/models}"
export VENV="$SUP_WORK/venv"
export PY="$VENV/bin/python"
export STATE="$SUP_WORK/state"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
# 各种缓存放到数据盘（云平台的系统盘通常只有 20–50 GB，放不下 pip、编译与内核缓存）
C="$SUP_WORK/cache"
export PIP_CACHE_DIR="$C/pip" VLLM_CACHE_ROOT="$C/vllm" TRITON_CACHE_DIR="$C/triton" \
       TORCHINDUCTOR_CACHE_DIR="$C/inductor" CUDA_CACHE_PATH="$C/nv" CUDA_CACHE_MAXSIZE=4294967296 \
       MODELSCOPE_CACHE="$C/modelscope" HF_HOME="$C/hf" XDG_CACHE_HOME="$C/xdg" TMPDIR="$SUP_WORK/tmp"
mkdir -p "$SUP_WORK/logs" "$STATE" "$SUP_MODELS" "$C" "$SUP_WORK/tmp"
# 预检得出的 NCCL 设置（例如 NCCL_P2P_DISABLE=1）
if [ -s "$STATE/nccl.env" ]; then set -a; . "$STATE/nccl.env"; set +a; fi
