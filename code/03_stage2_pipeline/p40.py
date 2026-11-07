#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tesla P40（Pascal，算力 6.1）上的推理设置。

P40 没有 FlashAttention，也不适合 fp16/bf16 计算，因此全部用 fp32，注意力走 PyTorch 的 SDPA。
transformers 在这种 GPU 上有两处默认做法很费显存或带宽，这里加以调整（只改计算路径，不改模型和解码设置）：

1. 预填充（一次输入多个词元）：单条输入时 transformers 会让 SDPA 走分组查询注意力（GQA，enable_gqa=True），
   PyTorch 只有 math 内核支持它，显存随提示长度平方增长。改为先展开 K/V，再用省显存内核。
2. 逐词解码（每步一个词元）：transformers 每一步都把整个 K/V 缓存展开到与查询头数相同（4–16 倍），
   批量较大时这一步的显存读写比模型权重本身还多。这里把同组的查询头排成“多个查询位置”，
   直接与未展开的 K/V 计算注意力，数学上完全等价。启用前在实际模型上与原实现逐项比对，不一致就不启用。
"""
import subprocess

_STATE = {'base': None, 'grouped': False}


def _base_sdpa():
    if _STATE['base'] is None:
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        _STATE['base'] = sdpa_attention_forward
    return _STATE['base']


def pascal_like(torch):
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8


def sdpa_without_gqa(torch, log, force=False):
    """预填充时先展开 K/V，使 SDPA 可以使用省显存内核（只在没有 FlashAttention 的 GPU 上启用）。"""
    if not (force or pascal_like(torch)):
        return False
    try:
        from transformers.integrations import sdpa_attention as S
    except Exception:  # noqa: BLE001
        return False
    if not hasattr(S, 'use_gqa_in_sdpa'):
        return False
    S.use_gqa_in_sdpa = lambda *a, **k: False
    log('  已启用 SDPA 设置：预填充时先展开 K/V（P40 无 FlashAttention，避免退回 math 内核）')
    return True


def grouped_decode_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None,
                             is_causal=None, **kwargs):
    """解码步（查询长度为 1）且 K/V 头数少于查询头数时，不展开 K/V；其余情况交给 transformers 原来的 SDPA 实现。"""
    import torch.nn.functional as F
    base = _base_sdpa()
    b, h, q, d = query.shape
    kvh = key.shape[1]
    usable = (q == 1 and kvh < h and h % kvh == 0 and value.shape[1] == kvh and key.shape[-1] == d
              and value.shape[-1] == d and kwargs.get('cache') is None and not kwargs.get('output_attentions')
              and (attention_mask is None or attention_mask.ndim == 4))
    if not usable:
        return base(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling,
                    is_causal=is_causal, **kwargs)
    g = h // kvh
    qg = query.reshape(b, kvh, g, d)              # 查询头 i 对应 K/V 头 i // g，与 repeat_kv 的约定一致
    mask = attention_mask
    if mask is not None:
        mask = mask[:, :, -1:, :key.shape[-2]]    # (b, 1, 1, L)，对同组的各查询头广播
    out = F.scaled_dot_product_attention(qg, key, value, attn_mask=mask, dropout_p=0.0, scale=scaling)
    return out.reshape(b, h, 1, d).transpose(1, 2).contiguous(), None


def set_grouped_decode(on):
    from transformers.modeling_utils import AttentionInterface
    AttentionInterface.register('sdpa', grouped_decode_attention if on else _base_sdpa())
    _STATE['grouped'] = bool(on)


def _probe_ids(model, pad_id, device):
    import torch
    vocab = int(model.config.get_text_config().vocab_size) if hasattr(model.config, 'get_text_config') \
        else int(model.config.vocab_size)
    lo = min(1000, max(10, vocab // 4))
    a = [lo + (i * 37) % 997 for i in range(23)]
    b = [lo + (i * 53) % 991 for i in range(15)]
    n = max(len(a), len(b))
    ids = torch.full((2, n), int(pad_id), dtype=torch.long)
    mask = torch.zeros((2, n), dtype=torch.long)
    for r, seq in enumerate((a, b)):
        ids[r, n - len(seq):] = torch.tensor(seq)
        mask[r, n - len(seq):] = 1
    return ids.to(device), mask.to(device)


def _greedy_logits(model, ids, mask, steps, pad_id):
    import torch
    with torch.inference_mode():
        out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=steps, min_new_tokens=steps,
                             do_sample=False, num_beams=1, pad_token_id=int(pad_id), output_logits=True,
                             return_dict_in_generate=True, temperature=None, top_p=None, top_k=None)
    return out.sequences[:, ids.shape[1]:].tolist(), torch.stack([x.float() for x in out.logits]).cpu()


def enable_grouped_decode(model, pad_id, device, log, steps=6, tol=1e-3):
    """在实际模型上比对：带左侧填充的两条输入，各解码 steps 步，新旧两种实现的贪心结果必须相同、
    logits 的最大相对差 < tol，才启用；否则保留 transformers 原实现。返回 (是否启用, 比对记录)。"""
    try:
        ids, mask = _probe_ids(model, pad_id, device)
        set_grouped_decode(False)
        tok0, lg0 = _greedy_logits(model, ids, mask, steps, pad_id)
        set_grouped_decode(True)
        tok1, lg1 = _greedy_logits(model, ids, mask, steps, pad_id)
        scale = float(lg0.abs().max().clamp(min=1.0))
        rel = float((lg0 - lg1).abs().max()) / scale
        ok = tok0 == tok1 and rel < tol
        rec = {'same_tokens': tok0 == tok1, 'max_rel_diff': rel, 'enabled': ok}
    except Exception as e:  # noqa: BLE001
        ok, rec = False, {'error': '{}: {}'.format(type(e).__name__, str(e)[:200]), 'enabled': False}
    set_grouped_decode(ok)
    log('  解码注意力（不展开 K/V）：{}，比对 {}'.format('已启用' if ok else '未启用', rec))
    return ok, rec


def smallest_gpu_bytes(torch):
    """本机所有 GPU 中最小的显存容量（nvidia-smi 不受 CUDA_VISIBLE_DEVICES 影响，能看到全部 GPU）。"""
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=30).stdout
        vals = [int(x) for x in out.split() if x.strip().isdigit()]
        if vals:
            return min(vals) * 2 ** 20
    except Exception:  # noqa: BLE001
        pass
    return min(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count()))


def equalize_memory(torch, log, margin_gib=1.5):
    """两张卡显存不同（24 GB / 22.5 GB）。把本进程可用显存统一限制为“最小那张卡的容量 − 1.5 GiB”，
    这样“显存不足 → 拆批”在两张卡上的触发条件完全相同，同一任务无论在哪张卡上跑，分批与输出都一致。"""
    if not torch.cuda.is_available():
        return None
    cap = smallest_gpu_bytes(torch) - int(margin_gib * 2 ** 30)
    for i in range(torch.cuda.device_count()):
        total = torch.cuda.get_device_properties(i).total_memory
        torch.cuda.set_per_process_memory_fraction(max(0.1, min(1.0, cap / total)), i)
    log('  每张卡可用显存统一限制为 {:.1f} GiB'.format(cap / 2 ** 30))
    return round(cap / 2 ** 30, 2)
