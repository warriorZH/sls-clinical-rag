#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S4：用 stage4 的同一份训练数据，以 LoRA 微调 Qwen3-Reranker（0.6B 或 4B），标签为“支持”或“相关性”。

    python s4_train.py --model q06 --objective support --seed 42 [--smoke]

流程与 eval_v2 的 E2（train_bge.py）对齐：同一 prepared.pt（平衡训练集与 dev/test 切分）、有效批量 16（4×4 累积）、
最多 3 轮、warmup 10% 后线性衰减、weight decay 0.01、梯度裁剪 1.0、梯度检查点；每轮在 dev 上用 stage4 的
checkpoint_key 选检查点，连续 2 轮不提升即停止；最后用 dev 定阈值，在 test 上报告支持类的 P/R/F1。
与 E2 的不同（预先写定）：骨干为解码器式重排模型，输入格式与零样本打分完全相同，以“最后位置上 yes 与 no
两个词元”的二分类交叉熵训练；用 LoRA（r=32、alpha=64、dropout 0.05，注意力与 MLP 的全部线性层），
学习率 1e-4，bf16；训练时输入最长 1024 个词元（打分时 2048）。
"""
import argparse
import json
import math
import random
import shutil
import sys
import time
import traceback

from common_sup import log, write_json
from s4_common import MODELS, QWEN3_PREFIX, QWEN3_SUFFIX, S4, STAGE4, TRAIN_DIR, fmt, stage4_modules


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['q06', 'q4b'], required=True)
    ap.add_argument('--objective', choices=['support', 'relevance'], required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    name = '{}_{}_s{}'.format(a.model, a.objective, a.seed) + ('_smoke' if a.smoke else '')
    out = TRAIN_DIR / name
    if (out / 'DONE.json').exists():
        log('{} 已完成'.format(name), 's4')
        return
    if out.exists():
        shutil.move(str(out), str(out.with_name(name + '_incomplete_' + time.strftime('%Y%m%d_%H%M%S'))))
    out.mkdir(parents=True)
    try:
        train(a, name, out)
    except Exception:  # noqa: BLE001
        write_json(out / 'FAILED.json', {'traceback': traceback.format_exc()})
        log('{} 失败：{}'.format(name, traceback.format_exc()), 's4')
        sys.exit(1)


def train(a, name, out):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

    T = S4['train']
    classify, choose_threshold, evaluate, checkpoint_key = stage4_modules()
    seed = a.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    peak = lambda: torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0
    data = torch.load(str(STAGE4 / 'prepared.pt'), map_location='cpu', weights_only=False)
    label = 'y_relevance' if a.objective == 'relevance' else 'y_support'
    base = MODELS / S4['models'][a.model]['dir']
    tok = AutoTokenizer.from_pretrained(str(base), padding_side='left')
    yes_id, no_id = tok.convert_tokens_to_ids('yes'), tok.convert_tokens_to_ids('no')
    pre = tok.encode(QWEN3_PREFIX, add_special_tokens=False)
    suf = tok.encode(QWEN3_SUFFIX, add_special_tokens=False)
    max_len = int(T['max_length'])
    room = max_len - len(pre) - len(suf)

    def enc(rows):
        body = tok([fmt(p['question'], p['text']) for p in rows], padding=False, truncation='longest_first',
                   return_attention_mask=False, max_length=room, add_special_tokens=False)['input_ids']
        return [pre + list(x) + suf for x in body]

    train_rows = data['train']
    if a.smoke:
        train_rows = random.Random(seed).sample(train_rows, min(240, len(train_rows)))
    dev_rows = data['dev'][:96] if a.smoke else data['dev']
    log('{}：训练 {} 对，dev {} 对，test {} 对'.format(name, len(train_rows), len(dev_rows), len(data['test'])), 's4')
    tr_ids, dev_ids = enc(train_rows), enc(dev_rows)
    test_ids = None if a.smoke else enc(data['test'])
    n_trunc = sum(1 for x in tr_ids if len(x) >= max_len)
    micro, accum = int(T['micro']), int(T['accum'])
    epochs = 1 if a.smoke else int(T['epochs'])
    steps_per_epoch = math.ceil(math.ceil(len(train_rows) / micro) / accum)
    total_steps = steps_per_epoch * epochs
    L = S4['lora']
    config = {'model': a.model, 'base': str(base), 'objective': a.objective, 'label': label, 'seed': seed,
              'train_pairs': len(train_rows), 'expected_train_pairs': T['expected_train_pairs'], 'dev_pairs': len(dev_rows),
              'test_pairs': len(data['test']), 'train_truncated_pairs': n_trunc, 'max_length': max_len,
              'micro': micro, 'accum': accum, 'effective_batch': micro * accum, 'lr': T['lr'], 'weight_decay': T['weight_decay'],
              'warmup_ratio': T['warmup_ratio'], 'epochs': epochs, 'total_steps': total_steps, 'lora': L,
              'loss': 'CE over (no, yes) logits at last position', 'dtype': 'bf16', 'yes_id': yes_id, 'no_id': no_id,
              'instruction': S4['instruction'], 'smoke': a.smoke, 'torch': torch.__version__,
              'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu', 'started': time.strftime('%Y-%m-%d %H:%M:%S')}
    write_json(out / 'config.json', config)
    if not a.smoke and len(train_rows) != int(T['expected_train_pairs']):
        log('注意：训练对数 {} 与预期 {} 不同（已记入 config.json）'.format(len(train_rows), T['expected_train_pairs']), 's4')

    model = AutoModelForCausalLM.from_pretrained(str(base), torch_dtype=torch.bfloat16).to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=int(L['r']), lora_alpha=int(L['alpha']), lora_dropout=float(L['dropout']),
                                             target_modules=list(L['target_modules']), task_type='CAUSAL_LM'))
    params = [p for p in model.parameters() if p.requires_grad]
    config['trainable_params'] = sum(p.numel() for p in params)
    write_json(out / 'config.json', config)
    opt = torch.optim.AdamW(params, lr=float(T['lr']), weight_decay=float(T['weight_decay']))
    sch = get_linear_schedule_with_warmup(opt, int(float(T['warmup_ratio']) * total_steps), total_steps)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def batch(ix, ids):
        n = max(len(ids[i]) for i in ix)
        x = torch.full((len(ix), n), pad, dtype=torch.long)
        m = torch.zeros((len(ix), n), dtype=torch.long)
        for j, i in enumerate(ix):                    # 左侧填充：最后一个位置总是真实的最后一个词元
            x[j, n - len(ids[i]):] = torch.tensor(ids[i])
            m[j, n - len(ids[i]):] = 1
        return x.to(device), m.to(device)

    def two_logits(ix, ids):
        x, m = batch(ix, ids)
        last = model(input_ids=x, attention_mask=m, logits_to_keep=1).logits[:, -1, :].float()
        return torch.stack([last[:, no_id], last[:, yes_id]], dim=1)

    def predict(ids, bs=16):
        model.eval()
        res = np.empty(len(ids))
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
        with torch.no_grad():
            for k in range(0, len(order), bs):
                ix = order[k:k + bs]
                v = torch.softmax(two_logits(ix, ids), dim=1)[:, 1]
                if not torch.isfinite(v).all():
                    raise RuntimeError('预测出现非有限值')
                res[ix] = v.cpu().numpy()
        model.train()
        return res

    rng = random.Random(seed)
    ys = [int(p[label]) for p in train_rows]
    step, best, patience, history = 0, (-1,), 0, []
    t0 = time.time()
    model.train()
    for epoch in range(epochs):
        ix = list(range(len(train_rows)))
        rng.shuffle(ix)
        chunks = [sorted(ix[k:k + 256], key=lambda i: len(tr_ids[i])) for k in range(0, len(ix), 256)]
        batches = [c[k:k + micro] for c in chunks for k in range(0, len(c), micro)]
        rng.shuffle(batches)
        for k in range(0, len(batches), accum):
            group = batches[k:k + accum]
            n = sum(map(len, group))
            opt.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for b in group:
                logit2 = two_logits(b, tr_ids)
                y = torch.tensor([ys[i] for i in b], device=device)
                loss = F.cross_entropy(logit2, y, reduction='sum') / n
                if not torch.isfinite(loss):
                    raise RuntimeError('损失出现非有限值')
                loss.backward()
                loss_sum += float(loss.detach())
            grad = torch.nn.utils.clip_grad_norm_(params, float(T['grad_clip']))
            if not torch.isfinite(grad):
                raise RuntimeError('梯度出现非有限值')
            opt.step()
            sch.step()
            step += 1
            if step == 1 or step % 100 == 0:
                el = time.time() - t0
                log('{} 第 {} 轮 step {}/{} loss {:.4f} {:.2f} 秒/步 显存峰值 {:.1f} GB'.format(
                    name, epoch + 1, step, total_steps, loss_sum, el / step, peak()), 's4')
            if a.smoke and step >= 15:
                break
        if a.smoke:
            sc = predict(dev_ids)
            el = time.time() - t0
            write_json(out / 'DONE.json', {'smoke_only': True, 'steps': step, 'sec_per_step': round(el / step, 2),
                                           'estimated_hours_full_run': round(el / step * math.ceil(math.ceil(
                                               int(T['expected_train_pairs']) / micro) / accum) * int(T['epochs']) / 3600, 2),
                                           'finite_dev_predictions': bool(np.isfinite(sc).all()),
                                           'peak_memory_gib': round(peak(), 2)})
            log('{} 冒烟测试通过'.format(name), 's4')
            return
        scores = predict(dev_ids)
        metrics = classify(data['dev'], scores, .5)
        key, chosen, grid = checkpoint_key(data['dev'], scores, data['meta'])
        key = tuple(float(x) for x in key)
        history.append({'epoch': epoch + 1, 'step': step, 'dev_support': metrics, 'key': list(key),
                        'selected_policy': chosen, 'seconds': round(time.time() - t0, 1)})
        write_json(out / 'history.json', history)
        log('{} 第 {} 轮结束：dev 选择键 {}'.format(name, epoch + 1, key), 's4')
        if key > best:
            best, patience = key, 0
            model.save_pretrained(str(out / 'adapter'))
            np.save(str(out / 'best_dev_scores.npy'), scores)
            write_json(out / 'best.json', {'epoch': epoch + 1, 'step': step, 'key': list(key)})
        else:
            patience += 1
        if patience >= int(T['patience']):
            break
    del opt, sch, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # 载入最佳检查点，在 test 上报告（与 E2 相同）
    from peft import PeftModel
    base_model = AutoModelForCausalLM.from_pretrained(str(base), torch_dtype=torch.bfloat16).to(device)
    model = PeftModel.from_pretrained(base_model, str(out / 'adapter')).to(device).eval()
    best_info = json.loads((out / 'best.json').read_text(encoding='utf-8'))
    dev_scores = np.load(str(out / 'best_dev_scores.npy'))
    threshold = float(choose_threshold(data['dev'], dev_scores))
    evaluate(data['dev'], dev_scores, data['meta'], threshold, out / 'dev')
    test_scores = predict(test_ids)
    evaluate(data['test'], test_scores, data['meta'], threshold, out / 'test')
    np.save(str(out / 'test_scores.npy'), test_scores)
    own = [{**p, 'y_support': p[label]} for p in data['test']]
    write_json(out / 'test_metrics.json', {'threshold_dev': threshold,
                                           'test_support_label': classify(data['test'], test_scores, threshold),
                                           'test_own_label': classify(own, test_scores, threshold)})
    write_json(out / 'DONE.json', {'best_epoch': best_info['epoch'], 'best_key': best_info['key'], 'threshold': threshold,
                                   'total_steps': step, 'seconds': round(time.time() - t0, 1),
                                   'peak_memory_gib': round(peak(), 2)})
    log('{} 完成'.format(name), 's4')


if __name__ == '__main__':
    main()
