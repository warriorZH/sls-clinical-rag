#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E2：在 stage4 的同一份训练数据上微调 bge-reranker-v2-m3，分别用“支持”标签和“相关性”标签。

    python train_bge.py --objective support   --seed 42 --job e2_support_s42
    python train_bge.py --objective relevance --seed 42 --job e2_relevance_s42
    python train_bge.py --smoke --job e2_smoke

除骨干外，流程与 stage4 的 train_stage4.py 相同：同一 prepared.pt（平衡训练集与 dev/test 切分）、fp32、
学习率 1e-5、有效批量 16（4×4 累积）、3 轮、warmup 10%、weight decay 0.01、梯度裁剪 1.0、梯度检查点、
每轮在 dev 上用 stage4 的 checkpoint_key 选检查点、连续 2 轮不提升即停止。
bge 保留原有的单 logit 输出，用二元交叉熵训练（与两类 softmax 等价）；预测分数为 sigmoid 概率。
"""
import argparse
import json
import math
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

from common import RUNS, Log, dumps, environment, from_pretrained, load_config, mark_done, now, resolve, write_json
import conditions as C


def stage4_search_dirs(stage4_dir):
    """stage4 训练时的导入环境：脚本所在目录（policy_stage4.py）+ PYTHONPATH=项目根目录（evaluate_v2.py）。"""
    stage4_dir = Path(stage4_dir)
    dirs = [stage4_dir, stage4_dir.parents[1], stage4_dir.parent]
    return [d for d in dirs if d.exists()]


def stage4_modules(stage4_dir):
    for d in stage4_search_dirs(stage4_dir):
        if str(d) not in sys.path:
            sys.path.append(str(d))          # 追加在末尾，不遮蔽本目录和已安装的包
    from evaluate_v2 import choose_threshold, classify, evaluate  # noqa: E402  stage4 自带的评测函数
    from policy_stage4 import checkpoint_key  # noqa: E402
    return classify, choose_threshold, evaluate, checkpoint_key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--objective', choices=['support', 'relevance'], default='support')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--job', required=True)
    a = ap.parse_args()
    cfg = load_config()
    name = 'smoke' if a.smoke else '{}_s{}'.format(a.objective, a.seed)
    out = RUNS / 'e2' / name
    log = Log(RUNS / 'logs' / 'e2_{}.log'.format(name))
    if (out / 'DONE.json').exists():
        log('{} 已完成'.format(name))
        mark_done(a.job, json.loads((out / 'DONE.json').read_text(encoding='utf-8')))
        return
    if out.exists():
        dest = out.with_name('{}_incomplete_{}'.format(name, time.strftime('%Y%m%d_%H%M%S')))
        shutil.move(str(out), str(dest))
        log('上次未完成的运行已移到 {}，从头重新训练'.format(dest))
    out.mkdir(parents=True)
    try:
        train(a, cfg, out, log)
        mark_done(a.job, json.loads((out / 'DONE.json').read_text(encoding='utf-8')))
    except Exception as e:  # noqa: BLE001
        write_json(out / 'FAILED.json', {'error': str(e), 'traceback': traceback.format_exc(), 'time': now()})
        log('失败：{}'.format(traceback.format_exc()))
        sys.exit(1)


def train(a, cfg, out, log):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    e2 = cfg['e2']
    stage4 = resolve(e2['stage4_dir'])
    classify, choose_threshold, evaluate, checkpoint_key = stage4_modules(stage4)
    seed = a.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = torch.load(str(stage4 / 'prepared.pt'), map_location='cpu', weights_only=False)
    label = 'y_relevance' if a.objective == 'relevance' else 'y_support'
    bge = C.v1_models()['bge_reranker_v2_m3']['path']
    base = resolve(bge) if not Path(bge).is_absolute() else Path(bge)
    tok = AutoTokenizer.from_pretrained(str(base))
    max_len = int(e2.get('max_length', 512))

    def enc(rows):
        ids = tok([p['question'] for p in rows], [p['text'] for p in rows], truncation='only_second',
                  max_length=max_len)['input_ids']
        return [list(x) for x in ids]

    train_rows = data['train']
    if a.smoke:
        train_rows = sorted(train_rows, key=lambda p: len(p['text']), reverse=True)[:16] + \
            random.Random(seed).sample(train_rows, min(784, len(train_rows)))
    log('编码训练集 {} 对，dev {} 对，test {} 对 …'.format(len(train_rows), len(data['dev']), len(data['test'])))
    tr_ids = enc(train_rows)
    dev_ids = enc(data['dev'][:128] if a.smoke else data['dev'])
    test_ids = None if a.smoke else enc(data['test'])
    n_trunc = sum(1 for p, x in zip(train_rows, tr_ids) if len(x) >= max_len)
    micro, accum, epochs = int(e2.get('micro', 4)), int(e2.get('accum', 4)), 1 if a.smoke else int(e2.get('epochs', 3))
    steps_per_epoch = math.ceil(math.ceil(len(train_rows) / micro) / accum)
    total_steps = steps_per_epoch * epochs
    config = {'objective': a.objective, 'seed': seed, 'label': label, 'base_model': str(base), 'max_length': max_len,
              'micro': micro, 'accum': accum, 'effective_batch': micro * accum, 'lr': float(e2.get('lr', 1e-5)),
              'weight_decay': 0.01, 'warmup_ratio': 0.1, 'epochs': epochs, 'train_pairs': len(train_rows),
              'train_truncated_pairs': n_trunc, 'total_steps': total_steps, 'loss': 'BCEWithLogits(single logit)',
              'precision': 'fp32', 'selection': 'stage4 checkpoint_key on dev, patience 2', 'smoke': a.smoke,
              'env': environment(), 'started': now()}
    write_json(out / 'config.json', config)
    log('设置：' + dumps({k: v for k, v in config.items() if k != 'env'}))

    model = from_pretrained(AutoModelForSequenceClassification, base, num_labels=1).to(device)
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise RuntimeError('模型参数不是 fp32')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    opt = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.01, foreach=False)
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)
    pad = tok.pad_token_id

    def batch(ix, ids):
        n = max(len(ids[i]) for i in ix)
        x = torch.full((len(ix), n), pad, dtype=torch.long)
        m = torch.zeros((len(ix), n), dtype=torch.long)
        for j, i in enumerate(ix):
            x[j, :len(ids[i])] = torch.tensor(ids[i])
            m[j, :len(ids[i])] = 1
        return {'input_ids': x.to(device), 'attention_mask': m.to(device)}

    def predict(ids, bs=16):
        model.eval()
        res = np.empty(len(ids))
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
        with torch.no_grad():
            for k in range(0, len(order), bs):
                ix = order[k:k + bs]
                v = model(**batch(ix, ids)).logits[:, 0].float()
                if not torch.isfinite(v).all():
                    raise RuntimeError('预测出现非有限值')
                res[ix] = torch.sigmoid(v).cpu().numpy()
        return res

    rng = random.Random(seed)
    step, best, patience, history = 0, (-1,), 0, []
    t0 = time.time()
    ys = [float(p[label]) for p in train_rows]
    for epoch in range(epochs):
        model.train()
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
                logits = model(**batch(b, tr_ids)).logits[:, 0]
                y = torch.tensor([ys[i] for i in b], device=device)
                loss = F.binary_cross_entropy_with_logits(logits, y, reduction='sum') / n
                if not torch.isfinite(loss):
                    raise RuntimeError('损失出现非有限值')
                loss.backward()
                loss_sum += float(loss.detach())
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad):
                raise RuntimeError('梯度出现非有限值')
            opt.step()
            sch.step()
            step += 1
            if step == 1 or step % 50 == 0:
                el = time.time() - t0
                mem = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0
                log('  第 {} 轮 step {}/{}  loss {:.4f}  {:.2f} 秒/步  显存峰值 {:.1f} GB'.format(
                    epoch + 1, step, total_steps, loss_sum, el / step, mem))
            if a.smoke and step >= 50:
                break
        if a.smoke:
            scores = predict(dev_ids)
            el = time.time() - t0
            rep = {'smoke_only': True, 'steps': step, 'seconds': round(el, 1), 'sec_per_step': round(el / max(step, 1), 2),
                   'estimated_hours_per_run': round(el / max(step, 1) * steps_per_epoch * int(e2.get('epochs', 3)) / 3600, 1),
                   'peak_memory_gib': round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else None,
                   'finite_dev_predictions': bool(np.isfinite(scores).all())}
            write_json(out / 'DONE.json', rep)
            log('冒烟测试：{}'.format(dumps(rep)))
            return
        scores = predict(dev_ids)
        metrics = classify(data['dev'], scores, .5)
        key, chosen, grid = checkpoint_key(data['dev'], scores, data['meta'])
        key = tuple(float(x) for x in key)
        history.append({'epoch': epoch + 1, 'step': step, 'dev_support': metrics, 'key': list(key),
                        'selected_policy': chosen, 'seconds': round(time.time() - t0, 1)})
        write_json(out / 'history.json', history)
        log('  第 {} 轮结束：dev 选择键 {}'.format(epoch + 1, key))
        if key > best:
            best, patience = key, 0
            model.save_pretrained(str(out / 'model'))
            tok.save_pretrained(str(out / 'model'))
            np.save(str(out / 'best_dev_scores.npy'), scores)
            write_json(out / 'best.json', {'epoch': epoch + 1, 'step': step, 'key': list(key)})
        else:
            patience += 1
        if patience >= 2:
            break
    del opt, sch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    best_info = json.loads((out / 'best.json').read_text(encoding='utf-8'))
    model = from_pretrained(AutoModelForSequenceClassification, out / 'model', num_labels=1).to(device)
    dev_scores = np.load(str(out / 'best_dev_scores.npy'))
    threshold = choose_threshold(data['dev'], dev_scores)
    evaluate(data['dev'], dev_scores, data['meta'], threshold, out / 'dev')
    test_scores = predict(test_ids)
    evaluate(data['test'], test_scores, data['meta'], threshold, out / 'test')
    np.save(str(out / 'test_scores.npy'), test_scores)
    own_label = [{**p, 'y_support': p[label]} for p in data['test']]
    threshold = float(threshold)
    write_json(out / 'test_metrics.json', {'threshold_dev': threshold,
                                           'test_support_label': classify(data['test'], test_scores, threshold),
                                           'test_own_label': classify(own_label, test_scores, threshold)})
    done = {'best_epoch': best_info['epoch'], 'best_key': best_info['key'], 'threshold': threshold, 'total_steps': step,
            'seconds': round(time.time() - t0, 1),
            'peak_memory_gib': round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else None}
    write_json(out / 'DONE.json', done)
    log('完成：{}'.format(dumps(done)))


if __name__ == '__main__':
    main()
