#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S4：用原始权重或微调后的 Qwen3-Reranker 给封存集与外部集的候选池打分。

    python s4_score.py --name q06_zeroshot | q06_support_s42 | ... [--smoke]

打分方法与 eval_v1 的 server_score.py（score_qwen3）完全相同：同一提示格式与任务说明，分数为 log P(yes)，
最长 2048 个词元（截断文档末尾）；区别只在精度（这里统一用 bf16，零样本模型也在这里重新打分，保证可比）。
"""
import argparse
import sys
import time

from common_sup import append_jsonl, log, write_json
from s4_common import MODELS, QWEN3_PREFIX, QWEN3_SUFFIX, S4, SCORE_DIR, TRAIN_DIR, evidence_text, fmt, pools, \
    questions, score_models


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    spec = {n: (k, r) for n, k, r in score_models()}
    key, run = spec[a.name.replace('_smoke', '')] if not a.smoke else (a.name.split('_')[0], None)
    out_dir = SCORE_DIR / ('smoke' if a.smoke else a.name)
    if (out_dir / 'DONE.json').exists():
        log('{} 已打分'.format(a.name), 's4')
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    base = MODELS / S4['models'][key]['dir']
    tok = AutoTokenizer.from_pretrained(str(base), padding_side='left')
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForCausalLM.from_pretrained(str(base), torch_dtype=torch.bfloat16).to(dev).eval()
    if run:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(TRAIN_DIR / run / 'adapter')).to(dev).eval()
    yes_id, no_id = tok.convert_tokens_to_ids('yes'), tok.convert_tokens_to_ids('no')
    pre = tok.encode(QWEN3_PREFIX, add_special_tokens=False)
    suf = tok.encode(QWEN3_SUFFIX, add_special_tokens=False)
    max_len = int(S4['score']['max_length'])
    room = max_len - len(pre) - len(suf)
    bs = int(S4['score']['batch_size'])
    info = {'name': a.name, 'model': key, 'adapter': run, 'score': 'log P(yes)', 'max_length': max_len, 'dtype': 'bf16',
            'instruction': S4['instruction'], 'started': time.strftime('%Y-%m-%d %H:%M:%S')}
    for target in ('sealed', 'external'):
        qs, ps = questions(target), pools(target)
        pairs = [{'qid': q, 'document_id': str(c['document_id']), 'rank': int(c['rank']),
                  'query': qs[q]['question_final'], 'evidence': evidence_text(c)}
                 for q in sorted(ps) if q in qs for c in ps[q]]
        if a.smoke:
            pairs = pairs[:32]
        p_out = out_dir / '{}.jsonl'.format(target)
        if p_out.exists():
            p_out.unlink()
        n_trunc = 0
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i]['evidence']) + len(pairs[i]['query']))
        t0 = time.time()
        with torch.inference_mode():
            for k in range(0, len(order), bs):
                b = [pairs[i] for i in order[k:k + bs]]
                body = tok([fmt(x['query'], x['evidence']) for x in b], padding=False, truncation='longest_first',
                           return_attention_mask=False, max_length=room)['input_ids']
                ids = [pre + list(x) + suf for x in body]
                n_trunc += sum(1 for x in body if len(x) >= room)
                n = max(map(len, ids))
                x = torch.full((len(ids), n), tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
                               dtype=torch.long)
                m = torch.zeros((len(ids), n), dtype=torch.long)
                for j, s in enumerate(ids):
                    x[j, n - len(s):] = torch.tensor(s)
                    m[j, n - len(s):] = 1
                last = model(input_ids=x.to(dev), attention_mask=m.to(dev), logits_to_keep=1).logits[:, -1, :].float()
                s = torch.log_softmax(torch.stack([last[:, no_id], last[:, yes_id]], dim=1), dim=1)[:, 1].cpu().tolist()
                for p, v in zip(b, s):
                    append_jsonl(p_out, {'qid': p['qid'], 'document_id': p['document_id'], 'rank': p['rank'], 'score': v})
        info[target] = {'pairs': len(pairs), 'truncated': n_trunc, 'seconds': round(time.time() - t0, 1)}
        log('{} {} 打分完成：{} 对，{:.0f} 秒'.format(a.name, target, len(pairs), time.time() - t0), 's4')
    write_json(out_dir / 'DONE.json', info)


if __name__ == '__main__':
    sys.exit(main())
