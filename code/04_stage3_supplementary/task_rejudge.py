#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S3：用 Qwen3-32B（非思考模式）按 eval_v2 已登记的要点召回评审规程，重评 eval_v2 已有的全部答案。

评审提示词、要点顺序（按答案 aid 固定打乱）、输出检查、重写规则（最多 3 次）、引用核对，
都与 eval_v2 的 server/judge.py（API 模式）相同；这里的函数逐行照搬自该文件。
答案、题目与参考要点来自 eval_v2 的运行结果；不需要、也不上传证据支持标签。

用法：python task_rejudge.py [--smoke]
"""
import argparse
import concurrent.futures as cf
import json
import random
import re
import sys
import time
from pathlib import Path

from common_sup import (ABORT, CFG, DATA, EXIT_SERVER_DOWN, RESULTS, Watchdog, append_jsonl, call_retry,
                        check_abort_on_failure, judge_ports, log, read_json, read_jsonl)

LABELS_KPR = ('2', '1', '0', 'X')


# ----------------------------------------------------------------------------- 照搬自 eval_v2 server/judge.py
def parse_json(text):
    t = text.strip()
    t = re.sub(r'^```(?:json)?\s*', '', t)
    t = re.sub(r'\s*```$', '', t)
    i, j = t.find('{'), t.rfind('}')
    if i < 0 or j <= i:
        raise ValueError('输出中没有 json 对象')
    return json.loads(t[i:j + 1])


def norm(s):
    return re.sub(r'[\s　]+', '', str(s or ''))


def norm_loose(s):
    return re.sub(r'[\s　，。、；：！？,.;:!?“”"‘’\'（）()【】\[\]《》<>…—\-]+', '', str(s or ''))


def quote_in(quote, text):
    q = norm(quote)
    if not q:
        return False
    return q in norm(text) or (norm_loose(quote) and norm_loose(quote) in norm_loose(text))


def kpr_messages(P, q, answer, points, seed):
    pts = list(points)
    random.Random(seed).shuffle(pts)
    lines = '\n'.join(P['kpr']['point_line'].format(pid=p['pid'], type=p['type'], text=p['text']) for p in pts)
    user = P['kpr']['user'].format(question=q['question_final'], points=lines, answer=answer)
    return [{'role': 'system', 'content': P['kpr']['system']}, {'role': 'user', 'content': user}]


def check_kpr(obj, points, answer):
    items = obj.get('items')
    if not isinstance(items, list):
        return None, '缺少 items 列表'
    want = {p['pid'] for p in points}
    labels, quotes, qok = {}, {}, {}
    for it in items:
        if not isinstance(it, dict):
            continue
        pid = str(it.get('pid', '')).strip().upper()
        lab = str(it.get('label', '')).strip().upper()
        if pid in want and lab in LABELS_KPR:
            labels[pid] = lab
            quotes[pid] = str(it.get('quote') or '')
            qok[pid] = quote_in(quotes[pid], answer) if lab in ('2', '1') else True
    missing = sorted(want - set(labels))
    if missing:
        return None, '缺少要点 {}'.format(','.join(missing))
    bad = [p for p in labels if not qok[p]]
    res = {'labels': labels, 'quotes': quotes, 'quote_ok': qok, 'abstain': bool(obj.get('abstain', False))}
    return res, ('quote 不是答案原文：{}'.format(','.join(bad)) if bad else None)
# ----------------------------------------------------------------------------- 照搬结束


def load_questions():
    out = {}
    for q in read_json(DATA / 'rejudge' / 'final_questions.json'):
        if q.get('included', True):
            out[('sealed', q['qid'])] = q
    for q in read_json(DATA / 'external_questions.json'):
        if q.get('included', True):
            out[('external', q['qid'])] = q
    return out


def load_answers(targets):
    recs = []
    for t in targets:
        gens = ['G1', 'G2'] if t == 'sealed' else ['G1']
        for g in gens:
            d = DATA / 'rejudge' / 'gen' / t / g
            for f in sorted(d.glob('*.jsonl')):
                for r in read_jsonl(f):
                    r['target'] = t
                    recs.append(r)
    return recs


def judge_one(client, P, q, rec):
    model = CFG['models']['judge']['served_name']
    extra = {'chat_template_kwargs': {'enable_thinking': bool(CFG['rejudge']['enable_thinking'])}}
    points = list(q['points'])
    msgs = kpr_messages(P, q, rec['answer'], points, rec['aid'])
    best, problem, attempts, finishes = None, None, 0, []
    for _ in range(3):
        attempts += 1
        r = call_retry(lambda: client.chat.completions.create(model=model, messages=msgs, temperature=0.0,
                                                              max_tokens=int(CFG['rejudge']['max_tokens']), extra_body=extra))
        content = (r.choices[0].message.content or '') if r.choices else ''
        finishes.append(r.choices[0].finish_reason if r.choices else None)
        try:
            res, problem = check_kpr(parse_json(content), points, rec['answer'])
        except Exception as e:  # noqa: BLE001
            res, problem = None, '无法解析为 json（{}）'.format(type(e).__name__)
        if res is not None:
            best = res
            if problem is None:
                break
        msgs = msgs[:2] + [{'role': 'assistant', 'content': content},
                           {'role': 'user', 'content': P['kpr']['retry'].format(problem=problem)}]
    out = {'aid': rec['aid'], 'cond': rec['cond'], 'qid': rec['qid'], 'gen': rec.get('gen', 'G1'), 'set': rec.get('set'),
           'group': rec.get('group'), 'target': rec['target'], 'valid': best is not None, 'attempts': attempts,
           'problem': problem, 'finish_reasons': finishes, 'judge': model}
    if best:
        out.update(best)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--s4', action='store_true', help='评审 S4 在云端新生成的答案')
    a = ap.parse_args()
    P = read_json(DATA / 'rejudge' / 'judge_prompts.json')
    qs = load_questions()
    if a.s4:
        recs = []
        src = RESULTS / 'smoke' / 's4_answers' if a.smoke else RESULTS / 's4' / 'answers'
        for f in sorted(src.glob('*.jsonl')):
            recs += read_jsonl(f)
        recs = [r for r in recs if (r['target'], r['qid']) in qs]
        out_dir = RESULTS / ('smoke' if a.smoke else 's4')
        out_p = out_dir / ('kpr_qwen3_32b_s4_smoke.jsonl' if a.smoke else 'kpr_qwen3_32b_s4.jsonl')
    else:
        targets = list(CFG['rejudge']['targets'])
        recs = [r for r in load_answers(targets) if (r['target'], r['qid']) in qs]
        out_dir = RESULTS / ('smoke' if a.smoke else 'rejudge')
        out_p = out_dir / 'kpr_qwen3_32b.jsonl'
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.smoke:
        rng = random.Random(1)
        recs = rng.sample(recs, min(len(recs), int(CFG['smoke']['rejudge_answers'])))
        if not recs:
            raise SystemExit('试跑没有可评审的答案（S4 试跑生成的答案缺失？）')
    done = {r['aid'] for r in read_jsonl(out_p) if r.get('valid')}
    todo = [r for r in recs if r['aid'] not in done]
    log('Qwen3-32B 重评要点召回：共 {} 条答案，已完成 {}，本次待做 {}'.format(len(recs), len(recs) - len(todo), len(todo)), 'rejudge')
    from openai import OpenAI
    ports = judge_ports()
    clients = [OpenAI(api_key='EMPTY', base_url='http://127.0.0.1:{}/v1'.format(p), timeout=900, max_retries=0) for p in ports]
    log('评审服务端口：{}'.format(ports), 'rejudge')
    t0 = time.time()
    n_bad = 0
    dog = Watchdog('rejudge')

    def task(k, r):
        if ABORT.is_set():
            return None
        return judge_one(clients[k % len(clients)], P, qs[(r['target'], r['qid'])], r)

    with cf.ThreadPoolExecutor(max_workers=4 if a.smoke else int(CFG['rejudge']['workers']) * len(clients)) as ex:
        futs = {ex.submit(task, k, r): r for k, r in enumerate(todo)}
        for i, f in enumerate(cf.as_completed(futs), 1):
            dog.tick()
            try:
                out = f.result()
            except Exception as e:  # noqa: BLE001
                if check_abort_on_failure(['judge'], 'rejudge'):
                    continue
                r = futs[f]
                out = {'aid': r['aid'], 'cond': r['cond'], 'qid': r['qid'], 'target': r['target'], 'valid': False,
                       'problem': 'exception: {}'.format(repr(e)[:300])}
            if out is None:
                continue
            n_bad += not out.get('valid')
            append_jsonl(out_p, out)
            if i % 200 == 0 or i == len(futs):
                el = time.time() - t0
                log('重评进度 {}/{}（不合规 {}），已用 {:.1f} 小时，预计还需 {:.1f} 小时'.format(
                    len(recs) - len(todo) + i, len(recs), n_bad, el / 3600, el / i * (len(futs) - i) / 3600), 'rejudge')
    dog.stop()
    if ABORT.is_set():
        log('重评因服务中断暂停', 'rejudge')
        sys.exit(EXIT_SERVER_DOWN)
    if a.smoke:
        rs = read_jsonl(out_p)
        valid = {r['aid'] for r in rs if r.get('valid')}
        aids = {r['aid'] for r in recs}
        rep = {'smoke_answers': len(aids), 'valid': len(valid & aids),
               'truncated': sum(1 for r in rs if 'length' in (r.get('finish_reasons') or [])),
               'mean_attempts': round(sum(r.get('attempts', 0) for r in rs) / max(1, len(rs)), 2),
               'sec_total': round(time.time() - t0, 1),
               'example': {k: rs[0].get(k) for k in ('cond', 'labels', 'abstain', 'attempts')} if rs else None}
        print(json.dumps(rep, ensure_ascii=False, indent=1))
        ok = len(valid & aids) >= len(aids) - 1
        log('重评试跑（{}）：{}/{} 条有效，{}'.format('S4' if a.s4 else 'S3', rep['valid'], len(aids), '通过' if ok else '未通过'),
            'rejudge')
        if not ok:
            sys.exit(2)


if __name__ == '__main__':
    main()
