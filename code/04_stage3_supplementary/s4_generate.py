#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S4：用 Qwen3-4B-Instruct-2507（vLLM，bf16，贪心解码，最多 768 个词元）按 eval_v2 的主提示词（只依据证据作答）
为 S4 的新条件与参照条件生成答案。提示词与 eval_v2 的 generate.py 完全相同（data/s4/prompts.json）。

    python s4_generate.py [--smoke]
"""
import argparse
import concurrent.futures as cf
import time

from common_sup import (ABORT, CFG, DATA, EXIT_SERVER_DOWN, RESULTS, Watchdog, append_jsonl, call_retry,
                        check_abort_on_failure, log, make_client, read_json, read_jsonl, sha256_text)
import sys
from s4_common import ANS_DIR, S4, SEL_DIR, conditions, evidence_text, pools, questions


def build_messages(prompts, question, evid_texts, prompt='main'):
    """照搬自 eval_v2 server/generate.py。"""
    if not evid_texts:
        t = prompts['{}_closed_book'.format(prompt)]
        user = t['user'].format(question=question)
    else:
        t = prompts['{}_evidence'.format(prompt)]
        block = '\n\n'.join(t['evidence_item'].format(i=i + 1, text=x) for i, x in enumerate(evid_texts))
        user = t['user'].format(question=question, evidence=block)
    return [{'role': 'system', 'content': t['system']}, {'role': 'user', 'content': user}]


def all_conditions(target):
    return list(S4['reference'][target]) + list(conditions())


def selection(target, cond, smoke=False):
    """正式运行读 s4_select.py 的结果；试跑（在微调之前）直接读一键包里参照条件的冻结选择。"""
    if not smoke:
        return {r['qid']: r['docs'] for r in read_jsonl(SEL_DIR / target / '{}.jsonl'.format(cond))}
    if cond.endswith('00'):
        return {q: [] for q in questions(target)}
    src = DATA / 's4' / 'selections_sealed' if target == 'sealed' else DATA / 'selections_external'
    return {r['qid']: [str(d) for d in r['docs']] for r in read_jsonl(src / '{}.jsonl'.format(cond))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    prompts = read_json(DATA / 's4' / 'prompts.json')
    G = S4['gen']
    client = make_client('gen')
    model = CFG['models']['gen']['served_name']
    ans_dir = RESULTS / 'smoke' / 's4_answers' if a.smoke else ANS_DIR
    ans_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for target in ('sealed', 'external'):
        qs = questions(target)
        texts = {(q, str(c['document_id'])): evidence_text(c) for q, cs in pools(target).items() for c in cs}
        conds = all_conditions(target)
        if a.smoke:                               # 试跑：每个目标集取闭卷与一个有证据的参照条件，各 2 题
            ref = list(S4['reference'][target])
            conds = [ref[0], ref[-1]]
        for cond in conds:
            sel = selection(target, cond, a.smoke)
            if not a.smoke and not sel:
                raise SystemExit('缺少证据选择：{} {}（s4_select 未完成？）'.format(target, cond))
            out = ans_dir / '{}_{}.jsonl'.format(target, cond)
            done = {r['qid'] for r in read_jsonl(out)}
            n_c = 0
            for q in sorted(qs):
                if q not in sel:
                    continue
                n_c += 1
                if a.smoke and n_c > 2:
                    break
                if q in done:
                    continue
                jobs.append((target, cond, q, sel[q], [texts[(q, d)] for d in sel[q]], out, qs[q]['question_final']))
    log('S4 生成答案：待做 {} 条'.format(len(jobs)), 's4')

    def work(j):
        target, cond, q, docs, ev, out, question = j
        msgs = build_messages(prompts, question, ev, G['prompt'])
        r = call_retry(lambda: client.chat.completions.create(model=model, messages=msgs, temperature=float(G['temperature']),
                                                              max_tokens=int(G['max_tokens'])))
        ch = r.choices[0]
        aid = sha256_text('s4|{}|{}|{}'.format(target, cond, q))[:16]
        append_jsonl(out, {'aid': aid, 'cond': cond, 'qid': q, 'target': target, 'gen': 'G1', 'set': 's4',
                           'group': cond.rsplit('_s', 1)[0], 'docs': docs, 'answer': ch.message.content or '',
                           'finish_reason': ch.finish_reason, 'prompt': G['prompt'],
                           'n_out': getattr(r.usage, 'completion_tokens', None) if r.usage else None})

    def task(j):
        if ABORT.is_set():
            return
        try:
            work(j)
        except Exception:  # noqa: BLE001
            if check_abort_on_failure(['gen'], 's4'):
                return
            raise

    t0 = time.time()
    dog = Watchdog('s4')
    with cf.ThreadPoolExecutor(max_workers=4 if a.smoke else int(G['workers'])) as ex:
        for i, f in enumerate(cf.as_completed([ex.submit(task, j) for j in jobs]), 1):
            f.result()
            dog.tick()
            if i % 500 == 0 or i == len(jobs):
                log('S4 生成进度 {}/{}，已用 {:.1f} 分钟'.format(i, len(jobs), (time.time() - t0) / 60), 's4')
    dog.stop()
    if ABORT.is_set():
        sys.exit(EXIT_SERVER_DOWN)
    if a.smoke:
        rs = [r for f in ans_dir.glob('*.jsonl') for r in read_jsonl(f)]
        empty = sum(1 for r in rs if not r['answer'].strip())
        trunc = sum(1 for r in rs if r.get('finish_reason') == 'length')
        log('S4 生成试跑：{} 条答案，空答案 {}，截断 {}，用时 {:.0f} 秒'.format(len(rs), empty, trunc, time.time() - t0), 's4')
        if len(rs) < 8 or empty:
            sys.exit(2)


if __name__ == '__main__':
    main()
