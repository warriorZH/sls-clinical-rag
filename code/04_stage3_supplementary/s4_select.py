#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S4：按分数选 Top-2 证据（规则与 eval_v2 的 select_evidence.py 相同：分数从高到低，同分按检索名次），
并把参照条件（闭卷、Relevance-CE、Support-CE）的 eval_v2 冻结选择结果复制过来。"""
from common_sup import DATA, log, read_jsonl
from s4_common import S4, SCORE_DIR, SEL_DIR, conditions, pools, questions
import json


def write(target, cond, rows):
    d = SEL_DIR / target
    d.mkdir(parents=True, exist_ok=True)
    with open(d / '{}.jsonl'.format(cond), 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def main():
    k = int(S4['k'])
    for target in ('sealed', 'external'):
        qs, ps = questions(target), pools(target)
        for cond, sname in conditions().items():
            sc = {}
            for r in read_jsonl(SCORE_DIR / sname / '{}.jsonl'.format(target)):
                sc[(r['qid'], str(r['document_id']))] = float(r['score'])
            rows = []
            for q in sorted(ps):
                if q not in qs:
                    continue
                cand = ps[q]
                miss = [c for c in cand if (q, str(c['document_id'])) not in sc]
                if miss:
                    raise SystemExit('{} {} 缺少 {} 条候选的分数'.format(sname, q, len(miss)))
                order = sorted(cand, key=lambda c: (-sc[(q, str(c['document_id']))], int(c['rank'])))
                rows.append({'cond': cond, 'qid': q, 'docs': [str(c['document_id']) for c in order[:k]]})
            write(target, cond, rows)
        src = DATA / 's4' / 'selections_sealed' if target == 'sealed' else DATA / 'selections_external'
        for cond in S4['reference'][target]:
            rows = read_jsonl(src / '{}.jsonl'.format(cond)) if not cond.endswith('00') else \
                [{'cond': cond, 'qid': q, 'docs': []} for q in sorted(qs)]
            write(target, cond, rows)
    log('S4 证据选择已写入 {}'.format(SEL_DIR), 's4')


if __name__ == '__main__':
    main()
