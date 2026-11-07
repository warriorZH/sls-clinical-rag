#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按条件选出交给问答模型的证据（候选编号与顺序），写入 runs/selections/<target>/<条件>.jsonl。

    python select_evidence.py --sets main,ksens,palt --job e3_select
选择规则（预先固定）：
  score      按分数从高到低取前 k 条，同分按检索名次；
  score_mean 多个种子分数取平均后同上；
  bm25       检索名次前 k 条；all 全部 20 条（检索名次顺序）；none 闭卷；
  random     以“种子:题号”为随机种子打乱候选后取前 k 条；
  list       金标导出的候选编号清单（O1–O4），只有在非 O 条件的答案冻结之后才允许读取，顺序按固定种子打乱。
"""
import argparse
import random
import sys

from common import EVAL_V1, HERE, RUNS, Log, is_done, load_config, load_target, mark_done, read_jsonl, write_jsonl
import conditions as C

ORACLE_FILE = HERE / 'data' / 'oracle_lists.jsonl'
ORACLE_ORDER_SEED = 7


def score_file(name, source, target):
    if source == 'v1':
        return EVAL_V1 / 'scores' / target / '{}.jsonl'.format(name)
    return RUNS / 'scores' / target / '{}.jsonl'.format(name)


def load_scores(name, source, target):
    p = score_file(name, source, target)
    if not p.exists():
        raise FileNotFoundError('缺少分数文件 {}'.format(p))
    out = {}
    for r in read_jsonl(p):
        out.setdefault(r['qid'], {})[str(r['document_id'])] = float(r['score'])
    return out


def oracle_lists():
    if not is_done('e3_freeze'):
        raise RuntimeError('O 条件的证据清单只能在非 O 条件的答案冻结（e3_freeze）之后读取')
    out = {}
    for r in read_jsonl(ORACLE_FILE):
        out.setdefault(r['list'], {})[r['qid']] = [str(d) for d in r['docs']]
    return out


def select(cond, questions, pools, target):
    """返回 {qid: [document_id, ...]}；不适用的题目不出现在结果中（只有 O 条件会这样）。"""
    sel, k = cond['selector'], cond['k']
    t = sel['type']
    res = {}
    if t == 'list':
        lists = oracle_lists().get(sel['list'], {})
        for q in questions:
            docs = lists.get(q['qid'])
            if docs:
                docs = list(docs)[:k] if k else list(docs)
                random.Random('{}:{}:{}'.format(ORACLE_ORDER_SEED, sel['list'], q['qid'])).shuffle(docs)
                res[q['qid']] = docs
        return res
    scores = None
    if t == 'score':
        scores = load_scores(sel['name'], sel['source'], target)
    elif t == 'score_mean':
        parts = [load_scores(n, sel['source'], target) for n in sel['names']]
        scores = {}
        for qid in parts[0]:
            scores[qid] = {d: sum(p[qid][d] for p in parts) / len(parts) for d in parts[0][qid]}
    for q in questions:
        qid, pool = q['qid'], pools[q['qid']]
        if t == 'none':
            docs = []
        elif t == 'bm25':
            docs = [c['document_id'] for c in pool[:k]]
        elif t == 'all':
            docs = [c['document_id'] for c in pool]
        elif t == 'random':
            docs = [c['document_id'] for c in pool]
            random.Random('{}:{}'.format(sel['seed'], qid)).shuffle(docs)
            docs = docs[:k]
        elif t in ('score', 'score_mean'):
            s = scores[qid]
            if len(s) != len(pool):
                raise RuntimeError('{} 的分数条数 {} 与候选数 {} 不符'.format(qid, len(s), len(pool)))
            order = sorted(pool, key=lambda c: (-s[c['document_id']], c['rank']))
            docs = [c['document_id'] for c in order[:k]]
        else:
            raise ValueError('未知的选择方式 {}'.format(t))
        res[qid] = docs
    return res


def selection_path(cond):
    return RUNS / 'selections' / cond['target'] / '{}.jsonl'.format(cond['id'])


def ensure_selection(cond, cfg, cache=None):
    """已有选择文件就直接读取（保证同一条件只选一次），否则生成。"""
    p = selection_path(cond)
    if p.exists():
        return {r['qid']: r['docs'] for r in read_jsonl(p)}
    cache = cache if cache is not None else {}
    if cond['target'] not in cache:
        cache[cond['target']] = load_target(cfg, cond['target'])
    questions, pools = cache[cond['target']]
    res = select(cond, questions, pools, cond['target'])
    write_jsonl(p, [{'cond': cond['id'], 'qid': qid, 'docs': docs} for qid, docs in sorted(res.items())])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', required=True, help='逗号分隔：main,ksens,palt,e2,oracle,g2')
    ap.add_argument('--job', required=True)
    a = ap.parse_args()
    cfg = load_config()
    log = Log(RUNS / 'logs' / 'select.log')
    allc = C.conditions(cfg)
    cache, made, skipped = {}, [], []
    for s in a.sets.split(','):
        for cond in allc[s.strip()]:
            try:
                res = ensure_selection(cond, cfg, cache)
                made.append(cond['id'])
                log('{}：{} 题'.format(cond['id'], len(res)))
            except FileNotFoundError as e:
                skipped.append({'cond': cond['id'], 'reason': str(e)})
                log('[跳过] {}：{}'.format(cond['id'], e))
    required_missing = [x for x in skipped if not any(g in x['cond'] for g in ('G12', 'G13'))]
    if required_missing:
        log('必需条件缺少分数：{}'.format(required_missing))
        sys.exit(1)
    mark_done(a.job, {'selections': made, 'skipped_optional': skipped})


if __name__ == '__main__':
    main()
