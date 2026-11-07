#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S4 公用部分：条件名、路径、stage4 数据与评测函数的载入、Qwen3-Reranker 的输入格式（与 eval_v1 打分完全相同）。"""
import os
import sys
from pathlib import Path

from common_sup import CFG, DATA, WORK, read_json, read_jsonl

S4 = CFG['s4']
S4_WORK = WORK / 's4'
TRAIN_DIR = S4_WORK / 'train'
SCORE_DIR = S4_WORK / 'scores'
SEL_DIR = S4_WORK / 'selections'
ANS_DIR = WORK / 'results' / 's4' / 'answers'
STAGE4 = S4_WORK / 'stage4'                     # stage4_export.tar.gz 解压后的位置
MODELS = Path(os.environ.get('SUP_MODELS') or (WORK / 'models'))

# 与 eval_v1/server_score.py 完全相同的 Qwen3-Reranker 输入格式
QWEN3_PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the '
                'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
QWEN3_SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def fmt(query, doc):
    return '<Instruct>: {}\n<Query>: {}\n<Document>: {}'.format(S4['instruction'], query, doc)


def size_tag(key):
    return {'q06': 'Q06', 'q4b': 'Q4B'}[key]


def train_runs():
    """全部微调运行：(模型键, 目标, 种子, 运行名)。4B 在前，便于多卡并行时均衡。"""
    out = []
    for key in ('q4b', 'q06'):
        for obj in S4['objectives']:
            for seed in S4['seeds']:
                out.append((key, obj, seed, '{}_{}_s{}'.format(key, obj, seed)))
    return out


def score_models():
    """要打分的全部模型：(打分名, 模型键, 微调运行名或 None)。"""
    out = [('{}_zeroshot'.format(k), k, None) for k in ('q06', 'q4b')]
    out += [(name, key, name) for key, _, _, name in train_runs()]
    return out


def conditions():
    """S4 新条件：条件名 → 打分名。N = 原始权重（零样本），S = 支持标签微调，R = 相关性标签微调。"""
    out = {}
    for key in ('q06', 'q4b'):
        out['{}N'.format(size_tag(key))] = '{}_zeroshot'.format(key)
        for obj, t in (('support', 'S'), ('relevance', 'R')):
            for seed in S4['seeds']:
                out['{}{}_s{}'.format(size_tag(key), t, seed)] = '{}_{}_s{}'.format(key, obj, seed)
    return out


def questions(target):
    if target == 'sealed':
        return {q['qid']: q for q in read_json(DATA / 'rejudge' / 'final_questions.json') if q.get('included', True)}
    return {q['qid']: q for q in read_json(DATA / 'external_questions.json') if q.get('included', True)}


def pools(target):
    p = DATA / 's4' / 'sealed_pools.jsonl' if target == 'sealed' else DATA / 'external_pools.jsonl'
    out = {}
    for r in read_jsonl(p):
        out.setdefault(r['qid'], []).append(r)
    for v in out.values():
        v.sort(key=lambda r: int(r['rank']))
    return out


def evidence_text(c):
    return c.get('context_text') or c.get('text') or ''


def stage4_modules():
    """与 eval_v2 的 train_bge.py 相同：从 stage4 导出包里的三个目录（按原顺序）载入 stage4 自带的评测函数。"""
    for i in range(3):
        d = STAGE4 / 'lib{}'.format(i)
        if d.exists() and str(d) not in sys.path:
            sys.path.append(str(d))
    from evaluate_v2 import choose_threshold, classify, evaluate  # noqa: E402
    from policy_stage4 import checkpoint_key  # noqa: E402
    return classify, choose_threshold, evaluate, checkpoint_key
