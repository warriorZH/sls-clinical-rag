#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按预注册分析计划计算封存集评测指标（eval_v1），只用 Python 标准库。

    python eval_v1/evaluate.py --freeze     封存集打分之前运行：写出 eval_v1/FREEZE_MANIFEST.json（各文件 SHA256）
    python eval_v1/evaluate.py              读取 eval_v1/scores/sealed/，写出 eval_v1/results/sealed/

依据：预注册分析计划.md 与 eval_v1/实验对比方案.md。需要 Python 3.8+。
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
TESTSET = EVAL_DIR.parent
SEALED = TESTSET / '_sealed_负责人保存_勿发给标注员'

FROZEN = {
    'final_questions.json': 'af0b2b4d34a84a4be57524c84788df938e295fd3e9afd263b25d7b7c70b5aeb3',
    'sealed_pools.jsonl': '6c228959456ad06765adc84f86811eb0abe410901769fe5013bd7d6559de4659',
    'phase2_gold_labels.jsonl': '1311d616353f78fb843bfae58d5338877b5d4ec836d0ccd6ab47ec58334f4e7a',
}
B = 10000
BOOT_SEED = 20261026
TOPK = 2
NDCG_K = 10
ALPHA = 0.05

BASE = 'Support-CE'
PREREG = ['Support-CE', 'Relevance-CE', 'BM25', 'bge-reranker-v2-m3', 'Qwen3-Reranker-0.6B']
F1_OTHERS = ['BM25', 'bge-reranker-v2-m3', 'Qwen3-Reranker-0.6B']
F_OTHERS = ['Relevance-CE', 'BM25', 'bge-reranker-v2-m3', 'Qwen3-Reranker-0.6B']

METRICS = [  # (键, 名称, 方向说明)
    ('cov', '覆盖率（主指标）', '越高越好'),
    ('full', '全覆盖率', '越高越好'),
    ('nonsup', '选入非支持率', '越低越好'),
    ('ndcg', 'nDCG@10', '越高越好'),
    ('cov_partial', '部分支持口径覆盖率', '越高越好'),
    ('cov_graded', 'X1 分级覆盖率（探索）', '越高越好'),
    ('mislead', 'X3 误导率（探索）', '越低越好'),
    ('rel_nosup', 'X4 相关但不完整支持率（探索）', '越低越好'),
    ('rel_zero', 'X4 相关但全不支持率（探索）', '越低越好'),
    ('relevant', 'X5 所选证据相关率（探索）', '—'),
]
FAMILIES = [  # (家族, 指标, 对照方法)
    ('F1', 'cov', F1_OTHERS),
    ('F2', 'full', F_OTHERS),
    ('F3', 'nonsup', F_OTHERS),
    ('F4', 'ndcg', F_OTHERS),
    ('F5', 'cov_partial', F_OTHERS),
]
SOURCES = ['教材', '古籍', '其他病例']


def source_cat(src):
    src = src or ''
    if src.startswith('medical_train'):
        return '其他病例'
    if src.startswith('TCM Educational Materials'):
        return '教材'
    if src.startswith('TCM Ancient books'):
        return '古籍'
    return '其他'


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------- 读取

def load_questions(path, full_path=None):
    qs = [q for q in json.loads(Path(path).read_text(encoding='utf-8')) if q.get('included', True)]
    strata = {}
    if full_path and Path(full_path).exists():
        strata = {x['qid']: x.get('stratum') for x in json.loads(Path(full_path).read_text(encoding='utf-8'))}
    out = {}
    for q in qs:
        out[q['qid']] = {'type': q['question_type'], 'points': [p['pid'] for p in q['points']],
                         'point_types': {p['pid']: p['type'] for p in q['points']},
                         'stratum': strata.get(q['qid']) or q.get('stratum') or '未知'}
    return out


def load_pools(path):
    pools = defaultdict(list)
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                pools[r['qid']].append({'doc': str(r['document_id']), 'rank': int(r['rank']),
                                        'source': source_cat(r.get('source')),
                                        'empty': not (r.get('context_text') or r.get('text') or '').strip()})
    for v in pools.values():
        v.sort(key=lambda x: x['rank'])
    return dict(pools)


def load_gold(path):
    gold = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                g = json.loads(line)
                gold[(g['qid'], str(g['document_id']))] = {
                    'support': {k: int(v) for k, v in g['support'].items() if v is not None and str(v) != ''},
                    'relevance': int(g.get('relevance') or 0), 'mismatch': int(g.get('mismatch') or 0)}
    return gold


def load_runs(scores_dir):
    """返回 {方法: [(种子, {(qid, doc): score}, 运行记录)]}，以及运行清单。"""
    mpath = Path(scores_dir) / 'RUN_MANIFEST.json'
    if not mpath.exists():
        sys.exit('找不到 {}。请先把服务器上的 eval_v1/scores/ 下载到本机。'.format(mpath))
    manifest = json.loads(mpath.read_text(encoding='utf-8'))
    runs = defaultdict(list)
    for name, rec in manifest['runs'].items():
        if rec.get('status') != 'ok':
            continue
        f = Path(scores_dir) / '{}.jsonl'.format(name)
        if not f.exists():
            sys.exit('运行清单中有 {}，但找不到 {}。'.format(name, f))
        if rec.get('scores_sha256') and sha256_file(f) != rec['scores_sha256']:
            sys.exit('{} 的 SHA256 与运行清单不符，文件可能被改动。'.format(f))
        sc = {}
        with f.open(encoding='utf-8') as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    sc[(r['qid'], str(r['document_id']))] = float(r['score'])
        runs[rec.get('method') or name].append((rec.get('seed'), sc, rec))
    for m in runs:
        runs[m].sort(key=lambda x: (x[0] is None, x[0]))
    return dict(runs), manifest


# ----------------------------------------------------------------------------- 每题指标

def maxsup(gold, qid, doc):
    s = gold[(qid, doc)]['support']
    return max(s.values()) if s else 0


def pool_stats(q, qid, pool, gold):
    pts = q['points']
    best = {p: max(gold[(qid, c['doc'])]['support'].get(p, 0) for c in pool) for p in pts}
    gains = sorted((maxsup(gold, qid, c['doc']) for c in pool), reverse=True)
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains[:NDCG_K]))
    return {'ub': sum(best[p] == 2 for p in pts) / len(pts),
            'ub_partial': sum(best[p] >= 1 for p in pts) / len(pts),
            'ub_graded': sum({2: 1.0, 1: 0.5}.get(best[p], 0.0) for p in pts) / len(pts),
            'has_full': any(best[p] == 2 for p in pts), 'idcg': idcg,
            'all_coverable': all(best[p] == 2 for p in pts)}


def rank_pool(pool, qid, scores):
    missing = [c['doc'] for c in pool if (qid, c['doc']) not in scores]
    if missing:
        raise KeyError('{} 缺少 {} 条候选的分数'.format(qid, len(missing)))
    return sorted(pool, key=lambda c: (-scores[(qid, c['doc'])], c['rank']))


def run_metrics(q, qid, pool, gold, scores, idcg):
    ranked = rank_pool(pool, qid, scores)
    top = ranked[:TOPK]
    pts = q['points']
    best = {p: max(gold[(qid, c['doc'])]['support'].get(p, 0) for c in top) for p in pts}
    cov = sum(best[p] == 2 for p in pts) / len(pts)
    ms = [maxsup(gold, qid, c['doc']) for c in top]
    rel = [gold[(qid, c['doc'])]['relevance'] for c in top]
    mis = [gold[(qid, c['doc'])]['mismatch'] for c in top]
    gains = [maxsup(gold, qid, c['doc']) for c in ranked[:NDCG_K]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    out = {'cov': cov, 'full': 1.0 if cov == 1 else 0.0,
           'nonsup': sum(m < 2 for m in ms) / TOPK,
           'ndcg': dcg / idcg if idcg > 0 else None,
           'cov_partial': sum(best[p] >= 1 for p in pts) / len(pts),
           'cov_graded': sum({2: 1.0, 1: 0.5}.get(best[p], 0.0) for p in pts) / len(pts),
           'mislead': sum(mis) / TOPK,
           'rel_nosup': sum(r == 1 and m < 2 for r, m in zip(rel, ms)) / TOPK,
           'rel_zero': sum(r == 1 and m == 0 for r, m in zip(rel, ms)) / TOPK,
           'relevant': sum(rel) / TOPK}
    picks = [{'source': c['source'], 'full': maxsup(gold, qid, c['doc']) == 2} for c in top]
    return out, picks, [c['doc'] for c in top]


def per_question(questions, pools, gold, runs):
    """返回 pool[qid]、vals[方法][指标][qid]（种子平均）、seed_vals[方法][种子][指标][qid]、picks、top 记录。"""
    pool_info = {qid: pool_stats(questions[qid], qid, pools[qid], gold) for qid in questions}
    vals, seed_vals, picks, tops = {}, {}, {}, {}
    for method, rs in runs.items():
        vals[method] = defaultdict(dict)
        seed_vals[method] = {}
        picks[method] = []
        tops[method] = {}
        for seed, sc, _ in rs:
            sv = defaultdict(dict)
            for qid in questions:
                m, pk, top = run_metrics(questions[qid], qid, pools[qid], gold, sc, pool_info[qid]['idcg'])
                for k, v in m.items():
                    sv[k][qid] = v
                picks[method].extend(dict(x, qid=qid, seed=seed) for x in pk)
                tops[method].setdefault(qid, []).append(top)
            seed_vals[method][seed] = sv
        for k, _, _ in METRICS:
            for qid in questions:
                xs = [seed_vals[method][s][k][qid] for s in seed_vals[method]]
                xs = [x for x in xs if x is not None]
                vals[method][k][qid] = sum(xs) / len(xs) if xs else None
    return pool_info, vals, seed_vals, picks, tops


# ----------------------------------------------------------------------------- 统计

class Boot:
    """按题目配对 bootstrap。同一题目集合共用同一套重抽样下标。"""

    def __init__(self, reps=B, seed=BOOT_SEED):
        self.reps, self.seed, self.cache = reps, seed, {}

    def idx(self, n, tag):
        key = (n, tag)
        if key not in self.cache:
            rng = random.Random('{}|{}|{}'.format(self.seed, tag, n))
            self.cache[key] = [[rng.randrange(n) for _ in range(n)] for _ in range(self.reps)]
        return self.cache[key]

    def dist(self, x, tag):
        n = len(x)
        return sorted(sum(x[i] for i in row) / n for row in self.idx(n, tag))


def percentile(sorted_vals, p):
    pos = p * (len(sorted_vals) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def ci_of(sorted_vals):
    return [percentile(sorted_vals, ALPHA / 2), percentile(sorted_vals, 1 - ALPHA / 2)]


def boot_p(sorted_vals):
    n = len(sorted_vals)
    le = sum(1 for v in sorted_vals if v <= 0)
    ge = sum(1 for v in sorted_vals if v >= 0)
    return min(1.0, 2 * min((le + 1) / (n + 1), (ge + 1) / (n + 1)))


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float('nan')


def estimate(boot, values, qids, tag):
    x = [values[q] for q in qids if values[q] is not None]
    if not x:
        return {'mean': None, 'ci95': [None, None], 'n': 0}
    d = boot.dist(x, tag)
    return {'mean': sum(x) / len(x), 'ci95': ci_of(d), 'n': len(x)}


def paired(boot, va, vb, qids, tag):
    keep = [q for q in qids if va[q] is not None and vb[q] is not None]
    d = [va[q] - vb[q] for q in keep]
    dist = boot.dist(d, tag)
    return {'diff': sum(d) / len(d), 'ci95': ci_of(dist), 'p': boot_p(dist), 'n': len(d),
            'mean_a': mean(va[q] for q in keep), 'mean_b': mean(vb[q] for q in keep)}


# ----------------------------------------------------------------------------- 一致性补充（κ 悖论）

def binary_agreement(a, b):
    n = len(a)
    both1 = sum(1 for x, y in zip(a, b) if x and y)
    both0 = sum(1 for x, y in zip(a, b) if not x and not y)
    diff = n - both1 - both0
    po = (both1 + both0) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe_k = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    pi = (pa1 + pb1) / 2
    pe_ac1 = 2 * pi * (1 - pi)
    return {'n': n, 'positives_a': sum(a), 'positives_b': sum(b), 'both_positive': both1,
            'percent_agreement': po,
            'kappa': (po - pe_k) / (1 - pe_k) if pe_k < 1 else float('nan'),
            'pabak': 2 * po - 1,
            'gwet_ac1': (po - pe_ac1) / (1 - pe_ac1) if pe_ac1 < 1 else float('nan'),
            'positive_agreement': 2 * both1 / (2 * both1 + diff) if (2 * both1 + diff) else float('nan'),
            'negative_agreement': 2 * both0 / (2 * both0 + diff) if (2 * both0 + diff) else float('nan')}


def read_annotator_cells(folder, questions):
    """返回 {标注员: {(qid, 候选编号, pid): 0/1/2}}；只取题目中存在的要点。"""
    out = {}
    for who in 'ABCDEF':
        p = Path(folder) / 'phase2_证据标注_标注员{}.csv'.format(who)
        if not p.exists():
            continue
        cells = {}
        with p.open(encoding='utf-8-sig', newline='') as f:
            for r in csv.DictReader(f):
                qid = r['题目编号']
                if qid not in questions:
                    continue
                for i, pid in enumerate(questions[qid]['points'], 1):
                    col = next((c for c in r if c.startswith('【填】P{}支持'.format(i))), None)
                    v = (r.get(col) or '').strip() if col else ''
                    if v in ('0', '1', '2'):
                        cells[(qid, r['候选编号'], pid)] = int(v)
        out[who] = cells
    return out


def annotation_quality(questions, gold_path):
    res = {}
    a1 = TESTSET / '一致性报告' / 'agreement_phase1.json'
    a2 = TESTSET / '一致性报告' / 'agreement_phase2.json'
    if a1.exists():
        res['phase1'] = json.loads(a1.read_text(encoding='utf-8'))
    if a2.exists():
        res['phase2'] = json.loads(a2.read_text(encoding='utf-8'))
    arb = TESTSET / '仲裁' / 'phase2_仲裁表.csv'
    if arb.exists():
        with arb.open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        res['phase2_arbitrated_candidates'] = len(rows)
        res['phase2_primary_disagreement_rows'] = sum(1 for r in rows if (r.get('影响主指标(完整支持分歧)') or '').strip()
                                                      in ('1', '是', 'True', 'true'))
    cells = read_annotator_cells(TESTSET / 'phase2_证据标注', questions)
    if len(cells) >= 2:
        whos = sorted(cells)
        common = set.intersection(*[set(c) for c in cells.values()])
        keys = sorted(common)
        pair_stats = {}
        for i in range(len(whos)):
            for j in range(i + 1, len(whos)):
                x = [cells[whos[i]][k] == 2 for k in keys]
                y = [cells[whos[j]][k] == 2 for k in keys]
                pair_stats['{}-{}'.format(whos[i], whos[j])] = binary_agreement(x, y)
        res['full_support_binary'] = pair_stats
        # 金标与多数票的差异
        gold_rows = {}
        with open(gold_path, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    g = json.loads(line)
                    for pid, v in g['support'].items():
                        gold_rows[(g['qid'], g['cand'], pid)] = int(v)
        changed, total, maj_full, gold_full = 0, 0, 0, 0
        for k in keys:
            vs = [cells[w][k] for w in whos]
            top, cnt = Counter(vs).most_common(1)[0]
            if cnt * 2 > len(vs) and k in gold_rows:
                total += 1
                changed += gold_rows[k] != top
                maj_full += top == 2
                gold_full += gold_rows[k] == 2
        res['gold_vs_majority'] = {'cells_with_majority': total, 'gold_differs': changed,
                                   'majority_full': maj_full, 'gold_full_on_these': gold_full}
    return res


# ----------------------------------------------------------------------------- 主流程

def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return '—'
    return '{:.{}f}'.format(x, nd)


def fmt_ci(ci, nd=3):
    return '[{}, {}]'.format(fmt(ci[0], nd), fmt(ci[1], nd))


def fmt_p(p):
    if p is None:
        return '—'
    return '<0.001' if p < 0.001 else '{:.3f}'.format(p)


def check_frozen(paths):
    out = {}
    for name, p in paths.items():
        h = sha256_file(p)
        out[name] = {'path': str(p), 'sha256': h, 'expected': FROZEN.get(name), 'ok': FROZEN.get(name) in (None, h)}
    return out


def evaluate(a):
    qpath = Path(a.questions)
    ppath = Path(a.pools)
    gpath = Path(a.gold)
    frozen = check_frozen({'final_questions.json': qpath, 'sealed_pools.jsonl': ppath,
                           'phase2_gold_labels.jsonl': gpath}) if a.target == 'sealed' else {}
    bad = [k for k, v in frozen.items() if not v['ok']]
    if bad:
        sys.exit('冻结文件哈希不符：{}。已停止。'.format(', '.join(bad)))
    questions = load_questions(qpath, SEALED / 'sealed_questions_full.json' if a.target == 'sealed' else None)
    pools = {k: v for k, v in load_pools(ppath).items() if k in questions}
    gold = load_gold(gpath)
    runs, manifest = load_runs(a.scores)
    for qid in questions:
        if qid not in pools or len(pools[qid]) != 20:
            sys.exit('{} 的候选池不完整。'.format(qid))
        for c in pools[qid]:
            if (qid, c['doc']) not in gold:
                sys.exit('{} 的候选 {} 没有金标。'.format(qid, c['doc']))
    qids = sorted(questions)
    pool_info, vals, seed_vals, picks, tops = per_question(questions, pools, gold, runs)
    boot = Boot(a.reps, a.seed)
    methods = [m for m in PREREG if m in runs] + sorted(m for m in runs if m not in PREREG)
    missing = [m for m in PREREG if m not in runs]

    res = {'generated': dt.datetime.now().astimezone().isoformat(timespec='seconds'), 'target': a.target,
           'frozen': frozen, 'n_questions': len(qids), 'n_points': sum(len(questions[q]['points']) for q in qids),
           'bootstrap': {'reps': a.reps, 'seed': a.seed, 'ci': 'percentile, linear interpolation'},
           'methods': methods, 'missing_prereg_methods': missing,
           'runs': {m: [{'seed': s, 'name': r.get('name'), 'n_truncated': r.get('info', {}).get('n_truncated'),
                         'weights_sha256': r.get('weights_sha256'), 'spec': r.get('spec')} for s, _, r in rs]
                    for m, rs in runs.items()},
           'empty_evidence': sum(c['empty'] for q in qids for c in pools[q])}

    # 候选池上限
    res['pool'] = {k: estimate(boot, {q: pool_info[q][k] for q in qids}, qids, 'all')
                   for k in ('ub', 'ub_partial', 'ub_graded')}
    res['pool']['questions_with_full'] = sum(pool_info[q]['has_full'] for q in qids)
    res['pool']['questions_all_coverable'] = sum(pool_info[q]['all_coverable'] for q in qids)
    nd_q = [q for q in qids if pool_info[q]['idcg'] > 0]
    res['pool']['ndcg_questions'] = len(nd_q)
    res['pool']['ndcg_excluded'] = len(qids) - len(nd_q)

    # 各方法估计
    est = {}
    for m in methods:
        est[m] = {}
        for k, _, _ in METRICS:
            sub, tag = (nd_q, 'ndcg') if k == 'ndcg' else (qids, 'all')
            est[m][k] = estimate(boot, vals[m][k], sub, tag)
    res['estimates'] = est

    # H1
    if BASE in runs and 'Relevance-CE' in runs:
        h1 = paired(boot, vals[BASE]['cov'], vals['Relevance-CE']['cov'], qids, 'all')
        h1['supported'] = h1['ci95'][0] > 0
        h1['seeds'] = {'Support-CE': [s for s, _, _ in runs[BASE]], 'Relevance-CE': [s for s, _, _ in runs['Relevance-CE']]}
        res['H1'] = h1
    else:
        res['H1'] = None

    # 次要比较（Holm，每个指标一个家族）
    fams = []
    for fam, k, others in FAMILIES:
        if BASE not in runs:
            break
        sub, tag = (nd_q, 'ndcg') if k == 'ndcg' else (qids, 'all')
        comps = []
        for o in others:
            if o in runs:
                c = paired(boot, vals[BASE][k], vals[o][k], sub, tag)
                c.update({'vs': o})
                comps.append(c)
        adj = holm([c['p'] for c in comps]) if comps else []
        for c, p in zip(comps, adj):
            c['p_holm'] = p
            c['reject'] = p < ALPHA
        fams.append({'family': fam, 'metric': k, 'comparisons': comps,
                     'missing': [o for o in others if o not in runs]})
    res['families'] = fams

    # 探索性：不进入家族的比较
    expl = []
    for k in ('cov_graded', 'mislead', 'rel_nosup', 'rel_zero', 'relevant'):
        if BASE in runs and 'Relevance-CE' in runs:
            c = paired(boot, vals[BASE][k], vals['Relevance-CE'][k], qids, 'all')
            c.update({'metric': k, 'vs': 'Relevance-CE'})
            expl.append(c)
    for m in methods:
        if m not in PREREG and BASE in runs:
            c = paired(boot, vals[BASE]['cov'], vals[m]['cov'], qids, 'all')
            c.update({'metric': 'cov', 'vs': m})
            expl.append(c)
    cond_q = [q for q in qids if pool_info[q]['has_full']]
    if BASE in runs and 'Relevance-CE' in runs and cond_q:
        c = paired(boot, vals[BASE]['cov'], vals['Relevance-CE']['cov'], cond_q, 'cond')
        c.update({'metric': 'cov', 'vs': 'Relevance-CE', 'subset': 'X2 池中至少一个要点获得完整支持的题',
                  'n_questions': len(cond_q)})
        res['X2'] = c
        res['X2_by_method'] = {m: estimate(boot, vals[m]['cov'], cond_q, 'cond') for m in methods}
    res['exploratory'] = expl

    # 分层（描述性）
    strata = {}
    groups = {
        '题型': lambda q: {'diagnosis': '诊断解释', 'principle': '治疗原则'}.get(questions[q]['type'], questions[q]['type']),
        '来源层': lambda q: questions[q]['stratum'],
        '来源层×题型': lambda q: '{}/{}'.format(questions[q]['stratum'], questions[q]['type']),
    }
    for gname, fn in groups.items():
        cats = sorted({fn(q) for q in qids})
        strata[gname] = {}
        for cat in cats:
            sub = [q for q in qids if fn(q) == cat]
            row = {'n': len(sub), 'pool_ub': mean(pool_info[q]['ub'] for q in sub),
                   'pool_ub_partial': mean(pool_info[q]['ub_partial'] for q in sub),
                   'questions_with_full': sum(pool_info[q]['has_full'] for q in sub)}
            for m in methods:
                row[m] = {k: mean(vals[m][k][q] for q in sub) for k in ('cov', 'cov_partial', 'nonsup', 'full', 'ndcg')}
            strata[gname][cat] = row
    res['strata'] = strata

    # 证据来源
    src = {'pool': {}}
    for s in SOURCES + ['其他']:
        cands = [(q, c) for q in qids for c in pools[q] if c['source'] == s]
        if cands:
            src['pool'][s] = {'candidates': len(cands),
                              'full_any': sum(maxsup(gold, q, c['doc']) == 2 for q, c in cands),
                              'partial_any': sum(maxsup(gold, q, c['doc']) >= 1 for q, c in cands)}
    for m in methods:
        n = len(picks[m])
        row = {}
        for s in SOURCES + ['其他']:
            ps = [p for p in picks[m] if p['source'] == s]
            if ps or s in SOURCES:
                row[s] = {'share': len(ps) / n if n else float('nan'), 'n': len(ps),
                          'full_rate': (sum(p['full'] for p in ps) / len(ps)) if ps else None}
        src[m] = row
    res['evidence_source'] = src

    # 种子
    seeds = {}
    for m in methods:
        if len(seed_vals[m]) > 1:
            per = {}
            for s, sv in seed_vals[m].items():
                per[str(s)] = {k: mean(sv[k][q] for q in (nd_q if k == 'ndcg' else qids)) for k in ('cov', 'cov_partial', 'nonsup', 'ndcg', 'full')}
            sd = {}
            for k in ('cov', 'cov_partial', 'nonsup', 'ndcg', 'full'):
                xs = [per[s][k] for s in per]
                mu = sum(xs) / len(xs)
                sd[k] = math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))
            seeds[m] = {'per_seed': per, 'sd': sd}
    res['seeds'] = seeds
    res['annotation_quality'] = annotation_quality(questions, gpath) if a.target == 'sealed' else {}

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    with (out / 'per_question.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        keys = [k for k, _, _ in METRICS]
        w.writerow(['题目编号', '题型', '来源层', '要点数', '池上限', '池上限_部分支持', '池中有完整支持'] +
                   ['{}|{}'.format(m, k) for m in methods for k in keys] + ['{}|Top2文档'.format(m) for m in methods])
        for q in qids:
            w.writerow([q, questions[q]['type'], questions[q]['stratum'], len(questions[q]['points']),
                        fmt(pool_info[q]['ub'], 4), fmt(pool_info[q]['ub_partial'], 4), int(pool_info[q]['has_full'])] +
                       [fmt(vals[m][k][q], 4) for m in methods for k in keys] +
                       [' ; '.join(','.join(t) for t in tops[m][q]) for m in methods])
    (out / '评测结果报告.md').write_text(render(res, manifest), encoding='utf-8')
    print('报告已写出：{}'.format(out / '评测结果报告.md'))
    print('数值：{}'.format(out / 'results.json'))
    return res


# ----------------------------------------------------------------------------- 报告

def render(r, manifest):
    L = []
    label = {k: n for k, n, _ in METRICS}
    L += ['# 封存集模型对比评测结果（eval_v1）', '',
          '生成时间：{}。依据：`预注册分析计划.md`、`eval_v1/实验对比方案.md`。'.format(r['generated']), '']
    if r['target'] == 'sealed':
        L += ['**标注来源说明。** 本评测使用的金标 `phase2_gold_labels.jsonl` 由 AI 代理完成三方标注和仲裁，'
              '并非计划中的中医专业人员（见方案第 1 节）。负责人决定把它当作金标，用于验证模型方案。'
              '对外报告时必须如实写明标注来源，不能写成专家金标。', '']
    else:
        L += ['**这是测试运行（target={}），不是封存集结果。**'.format(r['target']), '']
    L += ['## 1. 数据与运行', '',
          '- 题目 {} 道，必要要点 {} 条，每题候选 20 条，每种方法选前 {} 条。'.format(r['n_questions'], r['n_points'], TOPK),
          '- 证据文本缺失的候选：{} 条（按不支持计）。'.format(r['empty_evidence']),
          '- bootstrap：按题配对重抽样 {} 次，随机种子 {}，百分位法 95% CI。'.format(r['bootstrap']['reps'], r['bootstrap']['seed'])]
    if r['frozen']:
        L += ['- 冻结文件哈希：' + '；'.join('{} {}'.format(k, '一致' if v['ok'] else '不符') for k, v in r['frozen'].items())]
    if r['missing_prereg_methods']:
        L += ['- **未得到结果的预注册方法**：{}（原因见运行清单）。'.format('、'.join(r['missing_prereg_methods']))]
    failed = [(n, x.get('error')) for n, x in manifest.get('runs', {}).items() if x.get('status') != 'ok']
    for n, e in failed:
        L += ['- 运行失败：`{}`：{}'.format(n, e)]
    if manifest.get('reruns'):
        L += ['- 重跑记录：' + '；'.join('{}（{}）'.format(x['name'], x['reason']) for x in manifest['reruns'])]
    L += ['', '| 方法 | 运行 | 截断条数 |', '|---|---|---|']
    for m in r['methods']:
        rs = r['runs'][m]
        L.append('| {} | {} | {} |'.format(m, '、'.join('种子 {}'.format(x['seed']) if x['seed'] is not None else x['name'] for x in rs),
                                           '、'.join(str(x['n_truncated'] if x['n_truncated'] is not None else 0) for x in rs)))
    L += ['']

    L += ['## 2. 主假设 H1', '']
    h = r.get('H1')
    if h:
        L += ['H1：Support-CE 的 Top-2 必要点覆盖率（严格口径，金标=2）高于 Relevance-CE。', '',
              '| Support-CE | Relevance-CE | 差值 | 95% CI | 判定 |', '|---|---|---|---|---|',
              '| {} | {} | {} | {} | {} |'.format(fmt(h['mean_a']), fmt(h['mean_b']), fmt(h['diff']), fmt_ci(h['ci95']),
                                                 '**支持 H1**（CI 下界 > 0）' if h['supported'] else '**未支持 H1**（CI 下界 ≤ 0）'),
              '', '覆盖率先在每题内对种子取平均（Support-CE 种子 {}；Relevance-CE 种子 {}），再在 {} 道题上取平均。'.format(
                  '/'.join(map(str, h['seeds']['Support-CE'])), '/'.join(map(str, h['seeds']['Relevance-CE'])), h['n']), '']
    else:
        L += ['Support-CE 或 Relevance-CE 没有结果，H1 无法检验。', '']

    L += ['## 3. 各方法指标（点估计与 95% CI）', '']
    keys = [k for k, _, _ in METRICS[:5]]
    L += ['| 方法 | ' + ' | '.join(label[k] for k in keys) + ' |', '|---' * (len(keys) + 1) + '|']
    for m in r['methods']:
        e = r['estimates'][m]
        L.append('| {} | '.format(m) + ' | '.join('{} {}'.format(fmt(e[k]['mean']), fmt_ci(e[k]['ci95'])) for k in keys) + ' |')
    L += ['', '- nDCG@10 只在池中有支持证据（IDCG>0）的 {} 道题上计算，另有 {} 道题池中没有任何支持证据，未参与。'.format(
        r['pool']['ndcg_questions'], r['pool']['ndcg_excluded']),
        '- 选入非支持率越低越好，其余越高越好。', '']

    L += ['## 4. 次要比较（Support-CE 对其他方法，Holm 校正）', '',
          '| 家族 | 指标 | 对照 | Support-CE | 对照方法 | 差值 | 95% CI | p | Holm p | 结论(α=0.05) |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for f in r['families']:
        for c in f['comparisons']:
            L.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
                f['family'], label[f['metric']], c['vs'], fmt(c['mean_a']), fmt(c['mean_b']), fmt(c['diff']),
                fmt_ci(c['ci95']), fmt_p(c['p']), fmt_p(c['p_holm']), '差异显著' if c['reject'] else '不显著'))
        if f['missing']:
            L.append('| {} | {} | {} | 无结果 | | | | | | |'.format(f['family'], label[f['metric']], '、'.join(f['missing'])))
    L += ['']

    p = r['pool']
    L += ['## 5. 候选池上限（与方法无关）', '',
          '| 口径 | 池上限 | 95% CI |', '|---|---|---|',
          '| 严格（完整支持） | {} | {} |'.format(fmt(p['ub']['mean']), fmt_ci(p['ub']['ci95'])),
          '| 部分支持也计入 | {} | {} |'.format(fmt(p['ub_partial']['mean']), fmt_ci(p['ub_partial']['ci95'])),
          '| 分级（1/0.5，探索） | {} | {} |'.format(fmt(p['ub_graded']['mean']), fmt_ci(p['ub_graded']['ci95'])), '',
          '- 池中至少一个要点获得完整支持的题：{} / {}；所有要点都能被完整支持的题：{} / {}。'.format(
              p['questions_with_full'], r['n_questions'], p['questions_all_coverable'], r['n_questions']), '']

    L += ['## 6. 分层（描述性，不做显著性结论）', '']
    for gname, cats in r['strata'].items():
        L += ['### {}'.format(gname), '', '| 类别 | 题数 | 池上限 | 池上限(部分) | ' + ' | '.join('{} 覆盖率'.format(m) for m in r['methods']) + ' |',
              '|---' * (4 + len(r['methods'])) + '|']
        for cat, row in cats.items():
            L.append('| {} | {} | {} | {} | '.format(cat, row['n'], fmt(row['pool_ub']), fmt(row['pool_ub_partial'])) +
                     ' | '.join(fmt(row[m]['cov']) for m in r['methods']) + ' |')
        L += ['', '| 类别 | ' + ' | '.join('{} 部分口径'.format(m) for m in r['methods']) + ' |', '|---' * (1 + len(r['methods'])) + '|']
        for cat, row in cats.items():
            L.append('| {} | '.format(cat) + ' | '.join(fmt(row[m]['cov_partial']) for m in r['methods']) + ' |')
        L += ['']
    src = r['evidence_source']
    L += ['### 证据来源', '', '候选池中各来源的候选数，以及其中完整支持（或至少部分支持）任一要点的条数：', '',
          '| 来源 | 候选数 | 完整支持任一要点 | 至少部分支持 |', '|---|---|---|---|']
    for s, row in src['pool'].items():
        L.append('| {} | {} | {} | {} |'.format(s, row['candidates'], row['full_any'], row['partial_any']))
    L += ['', '各方法所选证据的来源构成（括号内为该来源所选证据中完整支持任一要点的比例）：', '',
          '| 方法 | ' + ' | '.join(SOURCES) + ' |', '|---' * (1 + len(SOURCES)) + '|']
    for m in r['methods']:
        L.append('| {} | '.format(m) + ' | '.join('{:.1%}（{}）'.format(src[m][s]['share'], fmt(src[m][s]['full_rate'], 2))
                                                  if s in src[m] else '—' for s in SOURCES) + ' |')
    L += ['']

    if r['seeds']:
        L += ['## 7. 种子间差异', '', '| 方法 | 种子 | 覆盖率 | 部分口径 | 选入非支持率 | nDCG@10 |', '|---|---|---|---|---|---|']
        for m, s in r['seeds'].items():
            for seed, v in s['per_seed'].items():
                L.append('| {} | {} | {} | {} | {} | {} |'.format(m, seed, fmt(v['cov']), fmt(v['cov_partial']), fmt(v['nonsup']), fmt(v['ndcg'])))
            L.append('| {} | 标准差 | {} | {} | {} | {} |'.format(m, fmt(s['sd']['cov']), fmt(s['sd']['cov_partial']), fmt(s['sd']['nonsup']), fmt(s['sd']['ndcg'])))
        L += ['']

    L += ['## 8. 探索性分析（未预注册，不用于结论）', '']
    keys = [k for k, _, _ in METRICS[5:]]
    L += ['| 方法 | ' + ' | '.join(label[k] for k in keys) + ' |', '|---' * (len(keys) + 1) + '|']
    for m in r['methods']:
        e = r['estimates'][m]
        L.append('| {} | '.format(m) + ' | '.join('{} {}'.format(fmt(e[k]['mean']), fmt_ci(e[k]['ci95'])) for k in keys) + ' |')
    L += ['', '| 比较 | 指标 | 差值 | 95% CI（未校正） |', '|---|---|---|---|']
    for c in r['exploratory']:
        L.append('| Support-CE − {} | {} | {} | {} |'.format(c['vs'], label[c['metric']], fmt(c['diff']), fmt_ci(c['ci95'])))
    if r.get('X2'):
        c = r['X2']
        L += ['', '**X2 条件分析**：只看候选池中至少一个要点获得完整支持的 {} 道题。'.format(c['n_questions']), '',
              '| Support-CE | Relevance-CE | 差值 | 95% CI（未校正） |', '|---|---|---|---|',
              '| {} | {} | {} | {} |'.format(fmt(c['mean_a']), fmt(c['mean_b']), fmt(c['diff']), fmt_ci(c['ci95'])), '',
              '同一子集上各方法的覆盖率：' + '；'.join('{} {}'.format(m, fmt(v['mean'])) for m, v in r['X2_by_method'].items())]
    L += ['']

    aq = r.get('annotation_quality') or {}
    if aq:
        L += ['## 9. 标注质量', '']
        if 'phase1' in aq:
            L += ['**阶段1**（180 题；纳入 176 题，排除 4 题；要点由草稿 412 条修订为 547 条）', '',
                  '| 指标 | 比较 | κ/α | 95% CI | 原始一致率 |', '|---|---|---|---|---|']
            for meas in aq['phase1']['measures']:
                for pair, v in meas['pairs'].items():
                    L.append('| {} | {} | {} | {} | {} |'.format(meas['measure'], pair, fmt(v.get('kappa')), fmt_ci(v.get('ci95') or [None, None]),
                                                               fmt(v.get('percent_agreement'))))
                mu = meas.get('multi') or {}
                if mu.get('krippendorff_alpha') is not None:
                    L.append('| {} | Krippendorff α（{}，{}） | {} | {} | — |'.format(
                        meas['measure'], mu.get('alpha_level', ''), '/'.join(mu.get('raters', [])),
                        fmt(mu['krippendorff_alpha']), fmt_ci(mu.get('alpha_ci95') or [None, None])))
                if mu.get('fleiss_kappa') is not None:
                    L.append('| {} | Fleiss κ（{}） | {} | — | — |'.format(
                        meas['measure'], '/'.join(mu.get('raters', [])), fmt(mu['fleiss_kappa'])))
            L += ['', '| 标注员 | 草拟要点 | 保留 | 修改 | 删除 | 新增 | 排除题 |', '|---|---|---|---|---|---|---|']
            for who, v in aq['phase1'].get('draft_point_modification', {}).items():
                L.append('| {} | {} | {} | {} | {} | {} | {} |'.format(who, v['draft_points'], v['kept_unchanged'], v['modified'],
                                                                      v['deleted'], v['added_points'], v['questions_excluded']))
            L += ['']
        if 'phase2' in aq:
            L += ['**阶段2**', '', '| 指标 | 比较 | κ/α | 95% CI | 原始一致率 |', '|---|---|---|---|---|']
            for meas in aq['phase2']['measures']:
                for pair, v in meas['pairs'].items():
                    L.append('| {} | {} | {} | {} | {} |'.format(meas['measure'], pair, fmt(v.get('kappa')), fmt_ci(v.get('ci95') or [None, None]),
                                                               fmt(v.get('percent_agreement'))))
                mu = meas.get('multi') or {}
                if mu.get('krippendorff_alpha') is not None:
                    L.append('| {} | Krippendorff α（{}，{}） | {} | {} | — |'.format(
                        meas['measure'], mu.get('alpha_level', ''), '/'.join(mu.get('raters', [])),
                        fmt(mu['krippendorff_alpha']), fmt_ci(mu.get('alpha_ci95') or [None, None])))
                if mu.get('fleiss_kappa') is not None:
                    L.append('| {} | Fleiss κ（{}） | {} | — | — |'.format(
                        meas['measure'], '/'.join(mu.get('raters', [])), fmt(mu['fleiss_kappa'])))
            L += ['']
        if 'full_support_binary' in aq:
            L += ['**完整支持标签的补充统计**（应对正例稀少时的 κ 悖论）', '',
                  '| 比较 | 单元数 | 正例数(前/后) | 共同正例 | 原始一致率 | κ | PABAK | Gwet AC1 | 正一致率 | 负一致率 |',
                  '|---|---|---|---|---|---|---|---|---|---|']
            for pair, v in aq['full_support_binary'].items():
                L.append('| {} | {} | {}/{} | {} | {} | {} | {} | {} | {} | {} |'.format(
                    pair, v['n'], v['positives_a'], v['positives_b'], v['both_positive'], fmt(v['percent_agreement']),
                    fmt(v['kappa']), fmt(v['pabak']), fmt(v['gwet_ac1']), fmt(v['positive_agreement']), fmt(v['negative_agreement'])))
            L += ['']
        if 'phase2_arbitrated_candidates' in aq:
            L += ['- 阶段2 进入仲裁的候选：{} / 3520（{:.1%}）；其中涉及完整支持分歧的 {} 条。'.format(
                aq['phase2_arbitrated_candidates'], aq['phase2_arbitrated_candidates'] / 3520, aq['phase2_primary_disagreement_rows'])]
        if 'gold_vs_majority' in aq:
            g = aq['gold_vs_majority']
            L += ['- 有多数票的 {} 个“候选×要点”单元中，金标与多数票不同的有 {} 个；完整支持在多数票中有 {} 个，在金标中有 {} 个。'.format(
                g['cells_with_majority'], g['gold_differs'], g['majority_full'], g['gold_full_on_these'])]
        L += ['- 计划第 6 节可选项“AI 与专家金标的 κ”：本次金标由 AI 生成，不适用。', '']
    L += ['## 10. 说明', '',
          '- 所有方法都对同一 20 条候选池重排，取前 2 条；同分时按 BM25 名次靠前者优先。',
          '- 偏离与实现说明见 `eval_v1/实验对比方案.md` 第 8 节。',
          '- 每题每方法的数值见 `per_question.csv`，全部数值见 `results.json`。', '']
    return '\n'.join(L)


def freeze(a):
    files = {
        'final_questions.json': SEALED / 'final_questions.json',
        'sealed_pools.jsonl': SEALED / 'sealed_pools.jsonl',
        'phase2_gold_labels.jsonl': SEALED / 'phase2_gold_labels.jsonl',
        '预注册分析计划.md': TESTSET / '预注册分析计划.md',
        '标注规范_v1.1.md': TESTSET / '标注规范_v1.1.md',
        'eval_v1/实验对比方案.md': EVAL_DIR / '实验对比方案.md',
        'eval_v1/server_score.py': EVAL_DIR / 'server_score.py',
        'eval_v1/evaluate.py': EVAL_DIR / 'evaluate.py',
        'eval_v1/models.json': EVAL_DIR / 'models.json',
    }
    out = {'created': dt.datetime.now().astimezone().isoformat(timespec='seconds'),
           'purpose': '封存集打分之前的冻结记录；请与预注册计划、实验对比方案一起上传 OSF 或提交到带时间记录的 git 仓库。',
           'files': {}}
    for k, p in files.items():
        out['files'][k] = sha256_file(p) if p.exists() else '缺失'
    bad = [k for k in FROZEN if out['files'][k] != FROZEN[k]]
    if bad:
        sys.exit('冻结文件哈希不符：{}。'.format(', '.join(bad)))
    path = EVAL_DIR / 'FREEZE_MANIFEST.json'
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print('已写出 {}'.format(path))
    for k, v in out['files'].items():
        print('  {:<28} {}'.format(k, v))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--freeze', action='store_true', help='写出 FREEZE_MANIFEST.json 后退出')
    ap.add_argument('--target', default='sealed', choices=['sealed', 'calib'])
    ap.add_argument('--questions', default=None)
    ap.add_argument('--pools', default=None)
    ap.add_argument('--gold', default=None)
    ap.add_argument('--scores', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--reps', type=int, default=B)
    ap.add_argument('--seed', type=int, default=BOOT_SEED)
    a = ap.parse_args()
    if a.freeze:
        return freeze(a)
    if a.target == 'sealed':
        a.questions = a.questions or str(SEALED / 'final_questions.json')
        a.pools = a.pools or str(SEALED / 'sealed_pools.jsonl')
        a.gold = a.gold or str(SEALED / 'phase2_gold_labels.jsonl')
        if a.reps != B or a.seed != BOOT_SEED:
            print('注意：封存集评测应使用预先固定的 reps={}、seed={}。'.format(B, BOOT_SEED))
    else:
        a.questions = a.questions or str(EVAL_DIR / 'calib' / 'calibration_questions.json')
        a.pools = a.pools or str(EVAL_DIR / 'calib' / 'calibration_pools.jsonl')
        if not a.gold:
            sys.exit('--target calib 需要用 --gold 指定测试用金标。')
    a.scores = a.scores or str(EVAL_DIR / 'scores' / a.target)
    a.out = a.out or str(EVAL_DIR / 'results' / a.target)
    evaluate(a)


if __name__ == '__main__':
    main()
