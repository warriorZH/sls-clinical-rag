#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""服务器端快速汇总（不含任何证据金标）：各条件的要点召回、矛盾率、拒答率、断言忠实性、评审一致性，
以及预注册比较的配对 bootstrap 置信区间；另有外部验证集（TCM-BEST4SDT）的复现比较（方案第十六节）。
完整分析（含检索层指标和机制分析）由 Claude 在本机完成。

    python summarize.py [--job summary]
输出 results/summary_v2.md 与 results/summary_v2.json。
"""
import argparse
import math
import random
from collections import defaultdict

from common import HERE, RESULTS, RUNS, Log, load_config, load_target, mark_done, now, read_json, read_jsonl, write_json
import conditions as C

B = 10000
SEED = 20261026
DELTA = 0.05
SEEDED_GROUPS = ('G02', 'G04', 'G05', 'G06', 'G07', 'G10', 'G11')
ORDER = {'X': 0, '0': 1, '1': 2, '2': 3}
EXT_GROUPS = ('X00', 'X01', 'X02', 'X04', 'X05', 'X08', 'X09')
EXT_NAMES = {'X00': '闭卷', 'X01': 'BM25 Top-2', 'X02': '随机 2 条', 'X04': 'Relevance-CE（完整题干）',
             'X05': 'Support-CE（完整题干）', 'X08': 'bge-reranker-v2-m3', 'X09': 'Qwen3-Reranker-0.6B'}


def bootstrap(diffs, seed=SEED, b=B):
    n = len(diffs)
    if n == 0:
        return None
    rng = random.Random(seed)
    mean = sum(diffs) / n
    reps = []
    for _ in range(b):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        reps.append(s / n)
    reps.sort()

    def q(p):
        x = p * (b - 1)
        lo, hi = int(math.floor(x)), int(math.ceil(x))
        return reps[lo] + (reps[hi] - reps[lo]) * (x - lo)
    le = sum(1 for r in reps if r <= 0)
    ge = sum(1 for r in reps if r >= 0)
    p = min(1.0, 2 * min((le + 1) / (b + 1), (ge + 1) / (b + 1)))
    return {'n': n, 'diff': mean, 'ci95': [q(0.025), q(0.975)], 'ci90': [q(0.05), q(0.95)], 'p': p}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj = [None] * len(ps)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def kappa(pairs):
    if not pairs:
        return None
    cats = sorted({x for p in pairs for x in p})
    n = len(pairs)
    po = sum(1 for a, b in pairs if a == b) / n
    pa = {c: sum(1 for a, _ in pairs if a == c) / n for c in cats}
    pb = {c: sum(1 for _, b in pairs if b == c) / n for c in cats}
    pe = sum(pa[c] * pb[c] for c in cats)
    return {'n': n, 'agreement': po, 'kappa': (po - pe) / (1 - pe) if pe < 1 else None}


def kpr_row(f, n, r):
    labs = list(f['labels'].values())
    return {'kpr': sum(1 for x in labs if x == '2') / n,
            'kpr_graded': (sum(1 for x in labs if x == '2') + 0.5 * sum(1 for x in labs if x == '1')) / n,
            'contra': sum(1 for x in labs if x == 'X') / n, 'full': float(all(x == '2' for x in labs)),
            'abstain': float(bool(f.get('abstain'))), 'chars': len(r['answer']), 'n_out': r['n_out'],
            'hit_max': float(r['hit_max']), 'repeat': float(r['repeat'])}


def group_average(lst):
    """同一组的几个条件（种子、随机抽样）先在题内平均；只用所有条件都有结果的题。"""
    qids = set.intersection(*[set(x) for x in lst]) if lst else set()
    agg = {}
    for qid in qids:
        vals = [x[qid] for x in lst]
        agg[qid] = {k: sum(v[k] for v in vals) / len(vals) for k in vals[0] if all(k in v for v in vals)}
    return agg


def paired(gm, g1, g2, k, nb, qids=None):
    r1, r2 = gm.get(g1, {}), gm.get(g2, {})
    common = sorted(set(r1) & set(r2) & (set(qids) if qids is not None else set(r1)))
    d = [r1[q][k] - r2[q][k] for q in common if k in r1[q] and k in r2[q]]
    return bootstrap(d, b=nb) if d else None


def external_summary(cfg, nb):
    """外部验证（TCM-BEST4SDT）：复现 H3、H4、H7，并按病例来源与题型分层（描述性）。只用答案与评审结果。"""
    conds = C.conditions(cfg)['ext']
    if not conds:
        return None
    try:
        questions, _ = load_target(cfg, 'external', check_hash=False)
    except FileNotFoundError:
        return {'status': '外部验证集尚未生成'}
    qinfo = {q['qid']: q for q in questions}
    npts = {q['qid']: len(q['points']) for q in questions}
    final = {r['aid']: r for r in read_jsonl(RUNS / 'judge' / 'external' / 'kpr_final.jsonl')}
    progress, by_group, answers = [], defaultdict(list), []
    for c in conds:
        p = RUNS / 'gen' / 'external' / c['gen'] / '{}.jsonl'.format(c['id'])
        ans = [r for r in read_jsonl(p) if r['qid'] in npts]
        judged = [r for r in ans if r['aid'] in final]
        progress.append({'cond': c['id'], 'answers': len(ans), 'judged': len(judged),
                         'frozen': p.with_suffix('.done.json').exists()})
        rows = {r['qid']: kpr_row(final[r['aid']], npts[r['qid']], r) for r in judged}
        if rows:
            by_group[c['group']].append(rows)
        answers += [(c, r) for r in judged]
    gm = {g: group_average(lst) for g, lst in by_group.items()}
    table = []
    for g in EXT_GROUPS:
        rows = gm.get(g) or {}
        if rows:
            table.append({'group': g, 'n_q': len(rows), **{k: sum(v[k] for v in rows.values()) / len(rows)
                                                          for k in ('kpr', 'kpr_graded', 'contra', 'full', 'abstain',
                                                                    'chars')}})
    fams = {'外部验证：主比较（复现 H3）': [('X05', 'X04')],
            '外部验证：答案层比较（复现 H4）': [('X05', 'X01'), ('X05', 'X02'), ('X05', 'X00')]}
    comps = {}
    for fam, items in fams.items():
        res = []
        for g1, g2 in items:
            r = paired(gm, g1, g2, 'kpr', nb)
            if r:
                res.append({'a': g1, 'b': g2, 'metric': 'kpr', **r})
        for x, h in zip(res, holm([x['p'] for x in res]) if res else []):
            x['p_holm'] = h if len(res) > 1 else x['p']
        comps[fam] = res
    equiv = []
    for g2 in ('X08', 'X09'):
        r = paired(gm, 'X05', g2, 'kpr', nb)
        if r:
            equiv.append({'a': 'X05', 'b': g2, 'delta': DELTA, 'ci90': r['ci90'],
                          'equivalent': r['ci90'][0] > -DELTA and r['ci90'][1] < DELTA})
    strata = []
    # CE（B 编码）输入总长 512：病例很长时留给证据的词元很少。按 CE 打分时记录的题干词元数算出每题的证据预算
    budget = {}
    for r in read_jsonl(RUNS / 'scores' / 'external' / 'support_ce_B_s42.jsonl'):
        if 'q_tokens' in r:
            budget[r['qid']] = 509 - int(r['q_tokens'])
    subsets = [('名家医案改写（编号 1–150）', lambda q: q.get('stratum') == 'classical'),
               ('现代临床病例（编号 151–300）', lambda q: q.get('stratum') == 'modern'),
               ('诊断题', lambda q: q.get('question_type') == 'diagnosis'),
               ('治则题', lambda q: q.get('question_type') == 'principle'),
               ('题干无提示性词句', lambda q: not q.get('leak_flags')),
               ('CE 输入中证据至少 64 个词元', lambda q: budget.get(q['qid'], -1) >= 64)]
    for name, fn in subsets:
        qs = {qid for qid, q in qinfo.items() if fn(q)}
        for g1, g2 in (('X05', 'X04'), ('X05', 'X01'), ('X05', 'X02'), ('X05', 'X00'), ('X05', 'X08'), ('X05', 'X09')):
            r = paired(gm, g1, g2, 'kpr', nb, qs)
            if r:
                strata.append({'subset': name, 'a': g1, 'b': g2, **r})
    pairs4, pairs2 = [], []
    sens_rows = {v: defaultdict(lambda: defaultdict(list)) for v in ('A', 'B', 'min')}
    for c, r in answers:
        f = final[r['aid']]
        la, lb = f.get('labels_A') or {}, f.get('labels_B')
        if not lb:
            continue
        for pid, x in la.items():
            if pid in lb:
                pairs4.append((x, lb[pid]))
                pairs2.append((x == '2', lb[pid] == '2'))
        n = npts[r['qid']]
        variants = {'A': la, 'B': lb, 'min': {p: min(la[p], lb.get(p, '0'), key=lambda y: ORDER[y]) for p in la}}
        for v, labs in variants.items():
            sens_rows[v][c['group']][r['qid']].append(sum(1 for y in labs.values() if y == '2') / n)
    sensitivity = []
    for g2 in ('X04', 'X01', 'X02', 'X00', 'X08', 'X09'):
        for v in ('A', 'B', 'min'):
            r1, r2 = sens_rows[v].get('X05', {}), sens_rows[v].get(g2, {})
            common = sorted(set(r1) & set(r2))
            d = [sum(r1[q]) / len(r1[q]) - sum(r2[q]) / len(r2[q]) for q in common]
            r = bootstrap(d, b=nb) if d else None
            if r:
                sensitivity.append({'a': 'X05', 'b': g2, 'labels': v, **r})
    man = read_json(RUNS / 'external' / 'EXT_MANIFEST.json') or {}
    return {'progress': progress, 'n_questions': len(questions), 'table': table, 'comparisons': comps,
            'equivalence': equiv, 'strata': strata, 'agreement': {'kpr_4class': kappa(pairs4), 'kpr_hit': kappa(pairs2)},
            'sensitivity_B': sensitivity,
            'build': {k: man.get(k) for k in ('questions_all', 'questions_included', 'excluded', 'by_stratum_and_type',
                                               'own_record_filter')}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', default='')
    ap.add_argument('--quick', action='store_true', help='中途查看用：bootstrap 只抽 2000 次')
    a = ap.parse_args()
    nb = 2000 if a.quick else B
    cfg = load_config()
    log = Log(RUNS / 'logs' / 'summary.log')
    questions, _ = load_target(cfg, 'sealed', check_hash=False)
    npts = {q['qid']: len(q['points']) for q in questions}
    allc = [c for c in C.all_conditions(cfg) if c['target'] == 'sealed']
    final = {r['aid']: r for r in read_jsonl(RUNS / 'judge' / 'sealed' / 'kpr_final.jsonl')}
    split = {r['aid']: r for r in read_jsonl(RUNS / 'judge' / 'sealed' / 'scr_split_A.jsonl') if r.get('valid')}
    verA = {r['aid']: r for r in read_jsonl(RUNS / 'judge' / 'sealed' / 'scr_verify_A.jsonl') if r.get('valid')}
    verB = {r['aid']: r for r in read_jsonl(RUNS / 'judge' / 'sealed' / 'scr_verify_B.jsonl') if r.get('valid')}

    per_cond = {}
    progress = []
    for c in allc:
        p = RUNS / 'gen' / 'sealed' / c['gen'] / '{}.jsonl'.format(c['id'])
        ans = read_jsonl(p)
        judged = [r for r in ans if r['aid'] in final]
        progress.append({'cond': c['id'], 'set': c['set'], 'answers': len(ans), 'judged': len(judged),
                         'frozen': p.with_suffix('.done.json').exists()})
        if not judged:
            continue
        rows = {}
        for r in judged:
            f = final[r['aid']]
            labs = list(f['labels'].values())
            n = npts[r['qid']]
            row = {'kpr': sum(1 for x in labs if x == '2') / n,
                   'kpr_graded': (sum(1 for x in labs if x == '2') + 0.5 * sum(1 for x in labs if x == '1')) / n,
                   'contra': sum(1 for x in labs if x == 'X') / n, 'full': float(all(x == '2' for x in labs)),
                   'abstain': float(bool(f.get('abstain'))), 'chars': len(r['answer']), 'n_out': r['n_out'],
                   'hit_max': float(r['hit_max']), 'repeat': float(r['repeat'])}
            sp, va = split.get(r['aid']), verA.get(r['aid'])
            if c.get('scr') and sp and va:
                labsA = [lab if (lab == 'N' or va.get('quote_ok', {}).get(cid, True)) else 'N'
                         for cid, lab in va['labels'].items()]
                if labsA:
                    row['scr'] = sum(1 for x in labsA if x == 'S') / len(labsA)
                    row['ghr'] = sum(1 for x in labsA if x in ('C', 'N')) / len(labsA)
                    row['c_rate'] = sum(1 for x in labsA if x == 'C') / len(labsA)
                    row['n_claims'] = len(labsA)
            rows[r['qid']] = row
        per_cond[c['id']] = {'cond': c, 'rows': rows}

    # 组级（种子、随机抽样取题内平均）
    groups = defaultdict(list)
    for cid, v in per_cond.items():
        c = v['cond']
        key = c['group'] if c['set'] in ('main', 'oracle', 'e2') else cid
        if c['set'] == 'main' or c['set'] in ('oracle', 'e2'):
            groups[key].append(v['rows'])
    gmetrics = {}
    for g, lst in groups.items():
        qids = set.intersection(*[set(x) for x in lst]) if lst else set()
        agg = {}
        for qid in qids:
            vals = [x[qid] for x in lst]
            agg[qid] = {k: sum(v[k] for v in vals if k in v) / max(1, sum(1 for v in vals if k in v))
                        for k in vals[0] if all(k in v for v in vals)}
        gmetrics[g] = agg

    def mean(g, k, qids=None):
        rows = gmetrics.get(g, {})
        xs = [rows[q][k] for q in (qids or rows) if q in rows and k in rows[q]]
        return sum(xs) / len(xs) if xs else None

    table = []
    for g in ['G00', 'G01', 'G02', 'G03', 'G03b', 'G04', 'G05', 'G06', 'G07', 'G08', 'G09', 'G10', 'G11', 'G12',
              'G13', 'O1', 'O2', 'O3', 'O4']:
        if g in gmetrics and gmetrics[g]:
            table.append({'group': g, 'n_q': len(gmetrics[g]),
                          **{k: mean(g, k) for k in ('kpr', 'kpr_graded', 'contra', 'full', 'abstain', 'scr', 'ghr',
                                                     'c_rate', 'chars', 'n_out', 'hit_max', 'repeat')}})

    def diff(g1, g2, k, qids=None):
        r1, r2 = gmetrics.get(g1, {}), gmetrics.get(g2, {})
        common = sorted(set(r1) & set(r2) & (set(qids) if qids else set(r1)))
        d = [r1[q][k] - r2[q][k] for q in common if k in r1[q] and k in r2[q]]
        return bootstrap(d, b=nb) if d else None

    oracle = defaultdict(set)
    for r in read_jsonl(HERE / 'data' / 'oracle_lists.jsonl'):
        oracle[r['list']].add(r['qid'])
    sub70 = sorted(oracle['O1'] & oracle['O2'] & oracle['O3'])
    families = {
        '主假设 H3': [('G05', 'G04', 'kpr', None)],
        '答案层比较（H4、H7）': [('G05', 'G01', 'kpr', None), ('G05', 'G02', 'kpr', None), ('G05', 'G00', 'kpr', None),
                          ('G05', 'G08', 'kpr', None), ('G05', 'G09', 'kpr', None)],
        '忠实性（H5）': [('G05', 'G04', 'scr', None), ('G05', 'G01', 'scr', None), ('G05', 'G04', 'ghr', None),
                     ('G05', 'G01', 'ghr', None)],
        '机制（H6）': [('O1', 'O2', 'kpr', sub70), ('O2', 'G00', 'kpr', None), ('O1', 'G00', 'kpr', None),
                    ('O3', 'G00', 'kpr', None)],
    }
    comps = {}
    for fam, items in families.items():
        res = []
        for g1, g2, k, qs in items:
            r = diff(g1, g2, k, qs)
            if r:
                res.append({'a': g1, 'b': g2, 'metric': k, **r})
        ps = [x['p'] for x in res]
        for x, h in zip(res, holm(ps) if ps else []):
            x['p_holm'] = h if len(res) > 1 else x['p']
        comps[fam] = res
    equiv = []
    for g2 in ('G08', 'G09'):
        r = diff('G05', g2, 'kpr')
        if r:
            equiv.append({'a': 'G05', 'b': g2, 'delta': DELTA, 'ci90': r['ci90'],
                          'equivalent': r['ci90'][0] > -DELTA and r['ci90'][1] < DELTA})

    # 评审一致性
    pairs4, pairs2 = [], []
    for f in final.values():
        for pid, la in (f.get('labels_A') or {}).items():
            lb = (f.get('labels_B') or {}).get(pid)
            if lb is not None:
                pairs4.append((la, lb))
                pairs2.append((la == '2', lb == '2'))
    scr_pairs = []
    for aid, vb in verB.items():
        va = verA.get(aid)
        if va:
            for cid, lb in vb['labels'].items():
                if cid in va['labels']:
                    scr_pairs.append((va['labels'][cid], lb))
    agreement = {'kpr_4class': kappa(pairs4), 'kpr_hit': kappa(pairs2), 'scr': kappa(scr_pairs)}

    # 评审 B 子集上的敏感性分析：同一批题目，分别用 A、B 以及两者取低（min）的标签重算主要比较
    order = {'X': 0, '0': 1, '1': 2, '2': 3}
    sens_rows = {v: defaultdict(lambda: defaultdict(list)) for v in ('A', 'B', 'min')}
    for c in allc:
        if c['set'] != 'main':
            continue
        for r in read_jsonl(RUNS / 'gen' / 'sealed' / c['gen'] / '{}.jsonl'.format(c['id'])):
            f = final.get(r['aid'])
            if not f or not f.get('labels_B'):
                continue
            la, lb = f['labels_A'], f['labels_B']
            n = npts[r['qid']]
            variants = {'A': la, 'B': lb,
                        'min': {p: min(la[p], lb.get(p, '0'), key=lambda x: order[x]) for p in la}}
            for v, labs in variants.items():
                sens_rows[v][c['group']][r['qid']].append(sum(1 for x in labs.values() if x == '2') / n)
    sensitivity = []
    for g1, g2 in (('G05', 'G04'), ('G05', 'G01'), ('G05', 'G02'), ('G05', 'G00'), ('G05', 'G08'), ('G05', 'G09')):
        for v in ('A', 'B', 'min'):
            r1, r2 = sens_rows[v].get(g1, {}), sens_rows[v].get(g2, {})
            common = sorted(set(r1) & set(r2))
            d = [sum(r1[q]) / len(r1[q]) - sum(r2[q]) / len(r2[q]) for q in common]
            r = bootstrap(d, b=nb) if d else None
            if r:
                sensitivity.append({'a': g1, 'b': g2, 'labels': v, **r})

    out = {'time': now(), 'bootstrap_B': nb, 'progress': progress, 'table': table, 'comparisons': comps, 'equivalence': equiv,
           'agreement': agreement, 'sensitivity_B': sensitivity, 'subset_O_n': len(sub70),
           'external': external_summary(cfg, nb)}
    write_json(RESULTS / 'summary_v2.json', out)
    write_md(out)
    log('汇总已写出：{}'.format(RESULTS / 'summary_v2.md'))
    if a.job:
        mark_done(a.job, {'groups': len(table)})


def f3(x):
    return '—' if x is None else '{:.3f}'.format(x)


def write_md(o):
    names = {'G00': '闭卷', 'G01': 'BM25 Top-2', 'G02': '随机 2 条', 'G03': '全池 20 条', 'G03b': '全池（Support-CE 排序）',
             'G04': 'Relevance-CE（完整题干）', 'G05': 'Support-CE（完整题干）', 'G06': 'Relevance-CE（160 词元）',
             'G07': 'Support-CE（160 词元）', 'G08': 'bge-reranker-v2-m3', 'G09': 'Qwen3-Reranker-0.6B',
             'G10': 'bge 支持标签微调', 'G11': 'bge 相关性标签微调', 'G12': 'LLM 零样本判断', 'G13': 'Qwen3-Reranker-4B',
             'O1': '金标：支持证据', 'O2': '金标：相关但不支持', 'O3': '金标：无关', 'O4': '金标：条件不符'}
    L = ['# eval_v2 服务器端汇总', '', '生成时间：{}。本摘要只用答案与评审结果，不含证据金标；检索层指标和机制分析由 Claude 补充。'.format(o['time']), '']
    done = sum(1 for p in o['progress'] if p['frozen'])
    L += ['## 进度', '', '答案已冻结的条件 {}/{}；已评审答案 {} 条。'.format(
        done, len(o['progress']), sum(p['judged'] for p in o['progress'])), '']
    L += ['## 各条件结果（Qwen3-4B-Instruct-2507，k=2，主提示词；种子与随机抽样已在题内平均）', '',
          '| 条件 | 题数 | 严格要点召回 | 分级召回 | 矛盾率 | 全对率 | 拒答率 | 支持断言率 | 无依据断言率 | 平均字数 |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for r in o['table']:
        L.append('| {} {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            r['group'], names.get(r['group'], ''), r['n_q'], f3(r['kpr']), f3(r['kpr_graded']), f3(r['contra']),
            f3(r['full']), f3(r['abstain']), f3(r['scr']), f3(r['ghr']),
            '—' if r['chars'] is None else '{:.0f}'.format(r['chars'])))
    L += ['', '## 预注册比较（配对 bootstrap {} 次，族内 Holm 校正）'.format(o['bootstrap_B']), '',
          '| 族 | 比较 | 指标 | 题数 | 差值 | 95% CI | p（Holm） |', '|---|---|---|---|---|---|---|']
    for fam, rows in o['comparisons'].items():
        for x in rows:
            L.append('| {} | {} − {} | {} | {} | {:+.3f} | [{:+.3f}, {:+.3f}] | {:.4f} |'.format(
                fam, x['a'], x['b'], x['metric'], x['n'], x['diff'], x['ci95'][0], x['ci95'][1], x['p_holm']))
    if o['equivalence']:
        L += ['', '## 等效检验（H7，界值 ±0.05，90% CI）', '']
        for e in o['equivalence']:
            L.append('- {} − {}：90% CI [{:+.3f}, {:+.3f}] → {}'.format(e['a'], e['b'], e['ci90'][0], e['ci90'][1],
                                                                       '等效' if e['equivalent'] else '不能判定等效'))
    ag = o['agreement']
    L += ['', '## 评审一致性（评审 A 与评审 B，B 只评约 25% 的题目）', '']
    if not any(ag.get(k) for k in ('kpr_4class', 'kpr_hit', 'scr')):
        L.append('- 暂无：评审 B 尚未完成或在 E0 中不可用。')
    for k, label in (('kpr_4class', '要点四分类'), ('kpr_hit', '是否命中'), ('scr', '断言核对')):
        v = ag.get(k)
        if v:
            L.append('- {}：{} 对，一致率 {:.3f}，κ = {}'.format(label, v['n'], v['agreement'],
                                                          '—' if v['kappa'] is None else '{:.3f}'.format(v['kappa'])))
    if o.get('sensitivity_B'):
        L += ['', '## 敏感性：在评审 B 的题目子集上换用不同评审的标签（严格要点召回差值，95% CI）', '',
              '| 比较 | 题数 | 评审 A | 评审 B | 两者取低 |', '|---|---|---|---|---|']
        by = defaultdict(dict)
        for x in o['sensitivity_B']:
            by[(x['a'], x['b'])][x['labels']] = x
        for (a, b), d in by.items():
            cell = lambda v: '—' if v not in d else '{:+.3f} [{:+.3f}, {:+.3f}]'.format(  # noqa: E731
                d[v]['diff'], d[v]['ci95'][0], d[v]['ci95'][1])
            n = max(x['n'] for x in d.values())
            L.append('| {} − {} | {} | {} | {} | {} |'.format(a, b, n, cell('A'), cell('B'), cell('min')))
    L += external_md(o.get('external'))
    (RESULTS / 'summary_v2.md').write_text('\n'.join(L) + '\n', encoding='utf-8')


def ci(x):
    return '{:+.3f} [{:+.3f}, {:+.3f}]'.format(x['diff'], x['ci95'][0], x['ci95'][1])


def external_md(e):
    if not e:
        return []
    L = ['', '## 外部验证：TCM-BEST4SDT（方案第十六节）', '']
    if e.get('status'):
        return L + ['- ' + e['status']]
    b = e.get('build') or {}
    if b.get('questions_all'):
        L.append('- 病例 {} 例，纳入 {} 例（排除与 CE 训练病例或集内近重复的 {} 例）；候选池中按规则排除病例自身原始记录 {} 条。'.format(
            b['questions_all'], b['questions_included'], len(b.get('excluded') or {}),
            sum(((b.get('own_record_filter') or {}).get('candidates_excluded') or {}).values())))
    done = sum(1 for p in e['progress'] if p['frozen'])
    L.append('- 答案已完成的条件 {}/{}；已评审答案 {} 条。参考要点来自数据集专家的标准答案，只报告要点召回。'.format(
        done, len(e['progress']), sum(p['judged'] for p in e['progress'])))
    if e['table']:
        L += ['', '| 条件 | 题数 | 严格要点召回 | 分级召回 | 矛盾率 | 全对率 | 拒答率 | 平均字数 |', '|---|---|---|---|---|---|---|---|']
        for r in e['table']:
            L.append('| {} {} | {} | {} | {} | {} | {} | {} | {:.0f} |'.format(
                r['group'], EXT_NAMES.get(r['group'], ''), r['n_q'], f3(r['kpr']), f3(r['kpr_graded']), f3(r['contra']),
                f3(r['full']), f3(r['abstain']), r['chars']))
    rows = [(fam, x) for fam, xs in e['comparisons'].items() for x in xs]
    if rows:
        L += ['', '| 族 | 比较 | 题数 | 差值 | 95% CI | p（Holm） |', '|---|---|---|---|---|---|']
        for fam, x in rows:
            L.append('| {} | {} − {} | {} | {:+.3f} | [{:+.3f}, {:+.3f}] | {:.4f} |'.format(
                fam, x['a'], x['b'], x['n'], x['diff'], x['ci95'][0], x['ci95'][1], x['p_holm']))
    for q in e['equivalence']:
        L.append('- 等效（复现 H7）{} − {}：90% CI [{:+.3f}, {:+.3f}] → {}'.format(
            q['a'], q['b'], q['ci90'][0], q['ci90'][1], '等效' if q['equivalent'] else '不能判定等效'))
    if e['strata']:
        gs = ('X04', 'X01', 'X02', 'X00', 'X08', 'X09')
        L += ['', '分层与敏感性（严格要点召回差值与 95% CI；分层为描述性）：', '',
              '| 子集 | 题数 | ' + ' | '.join('X05 − ' + g for g in gs) + ' |', '|---|---|' + '---|' * len(gs)]
        by = defaultdict(dict)
        for x in e['strata']:
            by[x['subset']][x['b']] = x
        for name, d in by.items():
            L.append('| {} | {} | {} |'.format(name, max(x['n'] for x in d.values()),
                                              ' | '.join(ci(d[g]) if g in d else '—' for g in gs)))
    ag = e.get('agreement') or {}
    for k, label in (('kpr_4class', '要点四分类'), ('kpr_hit', '是否命中')):
        v = ag.get(k)
        if v:
            L.append('- 评审 A 与 B 一致性（{}）：{} 对，一致率 {:.3f}，κ = {}'.format(
                label, v['n'], v['agreement'], '—' if v['kappa'] is None else '{:.3f}'.format(v['kappa'])))
    if e.get('sensitivity_B'):
        L += ['', '| 比较（评审 B 子集） | 题数 | 评审 A | 评审 B | 两者取低 |', '|---|---|---|---|---|']
        by = defaultdict(dict)
        for x in e['sensitivity_B']:
            by[(x['a'], x['b'])][x['labels']] = x
        for (a_, b_), d in by.items():
            L.append('| {} − {} | {} | {} | {} | {} |'.format(a_, b_, max(x['n'] for x in d.values()),
                                                           *[ci(d[v]) if v in d else '—' for v in ('A', 'B', 'min')]))
    return L


if __name__ == '__main__':
    main()
