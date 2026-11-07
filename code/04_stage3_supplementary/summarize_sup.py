#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补充分析汇总（补充登记第五、六节）：S1 官方选择题、S2 官方开放题总分、S3 Qwen3-32B 重评要点召回。

统计方法与 eval_v2 相同：按题配对 bootstrap 10,000 次（种子 20261026），95% 百分位 CI；
族内 Holm 校正；等效用 90% CI 与 ±0.05 比较。种子与随机抽样先在题内平均（同一组的各次运行都有结果的题才纳入）。
"""
import json
import random
import re
from collections import defaultdict

from common_sup import CFG, DATA, LOGS, RESULTS, WORK, log, read_json, read_jsonl, write_json

B = int(CFG['stats']['bootstrap_reps'])
SEED = int(CFG['stats']['bootstrap_seed'])
DELTA = float(CFG['stats']['equivalence_delta'])
MC_DIMS = ['证型', '病性', '病位', '治则治法']
# 官方 SDT 总分所用的 14 个维度（skip_think 时；顺序与官方 tcm_benchmark.py 相同）
SDT_DIMS = ['证型', '病性', '病位', '治则治法', '病因', '病机', '方证契合度', '方剂配伍规律', '配伍禁忌', '药材安全性分析',
            '妊娠禁忌', '煎服方法', '注意事项', '随症加减']


def base(c):
    return re.sub(r'_(s\d+|r\d+)$', '', c)


def boot(va, vb, tag):
    qs = sorted(set(va) & set(vb))
    d = [round(va[q] - vb[q], 10) for q in qs]      # 去掉浮点舍入噪声（种子平均时产生的 1e-17 级差值）
    n = len(d)
    if n < 2:
        return None
    rng = random.Random('{}|{}|{}'.format(SEED, tag, n))
    dist = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    pct = lambda p: dist[int(round(p * (B - 1)))]
    le = sum(1 for x in dist if x <= 0)
    ge = sum(1 for x in dist if x >= 0)
    return {'n': n, 'diff': sum(d) / n, 'ci95': [pct(.025), pct(.975)], 'ci90': [pct(.05), pct(.95)],
            'p': min(1.0, 2 * min((le + 1) / (B + 1), (ge + 1) / (B + 1)))}


def holm(ps):
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    out, run = [0.0] * m, 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[i]))
        out[i] = run
    return out


def group_means(per_run):
    """per_run: {cond: {qid: value}} → {组: {qid: 题内平均}}，只保留该组每次运行都有值的题。"""
    groups = defaultdict(list)
    for c in per_run:
        groups[base(c)].append(c)
    out = {}
    for g, cs in groups.items():
        qs = set.intersection(*[set(per_run[c]) for c in cs])
        out[g] = {q: sum(per_run[c][q] for c in cs) / len(cs) for q in qs}
    return out


def family_tests(G, prefix, closed, a, ones, b_main, equiv):
    """主比较（单独检验）、对照族（Holm）、等效（TOST）。"""
    res = {}
    r = boot(G.get(a, {}), G.get(b_main, {}), prefix + 'main')
    if r:
        res['main'] = dict(r, a=a, b=b_main, supported=r['ci95'][0] > 0)
    fam = []
    for b in ones + [closed]:
        r = boot(G.get(a, {}), G.get(b, {}), prefix + b)
        if r:
            fam.append(dict(r, a=a, b=b))
    for x, h in zip(fam, holm([x['p'] for x in fam])):
        x['p_holm'] = h
    res['family'] = fam
    eq = []
    for b in equiv:
        r = boot(G.get(a, {}), G.get(b, {}), prefix + 'eq' + b)
        if r:
            eq.append(dict(r, a=a, b=b, equivalent=r['ci90'][0] > -DELTA and r['ci90'][1] < DELTA))
    res['equivalence'] = eq
    return res


def official_part():
    recs = {}
    for r in read_jsonl(RESULTS / 'official' / 'official_results.jsonl'):
        recs[(r['cond'], r['qid'])] = r            # 以最后一条为准
    if not recs:
        return None
    out = {'n_records': len(recs), 'status': defaultdict(int), 'audit': defaultdict(int)}
    for r in recs.values():
        out['status'][r.get('status', '?')] += 1
        for k, v in (r.get('audit_summary') or {}).items():
            out['audit'][k] += v
        for run in ((r.get('multiple_choice') or {}).get('runs') or []):
            pa = run.get('parsed_answers') or {}
            if any(not pa.get(d) for d in MC_DIMS):
                out['audit']['mc_runs_with_unparsed_dim'] += 1
            out['audit']['mc_runs'] += 1

    def per_run(metric, drop_failed=False):
        pr = defaultdict(dict)
        for (c, q), r in recs.items():
            if drop_failed and r.get('status') != 'ok':
                continue
            ds = r.get('dimension_scores') or {}
            if metric == 'mc':
                v = sum(float(ds.get(d, 0.0)) for d in MC_DIMS) / len(MC_DIMS)
            elif metric == 'sdt':
                v = sum(float(ds.get(d, 0.0)) for d in SDT_DIMS) / len(SDT_DIMS)
            else:
                v = float(ds.get(metric, 0.0))
            pr[c][q] = v
        return group_means(pr)

    ones = ['X01', 'X02']
    res = {}
    for metric, tag in (('mc', 'S1'), ('sdt', 'S2')):
        G = per_run(metric)
        res[tag] = {'means': {g: (sum(v.values()) / len(v), len(v)) for g, v in G.items() if v},
                    'tests': family_tests(G, tag, 'X00', 'X05', ones, 'X04', ['X08', 'X09'])}
        Gs = per_run(metric, drop_failed=True)
        res[tag]['tests_excluding_failed'] = family_tests(Gs, tag + 'x', 'X00', 'X05', ones, 'X04', ['X08', 'X09'])
    res['dimensions'] = {}
    for d in SDT_DIMS:
        G = per_run(d)
        res['dimensions'][d] = {g: sum(v.values()) / len(v) for g, v in G.items() if v}
    out['results'] = res
    out['status'] = dict(out['status'])
    out['audit'] = dict(out['audit'])
    return out


def kappa(pairs):
    if not pairs:
        return None
    n = len(pairs)
    cats = sorted(set(a for a, _ in pairs) | set(b for _, b in pairs))
    po = sum(1 for a, b in pairs if a == b) / n
    pa = {c: sum(1 for a, _ in pairs if a == c) / n for c in cats}
    pb = {c: sum(1 for _, b in pairs if b == c) / n for c in cats}
    pe = sum(pa[c] * pb[c] for c in cats)
    return {'n': n, 'agree': po, 'kappa': (po - pe) / (1 - pe) if pe < 1 else None}


def effective(labels, qok):
    return {p: ('0' if l in ('2', '1') and not (qok or {}).get(p, True) else l) for p, l in labels.items()}


def rejudge_part():
    rows = {}
    for r in read_jsonl(RESULTS / 'rejudge' / 'kpr_qwen3_32b.jsonl'):
        if r.get('valid') or r['aid'] not in rows:
            rows[r['aid']] = r
    if not rows:
        return None
    npts = {}
    for q in read_json(DATA / 'rejudge' / 'final_questions.json'):
        npts[('sealed', q['qid'])] = len(q['points'])
    for q in read_json(DATA / 'external_questions.json'):
        npts[('external', q['qid'])] = len(q['points'])
    orig = {}
    for t in ('sealed', 'external'):
        for r in read_jsonl(DATA / 'rejudge' / 'kpr_final_{}.jsonl'.format(t)):
            orig[r['aid']] = dict(r, target=t)

    def kpr_of(labels, abstain, n):
        return 0.0 if abstain else sum(1 for v in labels.values() if v == '2') / n

    per = {'32B': {'sealed': defaultdict(dict), 'external': defaultdict(dict)},
           'A': {'sealed': defaultdict(dict), 'external': defaultdict(dict)}}
    pairs4, pairs2, pairsB4 = [], [], []
    n_valid = 0
    for aid, r in rows.items():
        if not r.get('valid'):
            continue
        n_valid += 1
        t = r['target']
        if r.get('gen', 'G1') != 'G1' and t == 'sealed':
            cond_key = r['cond']                 # G2 条件名本身带 G2_ 前缀
        else:
            cond_key = r['cond']
        n = npts[(t, r['qid'])]
        lab32 = effective(r['labels'], r.get('quote_ok'))
        per['32B'][t][cond_key][r['qid']] = kpr_of(lab32, r.get('abstain'), n)
        o = orig.get(aid)
        if o and o.get('labels') is not None:
            per['A'][t][cond_key][r['qid']] = kpr_of(o['labels'], o.get('abstain'), n)
            for p, la in o['labels'].items():
                if p in lab32:
                    pairs4.append((la, lab32[p]))
                    pairs2.append((la == '2', lab32[p] == '2'))
            for p, lb in (o.get('labels_B') or {}).items():
                if p in lab32:
                    pairsB4.append((lb, lab32[p]))
    out = {'n_answers': len(rows), 'n_valid': n_valid,
           'agreement': {'A_vs_32B_4class': kappa(pairs4), 'A_vs_32B_hit': kappa(pairs2), 'B_vs_32B_4class': kappa(pairsB4)},
           'label_dist_32B': dict(sorted(((k, v) for k, v in
                                          __import__('collections').Counter(b for _, b in pairs4).items()))),
           'results': {}}
    for judge in ('32B', 'A'):
        GS = group_means(per[judge]['sealed'])
        GE = group_means(per[judge]['external'])
        out['results'][judge] = {
            'sealed_means': {g: (sum(v.values()) / len(v), len(v)) for g, v in GS.items() if v},
            'external_means': {g: (sum(v.values()) / len(v), len(v)) for g, v in GE.items() if v},
            'sealed_tests': family_tests(GS, 'S3s' + judge, 'G00', 'G05', ['G01', 'G02', 'G08', 'G09'], 'G04', ['G08', 'G09']),
            'external_tests': family_tests(GE, 'S3e' + judge, 'X00', 'X05', ['X01', 'X02'], 'X04', ['X08', 'X09']),
        }
    return out


def fmt_test(x):
    if not x:
        return '—'
    s = '{:+.3f} [{:+.3f}, {:+.3f}]'.format(x['diff'], *x['ci95'])
    if 'p_holm' in x:
        s += '，Holm p = {:.4f}'.format(x['p_holm'])
    return s + '（{} 题）'.format(x['n'])


def write_md(summary):
    L = ['# eval_v2 补充分析结果（v2.4-sup）', '', '生成时间：{}。'.format(summary['time']), '']
    off = summary.get('official')
    if off:
        L += ['## S1、S2：TCM-BEST4SDT 官方评测（外部集 300 例）', '',
              '记录 {} 条；状态：{}。审计：{}。'.format(off['n_records'], json.dumps(off['status'], ensure_ascii=False),
                                               json.dumps(off['audit'], ensure_ascii=False)), '']
        for tag, name in (('S1', '官方选择题均分（证型、病性、病位、治则治法）'), ('S2', '官方 SDT 总分（14 个维度平均）')):
            r = off['results'][tag]
            L += ['### {} {}'.format(tag, name), '', '| 条件 | 均分 | 题数 |', '|---|---|---|']
            for g in ['X00', 'X01', 'X02', 'X04', 'X05', 'X08', 'X09']:
                if g in r['means']:
                    L.append('| {} | {:.3f} | {} |'.format(g, *r['means'][g]))
            t = r['tests']
            L += ['', '- 主比较 X05 − X04：{}，{}'.format(fmt_test(t.get('main')), '成立' if (t.get('main') or {}).get('supported') else '不成立')]
            for x in t['family']:
                L.append('- X05 − {}：{}'.format(x['b'], fmt_test(x)))
            for x in t['equivalence']:
                L.append('- 等效 X05 − {}：90% CI [{:+.3f}, {:+.3f}] → {}'.format(x['b'], *x['ci90'], '等效' if x['equivalent'] else '不能判定等效'))
            L.append('')
        L += ['### 各维度均分', '', '| 维度 | ' + ' | '.join(['X00', 'X01', 'X02', 'X04', 'X05', 'X08', 'X09']) + ' |',
              '|---|' + '---|' * 7]
        for d, m in off['results']['dimensions'].items():
            L.append('| {} | '.format(d) + ' | '.join('{:.3f}'.format(m[g]) if g in m else '—'
                                                       for g in ['X00', 'X01', 'X02', 'X04', 'X05', 'X08', 'X09']) + ' |')
        L.append('')
    rj = summary.get('rejudge')
    if rj:
        ag = rj['agreement']
        L += ['## S3：Qwen3-32B 重评要点召回', '', '答案 {} 条，有效 {} 条。'.format(rj['n_answers'], rj['n_valid']),
              '一致性：评审 A 与 32B（四分类）κ = {:.3f}；是否命中 κ = {:.3f}；评审 B 与 32B（四分类）κ = {}。'.format(
                  ag['A_vs_32B_4class']['kappa'], ag['A_vs_32B_hit']['kappa'],
                  '{:.3f}'.format(ag['B_vs_32B_4class']['kappa']) if ag['B_vs_32B_4class'] else '—'),
              '32B 的标签分布：{}。'.format(json.dumps(rj['label_dist_32B'], ensure_ascii=False)), '']
        for scope, gs, closed in (('sealed', ['G00', 'G01', 'G02', 'G04', 'G05', 'G08', 'G09', 'G10', 'G11', 'G12', 'G13', 'O1', 'O2', 'O3'], 'G00'),
                                  ('external', ['X00', 'X01', 'X02', 'X04', 'X05', 'X08', 'X09'], 'X00')):
            L += ['### {}'.format('封存集' if scope == 'sealed' else '外部集'), '', '| 条件 | 评审 A | Qwen3-32B |', '|---|---|---|']
            for g in gs:
                a = rj['results']['A'][scope + '_means'].get(g)
                b = rj['results']['32B'][scope + '_means'].get(g)
                if a or b:
                    L.append('| {} | {} | {} |'.format(g, '{:.3f}'.format(a[0]) if a else '—', '{:.3f}'.format(b[0]) if b else '—'))
            L.append('')
            for judge in ('A', '32B'):
                t = rj['results'][judge][scope + '_tests']
                L.append('- 评审 {}：主比较 {}；'.format(judge, fmt_test(t.get('main'))) +
                         '；'.join('对 {} {}'.format(x['b'], fmt_test(x)) for x in t['family']) + '；' +
                         '；'.join('等效 {} 90% CI [{:+.3f}, {:+.3f}]'.format(x['b'], *x['ci90']) for x in t['equivalence']))
            L.append('')
    if summary.get('s4'):
        L += s4_md(summary['s4'])
    rt = summary.get('runtime') or {}
    if rt.get('deployments'):
        L += ['', '## 运行环境', '']
        for d in rt['deployments']:
            L.append('- {} {} 模式（方案 {}{}）：{}'.format(d['started'], d['mode'], d['layout'],
                                                    '，启动失败' if d['failed'] else ('，备用部署' if d.get('deployment') else ''),
                                                    '；'.join('{} 卡{} 比例{} 方式{}'.format(k, v['devices'], v['util'], v['variant'])
                                                             for k, v in d['instances'].items())))
    (RESULTS / 'summary_sup.md').write_text('\n'.join(L), encoding='utf-8')


# ----------------------------------------------------------------------------- S4
def s4_tests(G, ref):
    """S4 预先写定的比较。G：{组: {题: 值}}；ref：Support-CE 参照组名（G05 或 X05）。"""
    def fam(pairs, tag, holm_it=True):
        res = []
        for a, b in pairs:
            r = boot(G.get(a, {}), G.get(b, {}), 'S4' + tag + a + b)
            if r:
                res.append(dict(r, a=a, b=b))
        if holm_it and res:
            for x, h in zip(res, holm([x['p'] for x in res])):
                x['p_holm'] = h
        return res
    out = {'H1_support_vs_zeroshot': fam([('Q06S', 'Q06N'), ('Q4BS', 'Q4BN')], 'h1'),
           'H2_support_vs_relevance': fam([('Q06S', 'Q06R'), ('Q4BS', 'Q4BR')], 'h2'),
           'H3_scale': fam([('Q4BS', 'Q06S'), ('Q4BN', 'Q06N'), ('Q4BR', 'Q06R')], 'h3', holm_it=False),
           'H4_vs_support_ce': fam([('Q06S', ref), ('Q4BS', ref)], 'h4')}
    a, b, c, d = (G.get(x, {}) for x in ('Q4BS', 'Q4BN', 'Q06S', 'Q06N'))
    qs = set(a) & set(b) & set(c) & set(d)
    inter = {q: (a[q] - b[q]) - (c[q] - d[q]) for q in qs}
    r = boot(inter, {q: 0.0 for q in qs}, 'S4inter')
    out['H3_interaction_size_x_support'] = dict(r, a='(Q4BS-Q4BN)', b='(Q06S-Q06N)') if r else None
    return out


def s4_part():
    if not CFG.get('s4', {}).get('enabled'):
        return None
    from s4_common import TRAIN_DIR, train_runs
    out = {'training': {}}
    for key, obj, seed, name in train_runs():
        d = TRAIN_DIR / name
        if (d / 'DONE.json').exists():
            row = {'done': read_json(d / 'DONE.json')}
            for f in ('test_metrics.json', 'config.json'):
                if (d / f).exists():
                    row[f.split('.')[0]] = read_json(d / f)
            if 'config' in row:
                row['config'] = {k: row['config'].get(k) for k in ('train_pairs', 'trainable_params', 'train_truncated_pairs', 'gpu')}
            out['training'][name] = row
    # 答案层：Qwen3-32B 评审的要点召回
    npts = {}
    for q in read_json(DATA / 'rejudge' / 'final_questions.json'):
        npts[('sealed', q['qid'])] = len(q['points'])
    for q in read_json(DATA / 'external_questions.json'):
        npts[('external', q['qid'])] = len(q['points'])
    rows = {}
    for r in read_jsonl(RESULTS / 's4' / 'kpr_qwen3_32b_s4.jsonl'):
        if r.get('valid') or r['aid'] not in rows:
            rows[r['aid']] = r
    per = {'sealed': defaultdict(dict), 'external': defaultdict(dict)}
    for r in rows.values():
        if not r.get('valid'):
            continue
        lab = effective(r['labels'], r.get('quote_ok'))
        n = npts[(r['target'], r['qid'])]
        per[r['target']][r['cond']][r['qid']] = 0.0 if r.get('abstain') else sum(1 for v in lab.values() if v == '2') / n
    out['kpr'] = {}
    for t, ref in (('sealed', 'G05'), ('external', 'X05')):
        G = group_means(per[t])
        out['kpr'][t] = {'means': {g: (sum(v.values()) / len(v), len(v)) for g, v in G.items() if v}, 'tests': s4_tests(G, ref)}
    # 外部集：官方选择题均分
    pr = defaultdict(dict)
    for r in read_jsonl(RESULTS / 's4' / 'mcq_results.jsonl'):
        if r.get('status') == 'ok':
            ds = r['dimension_scores']
            pr[r['cond']][r['qid']] = sum(float(ds.get(d, 0.0)) for d in MC_DIMS) / len(MC_DIMS)
    for r in read_jsonl(RESULTS / 'official' / 'official_results.jsonl'):
        if base(r['cond']) in ('X00', 'X04', 'X05'):
            ds = r.get('dimension_scores') or {}
            pr[r['cond']][r['qid']] = sum(float(ds.get(d, 0.0)) for d in MC_DIMS) / len(MC_DIMS)
    G = group_means(pr)
    out['mcq_external'] = {'means': {g: (sum(v.values()) / len(v), len(v)) for g, v in G.items() if v}, 'tests': s4_tests(G, 'X05')}
    out['note'] = '检索层（封存集上用证据金标计算的覆盖率等）在结果下载后于本地分析环境计算，不在服务器上进行。'
    return out


def s4_md(s4):
    L = ['## S4：Qwen3-Reranker 支持标签微调与原始权重的比较', '']
    L += ['### 微调结果（stage4 test 上支持类，dev 定阈值）', '', '| 运行 | 最佳轮次 | test P | test R | test F1 | 训练对数 |', '|---|---|---|---|---|---|']
    for name, r in s4['training'].items():
        m = (r.get('test_metrics') or {}).get('test_support_label') or {}
        f = lambda k: '{:.3f}'.format(m[k]) if isinstance(m.get(k), (int, float)) else '—'
        L.append('| {} | {} | {} | {} | {} | {} |'.format(name, r['done'].get('best_epoch'), f('precision'), f('recall'), f('f1'),
                                                         (r.get('config') or {}).get('train_pairs')))
    def block(title, part):
        L.extend(['', '### ' + title, '', '| 条件 | 均分 | 题数 |', '|---|---|---|'])
        for g in sorted(part['means']):
            L.append('| {} | {:.3f} | {} |'.format(g, *part['means'][g]))
        for fam, xs in part['tests'].items():
            if isinstance(xs, dict):
                L.append('- {}：{} − {} = {}'.format(fam, xs['a'], xs['b'], fmt_test(xs)))
            elif xs:
                L.append('- {}：'.format(fam) + '；'.join('{} − {} = {}'.format(x['a'], x['b'], fmt_test(x)) for x in xs))
    for t in ('external', 'sealed'):
        if s4['kpr'].get(t):
            block('要点召回（Qwen3-32B 评审，{}）'.format('外部集' if t == 'external' else '封存集'), s4['kpr'][t])
    if s4.get('mcq_external'):
        block('官方选择题均分（外部集）', s4['mcq_external'])
    L += ['', s4['note'], '']
    return L


def runtime_part():
    """运行环境记录：显卡、实际采用的服务部署与启动方式（供方法部分报告）。"""
    out = {'deployments': [], 'preflight': {}}
    p = LOGS / 'servers_history.jsonl'
    if p.exists():
        for line in p.read_text(encoding='utf-8').splitlines():
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            out['deployments'].append({'mode': r.get('mode'), 'layout': r.get('layout'), 'deployment': r.get('deployment'),
                                       'failed': r.get('failed', False), 'started': r.get('started') or r.get('time'),
                                       'instances': {k: {'devices': v.get('devices'), 'util': v.get('util'),
                                                         'variant': v.get('variant'), 'tp': len(v.get('devices') or [])}
                                                     for k, v in (r.get('instances') or {}).items()}})
    for f in ('vllm_variants', 'preflight_gpu'):
        q = WORK / 'state' / '{}.json'.format(f)
        if q.exists():
            try:
                out['preflight'][f] = json.loads(q.read_text(encoding='utf-8'))
            except Exception:  # noqa: BLE001
                pass
    variants = sorted({v['variant'] for d in out['deployments'] for v in d['instances'].values() if v.get('variant')})
    out['variants_used'] = variants
    return out


def main():
    import time
    summary = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'version': CFG['version'],
               'official': official_part(), 'rejudge': rejudge_part(), 's4': s4_part(), 'runtime': runtime_part()}
    write_json(RESULTS / 'summary_sup.json', summary)
    write_md(summary)
    log('汇总已写入 {}'.format(RESULTS / 'summary_sup.md'), 'pipeline')


if __name__ == '__main__':
    main()
