"""Inter-annotator agreement report for Phase 1 (questions/points) and Phase 2 (evidence support).

Usage (from the expert_testset_v1 folder):
    python scripts/compute_agreement.py --phase 1 --dir phase1_题目与要点核验
    python scripts/compute_agreement.py --phase 2 --dir phase2_证据标注
    python scripts/compute_agreement.py --phase 2 --dir phase0_校准练习 --prefix calib_phase2_证据标注

Annotators are discovered from file names (..._标注员A.csv, ..._标注员B.csv, ...).
Files whose annotator name starts with 'AI' are compared pairwise with each expert
but are left out of the multi-rater statistics.

Reports: Cohen's kappa for every annotator pair, Fleiss' kappa and Krippendorff's
alpha when three or more experts labelled the same units, raw agreement, and a
95% CI from a cluster bootstrap that resamples whole questions.
"""
import argparse, itertools, json, math, re, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (cohen_kappa, fleiss_kappa, krippendorff_alpha, percent_agreement, landis_koch,
                    kappa_from_confusion, coincidence, alpha_from_coincidence, bootstrap_sum,
                    PKG_ROOT, resolve_path, csv_listing)
from annot_io import load_phase1, load_phase2, final_points_of, normalize_text, clause_set

REPS = 2000


def fmt(x):
    return '—' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{x:.3f}'


def pair_stats(units, a, b, weights=None, categories=None):
    """units: list of (qid, {annotator: label}); returns dict with kappa, CI, agreement, n."""
    use = [(q, lab) for q, lab in units if lab.get(a) is not None and lab.get(b) is not None]
    if not use:
        return None
    xa = [lab[a] for _, lab in use]
    xb = [lab[b] for _, lab in use]
    cats = sorted(set(categories or []) | set(xa) | set(xb))
    k = cohen_kappa(xa, xb, weights=weights, categories=cats)
    per_q = defaultdict(lambda: defaultdict(float))
    for q, lab in use:
        per_q[q][(lab[a], lab[b])] += 1
    ci = bootstrap_sum(per_q, lambda conf: kappa_from_confusion(conf, cats, weights=weights), reps=REPS)
    return {'n_units': len(use), 'n_questions': len(per_q), 'kappa': k, 'ci95': ci,
            'percent_agreement': percent_agreement(xa, xb), 'interpretation': landis_koch(k)}


def multi_stats(units, raters, level='nominal'):
    complete = [(q, lab) for q, lab in units if all(lab.get(r) is not None for r in raters)]
    out = {'raters': raters}
    if len(raters) >= 3 and complete:
        out['fleiss_kappa'] = fleiss_kappa([[lab[r] for r in raters] for _, lab in complete])
    rows = [(q, [lab.get(r) for r in raters]) for q, lab in units]
    alpha = krippendorff_alpha([v for _, v in rows], level=level)
    per_q = defaultdict(list)
    for q, v in rows:
        per_q[q].append(v)
    per_q = {q: coincidence(v) for q, v in per_q.items()}
    out['krippendorff_alpha'] = alpha
    out['alpha_ci95'] = bootstrap_sum(per_q, lambda o: alpha_from_coincidence(o, level=level), reps=REPS)
    out['alpha_level'] = level
    out['n_units_any'] = sum(1 for _, v in rows if sum(x is not None for x in v) >= 2)
    out['n_units_complete'] = len(complete)
    return out


def analyse(name, units, annotators, weights=None, level='nominal', categories=None):
    experts = [a for a in annotators if not a.upper().startswith('AI')]
    res = {'measure': name, 'pairs': {}}
    for a, b in itertools.combinations(annotators, 2):
        s = pair_stats(units, a, b, weights=weights, categories=categories)
        if s:
            res['pairs'][f'{a}-{b}'] = s
    if len(experts) >= 2:
        res['multi'] = multi_stats(units, experts, level=level)
    return res


def phase1_units(data):
    annotators = sorted(data)
    qids = sorted({q for a in data.values() for q in a})
    inc, leak, dec3, keep = [], [], [], []
    for q in qids:
        inc.append((q, {a: data[a].get(q, {}).get('include') for a in annotators}))
        leak.append((q, {a: data[a].get(q, {}).get('leak') for a in annotators}))
        pids = sorted({p for a in annotators for p in data[a].get(q, {}).get('points', {})})
        for p in pids:
            labs = {a: data[a].get(q, {}).get('points', {}).get(p, {}).get('decision') for a in annotators}
            dec3.append((q, labs))
            keep.append((q, {a: (None if v is None else int(v != 0)) for a, v in labs.items()}))
    modification = {}
    for a in annotators:
        decisions = [p['decision'] for q in data[a].values() for p in q.get('points', {}).values()]
        filled = [d for d in decisions if d is not None]
        added = sum(len(q.get('missing_points', [])) for q in data[a].values())
        modification[a] = {'draft_points': len(decisions), 'filled': len(filled),
                           'kept_unchanged': sum(d == 1 for d in filled), 'modified': sum(d == 2 for d in filled),
                           'deleted': sum(d == 0 for d in filled), 'added_points': added,
                           'questions_excluded': sum(q.get('include') == 0 for q in data[a].values()),
                           'questions_with_leak': sum(q.get('leak') == 1 for q in data[a].values())}
    measures = [analyse('题目纳入(1/0)', inc, annotators),
                analyse('问题泄露(1/0)', leak, annotators),
                analyse('要点判定(保留/修改/删除，三分类)', dec3, annotators, categories=[0, 1, 2]),
                analyse('要点保留与否(保留或修改 vs 删除)', keep, annotators, categories=[0, 1])]
    return annotators, measures, {'draft_point_modification': modification,
                                  'text_agreement': phase1_text_agreement(data)}


def _overlap(x, y):
    """(Jaccard, F1) of two sets; two empty sets count as identical."""
    if not x and not y:
        return 1.0, 1.0
    inter = len(x & y)
    return inter / len(x | y), 2 * inter / (len(x) + len(y))


def phase1_text_agreement(data):
    """Pairwise agreement on the final wording, which the decision codes do not capture.
    Only questions that both annotators included are counted."""
    who = sorted(data)
    qids = sorted({q for a in data.values() for q in a})
    res = {}
    for a, b in itertools.combinations(who, 2):
        both = [q for q in qids if data[a].get(q, {}).get('include') == 1 and data[b].get(q, {}).get('include') == 1]
        if not both:
            continue
        pts_same = pts_f1 = add_same = add_j = del_same = 0.0
        for q in both:
            pa = {normalize_text(f'{t}{x}') for t, x in final_points_of(data[a][q])}
            pb = {normalize_text(f'{t}{x}') for t, x in final_points_of(data[b][q])}
            pts_same += pa == pb
            pts_f1 += _overlap(pa, pb)[1]
            aa, ab = clause_set(data[a][q].get('add_info', [])), clause_set(data[b][q].get('add_info', []))
            add_same += aa == ab
            add_j += _overlap(aa, ab)[0]
            del_same += clause_set(data[a][q].get('leak_fragments', [])) == clause_set(data[b][q].get('leak_fragments', []))
        k = len(both)
        res[f'{a}-{b}'] = {'n_questions': k, 'points_identical': pts_same / k, 'points_mean_f1': pts_f1 / k,
                           'add_info_identical': add_same / k, 'add_info_mean_jaccard': add_j / k,
                           'leak_fragments_identical': del_same / k}
    return res


def phase2_units(data):
    annotators = sorted(data)
    keys = sorted({k for a in data.values() for k in a})
    sup3, full, anyfull, rel, mis = [], [], [], [], []
    for key in keys:
        q = key[0]
        rel.append((q, {a: data[a].get(key, {}).get('relevance') for a in annotators}))
        mis.append((q, {a: data[a].get(key, {}).get('mismatch') for a in annotators}))
        pids = sorted({p for a in annotators for p in data[a].get(key, {}).get('support', {})})
        any_lab = {}
        for p in pids:
            labs = {a: data[a].get(key, {}).get('support', {}).get(p) for a in annotators}
            sup3.append((q, labs))
            full.append((q, {a: (None if v is None else int(v == 2)) for a, v in labs.items()}))
        for a in annotators:
            vals = [data[a].get(key, {}).get('support', {}).get(p) for p in pids]
            any_lab[a] = None if (not vals or any(v is None for v in vals)) else int(any(v == 2 for v in vals))
        anyfull.append((q, any_lab))
    counts = {}
    for a in annotators:
        labs = [v for rec in data[a].values() for v in rec['support'].values() if v is not None]
        counts[a] = {'candidates': len(data[a]), 'point_judgements': len(labs),
                     'full': sum(v == 2 for v in labs), 'partial': sum(v == 1 for v in labs), 'none': sum(v == 0 for v in labs),
                     'blank_cells': sum(v is None for rec in data[a].values() for v in rec['support'].values()),
                     'uncertain_rows': sum(rec['uncertain'] for rec in data[a].values())}
    measures = [analyse('要点支持度(2/1/0，名义κ)', sup3, annotators, categories=[0, 1, 2]),
                analyse('要点支持度(2/1/0，线性加权κ / 序数α)', sup3, annotators, weights='linear', level='ordinal', categories=[0, 1, 2]),
                analyse('完整支持 vs 非完整支持(主指标所用标签)', full, annotators, categories=[0, 1]),
                analyse('候选是否完整支持任一要点', anyfull, annotators, categories=[0, 1]),
                analyse('主题相关性(1/0)', rel, annotators, categories=[0, 1]),
                analyse('同病异证或条件不符(1/0)', mis, annotators, categories=[0, 1])]
    return annotators, measures, {'label_counts': counts}


def write_report(out_dir, phase, annotators, measures, extra):
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [f'# 标注一致性报告（Phase {phase}）', '',
             f'标注员：{", ".join(annotators)}；置信区间为按题目整体重抽样的 bootstrap（{REPS} 次，95%）。', '',
             '| 指标 | 比较 | 题数 | 单元数 | κ / α | 95% CI | 原始一致率 | 解读 |', '|---|---|---:|---:|---:|---|---:|---|']
    for m in measures:
        for pair, s in m['pairs'].items():
            lines.append(f"| {m['measure']} | Cohen κ {pair} | {s['n_questions']} | {s['n_units']} | {fmt(s['kappa'])} | "
                         f"[{fmt(s['ci95'][0])}, {fmt(s['ci95'][1])}] | {fmt(s['percent_agreement'])} | {s['interpretation']} |")
        mm = m.get('multi')
        if mm:
            lines.append(f"| {m['measure']} | Krippendorff α（{mm['alpha_level']}，{'/'.join(mm['raters'])}） | — | {mm['n_units_any']} | "
                         f"{fmt(mm['krippendorff_alpha'])} | [{fmt(mm['alpha_ci95'][0])}, {fmt(mm['alpha_ci95'][1])}] | — | {landis_koch(mm['krippendorff_alpha'])} |")
            if 'fleiss_kappa' in mm:
                lines.append(f"| {m['measure']} | Fleiss κ（{'/'.join(mm['raters'])}） | — | {mm['n_units_complete']} | {fmt(mm['fleiss_kappa'])} | — | — | {landis_koch(mm['fleiss_kappa'])} |")
    lines += ['', '解读参考 Landis & Koch (1977)：0.41–0.60 中等，0.61–0.80 较强，>0.80 几乎完全一致。'
              '论文中请报告 κ/α 及置信区间、单元数与题数，并说明分歧由仲裁人裁定。', '']
    text = extra.get('text_agreement')
    if text:
        lines += ['## 文本层面的一致性（阶段1）', '',
                  '判定码一致不代表最终文字一致。下表只统计两人都纳入的题；要点按“类型+内容”去掉标点后逐字比较，'
                  '补充信息和删除片段按分句比较。同义改写也会被算作不同，所以这些数值是下限。', '',
                  '| 比较 | 题数 | 最终要点完全相同 | 要点集合 F1 均值 | 补充信息完全相同 | 补充信息 Jaccard 均值 | 删除片段完全相同 |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for pair, t in text.items():
            lines.append(f"| {pair} | {t['n_questions']} | {t['points_identical']:.1%} | {t['points_mean_f1']:.3f} | "
                         f"{t['add_info_identical']:.1%} | {t['add_info_mean_jaccard']:.3f} | {t['leak_fragments_identical']:.1%} |")
        lines.append('')
    rest = {k: v for k, v in extra.items() if k != 'text_agreement'}
    lines += ['## 标注量统计', '', '```json', json.dumps(rest, ensure_ascii=False, indent=2), '```', '']
    (out_dir / f'agreement_phase{phase}.md').write_text('\n'.join(lines), encoding='utf-8')
    (out_dir / f'agreement_phase{phase}.json').write_text(json.dumps({'annotators': annotators, 'measures': measures, **extra},
                                                                      ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    print('\n'.join(lines))
    print(f"\n报告已写出：{(out_dir / f'agreement_phase{phase}.md').resolve()}\n"
          f"          {(out_dir / f'agreement_phase{phase}.json').resolve()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', type=int, choices=[1, 2], required=True)
    ap.add_argument('--dir', required=True, help='folder holding the filled annotator CSV files')
    ap.add_argument('--prefix', default=None, help='file prefix (default phase1/phase2 names)')
    ap.add_argument('--out', default='一致性报告', help='output folder (relative paths are inside expert_testset_v1)')
    a = ap.parse_args()
    folder = resolve_path(a.dir)
    out = Path(a.out) if Path(a.out).is_absolute() else PKG_ROOT / a.out
    if not folder.is_dir():
        sys.exit(f'找不到标注文件夹：{folder}')
    print(f'读取：{folder}')
    if a.phase == 1:
        data = load_phase1(folder, prefix='' if a.prefix is None else a.prefix)
    else:
        data = load_phase2(folder, prefix=a.prefix or 'phase2_证据标注')
    if len(data) < 2:
        sys.exit(f'至少需要两位已填写的标注员文件才能计算一致性，实际找到 {len(data)} 位。\n' + csv_listing(folder))
    annotators, measures, extra = phase1_units(data) if a.phase == 1 else phase2_units(data)
    write_report(out, a.phase, annotators, measures, extra)


if __name__ == '__main__':
    main()
