"""Adjudication sheets and final (gold) files.

Phase 1
  prepare : merge annotators -> 仲裁/phase1_仲裁表.csv
            Each 【仲裁】 field is pre-filled when all annotators agree on that field
            (copied fragments are compared clause by clause, so different cutting of the
            same text counts as agreement); disputed fields are left blank and listed in
            “需仲裁字段”. 参考 columns show how many annotators proposed each point/clause.
  finalize: filled 仲裁表 -> _sealed.../final_questions.json  (final question text + final necessary points)
Phase 2
  prepare : merge annotators -> 仲裁/phase2_仲裁表.csv  (only rows with a disagreement or an "无法判断" flag)
  finalize: -> _sealed.../phase2_gold_labels.jsonl and phase2_gold.csv

Usage (the expert_testset_v1 folder is found automatically, so any working directory works):
    python scripts/adjudicate.py --phase 1 --step prepare
    python scripts/adjudicate.py --phase 1 --step finalize
    python scripts/adjudicate.py --phase 2 --step prepare
    python scripts/adjudicate.py --phase 2 --step finalize
"""
import argparse, json, re, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import read_csv, write_csv, find_col, norm_int, PKG_ROOT, resolve_path, csv_listing
from annot_io import (load_phase1, load_phase2, final_points_of, render_point_list, parse_point_list,
                      normalize_text, split_multi, clauses, clause_set, POINT_TYPES)

SEALED = '_sealed_负责人保存_勿发给标注员'
NONE = '无'
FIELD_COLS = {'纳入': '【仲裁】最终纳入', '删除片段': '【仲裁】最终删除片段', '补充信息': '【仲裁】最终补充信息',
              '要点': '【仲裁】最终要点列表'}


def _multi(cell):
    """Arbiter cell for fragments/additions: '无' (or blank) means none."""
    cell = (cell or '').strip()
    return [] if cell in ('', NONE, '无。', '没有') else split_multi(cell)
QTYPE_CN = {'diagnosis': '诊断解释（诊断/证候/病机）', 'principle': '治疗原则'}


def load_sealed_questions(root):
    full = json.loads((root / SEALED / 'sealed_questions_full.json').read_text(encoding='utf-8'))
    return {q['qid']: q for q in full}


# ------------------------------------------------------------------ phase 1
def _votes(lists):
    """lists: {annotator: [(key, display), ...]} -> [(display, count)] in first-seen order, most votes first."""
    count, shown, order = Counter(), {}, []
    for items in lists.values():
        for key, disp in dict(items).items():
            if key not in shown:
                shown[key] = disp
                order.append(key)
            count[key] += 1
    return sorted(((shown[k], count[k]) for k in order), key=lambda x: -x[1])


def p1_prepare(root, src, out):
    data = load_phase1(src)
    who = sorted(data)
    if len(who) < 2:
        sys.exit('需要至少两位已填写的 Phase 1 标注员文件，实际找到 '
                 f'{len(who)} 位（{"、".join(who) or "无"}）。\n' + csv_listing(src))
    sealed = load_sealed_questions(root)
    n = len(who)
    need_majority = n // 2 + 1
    rows, n_disagree, field_count = [], 0, Counter()
    for qid in sorted(sealed):
        q = sealed[qid]
        per = {a: data[a].get(qid) or {} for a in who}
        incs = {a: per[a].get('include') for a in who}
        pts = {a: final_points_of(per[a]) if per[a] else [] for a in who}
        frags = {a: list(dict.fromkeys(per[a].get('leak_fragments', []))) for a in who}
        adds = {a: list(dict.fromkeys(per[a].get('add_info', []))) for a in who}
        same = {
            '纳入': len(set(incs.values())) == 1 and None not in incs.values(),
            '删除片段': len({clause_set(v) for v in frags.values()}) == 1,
            '补充信息': len({clause_set(v) for v in adds.values()}) == 1,
            '要点': len({json.dumps(sorted(normalize_text(f'{t}{x}') for t, x in v), ensure_ascii=False)
                       for v in pts.values()}) == 1,
        }
        all_excluded = same['纳入'] and incs[who[0]] == 0
        diffs = [] if all_excluded else [k for k, ok in same.items() if not ok]
        n_disagree += bool(diffs)
        field_count.update(diffs)
        row = {'题目编号': qid, '题型': QTYPE_CN[q['question_type']], '检索用问题文本(草稿)': q['question_draft'],
               '原始病情描述(参考)': q['raw_description'], '原始辨证分析(要点来源)': q['reference_analysis'],
               '是否一致': '需仲裁' if diffs else ('一致(均排除)' if all_excluded else '一致'),
               '需仲裁字段': '、'.join(diffs)}
        for a in who:
            p = per[a]
            row[f'{a}_纳入'] = '' if incs[a] is None else incs[a]
            row[f'{a}_排除原因'] = p.get('exclude_reason', '')
            row[f'{a}_删除片段'] = ' || '.join(frags[a])
            row[f'{a}_补充信息'] = ' || '.join(adds[a])
            row[f'{a}_最终要点'] = render_point_list(pts[a])
            row[f'{a}_备注'] = p.get('note', '')
        # reference columns: how many annotators proposed each point / clause (decision stays with the arbiter)
        pv = _votes({a: [(normalize_text(f'{t}{x}'), f'{t}:{x}' if t else x) for t, x in v] for a, v in pts.items()})
        av = _votes({a: clauses(v) for a, v in adds.items()})
        dv = _votes({a: clauses(v) for a, v in frags.items()})
        row['参考_要点(提出人数)'] = '；'.join(f'{d}〔{c}/{n}〕' for d, c in pv)
        row['参考_多数要点(可直接复制)'] = '；'.join(d for d, c in pv if c >= need_majority)
        row['参考_补充信息(提出人数)'] = '；'.join(f'{d}〔{c}/{n}〕' for d, c in av)
        row['参考_删除片段(提出人数)'] = '；'.join(f'{d}〔{c}/{n}〕' for d, c in dv)
        first = who[0]
        # blank = still to be decided; “无” = decided, nothing to delete/add
        row['【仲裁】最终纳入(1/0)'] = incs[first] if same['纳入'] else ''
        row['【仲裁】最终删除片段(||分隔)'] = (' || '.join(frags[first]) or NONE) if same['删除片段'] else ''
        row['【仲裁】最终补充信息(||分隔)'] = (' || '.join(adds[first]) or NONE) if same['补充信息'] else ''
        row['【仲裁】最终要点列表(类型:内容,用；分隔)'] = render_point_list(pts[first]) if same['要点'] else ''
        row['【仲裁】备注'] = ''
        rows.append(row)
    path = out / 'phase1_仲裁表.csv'
    write_csv(path, rows)
    detail = '，'.join(f'{k} {field_count[k]} 题' for k in ('纳入', '删除片段', '补充信息', '要点') if field_count[k])
    print(f'已生成 {path.resolve()}\n共 {len(rows)} 题（标注员：{"、".join(who)}），其中需仲裁 {n_disagree} 题'
          + (f'；有分歧的字段：{detail}' if detail else '') + '。\n'
          '一致的字段已预填在【仲裁】列；仲裁人只需填写空白的【仲裁】列（“需仲裁字段”列写明了要裁定哪几项）。\n'
          '删除片段、补充信息裁定为“没有”时请填“无”，留空表示尚未裁定，定稿时会被提示。')


def apply_edits(draft, deletions, additions):
    background, _, instruction = draft.partition('\n')
    missing = []
    for frag in deletions:
        if frag in background:
            background = background.replace(frag, '')
        else:
            missing.append(frag)
    background = re.sub(r'[；;]{2,}', '；', background).strip('；; ')
    if additions:
        background = background.rstrip('。') + '；' + '；'.join(a.strip('；。 ') for a in additions) + '。'
    if not background.endswith('。'):
        background += '。'
    return background + ('\n' + instruction if instruction else ''), missing


def p1_finalize(root, out):
    if not (out / 'phase1_仲裁表.csv').exists():
        sys.exit(f'找不到 {out / "phase1_仲裁表.csv"}，请先运行 --phase 1 --step prepare。')
    sheet = read_csv(out / 'phase1_仲裁表.csv')
    sealed = load_sealed_questions(root)
    h = sheet[0]
    c_inc, c_del = find_col(h, '【仲裁】最终纳入'), find_col(h, '【仲裁】最终删除片段')
    c_add, c_pts = find_col(h, '【仲裁】最终补充信息'), find_col(h, '【仲裁】最终要点列表')
    final, problems = [], []
    for r in sheet:
        qid = r['题目编号']
        q = sealed[qid]
        inc = norm_int(r[c_inc], {0, 1})
        if inc is None:
            problems.append(f'{qid}: 未填写最终纳入')
            continue
        disputed = [x for x in re.split(r'[、,，\s]+', r.get('需仲裁字段', '')) if x]
        if inc == 1:
            for field in disputed:
                col = find_col(h, FIELD_COLS[field]) if field in FIELD_COLS else None
                if col and not r[col].strip():
                    hint = '（裁定为没有时请填“无”）' if field in ('删除片段', '补充信息') else ''
                    problems.append(f'{qid}: “{field}”有分歧但【仲裁】列仍为空，尚未裁定{hint}')
        text, missing = apply_edits(q['question_draft'], _multi(r[c_del]), _multi(r[c_add]))
        if missing:
            problems.append(f'{qid}: 以下删除片段在问题文本中找不到，请核对原文照抄: {missing}')
        pts = parse_point_list(r[c_pts])
        bad = [x for t, x in pts if t not in POINT_TYPES]
        if inc == 1 and not pts:
            problems.append(f'{qid}: 纳入但没有最终要点')
        if inc == 1 and bad:
            problems.append(f'{qid}: 以下要点缺少有效类型(疾病诊断/证候/病机/治疗原则): {bad}')
        final.append({'qid': qid, 'question_type': q['question_type'], 'included': inc == 1,
                      'question_draft': q['question_draft'], 'question_final': text,
                      'points': [{'pid': f'P{i}', 'type': t, 'text': x} for i, (t, x) in enumerate(pts, 1)],
                      'arbiter_note': r.get('【仲裁】备注', '')})
    if problems:
        print('仲裁表存在以下问题，请修改后重新运行：\n' + '\n'.join(problems))
        sys.exit(1)
    path = root / SEALED / 'final_questions.json'
    path.write_text(json.dumps(final, ensure_ascii=False, indent=1), encoding='utf-8')
    n_in = sum(q['included'] for q in final)
    print(f'已写出 {path}：纳入 {n_in} 题，排除 {len(final) - n_in} 题，要点共 {sum(len(q["points"]) for q in final if q["included"])} 条。')


# ------------------------------------------------------------------ phase 2
FIELDS = ('relevance', 'mismatch')
CN = {'relevance': '相关性', 'mismatch': '同病异证或条件不符'}


def p2_prepare(root, src, out, prefix):
    data = load_phase2(src, prefix=prefix)
    who = sorted(a for a in data if not a.upper().startswith('AI'))
    keys = sorted({k for a in who for k in data[a]})
    max_p = max((max(rec['support'], default=0) for a in who for rec in data[a].values()), default=0)
    rows, n = [], 0
    for key in keys:
        recs = {a: data[a].get(key) for a in who if data[a].get(key)}
        if len(recs) < 2:
            continue
        base = next(iter(recs.values()))['row']
        npts = len(next(iter(recs.values()))['support'])
        diffs, major = [], False
        row = {'题目编号': key[0], '候选编号': key[1], '问题文本': base.get('问题文本', ''),
               '必要回答要点': base.get('必要回答要点', ''), '证据来源': base.get('证据来源', ''), '证据文本': base.get('证据文本', '')}
        for f in FIELDS:
            vals = {a: r[f] for a, r in recs.items()}
            for a, v in vals.items():
                row[f'{a}_{CN[f]}'] = '' if v is None else v
            agree = len(set(vals.values())) == 1 and None not in vals.values()
            row[f'【仲裁】{CN[f]}'] = next(iter(vals.values())) if agree else ''
            diffs += [] if agree else [CN[f]]
        for i in range(1, max_p + 1):
            if i > npts:
                for a in recs:
                    row[f'{a}_P{i}'] = 'NA'
                row[f'【仲裁】P{i}'] = 'NA'
                continue
            vals = {a: r['support'].get(i) for a, r in recs.items()}
            for a, v in vals.items():
                row[f'{a}_P{i}'] = '' if v is None else v
            agree = len(set(vals.values())) == 1 and None not in vals.values()
            row[f'【仲裁】P{i}'] = next(iter(vals.values())) if agree else ''
            diffs += [] if agree else [f'P{i}']
            if not agree and len({v == 2 for v in vals.values()}) > 1:
                major = True
        unc = [a for a, r in recs.items() if r['uncertain']]
        notes = ' | '.join(f'{a}:{r["note"]}' for a, r in recs.items() if r['note'])
        if diffs or unc:
            row['影响主指标(完整支持分歧)'] = '是' if major else '否'
            row['需仲裁项'] = '、'.join(diffs + ([f'标注员{"/".join(unc)}标记无法判断'] if unc else []))
            row['标注员备注'] = notes
            row['【仲裁】备注'] = ''
            rows.append(row)
            n += 1
    rows.sort(key=lambda r: (r['影响主指标(完整支持分歧)'] != '是', r['题目编号'], r['候选编号']))
    write_csv(out / 'phase2_仲裁表.csv', rows)
    n_major = sum(r['影响主指标(完整支持分歧)'] == '是' for r in rows)
    print(f'已生成 {out / "phase2_仲裁表.csv"}：需仲裁 {n} 行（共 {len(keys)} 个候选），其中影响主指标的完整支持分歧 {n_major} 行已排在前面。')


def p2_finalize(root, src, out, prefix):
    data = load_phase2(src, prefix=prefix)
    who = sorted(a for a in data if not a.upper().startswith('AI'))
    sheet_path = out / 'phase2_仲裁表.csv'
    arb = {(r['题目编号'], r['候选编号']): r for r in read_csv(sheet_path)} if sheet_path.exists() else {}
    keyfile = root / SEALED / f'{prefix}_candidate_key.csv'
    key = {(r['题目编号'], r['候选编号']): r for r in read_csv(keyfile)} if keyfile.exists() else {}
    gold, problems = [], []
    for k in sorted({k for a in who for k in data[a]}):
        recs = [data[a][k] for a in who if k in data[a]]
        npts = len(recs[0]['support'])
        g = {'qid': k[0], 'cand': k[1], 'support': {}, 'annotators': [a for a in who if k in data[a]]}
        a_row = arb.get(k)
        for f in FIELDS:
            vals = {r[f] for r in recs}
            if a_row is not None and a_row.get(f'【仲裁】{CN[f]}', '') != '':
                g[f] = norm_int(a_row[f'【仲裁】{CN[f]}'], {0, 1})
            elif len(vals) == 1 and None not in vals:
                g[f] = recs[0][f]
            else:
                g[f] = None
                problems.append(f'{k}: {CN[f]} 未裁定')
        for i in range(1, npts + 1):
            vals = {r['support'].get(i) for r in recs}
            if a_row is not None and a_row.get(f'【仲裁】P{i}', '') not in ('', 'NA'):
                g['support'][f'P{i}'] = norm_int(a_row[f'【仲裁】P{i}'], {0, 1, 2})
            elif len(vals) == 1 and None not in vals:
                g['support'][f'P{i}'] = recs[0]['support'][i]
            else:
                g['support'][f'P{i}'] = None
                problems.append(f'{k}: P{i} 未裁定')
        if k in key:
            g['retrieval_rank'] = int(key[k]['retrieval_rank'])
            g['document_id'] = key[k]['document_id']
            g['source'] = key[k]['source']
        gold.append(g)
    if problems:
        print(f'仍有 {len(problems)} 处未裁定，示例：\n' + '\n'.join(problems[:20]))
        sys.exit(1)
    path = root / SEALED / 'phase2_gold_labels.jsonl'
    with path.open('w', encoding='utf-8') as f:
        for g in gold:
            f.write(json.dumps(g, ensure_ascii=False) + '\n')
    flat = [{'题目编号': g['qid'], '候选编号': g['cand'], '相关性': g['relevance'], '同病异证或条件不符': g['mismatch'],
             **{f'{p}支持': v for p, v in g['support'].items()}, 'retrieval_rank': g.get('retrieval_rank', '')} for g in gold]
    pcols = sorted({k for r in flat for k in r if k.startswith('P')}, key=lambda c: int(re.sub(r'\D', '', c)))
    write_csv(root / SEALED / 'phase2_gold.csv', flat,
              fieldnames=['题目编号', '候选编号'] + pcols + ['相关性', '同病异证或条件不符', 'retrieval_rank'])
    print(f'已写出金标：{path}（{len(gold)} 个候选）。')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', type=int, choices=[1, 2], required=True)
    ap.add_argument('--step', choices=['prepare', 'finalize'], required=True)
    ap.add_argument('--root', default=None, help='expert_testset_v1 folder (default: the folder containing scripts/)')
    ap.add_argument('--src', default=None, help='folder with filled annotator files')
    ap.add_argument('--prefix', default='phase2_证据标注')
    a = ap.parse_args()
    root = Path(a.root).resolve() if a.root else PKG_ROOT
    if not (root / SEALED / 'sealed_questions_full.json').exists():
        sys.exit(f'在 {root} 中找不到 {SEALED}/sealed_questions_full.json，请用 --root 指定 expert_testset_v1 文件夹。')
    out = root / '仲裁'
    out.mkdir(exist_ok=True)
    print(f'工作文件夹：{root}')
    if a.phase == 1:
        src = resolve_path(a.src, root) if a.src else root / 'phase1_题目与要点核验'
        p1_prepare(root, src, out) if a.step == 'prepare' else p1_finalize(root, out)
    else:
        src = resolve_path(a.src, root) if a.src else root / 'phase2_证据标注'
        p2_prepare(root, src, out, a.prefix) if a.step == 'prepare' else p2_finalize(root, src, out, a.prefix)


if __name__ == '__main__':
    main()
