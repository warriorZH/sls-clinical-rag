"""Build the Phase 2 evidence-support annotation sheets.

Inputs
  --questions  final_questions.json   (written by adjudicate.py --phase 1 --step finalize)
  --pools      sealed_pools.jsonl     (written on the server by server_build_pools.py)
Outputs (in --out)
  phase2_证据标注_标注员A.csv / B / C   one row per (question, candidate), candidate order shuffled
  ../_sealed_负责人保存_勿发给标注员/phase2_candidate_key.csv   candidate code -> retrieval rank/document
  ../_sealed_负责人保存_勿发给标注员/PHASE2_MANIFEST.json

Annotators never see retrieval ranks, scores or any model output.
"""
import argparse, hashlib, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import write_csv

SOURCE_LABEL = {
    'TCM Educational Materials.txt': '中医教材',
    'TCM Ancient books.txt': '中医古籍（白话译文）',
}
QTYPE_CN = {'diagnosis': '诊断解释（诊断/证候/病机）', 'principle': '治疗原则'}
SUPPORT_HELP = '(2完整支持/1部分支持/0不支持)'


def source_label(src):
    if src.startswith('medical_train'):
        return '其他患者病例记录'
    return SOURCE_LABEL.get(src, src)


def point_header(i):
    return f'【填】P{i}支持{SUPPORT_HELP}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--questions', required=True)
    ap.add_argument('--pools', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--sealed-dir', default=None)
    ap.add_argument('--k', type=int, default=20)
    ap.add_argument('--annotators', default='A,B,C')
    ap.add_argument('--overlap', type=float, default=1.0,
                    help='Share of questions given to annotators other than A (1.0 = full double annotation)')
    ap.add_argument('--seed', default='20261024')
    ap.add_argument('--prefix', default='phase2_证据标注')
    a = ap.parse_args()

    questions = [q for q in json.loads(Path(a.questions).read_text(encoding='utf-8')) if q.get('included', True)]
    pools = {}
    with open(a.pools, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                pools.setdefault(r['qid'], []).append(r)
    max_points = max(len(q['points']) for q in questions)
    point_cols = [point_header(i) for i in range(1, max_points + 1)]

    rows, key = [], []
    for q in sorted(questions, key=lambda x: x['qid']):
        cands = sorted(pools.get(q['qid'], []), key=lambda r: r['rank'])[:a.k]
        if len(cands) < a.k:
            print(f"警告: {q['qid']} 只有 {len(cands)} 个候选", file=sys.stderr)
        rng = random.Random(f"{a.seed}|{q['qid']}")
        rng.shuffle(cands)
        points_text = '\n'.join(f"P{i}［{p['type']}］{p['text']}" for i, p in enumerate(q['points'], 1))
        for j, c in enumerate(cands, 1):
            code = f'C{j:02d}'
            evidence = c.get('context_text') or c['text']
            row = {'题目编号': q['qid'], '候选编号': code, '题型': QTYPE_CN.get(q['question_type'], q['question_type']),
                   '问题文本': q['question_final'], '必要回答要点': points_text,
                   '证据来源': source_label(c['source']), '证据文本': evidence,
                   '【填】相关性(1相关/0无关)': ''}
            for i in range(1, max_points + 1):
                row[point_header(i)] = '' if i <= len(q['points']) else 'NA'
            row['【填】同病异证或条件不符(1是/0否)'] = ''
            row['【填】无法判断(1是,留空为否)'] = ''
            row['【填】备注'] = ''
            rows.append(row)
            key.append({'题目编号': q['qid'], '候选编号': code, 'retrieval_rank': c['rank'],
                        'document_id': c.get('document_id'), 'source': c['source'],
                        'bm25_score': c.get('bm25_score'), 'text_sha256': c.get('sha256'),
                        'context_status': c.get('context_status')})

    out = Path(a.out)
    sealed = Path(a.sealed_dir) if a.sealed_dir else out.parent / '_sealed_负责人保存_勿发给标注员'
    names = [x.strip() for x in a.annotators.split(',') if x.strip()]
    qids = sorted({r['题目编号'] for r in rows})
    overlap_q = set(sorted(qids, key=lambda x: hashlib.sha256(f'{a.seed}|overlap|{x}'.encode()).hexdigest())[:round(len(qids) * a.overlap)])
    for n, who in enumerate(names):
        subset = rows if n == 0 or a.overlap >= 1 else [r for r in rows if r['题目编号'] in overlap_q]
        write_csv(out / f'{a.prefix}_标注员{who}.csv', subset)
    write_csv(sealed / f'{a.prefix}_candidate_key.csv', key)
    manifest = {'questions': len(qids), 'rows_per_full_annotator': len(rows), 'k': a.k,
                'annotators': names, 'overlap_share_for_non_A': a.overlap,
                'overlap_questions': sorted(overlap_q) if a.overlap < 1 else 'all',
                'evidence_shown': 'context_text (heading + surrounding source text) when available, else fragment',
                'order': 'candidates shuffled per question with seed; ranks/scores hidden',
                'max_points': max_points,
                'questions_sha256': hashlib.sha256(Path(a.questions).read_bytes()).hexdigest(),
                'pools_sha256': hashlib.sha256(Path(a.pools).read_bytes()).hexdigest()}
    (sealed / f'{a.prefix.upper()}_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2)[:1500])


if __name__ == '__main__':
    main()
