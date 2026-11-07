"""Build the Phase 0 calibration (practice) set from DEV questions only.

Dev questions never enter the sealed test set.  Practice points are the
earlier AI-revised standards and are labelled as practice material, not gold.

Usage (from the EEEQA folder):
    python expert_testset_v1/scripts/build_calibration_set.py --experiments experiments --out expert_testset_v1
"""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import write_csv

PICK = {'diagnosis': 3, 'principle': 2}
QTYPE_CN = {'diagnosis': '诊断解释（诊断/证候/病机）', 'principle': '治疗原则'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiments', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    exp, out = Path(a.experiments), Path(a.out)
    s5 = exp / 'stage5_real_retrieval_v1_2'
    std = json.loads((s5 / 'CODEX_ALL_TASK_STANDARDS.json').read_text(encoding='utf-8'))['tasks']
    review = {t['task_id']: t for t in json.loads((s5 / 'review_tasks.json').read_text(encoding='utf-8'))}
    dev = [t for t in std if t['split'] == 'dev' and t['source'] == 'med' and t['eligible'] and len(t['necessary_points']) >= 2]
    dev.sort(key=lambda t: hashlib.sha256(f"calib|{t['task_id']}".encode()).hexdigest())
    chosen = []
    for qtype, n in PICK.items():
        chosen += [t for t in dev if t['question_type'] == qtype][:n]

    pool_rows = {}
    with (s5 / 'candidates_unlabeled.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if r['task_id'] in {t['task_id'] for t in chosen} and r['retrieval']['rank'] <= 20:
                pool_rows.setdefault(r['task_id'], []).append(r)

    questions, pools, q1, p1 = [], [], [], []
    for k, t in enumerate(chosen, 1):
        qid = f'CAL{k}'
        types = {p['text']: p.get('type') for p in t.get('original_necessary_points', [])}
        points = [{'pid': f'P{i}', 'type': types.get(p['text']) or ('治疗原则' if t['question_type'] == 'principle' else '病机'),
                   'text': p['text']} for i, p in enumerate(t['necessary_points'], 1)]
        questions.append({'qid': qid, 'origin_id': t['task_id'], 'question_type': t['question_type'],
                          'question_final': t['question'], 'points': points, 'included': True})
        for r in pool_rows[t['task_id']]:
            pools.append({'qid': qid, 'rank': r['retrieval']['rank'], 'document_id': r['document_id'],
                          'source': r['retrieval']['source'], 'bm25_score': r['retrieval']['score'],
                          'sha256': r['retrieval']['sha256'], 'text': r['text'], 'title': r.get('title'),
                          'context_text': r.get('context_text'), 'context_status': r.get('context_status')})
        rv = review.get(t['task_id'], {})
        q1.append({'题目编号': qid, '题型': QTYPE_CN[t['question_type']], '检索用问题文本(草稿)': t['question'],
                   '原始病情描述(参考)': (rv.get('original_question') or '').split('\n')[0],
                   '原始辨证分析(要点来源)': rv.get('reference_analysis') or '', '原始治疗记录(参考)': rv.get('reference_treatment') or '',
                   '草拟要点汇总': ' | '.join(f"{p['pid']}[{p['type']}] {p['text']}" for p in points),
                   '自动标记的疑似泄露片段': '', '【填】纳入(1纳入/0排除)': '', '【填】排除原因': '',
                   '【填】问题含答案泄露(1是/0否)': '', '【填】应删除的泄露片段(原文照抄,多处用||分隔)': '',
                   '【填】应补充的关键病情信息(从原始病情描述照抄,多处用||分隔)': '',
                   '【填】遗漏的必要要点(类型:内容,多条用；分隔)': '', '【填】备注': ''})
        for p in points:
            p1.append({'题目编号': qid, '要点编号': p['pid'], '题型': QTYPE_CN[t['question_type']],
                       '检索用问题文本(草稿)': t['question'], '原始辨证分析(要点来源)': rv.get('reference_analysis') or '',
                       '草拟要点类型': p['type'], '草拟要点内容': p['text'], '来源原句': '', '需精简提示': '',
                       '【填】判定(1保留/2修改后保留/0删除)': '', '【填】修改后内容(拆分为多条用；分隔)': '',
                       '【填】修改后类型(疾病诊断/证候/病机/治疗原则)': '', '【填】备注': ''})

    cal = out / 'phase0_校准练习'
    cal.mkdir(parents=True, exist_ok=True)
    sealed = out / '_sealed_负责人保存_勿发给标注员'
    sealed.mkdir(parents=True, exist_ok=True)
    (sealed / 'calibration_questions.json').write_text(json.dumps(questions, ensure_ascii=False, indent=1), encoding='utf-8')
    with (sealed / 'calibration_pools.jsonl').open('w', encoding='utf-8') as f:
        for r in pools:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    for who in ('A', 'B', 'C'):
        write_csv(cal / f'calib_phase1_题目核验_标注员{who}.csv', q1)
        write_csv(cal / f'calib_phase1_要点核验_标注员{who}.csv', p1)
    subprocess.run([sys.executable, str(Path(__file__).resolve().parent / 'make_phase2_csv.py'),
                    '--questions', str(sealed / 'calibration_questions.json'),
                    '--pools', str(sealed / 'calibration_pools.jsonl'),
                    '--out', str(cal), '--sealed-dir', str(sealed), '--k', '20',
                    '--prefix', 'calib_phase2_证据标注', '--seed', 'calib-20261024'], check=True)
    print('calibration questions:', [(q['qid'], q['origin_id'], q['question_type']) for q in questions])


if __name__ == '__main__':
    main()
