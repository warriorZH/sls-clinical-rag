"""Select the sealed expert test set and write the Phase 1 annotation files.

Reproducible, score-blind selection:
  * no model score, prediction or label is read;
  * any task that appears in an earlier evaluation / review / audit file is
    excluded ("hard-touched");
  * one question per case; near-duplicates of training cases are excluded.

Usage (from the EEEQA folder):
    python expert_testset_v1/scripts/build_sealed_testset.py --experiments experiments --out expert_testset_v1
"""
import argparse, collections, hashlib, json, random, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from draft_points import (draft_diagnosis_points, atomize_principle_sentence, leak_fragments,
                          polish_points, PLACEHOLDER, EXPLICIT_PRINCIPLE)
from common import write_csv

SEED = '20261024'
TARGETS = {('fresh_med', 'diagnosis'): None,      # take every eligible fresh case
           ('heldout_med', 'diagnosis'): 100,     # fill diagnosis questions to 100 in total
           ('heldout_med', 'principle'): 80}
DIAG_Q = '\n请解释该病例的诊断、证候与病因病机。'
PRIN_Q = '\n请说明适用于该病例的治疗原则，不要求方药剂量。'
IDRE = re.compile(r'(?<![A-Za-z0-9])(fresh_med_\d{5}|med_\d{5}|hallu_\d{5})(?![0-9])')
# Every earlier evaluation / review / audit file that names individual tasks.
# A task named in any of these files is "hard-touched" and cannot enter the
# sealed set.  (Bulk prediction dumps of the reference-guided evaluation, the
# split manifests and rule-based exclusion lists are deliberately not listed:
# they enumerate whole splits without per-question inspection.)
HARD_FILES = [
    'qa_validation_v1/current48_references_sealed.json',
    'qa_validation_v1/server_results/current48_AUTOMATIC_PER_OUTPUT.json',
    'stage2_final_analysis/CASE_AUDIT.md',
    'stage2_final_analysis/DETAILED_ANALYSIS.md',
    'stage2_final_analysis/uploaddir/report/relevance_2026_failure_cases.json',
    'stage2_final_analysis/uploaddir/report/relevance_3407_failure_cases.json',
    'stage2_final_analysis/uploaddir/report/relevance_42_failure_cases.json',
    'stage2_final_analysis/uploaddir/report/shuffled_42_failure_cases.json',
    'stage2_final_analysis/uploaddir/report/support_2026_failure_cases.json',
    'stage2_final_analysis/uploaddir/report/support_3407_failure_cases.json',
    'stage2_final_analysis/uploaddir/report/support_42_failure_cases.json',
    'stage2_optimized_v2/report/bm25_failure_cases.json',
    'stage2_optimized_v2/report/length_overlap_failure_cases.json',
    'stage2_optimized_v2/report/majority_failure_cases.json',
    'stage2_optimized_v2/report/random_failure_cases.json',
    'stage2_optimized_v2/report/tfidf_failure_cases.json',
    'stage3_validation/actual_retrieval_candidates.jsonl',
    'stage3_validation/blind_20_comparison_after_freeze.json',
    'stage3_validation/character_near_duplicates.jsonl',
    'stage3_validation/fresh_to_old_neighbor_audit.json',
    'stage3_validation/near_duplicate_review_sample.json',
    'stage3_validation/near_duplicate_reviewed_examples.json',
    'stage3_validation/retrieval_verification_tasks.json',
    'stage3_validation/scope_residual_issues.json',
    'stage3_validation/template_duplicates.json',
    'stage3_validation/v3_frozen_reevaluation/report/bm25_failure_cases.json',
    'stage3_validation/v3_frozen_reevaluation/report/length_overlap_failure_cases.json',
    'stage3_validation/v3_frozen_reevaluation/report/majority_failure_cases.json',
    'stage3_validation/v3_frozen_reevaluation/report/random_failure_cases.json',
    'stage3_validation/v3_frozen_reevaluation/report/relevance_2026_failure_cases.json',
    'stage3_validation/v3_frozen_reevaluation/report/relevance_3407_failure_cases.json',
    'stage3_validation/v3_frozen_reevaluation/report/relevance_42_failure_cases.json',
    'stage3_validation/v3_frozen_reevaluation/report/shuffled_42_failure_cases.json',
    'stage3_validation/v3_frozen_reevaluation/report/support_2026_failure_cases.json',
    'stage3_validation/v3_frozen_reevaluation/report/support_3407_failure_cases.json',
    'stage3_validation/v3_frozen_reevaluation/report/support_42_failure_cases.json',
    'stage3_validation/v3_frozen_reevaluation/report/tfidf_failure_cases.json',
    'stage4_analysis/ANALYSIS_REPORT.md',
    'stage4_analysis/CASE_AUDIT.md',
    'stage4_analysis/annotation_contract_flags.json',
    'stage4_analysis/case_comparison_examples.json',
    'stage4_analysis/cross_split_neighbor_text_review.json',
    'stage4_analysis/necessary_point_completeness_audit.json',
    'stage4_analysis/placeholder_candidate_audit.json',
    'stage4_analysis/post_training_character_neighbors.json',
    'stage4_verification/SAMPLING_MANIFEST.json',
    'stage4_verification/blind_full_pool_review.json',
    'stage4_verification_v1/README.md',
    'stage4_verification_v1/blind_full_pool_review.json',
    'stage4_verification_v1/candidates_reviewed.jsonl',
    'stage4_verification_v1/reviewed_tasks.json',
    'stage4_verification_v1/server_snapshot_seed42/README.md',
    'stage5_real_retrieval/review_tasks.json',
    'stage5_real_retrieval_v1_2/BLIND_AUDIT_COMPARISON.csv',
    'stage5_real_retrieval_v1_2/BLIND_AUDIT_INPUTS.json',
    'stage5_real_retrieval_v1_2/CODEX_ALL_TASK_STANDARDS.json',
    'stage5_real_retrieval_v1_2/CODEX_PILOT_LABELS.jsonl',
    'stage5_real_retrieval_v1_2/CODEX_PILOT_README.md',
    'stage5_real_retrieval_v1_2/CODEX_PILOT_SELECTED_TASKS_ORIGINAL.json',
    'stage5_real_retrieval_v1_2/CODEX_PILOT_TASKS.json',
    'stage5_real_retrieval_v1_2/CODEX_TEST48_LABELS.jsonl',
    'stage5_real_retrieval_v1_2/CODEX_TEST48_REPORT.md',
    'stage5_real_retrieval_v1_2/CODEX_TEST48_RESULTS.json',
    'stage5_real_retrieval_v1_2/CODEX_TEST53_SELECTION_FREEZE.json',
    'stage5_real_retrieval_v1_2/CODEX_TEST53_TASK_STANDARD_FREEZE.json',
    'stage5_real_retrieval_v1_2/CODEX_THREE_SEED_OUTPUT_RESULTS.json',
    'stage5_real_retrieval_v1_2/DIRECT_CASE_GROUPS.json',
    'stage5_real_retrieval_v1_2/EXPANDED_BLIND_AUDIT_PACK.json',
    'stage5_real_retrieval_v1_2/TRAIN_SUPPLEMENT_FREEZE.json',
    'stage5_real_retrieval_v1_2/expanded_review_tasks.json',
    'stage5_real_retrieval_v1_2/pilot_train_review_pack.json',
    'stage5_real_retrieval_v1_2/review_tasks.json',
    'stage5_real_retrieval_v1_2/train_supplement_reviewed.jsonl',
]


def digest(*parts):
    return hashlib.sha256('|'.join(parts).encode('utf-8')).hexdigest()


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def scan_touched(exp):
    hard = collections.defaultdict(set)
    missing = []
    for rel in HARD_FILES:
        p = exp / rel
        if not p.exists():
            missing.append(rel)
            continue
        for tid in set(IDRE.findall(p.read_text(encoding='utf-8', errors='ignore'))):
            hard[tid].add(rel)
    s4v = exp / 'stage4_verification_v1/SAMPLING_MANIFEST.json'
    for t in load(s4v)['tasks']:
        tid = t if isinstance(t, str) else t.get('task_id')
        hard[tid].add('stage4_verification_v1/SAMPLING_MANIFEST.json#tasks')
    if missing:
        raise FileNotFoundError(f'缺少用于排除已接触题目的文件: {missing}')
    return hard


def background_of(question):
    return question.split('\n')[0].strip()


def char_vectors(texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(analyzer='char', ngram_range=(2, 3), min_df=1, sublinear_tf=True)
    return vec, vec.fit_transform(texts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiments', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    exp, out = Path(a.experiments), Path(a.out)

    hard = scan_touched(exp)
    dataset = load(exp / 'stage4_retraining/dataset.json')
    splits = load(exp / 'stage4_retraining/split_manifest.json')['task_ids']
    fold = {tid: s for s, ids in splits.items() for tid in ids}
    pool = load(exp / 'stage4_analysis/evidence/actual_retrieval/retrieval_verification_tasks.json')
    fresh_all = [t for t in pool if t['evaluation_split'] == 'test_unseen_med']
    train = [t for t in dataset if fold.get(t['id']) == 'train']
    stats = collections.Counter()

    candidates = []   # (stratum, qtype, record)
    for t in fresh_all:
        if t['id'] in hard:
            stats['fresh_excluded_touched'] += 1
            continue
        pts = draft_diagnosis_points(t['reference_analysis'])
        if not any(p['type'] == '疾病诊断' for p in pts) or len(pts) < 2:
            stats['fresh_excluded_too_few_source_points'] += 1
            continue
        candidates.append(('fresh_med', 'diagnosis', {
            'origin_id': t['id'], 'stage1_index': t['stage1_index'], 'group': f"stage1_{t['stage1_index']}",
            'background': background_of(t['question']), 'raw_description': t['original_question'],
            'reference_analysis': t['reference_analysis'], 'reference_treatment': t['reference_treatment'],
            'draft_points': pts}))
    for t in dataset:
        if fold.get(t['id']) != 'test' or not t['id'].startswith('med_'):
            continue
        if t['id'] in hard:
            stats['heldout_excluded_touched'] += 1
            continue
        ans = t.get('answer') or ''
        treat = t['treatment_reference'] if isinstance(t.get('treatment_reference'), str) else ''
        base = {'origin_id': t['id'], 'stage1_index': t['source']['index'], 'group': t['group_key'],
                'background': background_of(t['question']), 'raw_description': t['original_question'].split('\n')[0],
                'reference_analysis': ans, 'reference_treatment': treat}
        dpts = [{'type': {'病机解释': '病机', '诊断解释': '病机'}.get(p['type'], p['type']), 'text': p['text'],
                 'source': p.get('source_assertion', p['text'])}
                for p in t['answer_points'] if p['type'] in ('疾病诊断', '证候', '病机解释', '诊断解释')]
        if any(p['type'] in ('疾病诊断', '证候') for p in dpts) and len(dpts) >= 2:
            candidates.append(('heldout_med', 'diagnosis', dict(base, draft_points=dpts)))
        ppts = []
        for p in t['answer_points']:
            if p['type'] != '治疗原则' or PLACEHOLDER.search(p['text']):
                continue
            atoms = atomize_principle_sentence(p['text'])
            for a_ in atoms:
                ppts.append({'type': '治疗原则', 'text': a_, 'source': p.get('source_assertion', p['text'])})
        if ppts and EXPLICIT_PRINCIPLE.search(ans + treat):
            candidates.append(('heldout_med', 'principle', dict(base, draft_points=ppts)))
    stats['eligible'] = collections.Counter(f'{s}/{q}' for s, q, _ in candidates)

    # Near-duplicate screen against every training case (char 2-3gram TF-IDF).
    from sklearn.metrics.pairwise import cosine_similarity
    texts = [c[2]['background'] for c in candidates] + [background_of(t['question']) for t in train]
    _, mat = char_vectors(texts)
    cand_mat, train_mat = mat[:len(candidates)], mat[len(candidates):]
    sim_train = cosine_similarity(cand_mat, train_mat).max(axis=1)

    order = sorted(range(len(candidates)), key=lambda i: digest(SEED, candidates[i][0], candidates[i][1], candidates[i][2]['origin_id']))
    chosen, used_groups, used_records = [], set(), set()
    counts = collections.Counter()
    targets = dict(TARGETS)
    for phase in [('fresh_med', 'diagnosis'), ('heldout_med', 'principle'), ('heldout_med', 'diagnosis')]:
        for i in order:
            s, q, rec = candidates[i]
            if (s, q) != phase:
                continue
            if phase == ('heldout_med', 'diagnosis') and sum(1 for c in chosen if c['question_type'] == 'diagnosis') >= targets[phase]:
                break
            if phase == ('heldout_med', 'principle') and counts[phase] >= targets[phase]:
                break
            if sim_train[i] >= 0.85:
                stats['excluded_near_duplicate_of_training_case'] += 1
                continue
            if rec['group'] in used_groups or rec['stage1_index'] in used_records:
                stats['excluded_same_case_or_group'] += 1
                continue
            if chosen:
                sims = cosine_similarity(cand_mat[i], cand_mat[[c['_row'] for c in chosen]]).ravel()
                if sims.max() >= 0.85:
                    stats['excluded_near_duplicate_within_sealed_set'] += 1
                    continue
            used_groups.add(rec['group']); used_records.add(rec['stage1_index'])
            counts[phase] += 1
            question = rec['background'] + (DIAG_Q if q == 'diagnosis' else PRIN_Q)
            leaks = leak_fragments(rec['background'], rec['draft_points'])
            chosen.append(dict(rec, _row=i, stratum=s, question_type=q, question_draft=question,
                               leak_flags=leaks, max_sim_to_training=round(float(sim_train[i]), 4)))

    rng = random.Random(SEED)
    rng.shuffle(chosen)
    for k, c in enumerate(chosen, 1):
        c['qid'] = f'ST{k:03d}'
        c['draft_points'] = polish_points(c['draft_points'])
        for j, p in enumerate(c['draft_points'], 1):
            p['pid'] = f'P{j}'
        c.pop('_row')

    write_outputs(out, chosen, stats, counts, hard)


def render_points(points):
    return ' | '.join(f"{p['pid']}[{p['type']}] {p['text']}" for p in points)


def write_outputs(out, chosen, stats, counts, hard):
    sealed = out / '_sealed_负责人保存_勿发给标注员'
    sealed.mkdir(parents=True, exist_ok=True)
    p1 = out / 'phase1_题目与要点核验'
    p1.mkdir(parents=True, exist_ok=True)
    qtype_cn = {'diagnosis': '诊断解释（诊断/证候/病机）', 'principle': '治疗原则'}

    qrows, prow = [], []
    for c in sorted(chosen, key=lambda x: x['qid']):
        qrows.append({
            '题目编号': c['qid'], '题型': qtype_cn[c['question_type']],
            '检索用问题文本(草稿)': c['question_draft'],
            '原始病情描述(参考)': c['raw_description'],
            '原始辨证分析(要点来源)': c['reference_analysis'],
            '原始治疗记录(参考)': c['reference_treatment'],
            '草拟要点汇总': render_points(c['draft_points']),
            '自动标记的疑似泄露片段': ' || '.join(c['leak_flags']),
            '【填】纳入(1纳入/0排除)': '', '【填】排除原因': '',
            '【填】问题含答案泄露(1是/0否)': '', '【填】应删除的泄露片段(原文照抄,多处用||分隔)': '',
            '【填】应补充的关键病情信息(从原始病情描述照抄,多处用||分隔)': '',
            '【填】遗漏的必要要点(类型:内容,多条用；分隔)': '', '【填】备注': ''})
        for p in c['draft_points']:
            prow.append({
                '题目编号': c['qid'], '要点编号': p['pid'], '题型': qtype_cn[c['question_type']],
                '检索用问题文本(草稿)': c['question_draft'],
                '原始辨证分析(要点来源)': c['reference_analysis'],
                '草拟要点类型': p['type'], '草拟要点内容': p['text'], '来源原句': p.get('source', ''),
                '需精简提示': '较长，请精简为1-3个核心病机' if p.get('needs_condensing') or len(p['text']) > 30 else '',
                '【填】判定(1保留/2修改后保留/0删除)': '', '【填】修改后内容(拆分为多条用；分隔)': '',
                '【填】修改后类型(疾病诊断/证候/病机/治疗原则)': '', '【填】备注': ''})
    for who in ('A', 'B', 'C'):
        write_csv(p1 / f'phase1_题目核验_标注员{who}.csv', qrows)
        write_csv(p1 / f'phase1_要点核验_标注员{who}.csv', prow)

    full = [{k: v for k, v in c.items()} for c in sorted(chosen, key=lambda x: x['qid'])]
    (sealed / 'sealed_questions_full.json').write_text(json.dumps(full, ensure_ascii=False, indent=1), encoding='utf-8')
    write_csv(sealed / 'id_map.csv', [{'题目编号': c['qid'], 'origin_task_id': c['origin_id'], 'stratum': c['stratum'],
                                       'question_type': c['question_type'], 'stage1_index': c['stage1_index'],
                                       'max_char_sim_to_training': c['max_sim_to_training']} for c in full])
    manifest = {
        'version': 'expert_testset_v1', 'created': '2026-10-24', 'seed': SEED,
        'selection': 'score-blind; hash order sha256(seed|stratum|type|task_id); one question per case/group; '
                     'exclude tasks found in any earlier evaluation/review/audit file; exclude char 2-3gram TF-IDF '
                     'cosine >= 0.85 to any training case or to an already selected question',
        'strata': {'fresh_med': 'Medical Records cases never used as a task, evidence source or pilot in any earlier version '
                                '(from the 100 unseen cases of the stage-4 real-retrieval pool)',
                   'heldout_med': 'stage-4 held-out test split; never used for training/tuning; excluded if it appears in any '
                                  'real-retrieval, review, audit, near-duplicate or QA file'},
        'hallu_not_included': 'only 17 untouched Hallu test items remain and at most 4 fit the case diagnosis/principle scope; '
                              'Hallu results stay exploratory in the paper',
        'counts': {f'{s}/{q}': n for (s, q), n in counts.items()},
        'total_questions': len(chosen),
        'stats': {k: (dict(v) if isinstance(v, collections.Counter) else v) for k, v in stats.items()},
        'hard_touched_task_ids': len(hard),
        'model_scores_or_labels_used': False,
        'points_are_drafts_not_gold': True,
        'sha256_selection': hashlib.sha256(json.dumps([c['origin_id'] for c in full]).encode()).hexdigest(),
    }
    (sealed / 'SELECTION_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
