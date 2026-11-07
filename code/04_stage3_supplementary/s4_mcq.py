#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S4：外部集上用官方选择题（S1 的指标）评测 S4 的新条件。流程、提示词、证据段与 S1 完全相同，
只调用官方评测器的选择题部分（evaluate_combined：4 个维度、3 轮）。参照条件（X00、X04、X05）的选择题得分取自 S1。

    python s4_mcq.py [--smoke]
"""
import argparse
import concurrent.futures as cf
import copy
import time

import sys

from common_sup import (ABORT, CFG, EXIT_SERVER_DOWN, OFFICIAL, PROMPTS, RESULTS, Watchdog, append_jsonl,
                        check_abort_on_failure, log, read_jsonl)
import task_official as TO
from s4_common import SEL_DIR, conditions, evidence_text, pools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    from data_loader import TCMDataLoader
    from evaluators import MultipleChoiceEvaluator
    cases = sorted(TCMDataLoader(str(OFFICIAL / 'TCM_SDT.json')).load_cases(), key=lambda c: int(c['case_id']))
    texts = {(q, str(c['document_id'])): evidence_text(c) for q, cs in pools('external').items() for c in cs}
    sel = {}
    for cond in conditions():
        for r in read_jsonl(SEL_DIR / 'external' / '{}.jsonl'.format(cond)):
            sel[(cond, r['qid'])] = r['docs']
    out_dir = RESULTS / 's4'
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / ('mcq_smoke.jsonl' if a.smoke else 'mcq_results.jsonl')
    done = {(r['cond'], r['qid']) for r in read_jsonl(out) if r.get('status') == 'ok'}
    conds = list(conditions())
    if a.smoke:
        conds, cases = conds[:2], cases[:2]
    jobs = [(c, cs) for c in conds for cs in cases if (c, 'EX%03d' % int(cs['case_id'])) not in done]
    log('S4 选择题评测：待做 {} 个“条件×病例”'.format(len(jobs)), 's4')
    client = TO.shared_clients()['gen']

    def work(job):
        cond, case = job
        qid = 'EX%03d' % int(case['case_id'])
        docs = sel[(cond, qid)]
        items = [PROMPTS['evidence_item'].format(i=i + 1, text=texts[(qid, d)]) for i, d in enumerate(docs)]
        block = PROMPTS['evidence_header'].format(evidence=PROMPTS['evidence_sep'].join(items))
        err = ''
        for attempt in range(1, 4):
            if ABORT.is_set():
                return
            audit = []
            try:
                mc = MultipleChoiceEvaluator(CFG['official']['mc_option_seed'])
                scores, first, detail = mc.evaluate_combined(copy.deepcopy(case), TO.EvidenceModel(client, block, audit),
                                                             TO.QuietBar())
                append_jsonl(out, {'cond': cond, 'qid': qid, 'docs': docs, 'status': 'ok', 'attempt': attempt,
                                   'dimension_scores': scores, 'multiple_choice': detail,
                                   'gen_truncated': sum(1 for x in audit if x.get('finish_reason') == 'length')})
                return
            except Exception as e:  # noqa: BLE001
                if check_abort_on_failure(['gen'], 's4'):
                    return
                err = repr(e)[:300]
                time.sleep(5)
        append_jsonl(out, {'cond': cond, 'qid': qid, 'status': 'failed', 'error': err})

    t0 = time.time()
    dog = Watchdog('s4')
    with cf.ThreadPoolExecutor(max_workers=4 if a.smoke else int(CFG['s4']['mcq_workers'])) as ex:
        for i, f in enumerate(cf.as_completed([ex.submit(work, j) for j in jobs]), 1):
            f.result()
            dog.tick()
            if i % 300 == 0 or i == len(jobs):
                log('S4 选择题进度 {}/{}，已用 {:.1f} 分钟'.format(i, len(jobs), (time.time() - t0) / 60), 's4')

    dog.stop()
    if ABORT.is_set():
        sys.exit(EXIT_SERVER_DOWN)


if __name__ == '__main__':
    main()
