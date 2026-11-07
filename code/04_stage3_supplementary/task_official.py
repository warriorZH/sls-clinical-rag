#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S1/S2：在 TCM-BEST4SDT 的 300 例上，按官方评测代码对 13 个证据条件逐例评测。

做法（补充登记第三节）：
  - 每个“条件 × 病例”调用一次官方的 TCMBenchmark._evaluate_single_case（代码原样收录在 official/，未改动）；
  - 被测模型是 Qwen3-4B-Instruct-2507（vLLM 服务）。X00 的提示词与官方完全相同；其余条件在被测模型的
    每一次生成请求最前面加上该条件为本病例选出的 2 条证据（config/prompts_sup.json），证据沿用 eval_v2 的选择结果；
  - 评审 Qwen3-32B 与奖励模型 FangZheng-RM 由官方评测器调用，请求与官方相同，只额外记录返回内容（审计）；
  - 官方代码在调用失败时把该维度记 0 分。这里把含“评测失败”的结果视为未完成，最多重试 3 次；仍失败的按官方
    规则保留 0 分并标记 status=failed，汇总时单独计数并做剔除后的敏感性分析。

用法：python task_official.py [--smoke]
"""
import argparse
import concurrent.futures as cf
import copy
import json
import logging
import sys
import threading
import time
import types

from common_sup import (ABORT, CFG, DATA, EXIT_SERVER_DOWN, LOGS, OFFICIAL, PROMPTS, RESULTS, AuditedClient, Watchdog,
                        append_jsonl, call_retry, check_abort_on_failure, clean_think, log, make_client, read_json,
                        read_jsonl, usage_of, sha256_text)

ROLES = ('gen', 'judge', 'rm')

sys.path.insert(0, str(OFFICIAL))
import tcm_benchmark as TB                    # noqa: E402  官方代码
from data_loader import TCMDataLoader          # noqa: E402
from evaluators import LLMJudgeEvaluator, MultipleChoiceEvaluator, RewardModelEvaluator  # noqa: E402

# 官方模块在导入时把日志设到终端；改为写文件，避免几万行刷屏
_root = logging.getLogger()
for h in _root.handlers[:]:
    _root.removeHandler(h)
_fh = logging.FileHandler(LOGS / 'official_evaluators.log', encoding='utf-8')
_fh.setFormatter(logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s'))
_root.addHandler(_fh)
_root.setLevel(logging.WARNING)


class QuietBar:
    def write(self, *a, **k):
        pass

    def update(self, *a, **k):
        pass

    def set_description(self, *a, **k):
        pass


def evidence_text(c):
    """与 eval_v2 生成答案时相同：context_text，缺失时用 text。"""
    return c.get('context_text') or c.get('text') or ''


def load_evidence():
    pools = {}
    for r in read_jsonl(DATA / 'external_pools.jsonl'):
        pools[(r['qid'], str(r['document_id']))] = r
    sel = {}
    for cond in CFG['conditions']:
        if cond == 'X00':
            continue
        for r in read_jsonl(DATA / 'selections_external' / '{}.jsonl'.format(cond)):
            sel[(cond, r['qid'])] = [str(d) for d in r['docs']]
    return pools, sel


def evidence_block(cond, qid, pools, sel):
    if cond == 'X00':
        return '', []
    docs = sel.get((cond, qid))
    if docs is None:
        raise KeyError('缺少证据选择：{} {}'.format(cond, qid))
    items = [PROMPTS['evidence_item'].format(i=i + 1, text=evidence_text(pools[(qid, d)])) for i, d in enumerate(docs)]
    return PROMPTS['evidence_header'].format(evidence=PROMPTS['evidence_sep'].join(items)), docs


class EvidenceModel:
    """被测模型接口（与官方 ModelInterface 的 generate 签名相同）：在每次请求最前面加证据段。"""

    def __init__(self, client, block, audit):
        self.client, self.block, self.audit = client, block, audit
        self.name = CFG['models']['gen']['served_name']

    def generate(self, prompt, max_tokens=4096, temperature=0.0, clean_think=True, system_prompt=None):
        text = (self.block + prompt) if self.block else prompt
        msgs = ([{'role': 'system', 'content': system_prompt}] if system_prompt else []) + [{'role': 'user', 'content': text}]
        mt = min(int(max_tokens), int(CFG['official']['gen_max_tokens_cap']))
        t0 = time.time()
        try:
            r = call_retry(lambda: self.client.chat.completions.create(model=self.name, messages=msgs,
                                                                       temperature=float(temperature), max_tokens=mt))
        except Exception as e:  # noqa: BLE001  官方代码会捕获部分异常并继续评测；记下以便判为未完成
            self.audit.append({'tag': 'gen', 'error': repr(e)[:300]})
            raise
        ch = r.choices[0]
        content = ch.message.content or ''
        self.audit.append({'tag': 'gen', 'finish_reason': ch.finish_reason, 'content': content, 'usage': usage_of(r),
                           'prompt_sha': sha256_text(text)[:16], 'sec': round(time.time() - t0, 2)})
        return globals()['clean_think'](content) if clean_think else content.strip()

    def get_model_info(self):
        return {'model_name': self.name, 'type': 'api'}

    def get_last_reasoning(self):
        return ''


SHARED = {}
_shared_lock = threading.Lock()


def shared_clients():
    with _shared_lock:
        if not SHARED:
            for role in ('gen', 'judge', 'rm'):
                SHARED[role] = make_client(role)
        return SHARED


def run_one(cond, case, pools, sel):
    cl = shared_clients()
    audit = []
    block, docs = evidence_block(cond, 'EX%03d' % int(case['case_id']), pools, sel)
    m = CFG['models']
    cfg = TB.EvaluationConfig(data_path=str(OFFICIAL / 'TCM_SDT.json'), random_seed=CFG['official']['mc_option_seed'],
                              reward_model_name=m['rm']['served_name'], llm_judge_model_name=m['judge']['served_name'],
                              reward_api_port=m['rm']['port'], llm_judge_api_port=m['judge']['port'])
    bench = TB.TCMBenchmark.__new__(TB.TCMBenchmark)      # 不运行 __init__（它会读数据、打乱顺序）；只装配评测器
    bench.config = cfg
    bench.skip_think = bool(CFG['official']['skip_think'])
    bench.multiple_choice_evaluator = MultipleChoiceEvaluator(cfg.random_seed)
    rm = RewardModelEvaluator('127.0.0.1', m['rm']['port'], m['rm']['served_name'], api_key='EMPTY')
    rm.client = AuditedClient(cl['rm'], 'rm', audit)
    judge = LLMJudgeEvaluator('127.0.0.1', m['judge']['port'], m['judge']['served_name'], api_key='EMPTY')
    judge.client = AuditedClient(cl['judge'], 'judge', audit)
    bench.reward_model_evaluator = rm
    bench.llm_judge_evaluator = judge
    model = EvidenceModel(cl['gen'], block, audit)
    t0 = time.time()
    res = bench._evaluate_single_case(copy.deepcopy(case), model, QuietBar())
    failed = [k for k, v in (res.get('model_responses') or {}).items() if isinstance(v, str) and v.startswith('评测失败')]
    if isinstance(res.get('detailed_evaluation_results', {}).get('multiple_choice'), dict) and \
            'error' in res['detailed_evaluation_results']['multiple_choice']:
        failed.append('multiple_choice')
    call_errors = [a for a in audit if a.get('error')]
    if call_errors:          # 有调用最终失败（即使官方代码自行捕获后给出了分数），整例视为未完成、重做
        failed.append('call_error:' + ','.join(sorted({a['tag'] for a in call_errors})))
    rec = {'cond': cond, 'qid': 'EX%03d' % int(case['case_id']), 'case_id': case['case_id'], 'docs': docs,
           'dimension_scores': res.get('dimension_scores', {}), 'model_responses': res.get('model_responses', {}),
           'multiple_choice': res.get('detailed_evaluation_results', {}).get('multiple_choice'),
           'failed_parts': failed, 'sec': round(time.time() - t0, 1),
           'audit_summary': {
               'call_errors': len(call_errors),
               'gen_calls': sum(1 for a in audit if a['tag'] == 'gen' and not a.get('error')),
               'gen_truncated': sum(1 for a in audit if a['tag'] == 'gen' and a.get('finish_reason') == 'length'),
               'judge_calls': sum(1 for a in audit if a['tag'] == 'judge' and not a.get('error')),
               'judge_no_json': sum(1 for a in audit if a['tag'] == 'judge' and not a.get('error') and not a.get('stream')
                                    and not a.get('json_found')),
               'judge_truncated': sum(1 for a in audit if a['tag'] == 'judge' and not a.get('error') and a.get('finish_reason') == 'length'),
               'rm_calls': sum(1 for a in audit if a['tag'] == 'rm' and not a.get('error')),
               'rm_no_json': sum(1 for a in audit if a['tag'] == 'rm' and not a.get('error') and not a.get('json_found')),
               'rm_truncated': sum(1 for a in audit if a['tag'] == 'rm' and not a.get('error') and a.get('finish_reason') == 'length')}}
    return rec, audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    out_dir = RESULTS / ('smoke' if a.smoke else 'official')
    out_dir.mkdir(parents=True, exist_ok=True)
    res_p, err_p, aud_p = out_dir / 'official_results.jsonl', out_dir / 'official_errors.jsonl', out_dir / 'official_audit.jsonl'

    cases = TCMDataLoader(str(OFFICIAL / 'TCM_SDT.json')).load_cases()
    cases = sorted(cases, key=lambda c: int(c['case_id']))
    conds = list(CFG['conditions'])
    if a.smoke:
        cases = cases[:int(CFG['smoke']['official_cases'])]
        conds = list(CFG['smoke']['official_conditions'])
    pools, sel = load_evidence()

    done = {}
    for r in read_jsonl(res_p):
        done[(r['cond'], r['qid'])] = r
    attempts = {}
    for r in read_jsonl(err_p):
        if r.get('server_down'):          # 服务中断造成的失败不计入重试次数
            continue
        k = (r['cond'], r['qid'])
        attempts[k] = attempts.get(k, 0) + 1
    maxa = int(CFG['official']['max_attempts'])
    todo = [(c, cs) for c in conds for cs in cases if (c, 'EX%03d' % int(cs['case_id'])) not in done]
    total = len(conds) * len(cases)
    log('官方评测：共 {} 个“条件×病例”，已完成 {}，本次待做 {}'.format(total, total - len(todo), len(todo)), 'official')

    def work(item):
        cond, case = item
        k = (cond, 'EX%03d' % int(case['case_id']))
        while True:
            if ABORT.is_set():
                return 'aborted'
            n_prev = attempts.get(k, 0)
            try:
                rec, audit = run_one(cond, case, pools, sel)
            except Exception as e:  # noqa: BLE001
                rec, audit = {'cond': cond, 'qid': k[1], 'failed_parts': ['exception'], 'error': repr(e)[:500]}, []
            if rec.get('failed_parts') and check_abort_on_failure(ROLES, 'official'):
                append_jsonl(err_p, {'cond': cond, 'qid': k[1], 'server_down': True, 'failed_parts': rec.get('failed_parts')})
                return 'aborted'
            if not rec.get('failed_parts'):
                rec['status'] = 'ok'
                rec['attempt'] = n_prev + 1
                append_jsonl(aud_p, {'cond': cond, 'qid': k[1], 'audit': audit})
                append_jsonl(res_p, rec)
                return 'ok'
            attempts[k] = n_prev + 1
            append_jsonl(err_p, {'cond': cond, 'qid': k[1], 'attempt': n_prev + 1, 'failed_parts': rec.get('failed_parts'),
                                 'error': rec.get('error') or [rec.get('model_responses', {}).get(p) for p in rec.get('failed_parts', [])]})
            if attempts[k] >= maxa:
                if 'dimension_scores' in rec:          # 官方规则：失败维度记 0 分；保留并标记
                    rec['status'] = 'failed'
                    rec['attempt'] = attempts[k]
                    append_jsonl(aud_p, {'cond': cond, 'qid': k[1], 'audit': audit})
                    append_jsonl(res_p, rec)
                return 'failed'
            time.sleep(5)

    n_ok = n_fail = 0
    t0 = time.time()
    workers = 4 if a.smoke else int(CFG['official']['workers'])
    dog = Watchdog('official')
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, it) for it in todo]
        for i, f in enumerate(cf.as_completed(futs), 1):
            r = f.result()
            dog.tick()
            n_ok += r == 'ok'
            n_fail += r == 'failed'
            if i % 20 == 0 or i == len(futs):
                el = time.time() - t0
                eta = el / i * (len(futs) - i)
                log('官方评测进度 {}/{}（失败 {}），已用 {:.1f} 小时，预计还需 {:.1f} 小时'.format(
                    total - len(todo) + i, total, n_fail, el / 3600, eta / 3600), 'official')
    dog.stop()
    if ABORT.is_set():
        log('官方评测因服务中断暂停：本次成功 {}，失败 {}'.format(n_ok, n_fail), 'official')
        sys.exit(EXIT_SERVER_DOWN)
    log('官方评测结束：成功 {}，失败 {}'.format(n_ok, n_fail), 'official')
    if a.smoke:
        sys.exit(smoke_check(read_jsonl(res_p), len(conds) * len(cases)))


def smoke_check(recs, expected):
    """试跑验收：全部成功；评审与奖励模型输出缺 JSON 或被截断的合计各不超过 1 次；给出用时估计。"""
    bad = [r for r in recs if r.get('status') != 'ok']
    au = [r.get('audit_summary') or {} for r in recs]
    tot = {k: sum(a.get(k, 0) for a in au) for k in ('gen_calls', 'gen_truncated', 'judge_calls', 'judge_no_json',
                                                       'judge_truncated', 'rm_calls', 'rm_no_json', 'rm_truncated')}
    sec = [r.get('sec', 0) for r in recs if r.get('status') == 'ok']
    mean_sec = sum(sec) / len(sec) if sec else 0
    n_all = len(CFG['conditions']) * 300
    rep = {'smoke_records': len(recs), 'expected': expected, 'not_ok': len(bad), 'audit_totals': tot,
           'mean_sec_per_case_4_workers': round(mean_sec, 1),
           'rough_hours_full_run': round(mean_sec * n_all / max(1, int(CFG['official']['workers']) * 0.5) / 3600, 1),
           'example_scores': recs[0]['dimension_scores'] if recs else None}
    problems = []
    if len(recs) < expected or bad:
        problems.append('有病例未成功')
    if tot['judge_no_json'] + tot['judge_truncated'] > 1:
        problems.append('评审输出缺 JSON 或被截断 {} 次'.format(tot['judge_no_json'] + tot['judge_truncated']))
    if tot['rm_no_json'] + tot['rm_truncated'] > 1:
        problems.append('奖励模型输出缺 JSON 或被截断 {} 次'.format(tot['rm_no_json'] + tot['rm_truncated']))
    if tot['gen_calls'] and tot['gen_truncated'] > tot['gen_calls'] // 3:
        problems.append('被测模型输出被截断过多（{}/{}）'.format(tot['gen_truncated'], tot['gen_calls']))
    rep['problems'] = problems
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    log('官方评测试跑：{}；单例平均 {:.0f} 秒；粗略估计正式运行约 {} 小时'.format(
        '通过' if not problems else '未通过（{}）'.format('；'.join(problems)), mean_sec, rep['rough_hours_full_run']), 'official')
    return 2 if problems else 0


if __name__ == '__main__':
    main()
