"""S4 检索层（封存集，证据金标，本地计算）：口径与 eval_v1 evaluate.py / eval_v2 retrieval.py 相同。"""
import importlib.util, json, sys
from pathlib import Path
UP = Path('/mnt/user-data/uploads/expert_testset_v1')
PACK = Path('/tmp/claude-0/-home-claude/5905fdda-b74d-55f0-9180-2be067827c18/scratchpad/pack_1103')
SUP = Path('/home/claude/sup_results/s4/scores')
spec = importlib.util.spec_from_file_location('ev1', str(UP / 'eval_v1' / 'evaluate.py'))
ev1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev1)
S = UP / '_sealed_负责人保存_勿发给标注员'
questions = ev1.load_questions(S / 'final_questions.json')
pools = ev1.load_pools(S / 'sealed_pools.jsonl')
gold = ev1.load_gold(S / 'phase2_gold_labels.jsonl')
qids = sorted(questions)
def load(p):
    return {(r['qid'], str(r['document_id'])): float(r['score']) for r in map(json.loads, filter(str.strip, open(p, encoding='utf-8')))}
V2 = PACK / 'runs' / 'scores' / 'sealed'
seeds = (42, 2026, 3407)
methods = {
    'G05(S-B)': [V2 / 'support_ce_B_s{}.jsonl'.format(s) for s in seeds],
    'G04(R-B)': [V2 / 'relevance_ce_B_s{}.jsonl'.format(s) for s in seeds],
    'QR06-N(v2)': [V2 / 'qr06_B.jsonl'] if False else [UP / 'eval_v1' / 'scores' / 'sealed' / 'qwen3_reranker_0_6b.jsonl'],
    'QR4B-N(v2)': [V2 / 'qr4b_N.jsonl'],
    'Q06N': [SUP / 'q06_zeroshot' / 'sealed.jsonl'], 'Q4BN': [SUP / 'q4b_zeroshot' / 'sealed.jsonl'],
}
for k, n in (('06', 'q06'), ('4B', 'q4b')):
    methods['Q{}S'.format(k)] = [SUP / '{}_support_s{}'.format(n, s) / 'sealed.jsonl' for s in seeds]
    methods['Q{}R'.format(k)] = [SUP / '{}_relevance_s{}'.format(n, s) / 'sealed.jsonl' for s in seeds]
runs = {m: [(i, load(f), {}) for i, f in enumerate(fs)] for m, fs in methods.items()}
pool_info, vals, seed_vals, picks, tops = ev1.per_question(questions, pools, gold, runs)
boot = ev1.Boot()
MET = ('cov', 'cov_graded', 'cov_partial', 'ndcg', 'nonsup')
res = {'n': len(qids), 'means': {m: {k: ev1.mean(vals[m][k][q] for q in qids) for k in MET} for m in runs},
       'pool_ub': {k: ev1.mean(pool_info[q][k] for q in qids) for k in ('ub', 'ub_partial', 'ub_graded')}, 'tests': {}}
def test(a, b, metric, tag):
    keep = [q for q in qids if vals[a][metric][q] is not None and vals[b][metric][q] is not None]
    d = [vals[a][metric][q] - vals[b][metric][q] for q in keep]
    dist = boot.dist(d, tag)
    return {'a': a, 'b': b, 'n': len(d), 'diff': sum(d) / len(d), 'ci95': [ev1.percentile(dist, .025), ev1.percentile(dist, .975)], 'p': ev1.boot_p(dist)}
fams = {'H1': [('Q06S', 'Q06N'), ('Q4BS', 'Q4BN')], 'H2': [('Q06S', 'Q06R'), ('Q4BS', 'Q4BR')],
        'H3': [('Q4BS', 'Q06S'), ('Q4BN', 'Q06N'), ('Q4BR', 'Q06R')], 'H4': [('Q06S', 'G05(S-B)'), ('Q4BS', 'G05(S-B)')],
        'zs_check': [('Q06N', 'QR06-N(v2)'), ('Q4BN', 'QR4B-N(v2)')]}
for metric in ('cov', 'cov_graded'):
    for f, pairs in fams.items():
        rs = [test(a, b, metric, '{}|{}|{}-{}'.format(metric, f, a, b)) for a, b in pairs]
        if f in ('H1', 'H2', 'H4'):
            for r, p in zip(rs, ev1.holm([r['p'] for r in rs])):
                r['p_holm'] = p
        res['tests']['{}|{}'.format(metric, f)] = rs
    # 交互
    d = [(vals['Q4BS'][metric][q] - vals['Q4BN'][metric][q]) - (vals['Q06S'][metric][q] - vals['Q06N'][metric][q]) for q in qids]
    dist = boot.dist(d, metric + 'inter')
    res['tests'][metric + '|H3_inter'] = {'diff': sum(d) / len(d), 'ci95': [ev1.percentile(dist, .025), ev1.percentile(dist, .975)]}
json.dump(res, open('/home/claude/sup_ana/s4_retrieval_results.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(res['means'], ensure_ascii=False, indent=0)); print(res['pool_ub'])
for k, v in res['tests'].items():
    for r in (v if isinstance(v, list) else [v]):
        print(k, r.get('a', ''), '-', r.get('b', ''), round(r['diff'], 3), [round(x, 3) for x in r['ci95']], round(r.get('p_holm', r.get('p', -1)), 4))
