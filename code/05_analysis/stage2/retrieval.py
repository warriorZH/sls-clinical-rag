"""eval_v2 检索层分析（用证据金标）：E1（H1a–H1d）、E2（H2a、H2b），口径与 eval_v1 的 evaluate.py 相同。"""
import importlib.util, json, sys
from collections import defaultdict
from pathlib import Path

UP = Path('/mnt/user-data/uploads/expert_testset_v1')
PACK = Path(sys.argv[1])
OUT = Path(sys.argv[2])
spec = importlib.util.spec_from_file_location('ev1', str(UP / 'eval_v1' / 'evaluate.py'))
ev1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev1)

S = UP / '_sealed_负责人保存_勿发给标注员'
questions = ev1.load_questions(S / 'final_questions.json')
pools = ev1.load_pools(S / 'sealed_pools.jsonl')
gold = ev1.load_gold(S / 'phase2_gold_labels.jsonl')
qids = sorted(questions)
print('questions', len(qids))


def load_scores(path):
    sc, qtok = {}, {}
    for line in open(path, encoding='utf-8'):
        if line.strip():
            r = json.loads(line)
            sc[(r['qid'], str(r['document_id']))] = float(r['score'])
            if 'q_tokens' in r:
                qtok[r['qid']] = r['q_tokens']
    return sc, qtok


V1 = UP / 'eval_v1' / 'scores' / 'sealed'
V2 = PACK / 'runs' / 'scores' / 'sealed'
methods = {
    'S-A': [V1 / 'support_ce_s{}.jsonl'.format(s) for s in (42, 2026, 3407)],
    'R-A': [V1 / 'relevance_ce_s{}.jsonl'.format(s) for s in (42, 2026, 3407)],
    'S-B': [V2 / 'support_ce_B_s{}.jsonl'.format(s) for s in (42, 2026, 3407)],
    'R-B': [V2 / 'relevance_ce_B_s{}.jsonl'.format(s) for s in (42, 2026, 3407)],
    'S-C': [V2 / 'support_ce_C_s{}.jsonl'.format(s) for s in (42, 2026, 3407)],
    'R-C': [V2 / 'relevance_ce_C_s{}.jsonl'.format(s) for s in (42, 2026, 3407)],
    'BM25': [V1 / 'bm25.jsonl'],
    'BGE-N': [V1 / 'bge_reranker_v2_m3.jsonl'],
    'QR-N': [V1 / 'qwen3_reranker_0_6b.jsonl'],
    'BGE-A': [V2 / 'bge_A.jsonl'], 'BGE-B': [V2 / 'bge_B.jsonl'],
    'QR-A': [V2 / 'qr06_A.jsonl'], 'QR-B': [V2 / 'qr06_B.jsonl'],
    'QR4B-N': [V2 / 'qr4b_N.jsonl'], 'LLM-N': [V2 / 'llmsel_N.jsonl'],
    'BGE-S': [V2 / 'bge_support_ft_s{}.jsonl'.format(s) for s in (42, 2026, 3407)],
    'BGE-R': [V2 / 'bge_relevance_ft_s{}.jsonl'.format(s) for s in (42, 2026, 3407)],
}
runs, qtok_b = {}, {}
for m, files in methods.items():
    runs[m] = []
    for i, f in enumerate(files):
        sc, qt = load_scores(f)
        runs[m].append((i, sc, {}))
        if m == 'S-B':
            qtok_b.update(qt)
pool_info, vals, seed_vals, picks, tops = ev1.per_question(questions, pools, gold, runs)
trunc = sorted(q for q in qids if qtok_b.get(q, 0) > 160)
print('truncated questions (q_tokens>160 under B):', len(trunc))

boot = ev1.Boot()
res = {'n': len(qids), 'n_trunc': len(trunc), 'means': {}, 'tests': []}
for m in runs:
    res['means'][m] = {k: ev1.mean(vals[m][k][q] for q in qids) for k in ('cov', 'cov_graded', 'cov_partial', 'ndcg', 'nonsup', 'full', 'rel_nosup', 'relevant')}
res['pool_ub'] = {k: ev1.mean(pool_info[q][k] for q in qids) for k in ('ub', 'ub_partial', 'ub_graded')}


def pct(d, p):
    return ev1.percentile(d, p)


def test(name, a, b, metric, subset=None, alpha_ci=(0.025, 0.975), tag=None):
    qs = subset or qids
    va, vb = vals[a][metric], vals[b][metric]
    keep = [q for q in qs if va[q] is not None and vb[q] is not None]
    d = [va[q] - vb[q] for q in keep]
    dist = boot.dist(d, tag or name)
    r = {'test': name, 'a': a, 'b': b, 'metric': metric, 'n': len(d), 'diff': sum(d) / len(d),
         'ci95': [pct(dist, 0.025), pct(dist, 0.975)], 'ci90': [pct(dist, 0.05), pct(dist, 0.95)],
         'p': ev1.boot_p(dist), 'sd': (sum((x - sum(d) / len(d)) ** 2 for x in d) / (len(d) - 1)) ** 0.5}
    res['tests'].append(r)
    return r


T = []
T.append(test('H1a', 'S-B', 'S-A', 'cov_graded', trunc))
T.append(test('H1a_strict', 'S-B', 'S-A', 'cov', trunc))
for m in ('cov', 'cov_graded'):
    T.append(test('H1b_bge_' + m, 'S-B', 'BGE-N', m))
    T.append(test('H1b_qr_' + m, 'S-B', 'QR-N', m))
T.append(test('H1c_bge', 'BGE-A', 'BGE-N', 'cov'))
T.append(test('H1c_qr', 'QR-A', 'QR-N', 'cov'))
T.append(test('H1c_bge_graded', 'BGE-A', 'BGE-N', 'cov_graded'))
T.append(test('H1c_qr_graded', 'QR-A', 'QR-N', 'cov_graded'))
T.append(test('H1d', 'S-B', 'R-B', 'cov'))
T.append(test('H1d_graded', 'S-B', 'R-B', 'cov_graded'))
T.append(test('H1_v1', 'S-A', 'R-A', 'cov'))
T.append(test('H2a', 'BGE-S', 'BGE-R', 'cov'))
T.append(test('H2a_graded', 'BGE-S', 'BGE-R', 'cov_graded'))
T.append(test('H2b', 'BGE-S', 'BGE-N', 'cov'))
T.append(test('H2b_graded', 'BGE-S', 'BGE-N', 'cov_graded'))
T.append(test('SB_vs_QR4B', 'S-B', 'QR4B-N', 'cov'))
T.append(test('SB_vs_LLM', 'S-B', 'LLM-N', 'cov'))
T.append(test('SB_vs_BM25', 'S-B', 'BM25', 'cov'))
T.append(test('BGES_vs_SB', 'BGE-S', 'S-B', 'cov'))
T.append(test('BGES_vs_SB_graded', 'BGE-S', 'S-B', 'cov_graded'))
# Holm within families (per metric) as pre-registered: E = H1a, H1b, H1d ; F = H2a, H2b
def holm_family(names):
    ps = [next(t for t in res['tests'] if t['test'] == n)['p'] for n in names]
    adj = ev1.holm(ps)
    for n, a in zip(names, adj):
        next(t for t in res['tests'] if t['test'] == n)['p_holm'] = a
holm_family(['H1a', 'H1b_bge_cov_graded', 'H1b_qr_cov_graded'])
holm_family(['H1b_bge_cov', 'H1b_qr_cov', 'H1d'])
holm_family(['H2a', 'H2b'])
# per-seed stability: H1d per seed
res['seed_cov'] = {m: [ev1.mean(seed_vals[m][s]['cov'][q] for q in qids) for s in seed_vals[m]] for m in ('S-B', 'R-B', 'S-A', 'R-A', 'BGE-S', 'BGE-R')}
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
print('\nmeans (seed-avg):')
for m, d in res['means'].items():
    print('  {:7s} cov {:.3f}  graded {:.3f}  partial {:.3f}  ndcg {:.3f}  nonsup {:.3f}'.format(m, d['cov'], d['cov_graded'], d['cov_partial'], d['ndcg'] or 0, d['nonsup']))
print('pool upper bounds:', {k: round(v, 3) for k, v in res['pool_ub'].items()})
print('\ntests:')
for t in res['tests']:
    print('  {:20s} {:6s}-{:6s} {:10s} n={:3d} diff={:+.4f} ci95=[{:+.4f},{:+.4f}] ci90=[{:+.4f},{:+.4f}] p={:.4f}{}'.format(
        t['test'], t['a'], t['b'], t['metric'], t['n'], t['diff'], *t['ci95'], *t['ci90'], t['p'],
        ' holm={:.4f}'.format(t['p_holm']) if 'p_holm' in t else ''))
print('\nseed cov:', {k: [round(x, 3) for x in v] for k, v in res['seed_cov'].items()})
