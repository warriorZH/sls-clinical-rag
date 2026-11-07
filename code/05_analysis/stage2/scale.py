"""参数量与结果差异：同族不同规模、同规模不同族、同规模不同标签、同标签不同骨干；外部集按病例长度分层；G1/G2 在评审 B 子集上的对照。"""
import importlib.util, json, re, random, sys
from collections import defaultdict
from pathlib import Path
UP = Path('/mnt/user-data/uploads/expert_testset_v1'); PACK = Path(sys.argv[1]); EXT = Path(sys.argv[2]); OUT = Path(sys.argv[3])
B, SEED = 10000, 20261026
spec = importlib.util.spec_from_file_location('ev1', str(UP / 'eval_v1' / 'evaluate.py'))
ev1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev1)
S = UP / '_sealed_负责人保存_勿发给标注员'
questions = ev1.load_questions(S / 'final_questions.json'); pools = ev1.load_pools(S / 'sealed_pools.jsonl'); gold = ev1.load_gold(S / 'phase2_gold_labels.jsonl')
qids = sorted(questions)
def load_scores(path):
    sc = {}
    for line in open(path, encoding='utf-8'):
        if line.strip():
            r = json.loads(line); sc[(r['qid'], str(r['document_id']))] = float(r['score'])
    return sc
V1 = UP / 'eval_v1' / 'scores' / 'sealed'; V2 = PACK / 'runs' / 'scores' / 'sealed'
methods = {'S-B': [V2 / 'support_ce_B_s{}.jsonl'.format(s) for s in (42, 2026, 3407)],
           'R-B': [V2 / 'relevance_ce_B_s{}.jsonl'.format(s) for s in (42, 2026, 3407)],
           'BGE-N': [V1 / 'bge_reranker_v2_m3.jsonl'], 'QR-N': [V1 / 'qwen3_reranker_0_6b.jsonl'],
           'QR4B-N': [V2 / 'qr4b_N.jsonl'], 'LLM-N': [V2 / 'llmsel_N.jsonl'],
           'BGE-S': [V2 / 'bge_support_ft_s{}.jsonl'.format(s) for s in (42, 2026, 3407)],
           'BGE-R': [V2 / 'bge_relevance_ft_s{}.jsonl'.format(s) for s in (42, 2026, 3407)]}
runs = {m: [(i, load_scores(f), {}) for i, f in enumerate(fs)] for m, fs in methods.items()}
pool_info, vals, seed_vals, picks, tops = ev1.per_question(questions, pools, gold, runs)

def boot(d, tag):
    n = len(d); rng = random.Random('{}|{}|{}'.format(SEED, tag, n))
    dist = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    pct = lambda p: dist[int(round(p * (B - 1)))]
    le = sum(x <= 0 for x in dist); ge = sum(x >= 0 for x in dist)
    return {'n': n, 'diff': sum(d) / n, 'ci95': [pct(.025), pct(.975)], 'p': min(1, 2 * min((le + 1) / (B + 1), (ge + 1) / (B + 1)))}
res = {'retrieval': {}, 'answer': {}, 'external_len': {}, 'g1_g2_judgeB': {}}
for a, b in [('QR4B-N', 'QR-N'), ('QR-N', 'BGE-N'), ('BGE-S', 'S-B'), ('BGE-R', 'R-B'), ('LLM-N', 'QR4B-N'), ('S-B', 'R-B'), ('BGE-S', 'BGE-R')]:
    for k in ('cov', 'cov_graded', 'ndcg'):
        d = [vals[a][k][q] - vals[b][k][q] for q in qids if vals[a][k][q] is not None and vals[b][k][q] is not None]
        res['retrieval']['{} - {} [{}]'.format(a, b, k)] = boot(d, 'ret|{}|{}|{}'.format(a, b, k))
res['retrieval_means'] = {m: {k: ev1.mean(vals[m][k][q] for q in qids) for k in ('cov', 'cov_graded', 'ndcg')} for m in runs}

def kload(target):
    per = defaultdict(dict)
    for l in open(PACK / 'runs' / 'judge' / target / 'kpr_final.jsonl', encoding='utf-8'):
        r = json.loads(l); per[r['cond']][r['qid']] = r
    return per
def kpr(r, lab='labels'):
    L = r.get(lab)
    if not L: return None
    return 0.0 if r.get('abstain' if lab == 'labels' else 'abstain_' + lab[-1]) else sum(v == '2' for v in L.values()) / len(L)
def G(per, b, lab='labels', qsub=None):
    cs = [c for c in per if re.fullmatch(re.escape(b) + r'(_s\d+|_r\d+)?', c)]
    vals_ = defaultdict(list)
    for c in cs:
        for q, r in per[c].items():
            if qsub is not None and q not in qsub: continue
            v = kpr(r, lab)
            if v is not None: vals_[q].append(v)
    return {q: sum(v) / len(v) for q, v in vals_.items() if len(v) == len(cs)}
sealed = kload('sealed'); ext = kload('external')
for a, b in [('G13', 'G09'), ('G09', 'G08'), ('G10', 'G05'), ('G11', 'G04'), ('G12', 'G13'), ('G05', 'G04'), ('G10', 'G11')]:
    va, vb = G(sealed, a), G(sealed, b); qq = sorted(set(va) & set(vb))
    res['answer']['{} - {}'.format(a, b)] = boot([va[q] - vb[q] for q in qq], 'ans|' + a + b)
# 外部集：按病例长度三等分
eq = {q['qid']: q for q in json.load(open(EXT / 'external_questions_all.json', encoding='utf-8'))}
L = {q: len(eq[q]['question_final']) for q in eq}
xs = {b: G(ext, b) for b in ['X04', 'X05', 'X08', 'X09', 'X00']}
common = sorted(set.intersection(*[set(v) for v in xs.values()]))
cut = sorted(L[q] for q in common); t1, t2 = cut[len(cut) // 3], cut[2 * len(cut) // 3]
for name, sel in [('短（≤{}字）'.format(t1), lambda q: L[q] <= t1), ('中', lambda q: t1 < L[q] <= t2), ('长（>{}字）'.format(t2), lambda q: L[q] > t2)]:
    qq = [q for q in common if sel(q)]
    row = {'n': len(qq), 'means': {b: sum(xs[b][q] for q in qq) / len(qq) for b in xs}}
    for a, b in [('X09', 'X05'), ('X09', 'X08'), ('X08', 'X05'), ('X05', 'X04')]:
        row['{} - {}'.format(a, b)] = boot([xs[a][q] - xs[b][q] for q in qq], 'len|{}|{}|{}'.format(name, a, b))
    res['external_len'][name] = row
# G1 与 G2 在评审 B 子集上（排除评审 A 自评 G2 的影响）
subB = {q for c in sealed for q, r in sealed[c].items() if r.get('labels_B')}
for lab in ('labels', 'labels_B'):
    row = {}
    for b in ['G00', 'G04', 'G05', 'G2_G00', 'G2_G04', 'G2_G05']:
        v = G(sealed, b, lab, subB); row[b] = (sum(v.values()) / len(v), len(v)) if v else None
    res['g1_g2_judgeB'][lab] = row
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
print('retrieval means', json.dumps({m: {k: round(v, 3) for k, v in d.items()} for m, d in res['retrieval_means'].items()}, ensure_ascii=False))
for sec in ('retrieval', 'answer'):
    for k, t in res[sec].items():
        print('{:9s} {:34s} n={:3d} d={:+.4f} [{:+.4f},{:+.4f}] p={:.4f}'.format(sec, k, t['n'], t['diff'], *t['ci95'], t['p']))
for name, row in res['external_len'].items():
    print('EXT', name, 'n', row['n'], {b: round(v, 3) for b, v in row['means'].items()})
    for k in [x for x in row if ' - ' in x]:
        t = row[k]; print('      {:10s} d={:+.3f} [{:+.3f},{:+.3f}] p={:.4f}'.format(k, t['diff'], *t['ci95'], t['p']))
for lab, row in res['g1_g2_judgeB'].items():
    print('judge', lab, {b: (round(v[0], 3), v[1]) for b, v in row.items() if v})
