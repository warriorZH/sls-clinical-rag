"""按登记方案（bootstrap 10,000 次，种子 20261026，族内 Holm）重算全部预注册的答案层比较。"""
import json, re, random, sys
from collections import defaultdict
from pathlib import Path
PACK = Path(sys.argv[1]); HERE = Path(sys.argv[2]); B, SEED = 10000, 20261026
J = lambda t: PACK / 'runs' / 'judge' / t
rd = lambda p: [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]

def holm(ps):
    m = len(ps); order = sorted(range(m), key=lambda i: ps[i]); out = [0] * m; run = 0
    for k, i in enumerate(order):
        run = max(run, min(1, (m - k) * ps[i])); out[i] = run
    return out

def per_seed(target, with_scr=False):
    per = defaultdict(dict)
    split = ver = {}
    if with_scr:
        split = {r['aid']: r for r in rd(J(target) / 'scr_split_A.jsonl') if r.get('valid')}
        ver = {r['aid']: r for r in rd(J(target) / 'scr_verify_A.jsonl') if r.get('valid')}
    for r in rd(J(target) / 'kpr_final.jsonl'):
        if r.get('gen', 'G1') != 'G1' or r.get('set') not in ('main', 'oracle', 'e2', 'ext'):
            continue
        labs = list((r.get('labels') or {}).values())
        if not labs: continue
        row = {'kpr': sum(x == '2' for x in labs) / len(labs)}
        sp, va = split.get(r['aid']), ver.get(r['aid'])
        if sp and va:
            la = [l if (l == 'N' or va.get('quote_ok', {}).get(c, True)) else 'N' for c, l in va['labels'].items()]
            if la:
                row['scr'] = sum(x == 'S' for x in la) / len(la); row['ghr'] = sum(x in ('C', 'N') for x in la) / len(la)
        per[r['cond']][r['qid']] = row
    return per

def group(per, b, k):
    cs = [c for c in per if re.fullmatch(re.escape(b) + r'(_s\d+|_r\d+)?', c)]
    qs = set.intersection(*[set(per[c]) for c in cs])
    out = {}
    for q in qs:
        vals = [per[c][q] for c in cs]
        if all(k in v for v in vals):
            out[q] = sum(v[k] for v in vals) / len(vals)
    return out

def boot(va, vb, tag, qsub=None):
    qq = sorted(set(va) & set(vb) & (set(qsub) if qsub else set(va)))
    d = [va[q] - vb[q] for q in qq]; n = len(d)
    rng = random.Random('{}|{}|{}'.format(SEED, tag, n))
    dist = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    pct = lambda p: dist[int(round(p * (B - 1)))]
    le = sum(x <= 0 for x in dist); ge = sum(x >= 0 for x in dist)
    return {'n': n, 'diff': sum(d) / n, 'ci95': [pct(.025), pct(.975)], 'ci90': [pct(.05), pct(.95)], 'p': min(1, 2 * min((le + 1) / (B + 1), (ge + 1) / (B + 1)))}

sealed = per_seed('sealed', True); ext = per_seed('external')
oracle = defaultdict(set)
for r in rd(HERE / 'data' / 'oracle_lists.jsonl'):
    oracle[r['list']].add(r['qid'])
sub70 = oracle['O1'] & oracle['O2'] & oracle['O3']
fams = {
    'H3': [(sealed, 'G05', 'G04', 'kpr', None)],
    'B 答案层（H4、H7）': [(sealed, 'G05', 'G01', 'kpr', None), (sealed, 'G05', 'G00', 'kpr', None), (sealed, 'G05', 'G02', 'kpr', None), (sealed, 'G05', 'G08', 'kpr', None), (sealed, 'G05', 'G09', 'kpr', None)],
    'C 忠实性（H5）': [(sealed, 'G05', 'G04', 'scr', None), (sealed, 'G05', 'G01', 'scr', None), (sealed, 'G05', 'G04', 'ghr', None), (sealed, 'G05', 'G01', 'ghr', None)],
    'D 机制（H6）': [(sealed, 'O1', 'O2', 'kpr', sub70), (sealed, 'O2', 'G00', 'kpr', None), (sealed, 'O1', 'G00', 'kpr', None), (sealed, 'O3', 'G00', 'kpr', None)],
    'G HX3': [(ext, 'X05', 'X04', 'kpr', None)],
    'G HX4': [(ext, 'X05', 'X01', 'kpr', None), (ext, 'X05', 'X02', 'kpr', None), (ext, 'X05', 'X00', 'kpr', None)],
    'G HX7': [(ext, 'X05', 'X08', 'kpr', None), (ext, 'X05', 'X09', 'kpr', None)],
}
out = {}
for fam, items in fams.items():
    res = []
    for per, a, b, k, qs in items:
        r = boot(group(per, a, k), group(per, b, k), '{}|{}|{}'.format(a, b, k), qs); r.update(a=a, b=b, metric=k); res.append(r)
    hp = holm([r['p'] for r in res])
    for r, h in zip(res, hp):
        r['p_holm'] = h if len(res) > 1 else r['p']
    out[fam] = res
    for r in res:
        print('{:16s} {:4s}-{:4s} {:4s} n={:3d} d={:+.3f} 95%[{:+.3f},{:+.3f}] 90%[{:+.3f},{:+.3f}] p={:.4f} holm={:.4f}'.format(fam, r['a'], r['b'], r['metric'], r['n'], r['diff'], *r['ci95'], *r['ci90'], r['p'], r['p_holm']))
for b in ['G00', 'G01', 'G02', 'G04', 'G05', 'G08', 'G09', 'G10', 'G11', 'G12', 'G13', 'O1', 'O2', 'O3']:
    v = group(sealed, b, 'kpr'); print(b, len(v), round(sum(v.values()) / len(v), 3))
json.dump(out, open(Path(sys.argv[3]), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
