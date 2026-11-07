"""断言忠实性的敏感性分析：种子取“有值的种子”平均（登记版要求三个种子都有值）。"""
import json, re, random, sys
from collections import defaultdict
from pathlib import Path
PACK = Path(sys.argv[1]); B, SEED = 10000, 20261026
J = PACK / 'runs' / 'judge' / 'sealed'
rd = lambda p: [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]
split = {r['aid']: r for r in rd(J / 'scr_split_A.jsonl') if r.get('valid')}
ver = {r['aid']: r for r in rd(J / 'scr_verify_A.jsonl') if r.get('valid')}
per = defaultdict(lambda: defaultdict(list))   # base -> qid -> [scr per seed]
nclaims = defaultdict(list); nS = defaultdict(int); nAll = defaultdict(int); nC = defaultdict(int)
for aid, v in ver.items():
    if aid not in split: continue
    c = v['cond']; b = re.sub(r'_(s\d+|r\d+)$', '', c)
    labs = [lab if (lab == 'N' or v.get('quote_ok', {}).get(cid, True)) else 'N' for cid, lab in v['labels'].items()]
    if not labs: continue
    per[b][v['qid']].append(sum(x == 'S' for x in labs) / len(labs))
    nS[b] += sum(x == 'S' for x in labs); nC[b] += sum(x == 'C' for x in labs); nAll[b] += len(labs); nclaims[b].append(len(labs))
qv = {b: {q: sum(x) / len(x) for q, x in d.items()} for b, d in per.items()}
def boot(a, b, tag):
    qq = sorted(set(qv[a]) & set(qv[b])); d = [qv[a][q] - qv[b][q] for q in qq]; n = len(d)
    rng = random.Random('{}|{}|{}'.format(SEED, tag, n))
    dist = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    pct = lambda p: dist[int(round(p * (B - 1)))]
    le = sum(x <= 0 for x in dist); ge = sum(x >= 0 for x in dist)
    return n, sum(d) / n, pct(.025), pct(.975), min(1, 2 * min((le + 1) / (B + 1), (ge + 1) / (B + 1)))
print('cond  n_q  SCR(q-mean)  pooled_S  pooled_C  claims/ans')
for b in ['G01', 'G02', 'G03', 'G04', 'G05', 'G06', 'G07', 'G08', 'G09', 'G10', 'G11', 'G12', 'G13', 'O1', 'O2', 'O3', 'O4']:
    if b in qv:
        print('{:5s} {:4d} {:.3f} {:.3f} {:.3f} {:.1f}'.format(b, len(qv[b]), sum(qv[b].values()) / len(qv[b]), nS[b] / nAll[b], nC[b] / nAll[b], sum(nclaims[b]) / len(nclaims[b])))
for a, b in [('G05', 'G04'), ('G05', 'G01'), ('G05', 'G02'), ('G05', 'G08'), ('G05', 'G09'), ('G10', 'G11'), ('G10', 'G08'), ('G12', 'G05'), ('O1', 'O2'), ('O1', 'O3'), ('G05', 'G03')]:
    print('{} - {}: n={} diff={:+.3f} [{:+.3f},{:+.3f}] p={:.4f}'.format(a, b, *boot(a, b, a + b)))
