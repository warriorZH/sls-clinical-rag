"""答案层补充分析：E2 答案层、E5（k、P-alt、G2）、拒答与选择性准确率、截断与长上下文。"""
import json, random, sys
from collections import defaultdict
from pathlib import Path

PACK = Path(sys.argv[1]); OUT = Path(sys.argv[2])
B, SEED = 10000, 20261026


def load(target):
    rows = [json.loads(l) for l in open(PACK / 'runs' / 'judge' / target / 'kpr_final.jsonl', encoding='utf-8')]
    per = defaultdict(dict)       # cond -> qid -> record
    for r in rows:
        per[r['cond']][r['qid']] = r
    return per


def kpr_of(r, graded=False, labels='labels'):
    lab = r.get(labels)
    if lab is None:
        return None
    if r.get('abstain'):
        return 0.0
    n = len(lab)
    if graded:
        return sum({'2': 1.0, '1': 0.5}.get(v, 0.0) for v in lab.values()) / n
    return sum(1 for v in lab.values() if v == '2') / n


def group_values(per, prefix_list, graded=False):
    """把同一组（种子或抽样）的条件在题内平均。prefix_list：属于该组的条件名列表。"""
    vals = defaultdict(list)
    for c in prefix_list:
        for q, r in per.get(c, {}).items():
            v = kpr_of(r, graded)
            if v is not None:
                vals[q].append(v)
    return {q: sum(v) / len(v) for q, v in vals.items() if len(v) == len(prefix_list)}


def conds_of(per, base):
    import re
    return sorted(c for c in per if re.fullmatch(re.escape(base) + r'(_s\d+|_r\d+)?', c))


def boot_diff(va, vb, tag):
    qs = sorted(set(va) & set(vb))
    d = [va[q] - vb[q] for q in qs]
    n = len(d)
    rng = random.Random('{}|{}|{}'.format(SEED, tag, n))
    dist = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    pct = lambda p: dist[int(round(p * (B - 1)))]
    le = sum(1 for x in dist if x <= 0); ge = sum(1 for x in dist if x >= 0)
    p = min(1.0, 2 * min((le + 1) / (B + 1), (ge + 1) / (B + 1)))
    mean = sum(d) / n
    sd = (sum((x - mean) ** 2 for x in d) / (n - 1)) ** 0.5
    return {'n': n, 'diff': mean, 'ci95': [pct(0.025), pct(0.975)], 'ci90': [pct(0.05), pct(0.95)], 'p': p, 'sd': sd}


per = load('sealed')
res = {'means': {}, 'tests': {}, 'abstain': {}}

def G(base, graded=False):
    return group_values(per, conds_of(per, base), graded)

# validate main table
for base in ['G00', 'G01', 'G02', 'G03', 'G03b', 'G04', 'G05', 'G06', 'G07', 'G08', 'G09', 'G10', 'G11', 'G12', 'G13', 'O1', 'O2', 'O3', 'O4',
             'K1_G01', 'K1_G04', 'K1_G05', 'K1_G08', 'K1_G09', 'K3_G01', 'K3_G04', 'K3_G05', 'K3_G08', 'K3_G09',
             'K5_G01', 'K5_G04', 'K5_G05', 'K5_G08', 'K5_G09', 'PA_G00', 'PA_G01', 'PA_G02', 'PA_G04', 'PA_G05', 'PA_G08', 'PA_G09',
             'G2_G00', 'G2_G01', 'G2_G04', 'G2_G05', 'G2_G08', 'G2_G09', 'G2_O1', 'G2_O2', 'G2_O3']:
    v = G(base); vg = G(base, True)
    cs = conds_of(per, base)
    ab = [1.0 if per[c][q].get('abstain') else 0.0 for c in cs for q in per[c]]
    # selective KPR: among non-abstained answers
    sel = [kpr_of(per[c][q]) for c in cs for q in per[c] if not per[c][q].get('abstain') and kpr_of(per[c][q]) is not None]
    res['means'][base] = {'n': len(v), 'kpr': sum(v.values()) / len(v) if v else None,
                          'graded': sum(vg.values()) / len(vg) if vg else None,
                          'abstain': sum(ab) / len(ab) if ab else None,
                          'kpr_answered': sum(sel) / len(sel) if sel else None}

comps = [
    # E2 answer level
    ('G10', 'G11'), ('G10', 'G08'), ('G11', 'G08'), ('G10', 'G05'), ('G11', 'G04'),
    # stronger baselines
    ('G12', 'G05'), ('G13', 'G05'), ('G09', 'G08'),
    # truncation at answer level
    ('G05', 'G07'), ('G04', 'G06'), ('G07', 'G06'),
    # long context
    ('G03', 'G05'), ('G03b', 'G03'), ('G03', 'G00'),
    # k sensitivity
    ('K1_G05', 'K1_G04'), ('K3_G05', 'K3_G04'), ('K5_G05', 'K5_G04'),
    ('K1_G05', 'K1_G01'), ('K3_G05', 'K3_G01'), ('K5_G05', 'K5_G01'),
    ('K5_G05', 'K5_G08'), ('K5_G05', 'K5_G09'),
    # P-alt
    ('PA_G05', 'PA_G04'), ('PA_G05', 'PA_G01'), ('PA_G05', 'PA_G02'), ('PA_G05', 'PA_G00'), ('PA_G05', 'PA_G08'), ('PA_G05', 'PA_G09'),
    ('PA_G00', 'G00'), ('PA_G05', 'G05'), ('PA_G01', 'PA_G00'), ('PA_G04', 'PA_G00'), ('PA_G09', 'PA_G00'),
    # G2
    ('G2_G05', 'G2_G04'), ('G2_G05', 'G2_G01'), ('G2_G05', 'G2_G00'), ('G2_G05', 'G2_G08'), ('G2_G05', 'G2_G09'),
    ('G2_O1', 'G2_O2'), ('G2_O2', 'G2_G00'), ('G2_O1', 'G2_G00'),
    # oracle vs methods
    ('O1', 'G05'), ('O1', 'O3'), ('O2', 'O3'),
]
for a, b in comps:
    res['tests']['{} - {}'.format(a, b)] = boot_diff(G(a), G(b), '{}|{}'.format(a, b))

ext = load('external')
def GX(base):
    return group_values(ext, conds_of(ext, base))
res['external_means'] = {}
for base in ['X00', 'X01', 'X02', 'X04', 'X05', 'X08', 'X09']:
    v = GX(base); cs = conds_of(ext, base)
    ab = [1.0 if ext[c][q].get('abstain') else 0.0 for c in cs for q in ext[c]]
    sel = [kpr_of(ext[c][q]) for c in cs for q in ext[c] if not ext[c][q].get('abstain') and kpr_of(ext[c][q]) is not None]
    res['external_means'][base] = {'n': len(v), 'kpr': sum(v.values()) / len(v), 'abstain': sum(ab) / len(ab),
                                   'kpr_answered': sum(sel) / len(sel) if sel else None}
for a, b in [('X09', 'X08'), ('X01', 'X04'), ('X09', 'X01')]:
    res['tests']['ext {} - {}'.format(a, b)] = boot_diff(GX(a), GX(b), 'ext|{}|{}'.format(a, b))

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
print('{:8s} {:>4s} {:>6s} {:>6s} {:>7s} {:>8s}'.format('cond', 'n', 'kpr', 'graded', 'abstain', 'kpr_ans'))
for k, m in res['means'].items():
    print('{:8s} {:4d} {:6.3f} {:6.3f} {:7.3f} {:8.3f}'.format(k, m['n'], m['kpr'] or 0, m['graded'] or 0, m['abstain'] or 0, m['kpr_answered'] or 0))
print()
for k, m in res['external_means'].items():
    print('{:8s} {:4d} {:6.3f} abstain {:6.3f} kpr_ans {:6.3f}'.format(k, m['n'], m['kpr'], m['abstain'], m['kpr_answered'] or 0))
print()
for k, t in res['tests'].items():
    print('{:22s} n={:3d} diff={:+.4f} ci95=[{:+.4f},{:+.4f}] ci90=[{:+.4f},{:+.4f}] p={:.4f} sd={:.3f}'.format(k, t['n'], t['diff'], *t['ci95'], *t['ci90'], t['p'], t['sd']))
