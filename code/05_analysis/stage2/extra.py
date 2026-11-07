"""补充：H2c 交互（答案层）、Wilcoxon 稳健性、H3 分层（池中有无支持证据、题型）、McNemar（全对率）。"""
import json, re, random, sys
from collections import defaultdict
from pathlib import Path
from scipy.stats import wilcoxon, binomtest
PACK = Path(sys.argv[1]); UP = Path(sys.argv[2]); B, SEED = 10000, 20261026
S = UP / '_sealed_负责人保存_勿发给标注员'
qs = {q['qid']: q for q in json.load(open(S / 'final_questions.json', encoding='utf-8')) if q.get('included', True)}
gold = {}
for l in open(S / 'phase2_gold_labels.jsonl', encoding='utf-8'):
    if l.strip():
        g = json.loads(l); gold.setdefault(g['qid'], []).append(g)
has_sup = {q: any(int(v or 0) >= 1 for g in gold.get(q, []) for v in g['support'].values()) for q in qs}
has_full = {q: any(int(v or 0) == 2 for g in gold.get(q, []) for v in g['support'].values()) for q in qs}

def load(target):
    per = defaultdict(dict)
    for l in open(PACK / 'runs' / 'judge' / target / 'kpr_final.jsonl', encoding='utf-8'):
        r = json.loads(l); per[r['cond']][r['qid']] = r
    return per
def kpr(r):
    lab = r.get('labels')
    if lab is None: return None
    return 0.0 if r.get('abstain') else sum(v == '2' for v in lab.values()) / len(lab)
def full(r):
    lab = r.get('labels')
    return None if lab is None else float((not r.get('abstain')) and all(v == '2' for v in lab.values()))
def G(per, b, f=kpr):
    cs = [c for c in per if re.fullmatch(re.escape(b) + r'(_s\d+|_r\d+)?', c)]
    vals = defaultdict(list)
    for c in cs:
        for q, r in per[c].items():
            v = f(r)
            if v is not None: vals[q].append(v)
    return {q: sum(v) / len(v) for q, v in vals.items() if len(v) == len(cs)}
def boot(d, tag):
    n = len(d); rng = random.Random('{}|{}|{}'.format(SEED, tag, n))
    dist = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    pct = lambda p: dist[int(round(p * (B - 1)))]
    le = sum(x <= 0 for x in dist); ge = sum(x >= 0 for x in dist)
    return {'n': n, 'diff': sum(d) / n, 'ci95': [pct(.025), pct(.975)], 'p': min(1, 2 * min((le + 1) / (B + 1), (ge + 1) / (B + 1)))}
sealed = load('sealed'); ext = load('external')
g = {b: G(sealed, b) for b in ['G04', 'G05', 'G10', 'G11', 'G00', 'G08']}
# H2c 交互：(G10-G11) - (G05-G04)
qq = sorted(set(g['G04']) & set(g['G05']) & set(g['G10']) & set(g['G11']))
d = [(g['G10'][q] - g['G11'][q]) - (g['G05'][q] - g['G04'][q]) for q in qq]
print('H2c answer-level interaction (G10-G11)-(G05-G04):', boot(d, 'h2c'))
# Wilcoxon
for name, per, a, b in [('H3', sealed, 'G05', 'G04'), ('HX3', ext, 'X05', 'X04'), ('G10-G11', sealed, 'G10', 'G11')]:
    va, vb = G(per, a), G(per, b); q2 = sorted(set(va) & set(vb))
    dd = [va[q] - vb[q] for q in q2]
    print(name, 'Wilcoxon', wilcoxon(dd, zero_method='wilcox').pvalue, 'n_nonzero', sum(1 for x in dd if x != 0), 'pos', sum(1 for x in dd if x > 0), 'neg', sum(1 for x in dd if x < 0))
# H3 分层
for label, sel in [('池中有支持证据(部分或完整)', lambda q: has_sup[q]), ('池中无支持证据', lambda q: not has_sup[q]),
                   ('池中有完整支持', lambda q: has_full[q]), ('诊断题', lambda q: qs[q]['question_type'] == 'diagnosis'),
                   ('治则题', lambda q: qs[q]['question_type'] != 'diagnosis')]:
    q2 = [q for q in sorted(set(g['G05']) & set(g['G04'])) if sel(q)]
    print('H3 stratum', label, boot([g['G05'][q] - g['G04'][q] for q in q2], 'h3' + label))
# McNemar-type on full-correct (seed-level first seed only to keep binary): use seed-averaged -> sign test on questions
fa, fb = G(sealed, 'G05', full), G(sealed, 'G04', full)
q2 = sorted(set(fa) & set(fb)); pos = sum(fa[q] > fb[q] for q in q2); neg = sum(fa[q] < fb[q] for q in q2)
print('full-correct G05 vs G04: mean', sum(fa[q] for q in q2) / len(q2), sum(fb[q] for q in q2) / len(q2), 'sign test', pos, neg, binomtest(pos, pos + neg).pvalue)
