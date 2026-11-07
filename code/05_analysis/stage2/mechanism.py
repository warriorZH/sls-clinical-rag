"""机制分析：证据利用率 U_s、H8 要点层 GEE、相对闭卷的得失、G05/G04 的错误来源、字面匹配召回率。"""
import json, re, sys, random, warnings
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings('ignore')
PACK = Path(sys.argv[1]); UP = Path(sys.argv[2]); EXTDATA = Path(sys.argv[3]); OUT = Path(sys.argv[4])
S = UP / '_sealed_负责人保存_勿发给标注员'
B, SEED = 10000, 20261026

qs = {q['qid']: q for q in json.load(open(S / 'final_questions.json', encoding='utf-8')) if q.get('included', True)}
gold = {}
for l in open(S / 'phase2_gold_labels.jsonl', encoding='utf-8'):
    if l.strip():
        g = json.loads(l)
        gold[(g['qid'], str(g['document_id']))] = g
pool = defaultdict(list)
for l in open(S / 'sealed_pools.jsonl', encoding='utf-8'):
    if l.strip():
        r = json.loads(l); pool[r['qid']].append(str(r['document_id']))

def sup(qid, doc, pid):
    g = gold.get((qid, doc))
    if not g:
        return 0
    v = g['support'].get(pid)
    return int(v) if v not in (None, '') else 0

def cat(qid, doc):
    g = gold.get((qid, doc))
    if not g:
        return 'unknown'
    if any(int(v or 0) >= 1 for v in g['support'].values()):
        return 'support'
    if int(g.get('mismatch') or 0):
        return 'mismatch'
    if int(g.get('relevance') or 0):
        return 'related'
    return 'irrelevant'

pool_max = {(q, p['pid']): max([sup(q, d, p['pid']) for d in pool[q]] or [0]) for q in qs for p in qs[q]['points']}

def load_target(target, gens):
    judge = {}
    for l in open(PACK / 'runs' / 'judge' / target / 'kpr_final.jsonl', encoding='utf-8'):
        r = json.loads(l); judge[(r['cond'], r['qid'], r.get('gen', 'G1'))] = r
    gen = {}
    for g in gens:
        for f in sorted((PACK / 'runs' / 'gen' / target / g).glob('*.jsonl')):
            for l in open(f, encoding='utf-8'):
                r = json.loads(l); gen[(r['cond'], r['qid'], g)] = r
    return judge, gen

judge, gen = load_target('sealed', ['G1', 'G2'])
base = lambda c: re.sub(r'_(s\d+|r\d+)$', '', c)

# 闭卷命中（G1、G2 各自）
cb = {}
for (c, q, g), r in judge.items():
    if c in ('G00', 'G2_G00'):
        for pid, v in (r.get('labels') or {}).items():
            cb[(g, q, pid)] = int(v == '2' and not r.get('abstain'))

rows = []
for (c, q, g), r in judge.items():
    if q not in qs or r.get('labels') is None:
        continue
    gr = gen.get((c, q, g))
    docs = [str(d) for d in (gr or {}).get('docs') or []]
    for p in qs[q]['points']:
        pid = p['pid']
        lab = r['labels'].get(pid)
        h = int(lab == '2' and not r.get('abstain'))
        s = max([sup(q, d, pid) for d in docs] or [0]) if docs else None
        rows.append({'cond': c, 'base': base(c), 'gen': g, 'qid': q, 'pid': pid, 'ptype': p['type'], 'qtype': qs[q]['question_type'],
                     'h': h, 'partial': int(lab == '1' and not r.get('abstain')), 'abstain': int(bool(r.get('abstain'))),
                     's': s, 'ndocs': len(docs), 'cb': cb.get((g, q, pid)), 'pool_max': pool_max[(q, pid)],
                     'cats': ','.join(sorted(set(cat(q, d) for d in docs))) if docs else '',
                     'literal': int(p['text'] in ((gr or {}).get('answer') or ''))})
df = pd.DataFrame(rows)
res = {}
print('point-level rows', len(df), 'ptype', dict(Counter(df[df.cond == 'G00'].ptype)))

# ---------- U_s
def us_table(sub):
    t = sub.groupby('s').h.agg(['mean', 'count'])
    return {int(k): [float(v['mean']), int(v['count'])] for k, v in t.iterrows()}

families = {
    'E4 (O1–O4)': df[(df.gen == 'G1') & df.base.isin(['O1', 'O2', 'O3', 'O4'])],
    'G1 主提示词 k=2 全部有证据条件': df[(df.gen == 'G1') & df.base.isin(['G01', 'G02', 'G04', 'G05', 'G06', 'G07', 'G08', 'G09', 'G10', 'G11', 'G12', 'G13', 'O1', 'O2', 'O3', 'O4'])],
    'G1 P-alt': df[(df.gen == 'G1') & df.base.str.startswith('PA_') & (df.base != 'PA_G00')],
    'G2 有证据条件': df[(df.gen == 'G2') & (df.base != 'G2_G00')],
}
res['U_s'] = {k: us_table(v) for k, v in families.items()}
for b in ['G01', 'G04', 'G05', 'G08', 'G09', 'G10', 'G11', 'G12', 'G13', 'O1', 'O2']:
    res['U_s'][b] = us_table(df[(df.gen == 'G1') & (df.base == b)])
print(json.dumps(res['U_s'], ensure_ascii=False))

# ---------- H8 GEE
def gee(sub, label):
    d = sub.dropna(subset=['s', 'cb']).copy()
    d['s'] = d['s'].astype(int).astype(str)
    d['cb'] = d['cb'].astype(int)
    m = smf.gee('h ~ C(s, Treatment("0")) + cb + C(ptype) + C(qtype)', groups='qid', data=d,
                family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
    out = {'n_obs': int(len(d)), 'n_q': int(d.qid.nunique()), 'n_s2': int((d.s == '2').sum()), 'n_s1': int((d.s == '1').sum())}
    for term in m.params.index:
        if term.startswith('C(s'):
            k = term.split('T.')[-1].rstrip(']')
            lo, hi = m.conf_int().loc[term]
            out['OR_s{}_vs_s0'.format(k)] = [float(np.exp(m.params[term])), float(np.exp(lo)), float(np.exp(hi)), float(m.pvalues[term])]
    lo, hi = m.conf_int().loc['cb']
    out['OR_closedbook_hit'] = [float(np.exp(m.params['cb'])), float(np.exp(lo)), float(np.exp(hi))]
    print(label, json.dumps(out, ensure_ascii=False))
    return out

res['H8_GEE'] = {k: gee(v, k) for k, v in families.items()}

# ---------- 相对闭卷的得失（G1）
cbd = {(q, p): v for (g, q, p), v in cb.items() if g == 'G1'}
gl = {}
for b in ['G01', 'G02', 'G03', 'G04', 'G05', 'G08', 'G09', 'G10', 'G11', 'G12', 'G13', 'O1', 'O2', 'O3', 'O4',
          'PA_G00', 'PA_G01', 'PA_G04', 'PA_G05', 'PA_G08', 'PA_G09']:
    sub = df[(df.gen == 'G1') & (df.base == b)]
    sub = sub[sub.cb.notna()]
    gain = ((sub.h == 1) & (sub.cb == 0)).mean(); loss = ((sub.h == 0) & (sub.cb == 1)).mean()
    by_s = {}
    if b not in ('PA_G00', 'G03'):
        for s in [0, 1, 2]:
            ss = sub[sub.s == s]
            if len(ss):
                by_s[s] = {'n': int(len(ss)), 'gain': float(((ss.h == 1) & (ss.cb == 0)).mean()), 'loss': float(((ss.h == 0) & (ss.cb == 1)).mean()),
                           'keep_given_cb': float(ss[ss.cb == 1].h.mean()) if (ss.cb == 1).any() else None,
                           'gain_given_not_cb': float(ss[ss.cb == 0].h.mean()) if (ss.cb == 0).any() else None}
    gl[b] = {'n_points': int(len(sub)), 'gain_rate': float(gain), 'loss_rate': float(loss),
             'keep_rate_given_cb_hit': float(sub[sub.cb == 1].h.mean()), 'new_hit_rate_given_cb_miss': float(sub[sub.cb == 0].h.mean()), 'by_s': by_s}
res['gain_loss_vs_closed_book'] = gl
for b, v in gl.items():
    print('{:7s} gain {:.3f} loss {:.3f} keep|cb {:.3f} new|miss {:.3f}'.format(b, v['gain_rate'], v['loss_rate'], v['keep_rate_given_cb_hit'], v['new_hit_rate_given_cb_miss']), {k: (round(x['gain_given_not_cb'] or 0, 3), round(x['keep_given_cb'] or 0, 3), x['n']) for k, x in v['by_s'].items()})

# ---------- 错误来源（要点层，G1 主提示词）
def err(sub):
    c = Counter()
    for _, r in sub.iterrows():
        if r.h == 1: c['命中'] += 1
        elif r.abstain: c['整体拒答'] += 1
        elif r.pool_max == 0: c['召回缺失（池中无支持）'] += 1
        elif (r.s or 0) == 0: c['选择缺失（池中有、未选中）'] += 1
        else: c['利用失败（已选中支持证据仍未写出）'] += 1
    n = sum(c.values())
    return {k: [v, v / n] for k, v in c.most_common()}
res['error_sources'] = {}
for b in ['G01', 'G04', 'G05', 'G08', 'G09', 'G10', 'G12', 'O1']:
    res['error_sources'][b] = err(df[(df.gen == 'G1') & (df.base == b)])
    print(b, {k: round(v[1], 3) for k, v in res['error_sources'][b].items()})

# ---------- 字面匹配召回率（不依赖评审）
def qmean(sub, col):
    return sub.groupby(['qid', 'cond'])[col].mean().groupby('qid').mean().to_dict()

def boot(va, vb, tag):
    qq = sorted(set(va) & set(vb)); d = [va[q] - vb[q] for q in qq]; n = len(d)
    rng = random.Random('{}|{}|{}'.format(SEED, tag, n))
    dist = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    pct = lambda p: dist[int(round(p * (B - 1)))]
    le = sum(1 for x in dist if x <= 0); ge = sum(1 for x in dist if x >= 0)
    return {'n': n, 'diff': sum(d) / n, 'ci95': [pct(.025), pct(.975)], 'p': min(1, 2 * min((le + 1) / (B + 1), (ge + 1) / (B + 1)))}

lit = {}
g1 = df[df.gen == 'G1']
for b in ['G00', 'G01', 'G02', 'G04', 'G05', 'G08', 'G09', 'G10', 'G11', 'G12', 'G13', 'O1', 'O2', 'O3']:
    v = qmean(g1[g1.base == b], 'literal'); lit[b] = sum(v.values()) / len(v)
res['literal_means_sealed'] = lit
res['literal_tests_sealed'] = {}
for a, b in [('G05', 'G04'), ('G05', 'G01'), ('G05', 'G02'), ('G05', 'G00'), ('G05', 'G08'), ('G05', 'G09'), ('G10', 'G11'), ('O1', 'O2')]:
    res['literal_tests_sealed']['{} - {}'.format(a, b)] = boot(qmean(g1[g1.base == a], 'literal'), qmean(g1[g1.base == b], 'literal'), 'lit|' + a + b)

# 外部集
eq = {q['qid']: q for q in json.load(open(EXTDATA / 'external_questions_all.json', encoding='utf-8'))}
ej, eg = load_target('external', ['G1'])
erows = []
for (c, q, g), r in ej.items():
    gr = eg.get((c, q, g))
    if not gr or q not in eq:
        continue
    for p in eq[q]['points']:
        erows.append({'cond': c, 'base': base(c), 'qid': q, 'literal': int(p['text'] in (gr.get('answer') or ''))})
edf = pd.DataFrame(erows)
res['literal_means_external'] = {b: float(np.mean(list(qmean(edf[edf.base == b], 'literal').values()))) for b in sorted(edf.base.unique())}
res['literal_tests_external'] = {}
for a, b in [('X05', 'X04'), ('X05', 'X01'), ('X05', 'X02'), ('X05', 'X00'), ('X05', 'X08'), ('X05', 'X09')]:
    res['literal_tests_external']['{} - {}'.format(a, b)] = boot(qmean(edf[edf.base == a], 'literal'), qmean(edf[edf.base == b], 'literal'), 'elit|' + a + b)
print('literal sealed', {k: round(v, 3) for k, v in lit.items()})
print('literal external', {k: round(v, 3) for k, v in res['literal_means_external'].items()})
for k, t in list(res['literal_tests_sealed'].items()) + list(res['literal_tests_external'].items()):
    print('{:12s} n={} diff={:+.4f} [{:+.4f},{:+.4f}] p={:.4f}'.format(k, t['n'], t['diff'], *t['ci95'], t['p']))

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
