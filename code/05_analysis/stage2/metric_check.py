"""评分指标诊断：评审判定与逐字匹配的交叉表、只看非拒答答案的召回、抽样查看“评审判命中但没有逐字出现”的要点。"""
import json, re, random, sys
from collections import defaultdict, Counter
from pathlib import Path
PACK = Path(sys.argv[1]); UP = Path(sys.argv[2]); EXT = Path(sys.argv[3])
S = UP / '_sealed_负责人保存_勿发给标注员'
base = lambda c: re.sub(r'_(s\d+|r\d+)$', '', c)

def load(target, qfile, gens=('G1',)):
    qs = {q['qid']: q for q in json.load(open(qfile, encoding='utf-8')) if q.get('included', True)}
    judge = {}
    for l in open(PACK / 'runs' / 'judge' / target / 'kpr_final.jsonl', encoding='utf-8'):
        r = json.loads(l); judge[(r['cond'], r['qid'], r.get('gen', 'G1'))] = r
    ans = {}
    for g in gens:
        for f in (PACK / 'runs' / 'gen' / target / g).glob('*.jsonl'):
            for l in open(f, encoding='utf-8'):
                r = json.loads(l); ans[(r['cond'], r['qid'], g)] = r['answer']
    rows = []
    for (c, q, g), r in judge.items():
        if q not in qs or r.get('labels') is None or (c, q, g) not in ans: continue
        a = ans[(c, q, g)]
        for p in qs[q]['points']:
            lab = r['labels'].get(p['pid'])
            rows.append({'cond': c, 'base': base(c), 'qid': q, 'pid': p['pid'], 'ptype': p['type'], 'text': p['text'],
                         'lab': lab, 'abstain': bool(r.get('abstain')), 'lit': p['text'] in a, 'answer': a})
    return rows

def table(rows, conds):
    out = {}
    for b in conds:
        rs = [r for r in rows if r['base'] == b]
        if not rs: continue
        ans = [r for r in rs if not r['abstain']]
        hit = lambda r: r['lab'] == '2' and not r['abstain']
        n = len(rs)
        out[b] = {
            'KPR(评审)': sum(hit(r) for r in rs) / n,
            'KPR(只看未拒答)': sum(r['lab'] == '2' for r in ans) / max(1, len(ans)),
            '逐字(全部)': sum(r['lit'] for r in rs) / n,
            '逐字(只看未拒答)': sum(r['lit'] for r in ans) / max(1, len(ans)),
            '评审命中中逐字出现的比例': sum(r['lit'] for r in rs if hit(r)) / max(1, sum(hit(r) for r in rs)),
            '逐字出现但评审未判命中': sum(1 for r in rs if r['lit'] and not hit(r)) / n,
            '其中来自拒答答案': sum(1 for r in rs if r['lit'] and r['abstain']) / n,
            '要点数': n,
        }
    return out

sealed = load('sealed', S / 'final_questions.json')
ext = load('external', EXT / 'external_questions_all.json')
ts = table(sealed, ['G00', 'G01', 'G04', 'G05', 'G08', 'G09', 'G12', 'O1', 'O2', 'PA_G00', 'PA_G04', 'PA_G05'])
te = table(ext, ['X00', 'X01', 'X04', 'X05', 'X08', 'X09'])
cols = list(next(iter(ts.values())).keys())
print('cond   ' + ' | '.join(cols))
for b, v in list(ts.items()) + list(te.items()):
    print('{:7s}'.format(b) + ' | '.join('{:.3f}'.format(v[c]) if c != '要点数' else str(v[c]) for c in cols))

# 抽样：闭卷中“评审命中但未逐字出现”；G05 中“逐字出现但评审未判命中且未拒答”
rng = random.Random(20261103)
def show(rs, k, title):
    print('\n====', title, '（共 {} 条，抽 {} 条）'.format(len(rs), min(k, len(rs))))
    for r in rng.sample(rs, min(k, len(rs))):
        a = r['answer']
        # 找最可能对应的片段：取答案中与要点共享字符最多的一句
        sents = [s for s in re.split(r'[。；\n]', a) if s.strip()]
        best = max(sents, key=lambda s: len(set(s) & set(r['text']))) if sents else ''
        print('- [{}/{} {}] 要点「{}」 判定 {} | 答案片段：{}'.format(r['qid'], r['cond'], r['ptype'], r['text'], r['lab'], best.strip()[:90]))
show([r for r in sealed if r['base'] == 'G00' and r['lab'] == '2' and not r['lit']], 14, 'G00 评审判命中、要点原文未逐字出现')
show([r for r in sealed if r['base'] == 'G05' and r['lit'] and r['lab'] != '2' and not r['abstain']], 8, 'G05 要点原文出现、评审未判命中（未拒答）')
show([r for r in ext if r['base'] == 'X00' and r['lab'] == '2' and not r['lit']], 8, 'X00 评审判命中、要点原文未逐字出现')
