import json, collections, random, statistics
npts={}
for q in json.load(open('/home/claude/eval_v2_sup/data/rejudge/final_questions.json')): npts[('sealed',q['qid'])]=len(q['points'])
for q in json.load(open('/home/claude/eval_v2_sup/data/external_questions.json')): npts[('external',q['qid'])]=len(q['points'])
k=collections.defaultdict(lambda: collections.defaultdict(list))
seen=set()
for l in open('/home/claude/sup_results/results/rejudge/kpr_qwen3_32b.jsonl'):
    r=json.loads(l)
    if not r.get('valid') or r['aid'] in seen: continue
    seen.add(r['aid'])
    lab={p:(v if (v not in ('1','2') or r['quote_ok'].get(p)) else '0') for p,v in r['labels'].items()}
    g=r['cond'].split('_s')[0] if r['cond'].startswith(('G0','G1','X0','PA_G0','G2_G0','K')) else r['cond']
    k[(r['target'],g)][r['qid']].append(0 if r['abstain'] else sum(v=='2' for v in lab.values())/npts[(r['target'],r['qid'])])
m={g:{q:statistics.mean(v) for q,v in d.items()} for g,d in k.items()}
def boot(a,b,t):
    qs=sorted(set(m[(t,a)])&set(m[(t,b)])); d=[m[(t,a)][q]-m[(t,b)][q] for q in qs]
    rng=random.Random(20261026); n=len(d); bs=sorted(sum(d[rng.randrange(n)] for _ in range(n))/n for _ in range(10000))
    return round(sum(d)/n,3),[round(bs[249],3),round(bs[9749],3)],n
print(sorted({g for (t,g) in m if t=='sealed'})[:80])
for a,b,t in [('G10','G11','sealed'),('PA_G05','PA_G04','sealed'),('G2_G05','G2_G04','sealed'),('G13','G09','sealed'),('G12','G05','sealed'),('K1_G05','K1_G04','sealed'),('K3_G05','K3_G04','sealed'),('K5_G05','K5_G04','sealed'),('G10','G08','sealed'),('G05','G00','sealed'),('X05','X00','external')]:
    try: print(a,b,t,boot(a,b,t))
    except KeyError as e: print('missing',e)
