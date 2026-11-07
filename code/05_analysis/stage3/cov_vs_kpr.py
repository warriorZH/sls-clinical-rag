import json, collections, statistics, sys
sys.argv=['x']
exec(open('s4_retrieval.py').read().split("pool_info, vals")[0])   # 复用载入与 runs
pool_info, vals, seed_vals, picks, tops = ev1.per_question(questions, pools, gold, runs)
npts={q['qid']:len(q['points']) for q in json.load(open('/home/claude/eval_v2_sup/data/rejudge/final_questions.json'))}
k={}
for l in open('/home/claude/sup_results/results/s4/kpr_qwen3_32b_s4.jsonl'):
    r=json.loads(l)
    if r['target']!='sealed' or not r['valid']: continue
    lab={p:(v if (v not in ('1','2') or r['quote_ok'].get(p)) else '0') for p,v in r['labels'].items()}
    k[(r['cond'],r['qid'])]=0 if r['abstain'] else sum(v=='2' for v in lab.values())/npts[r['qid']]
seeds=(42,2026,3407)
cmap={'G05(S-B)':['G05_s%d'%s for s in seeds],'G04(R-B)':['G04_s%d'%s for s in seeds],'Q06N':['Q06N'],'Q4BN':['Q4BN']}
for m in ('Q06S','Q06R','Q4BS','Q4BR'): cmap[m]=['%s_s%d'%(m,s) for s in seeds]
rows=[]
for m,cs in cmap.items():
    sv=list(seed_vals[m].values())
    for i,c in enumerate(cs):
        for q in qids:
            if (c,q) in k: rows.append((sv[i]['cov_graded'][q], k[(c,q)], k.get(('G00',q))))
def b(x): return '0' if x==0 else ('(0,0.25]' if x<=0.25 else ('(0.25,0.5]' if x<=0.5 else '>0.5'))
g=collections.defaultdict(list)
for cov,kk,k0 in rows: g[b(cov)].append((kk,k0))
for key in ('0','(0,0.25]','(0.25,0.5]','>0.5'):
    v=g[key]; print(key, len(v), '有证据 KPR %.3f'%statistics.mean(x for x,_ in v), '同题闭卷 KPR %.3f'%statistics.mean(y for _,y in v))
ub=[pool_info[q]['ub_graded'] for q in qids]
print('pool graded UB: share of questions with UB==0: %.2f, mean %.3f'%(sum(u==0 for u in ub)/len(ub), statistics.mean(ub)))
print('strict UB==0 share %.2f'%(sum(pool_info[q]['ub']==0 for q in qids)/len(qids)))
