import json,collections,hashlib
from pathlib import Path
import numpy as np
from evaluate_v2 import per_question,summarize,classify,write_json
def policies(rows,scores,meta,budget=.1):
 grid=[]
 for tau in np.arange(.05,1,.05):
  for m in [1,2,4,6]:
   metrics=summarize(per_question(rows,scores,meta,float(tau),m));ok=metrics['negative_selection_rate']<=budget and metrics['selected_count']<=4 and metrics['empty']<=.02
   grid.append({'threshold':float(tau),'m':m,'feasible':bool(ok),**metrics})
 good=[p for p in grid if p['feasible']]
 chosen=max(good,key=lambda p:(p['full_coverage'],p['coverage'],-p['selected_count'],-p['negative_selection_rate'],-abs(p['threshold']-.5))) if good else None
 return chosen,grid
def checkpoint_key(rows,scores,meta):
 chosen,grid=policies(rows,scores,meta)
 if chosen:return (1,chosen['full_coverage'],chosen['coverage'],-chosen['selected_count']),chosen,grid
 top=summarize(per_question(rows,scores,meta,0,2,'top_m'))
 return (0,top['coverage'],top['full_coverage'],-top['negative_selection_rate']),None,grid
def export_run(data,devscores,testscores,out,objective):
 out=Path(out);rows=data['test'];meta=data['meta'];label='y_relevance' if objective=='relevance' else 'y_support'
 own_dev=[{**p,'y_support':p[label]} for p in data['dev']]
 taus=[i/20 for i in range(1,20)];f1tau=max(taus,key=lambda t:(classify(own_dev,devscores,t)['f1'],-abs(t-.5)))
 own_test=[{**p,'y_support':p[label]} for p in rows]
 write_json(out/'own_target_metrics.json',{'target':label,'threshold_dev_own_f1':f1tau,'dev':classify(own_dev,devscores,f1tau),'test':classify(own_test,testscores,f1tau)})
 reports={};pols={};allquestions={}
 for budget in [.05,.1,.15]:
  chosen,grid=policies(data['dev'],devscores,meta,budget);write_json(out/f'dev_grid_{budget}.json',grid);pols[str(budget)]=chosen
  if chosen:
   name=f'budget_{budget}';qs=per_question(rows,testscores,meta,chosen['threshold'],chosen['m']);allquestions[name]=qs
   reports[name]={s:summarize([q for q in qs if s=='all' or q['source']==s]) for s in ['all','med','hallu']}
 for m in [1,2,4,6]:
  allquestions[f'top_{m}']=per_question(rows,testscores,meta,0,m,'top_m')
 allquestions['own_f1_tau_m4']=per_question(rows,testscores,meta,f1tau,4)
 for name,qs in allquestions.items():
  with (out/f'{name}_questions.jsonl').open('w',encoding='utf-8') as f:
   for q in qs:f.write(json.dumps(q,ensure_ascii=False)+'\n')
  reports.setdefault(name,{s:summarize([q for q in qs if s=='all' or q['source']==s]) for s in ['all','med','hallu']})
 for split,rr,ss in [('dev',data['dev'],devscores),('test',rows,testscores)]:
  with (out/f'{split}_raw_predictions.jsonl').open('w',encoding='utf-8') as f:
   for p,s in zip(rr,ss):f.write(json.dumps({k:v for k,v in p.items() if k!='ids'}|{'score':float(s)},ensure_ascii=False)+'\n')
 ceiling=[]
 for tid in dict.fromkeys(p['task_id'] for p in rows):
  candidates=[p for p in rows if p['task_id']==tid];covered=set().union(*(set(p['covers']) for p in candidates));req=set(meta[tid]['required'])
  ceiling.append({'task_id':tid,'source':meta[tid]['source'],'coverage':len(covered&req)/len(req),'full_coverage':int(req<=covered),'missing_points':sorted(req-covered)})
 write_json(out/'pool_ceiling.json',ceiling);write_json(out/'selected_policies.json',pols);write_json(out/'set_metrics.json',reports)
