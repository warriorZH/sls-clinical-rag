"""Score naturally retrieved passages without inventing clinical gold labels."""
import json,time
from pathlib import Path
import numpy as np
import torch
from evaluate_v2 import write_json
BASE=Path('/home/<user>/UUM/EEE_research/experiments/stage4_retraining')
def export(model,predict,micro,out):
 p=BASE/'actual_retrieval/retrieval_prepared.pt'
 if not p.exists():
  write_json(out/'actual_retrieval_pending.json',{'reason':'retrieval pool not yet complete','clinical_label_status':'unresolved'});return
 data=torch.load(p,map_location='cpu',weights_only=False);rows=data['test'];scores=predict(model,rows,micro);np.save(out/'actual_retrieval_scores.npy',scores)
 pol=json.loads((out/'selected_policies.json').read_text(encoding='utf-8')).get('0.1');groups={}
 with (out/'actual_retrieval_predictions.jsonl').open('w',encoding='utf-8') as f:
  for r,s in zip(rows,scores):
   f.write(json.dumps({k:v for k,v in r.items() if k!='ids'}|{'model_score':float(s)},ensure_ascii=False)+'\n');groups.setdefault(r['task_id'],[]).append((r,float(s)))
 selected=[]
 for tid,rs in groups.items():
  import hashlib
  rs.sort(key=lambda x:(-x[1],hashlib.sha256(('42:'+tid+':'+x[0]['candidate_id']).encode()).hexdigest()))
  q={'task_id':tid,'source':rs[0][0]['source'],'question':rs[0][0]['question'],'common_top2':[{'candidate_id':r['candidate_id'],'text':r['text'],'score':s,'retrieval':r['retrieval']} for r,s in rs[:2]],'primary_budget_feasible_on_reference_dev':pol is not None,'clinical_gold_status':'UNRESOLVED','coverage':None,'negative_rate':None}
  if pol:q['dev_locked_selected']=[{'candidate_id':r['candidate_id'],'text':r['text'],'score':s,'retrieval':r['retrieval']} for r,s in rs[:pol['m']] if s>=pol['threshold']]
  selected.append(q)
 write_json(out/'actual_retrieval_questions.json',selected);write_json(out/'actual_retrieval_status.json',{'scored_pairs':len(rows),'questions':len(groups),'threshold_source':'reference_pool_dev_only','actual_pool_labels':'pending_semantic_conditional_review','clinical_accuracy_computed':False})
