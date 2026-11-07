"""Wait for scoring and export output IDs and case groups; never invent labels."""
import json,time,collections,traceback
from pathlib import Path
import numpy as np,torch
B=Path(__file__).resolve().parent
def save(n,v):(B/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 while not all((B/f'parent_{o}_{s}/DIRECT_REVIEW_SCORES_DONE.json').exists() for o in ['support','relevance'] for s in [42,2026,3407]):
  if any((B/f'DIRECT_SCORE_{o}_FAILED.json').exists() for o in ['support','relevance']):raise RuntimeError('Scoring failure; labels unchanged')
  time.sleep(30)
 tasks={t['task_id']:t for t in json.loads((B/'CODEX_ALL_TASK_STANDARDS.json').read_text(encoding='utf-8'))['tasks']}
 index=[json.loads(s) for s in (B/'parent_support_42/DIRECT_REVIEW_SCORE_INDEX.jsonl').read_text(encoding='utf-8').splitlines()]
 groups=collections.defaultdict(list)
 for i,r in enumerate(index):
  if r['task_id'] in tasks and tasks[r['task_id']]['eligible']:groups[r['task_id']].append((i,r))
 result=[]
 for seed in [42,2026,3407]:
  for obj in ['support','relevance']:
   for variant in ['fragment','context']:
    scores=np.load(B/f'parent_{obj}_{seed}/expanded_{variant}_scores.npy');assert len(scores)==len(index)
    for k in [20,50,100]:
     for tid,items in groups.items():
      subset=items[:k];assert len(subset)==k
      order=sorted(subset,key=lambda ir:(-float(scores[ir[0]]),ir[1]['candidate_id']))
      result.append({'task_id':tid,'split':tasks[tid]['split'],'objective':obj,'seed':seed,'variant':variant,'k':k,
          'top6_ids':[r['candidate_id'] for i,r in order[:6]],'top2_ids':[r['candidate_id'] for i,r in order[:2]]})
 save('DIRECT_ALL_SEED_SELECTIONS.json',result)
 old=torch.load(B.parents[1]/'experiments/stage4_retraining/prepared.pt',map_location='cpu',weights_only=False)
 meta=old['meta'];save('DIRECT_CASE_GROUPS.json',{tid:meta.get(tid,{}).get('group',tid) for tid in tasks})
 save('DIRECT_ALL_SEED_EXPORT_DONE.json',{'status':'ALL_SEED_OUTPUT_IDS_EXPORTED_REVIEW_PENDING','records':len(result),'labels_complete':False,'training':False})
if __name__=='__main__':
 try:main()
 except Exception:save('DIRECT_ALL_SEED_EXPORT_FAILED.json',{'traceback':traceback.format_exc()});raise
