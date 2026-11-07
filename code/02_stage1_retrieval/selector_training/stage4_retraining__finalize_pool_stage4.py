"""Validate query-only pool and prepare a reproducible blinded adjudication pack."""
import json,hashlib,random,collections,sys,time
from pathlib import Path
BASE=Path('/home/<user>/UUM/EEE_research/experiments/stage4_retraining');OUT=BASE/'actual_retrieval'
def save(n,v):(OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2))
def main():
 rows=[json.loads(s) for s in (OUT/'actual_retrieval_candidates.jsonl').read_text().splitlines() if s.strip()];tasks=json.loads((OUT/'retrieval_verification_tasks.json').read_text());tm={t['id']:t for t in tasks};split=json.loads((BASE/'split_manifest.json').read_text());train=set(split['task_ids']['train']);dev=set(split['task_ids']['dev']);test=set(split['task_ids']['test']);counts=collections.Counter();grouped=collections.defaultdict(list)
 root=BASE.parents[1];raw={name:(root/'dataset'/name).read_text(encoding='utf-8-sig') for name in ['TCM Ancient books.txt','TCM Educational Materials.txt']};records=json.loads((root/'dataset/stage1_sections_v1_1/medical_records_stage1.json').read_text());fixed=0
 for r in rows:
  assert r['y_support'] is None and r['y_relevance'] is None and r['covers']==[]
  assert 1<=r['retrieval']['rank']<=20 and not r['constructed']
  src=r['retrieval']['source']
  if src in raw:fulltext=raw[src]
  else:
   _,index,field=src.split(':',2);fulltext=records[int(index)][field]
  a=r['retrieval']['start_char'];offset=fulltext[a:a+420].find(r['text']);assert offset>=0,(r['candidate_id'],src,a)
  if offset:fixed+=1
  a+=offset;r['retrieval']['start_char']=a;r['retrieval']['end_char']=a+len(r['text']);assert fulltext[a:a+len(r['text'])]==r['text']
  if src.startswith('medical_train:'):
   tid='med_'+f"{int(src.split(':')[1]):05d}";assert tid in train and tid not in dev|test
   if r['task_id'] not in train:assert tid!=r['task_id']
  counts[tm[r['task_id']]['evaluation_split']]+=1;grouped[r['task_id']].append(r)
 assert all(len(v)==20 for v in grouped.values()) and len(grouped)==len(tasks)
 # Correct strip offsets only; never change query/passage/token input or rank.
 with (OUT/'actual_retrieval_candidates.jsonl').open('w') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 import torch
 p=OUT/'retrieval_prepared.pt';payload=torch.load(p,map_location='cpu',weights_only=False);rm={r['candidate_id']:r['retrieval'] for r in rows}
 for r in payload['test']:r['retrieval']=rm[r['candidate_id']]
 torch.save(payload,p)
 rng=random.Random(20261010);blind=[];sealed=[]
 # Sample by source and fold before model scores; two uniform candidates per task.
 for fold in ['train_retrospective','dev_retrospective','test_retrospective','test_unseen_med']:
  for source in ['med','hallu']:
   ts=[t for t in tasks if t['evaluation_split']==fold and t['source']==source];rng.shuffle(ts)
   for t in ts[:20]:
    for r in rng.sample(grouped[t['id']],2):
     bid=f"newblind_{len(blind)+1:04d}";blind.append({'blind_id':bid,'question':t['question'],'necessary_points':t['answer_points'],'passage':r['text'],'support':None,'relevant':None,'covers':None,'scope_conditions':None,'reason':None,'review_status':'pending_semantic_and_condition_adjudication'})
     sealed.append({'blind_id':bid,'task_id':t['id'],'candidate_id':r['candidate_id'],'source':source,'evaluation_split':fold,'anchor_match_hint':r.get('anchor_covers',[]),'retrieval':r['retrieval']})
 order=list(range(len(blind)));rng.shuffle(order)
 for name,items in [('blind_adjudication_pack.jsonl',[blind[i] for i in order]),('sealed_adjudication_key.jsonl',sealed)]:
  with (OUT/name).open('w') as f:
   for item in items:f.write(json.dumps(item,ensure_ascii=False)+'\n')
 save('pool_validation.json',{'status':'PASS','tasks':len(tasks),'pairs':len(rows),'source_fold_pairs':dict(counts),'twenty_candidates_each':True,'reference_insertion':False,'train_only_case_bank_checked':True,'source_spans_verified':len(rows),'strip_offsets_corrected_without_input_change':fixed,'null_support_is_unresolved_not_negative':True,'blinded_review_pairs':len(blind),'sampling':'uniform task sampling within source/fold and two uniform candidates each; seed20261010; no model outputs used','candidate_file_sha256':hashlib.sha256((OUT/'actual_retrieval_candidates.jsonl').read_bytes()).hexdigest()})
 print('POOL VALIDATED',len(tasks),len(rows),'BLIND',len(blind),flush=True)
if __name__=='__main__':main()
