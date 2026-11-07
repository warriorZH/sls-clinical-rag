"""Freeze task groups, remove cross-fold case references, balance train only."""
import json, hashlib, re, random, collections
from pathlib import Path
import sentencepiece as spm

BASE=Path('/home/<user>/UUM/EEE_research')
OUT=BASE/'experiments/stage4_retraining'
DATA=OUT/'dataset.json'
def norm(s): return re.sub(r'\s+|[^\w\u4e00-\u9fff]','',s)
def dump(path,obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'split_manifest.json').exists(): raise RuntimeError('Existing split: refusing overwrite')
 tasks=json.loads(DATA.read_text()); parent=list(range(len(tasks)))
 def root(i):
  while parent[i]!=i: parent[i]=parent[parent[i]];i=parent[i]
  return i
 def union(a,b): parent[root(b)]=root(a)
 seen={}; reasons=collections.Counter()
 for i,t in enumerate(tasks):
  keys=[('query',norm(t['question']))]
  original=t.get('original_question',t['question'])
  m=re.search(r'([\u4e00-\u9fff]{1,3}某)[，,]\s*([男女]).{0,25}?(\d{4}年\d{1,2}月\d{1,2}日)出生',original)
  if m: keys.append(('patient',m.group(1)+m.group(2)+m.group(3)))
  diagnosis=''.join(e['text'] for e in t['evidence'] if e.get('source')=='stage1诊断分析')
  if len(norm(diagnosis))>=80: keys.append(('diagnosis',norm(diagnosis)))
  for k in keys:
   if k in seen: union(i,seen[k]); reasons[k[0]]+=1
   else: seen[k]=i
 id_index={t['id']:i for i,t in enumerate(tasks)}
 for a,b in json.loads((OUT/'confirmed_group_links.json').read_text()):
  if a in id_index and b in id_index:union(id_index[a],id_index[b]);reasons['reviewed_same_case']+=1
 groups=collections.defaultdict(list)
 for i,t in enumerate(tasks): groups[root(i)].append(t)
 rng=random.Random(42); gs=list(groups.values());rng.shuffle(gs)
 gs.sort(key=len,reverse=True)
 total=collections.Counter(t['id'].split('_')[0] for t in tasks)
 counts={s:collections.Counter() for s in ['train','dev','test']};ratios={'train':.7,'dev':.15,'test':.15}
 splits={s:[] for s in counts}; assignment={}; groupmap={}
 for ts in gs:
  c=collections.Counter(t['id'].split('_')[0] for t in ts)
  # Greedy largest relative deficit; ties deterministic after seeded group shuffle.
  s=max(counts,key=lambda s:sum(c[k]*(ratios[s]*total[k]-counts[s][k])/(ratios[s]*total[k]) for k in c))
  gid='g_'+hashlib.sha256('|'.join(sorted(t['id'] for t in ts)).encode()).hexdigest()[:16]
  for t in ts: assignment[t['id']]=s;groupmap[t['id']]=gid;splits[s].append(t['id'])
  counts[s].update(c)
 sp=spm.SentencePieceProcessor(model_file='/home/<user>/models/mdeberta-v3-base/spm.model')
 assert [sp.piece_to_id(x) for x in ['[PAD]','[CLS]','[SEP]','[UNK]']]==[0,1,2,3]
 pools={s:[] for s in splits};meta={};removed=collections.Counter();lengths=[]
 for t in tasks:
  tid=t['id'];s=assignment[tid]
  meta[tid]={'required':[p['id'] for p in t['answer_points'] if p.get('required',True)],'source':tid.split('_')[0],'group':groupmap[tid]}
  q=sp.encode(t['question'])
  for e in t['evidence']:
   origins=set(re.findall(r'(?:med|hallu)_\d{5}',str(e.get('source',''))))|{e.get('origin_task_id')}
   if any(o in assignment and assignment[o]!=s for o in origins): removed[s]+=1;continue
   origin=e.get('origin_task_id')
   if origin in assignment and assignment[origin]!=s: removed[s]+=1;continue
   ids=[1]+q+[2]+sp.encode(e['text'])+[2]
   assert len(ids)<=512
   assert e['support'] in [0,1] and e['relevant'] in [0,1]
   pools[s].append({'task_id':tid,'candidate_id':e['id'],'group':groupmap[tid],'source':meta[tid]['source'],'ids':ids,'question':t['question'],'text':e['text'],'y_support':e['support'],'y_relevance':e['relevant'],'covers':e['covers'],'constructed':bool(e.get('constructed',False))})
   lengths.append(len(ids))
 # Identical pair input must not be shared across folds (question grouping guarantees this).
 hashes={s:{hashlib.sha256((p['question']+'\n'+p['text']).encode()).hexdigest() for p in rows} for s,rows in pools.items()}
 for a,b in [('train','dev'),('train','test'),('dev','test')]: assert not hashes[a]&hashes[b]
 balanced=[]
 for source in total:
  yes=[p for p in pools['train'] if p['source']==source and p['y_support']==1]
  no=[p for p in pools['train'] if p['source']==source and p['y_support']==0]
  n=min(len(yes),len(no));rng.shuffle(yes);rng.shuffle(no);balanced+=yes[:n]+no[:n]
 rng.shuffle(balanced)
 import torch
 payload={'train':balanced,'train_full':pools['train'],'dev':pools['dev'],'test':pools['test'],'meta':meta}
 torch.save(payload,OUT/'prepared.pt')
 report={'dataset_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),'split_seed':42,'groups':len(gs),'largest_group':max(map(len,gs)),'group_links':dict(reasons),'task_ids':splits,'task_group':groupmap,'source_tasks':{s:dict(c) for s,c in counts.items()},'cross_fold_reference_pairs_removed':dict(removed),'pair_counts':{s:len(p) for s,p in pools.items()},'balanced_train_pairs':len(balanced),'balanced_train_labels':{key:dict(collections.Counter(p[key] for p in balanced)) for key in ['y_support','y_relevance']},'min_max_mean_tokens':[min(lengths),max(lengths),sum(lengths)/len(lengths)],'zero_positive_or_negative_tasks':{s:sum(len({p['y_support'] for p in rows if p['task_id']==tid})<2 for tid in splits[s]) for s,rows in pools.items()},'checks':'PASS no group/pair overlap; no cross-fold case provenance; finite integer labels; raw SentencePiece <=512; train-only balancing'}
 dump(OUT/'split_manifest.json',report)
 dump(OUT/'preparation_summary.json',{k:v for k,v in report.items() if k not in ['task_ids','task_group']})
 print(json.dumps({k:v for k,v in report.items() if k not in ['task_ids','task_group']},ensure_ascii=False),flush=True)
if __name__=='__main__': main()
