"""Score frozen existing checkpoints; no training or label-dependent selection."""
import os,sys,json,hashlib,traceback,argparse
from pathlib import Path
import numpy as np,torch,sentencepiece as spm
from transformers import AutoModelForSequenceClassification
B=Path(__file__).resolve().parent;ROOT=B.parents[1];OLD=ROOT/'experiments/stage4_retraining'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(OLD))
from train_stage4 import predict
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def main(obj):
 torch.set_num_threads(4)
 rr=[json.loads(s) for s in (B/'candidates_unlabeled.jsonl').read_text(encoding='utf-8').splitlines()]
 rr=[r for r in rr if r['split']!='train'];assert len(rr)==11400
 sp=spm.SentencePieceProcessor(model_file='/home/<user>/models/mdeberta-v3-base/spm.model')
 for seed in [42,2026,3407]:
  out=B/f'parent_{obj}_{seed}';out.mkdir(exist_ok=True)
  index=out/'DIRECT_REVIEW_SCORE_INDEX.jsonl'
  index.write_text(''.join(json.dumps({k:r[k] for k in ['candidate_id','task_id','split']},ensure_ascii=False)+'\n' for r in rr),encoding='utf-8')
  if all((out/f'expanded_{v}_scores.npy').exists() for v in ['fragment','context']):
   for v in ['fragment','context']:
    a=np.load(out/f'expanded_{v}_scores.npy');assert a.shape==(11400,) and np.isfinite(a).all()
   save(out/'DIRECT_REVIEW_SCORES_DONE.json',{'status':'EXISTING_SCORES_VALIDATED','seed':seed,'objective':obj,'pairs':11400})
   print('EXISTING',obj,seed,flush=True);continue
  checkpoint=OLD/f'{obj}_{seed}/best_checkpoint.pt';frozen=sha(checkpoint)
  save(out/'DIRECT_REVIEW_SCORE_STATUS.json',{'status':'SCORING','seed':seed,'objective':obj,'checkpoint_sha256':frozen,'training':False})
  check=torch.load(checkpoint,map_location='cpu',weights_only=False)
  model=AutoModelForSequenceClassification.from_pretrained(check['config']['model'],num_labels=2,local_files_only=True,dtype=torch.float32)
  model.load_state_dict(check['model']);del check;model.cuda().eval()
  for variant in ['fragment','context']:
   target=out/f'expanded_{variant}_scores.npy'
   if target.exists():continue
   data=[]
   for r in rr:
    q=sp.encode(r['question'])[:160];t=sp.encode(r['text'] if variant=='fragment' else r['context_text'])[:509-len(q)]
    data.append(r|{'ids':[1]+q+[2]+t+[2]})
   scores=predict(model,data,4);assert scores.shape==(11400,) and np.isfinite(scores).all()
   np.save(target,scores);print('SCORED',obj,seed,variant,len(scores),flush=True)
  del model;torch.cuda.empty_cache();assert sha(checkpoint)==frozen
  save(out/'DIRECT_REVIEW_SCORES_DONE.json',{'status':'FROZEN_CHECKPOINT_SCORES_EXPORTED','seed':seed,'objective':obj,'checkpoint_sha256':frozen,'pairs':11400,'training':False})
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--objective',required=True,choices=['support','relevance']);a=ap.parse_args()
 try:main(a.objective)
 except Exception:save(B/f'DIRECT_SCORE_{a.objective}_FAILED.json',{'traceback':traceback.format_exc()});raise
