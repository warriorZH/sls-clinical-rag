"""Recover encoding-failed exports from frozen scores; never train or overwrite runs."""
import os,json,hashlib,importlib.util,time
from pathlib import Path
import numpy as np,torch
BASE=Path('/home/<user>/UUM/EEE_research/experiments/stage4_retraining')
OUT=BASE/'analysis_recovered_42'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 OUT.mkdir(exist_ok=True);assert not (OUT/'RECOVERY_DONE.json').exists()
 original=(BASE/'policy_stage4.py').read_text(encoding='utf-8');(OUT/'original_policy_stage4.py.txt').write_text(original,encoding='utf-8')
 corrected=original.replace(".open('w')",".open('w',encoding='utf-8')");modulepath=OUT/'policy_stage4_utf8.py';modulepath.write_text(corrected,encoding='utf-8')
 spec=importlib.util.spec_from_file_location('utf8policy',modulepath);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 data=torch.load(BASE/'prepared.pt',weights_only=False,map_location='cpu');records=[]
 for obj in ['support','relevance']:
  src=BASE/f'{obj}_42';dest=OUT/f'{obj}_42';dest.mkdir(exist_ok=True)
  failure=json.loads((src/'FAILED.json').read_text());assert "'ascii' codec" in failure['error']
  paths=[src/'best_dev_scores.npy',src/'test_scores.npy'];before={str(p.relative_to(BASE)):sha(p) for p in paths}
  ds=np.load(paths[0]);ts=np.load(paths[1]);assert len(ds)==len(data['dev']) and len(ts)==len(data['test']) and np.isfinite(ds).all() and np.isfinite(ts).all()
  module.export_run(data,ds,ts,dest,obj)
  old=json.loads((src/'own_target_metrics.json').read_text());new=json.loads((dest/'own_target_metrics.json').read_text());assert old==new,'Frozen own-target metrics changed'
  checkpoint=torch.load(src/'best_checkpoint.pt',weights_only=False,map_location='cpu');epoch=checkpoint['epoch'];steps=checkpoint['step'];config=checkpoint['config'];del checkpoint
  own=json.loads((dest/'own_target_metrics.json').read_text());pols=json.loads((dest/'selected_policies.json').read_text());met=json.loads((dest/'set_metrics.json').read_text())
  assert before=={str(p.relative_to(BASE)):sha(p) for p in paths}
  rec={'objective':obj,'seed':42,'status':'EXISTING_PREDICTIONS_RECOVERED_NOT_NEW_TRAINING','best_epoch':epoch,'checkpoint_step':steps,'config':config,'original_failure':failure,'score_hashes':before,'own_target_metrics_identical':True,'original_files_changed':False,'clinical_gold':False};save(dest/'RECOVERY.json',rec);records.append(rec)
  print('RECOVERED',obj,'epoch',epoch,'policies',pols,'top2',met['top_2']['all'],flush=True)
 save(OUT/'RECOVERY_DONE.json',{'runs':records,'original_queue_still_failed':True,'trained_seeds':[42],'unstarted_seeds':[2026,3407],'training_restarted':False,'actual_retrieval_model_predictions_complete':False})
 # Archive textual evidence, scores and prior baseline exports; exclude model/data tensors.
 import tarfile
 archive=BASE/'stage4_analysis_evidence.tar.gz'
 with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
  for p in sorted(BASE.rglob('*')):
   if not p.is_file() or p==archive or '__pycache__' in p.parts:continue
   if p.suffix not in ['.json','.jsonl','.csv','.npy','.png','.md','.py','.log','.txt']:continue
   if p.name=='dataset.json':continue
   tar.add(p,arcname=str(p.relative_to(BASE)))
 print('ARCHIVE',archive,archive.stat().st_size,sha(archive),flush=True)
if __name__=='__main__':main()
