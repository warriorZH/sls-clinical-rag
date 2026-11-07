"""Full FP32 mDeBERTa CE; no label/coverage metadata enters model inputs."""
import os,time,json,random,argparse,math,traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification,get_linear_schedule_with_warmup
from evaluate_v2 import classify,choose_threshold,evaluate,write_json
BASE=Path('/home/<user>/UUM/EEE_research/experiments/stage4_retraining')
from policy_stage4 import checkpoint_key,export_run
def emit(path,obj):
 obj={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**obj}
 with path.open('a',encoding='utf-8') as f:f.write(json.dumps(obj)+'\n')
 print(json.dumps(obj),flush=True)
def batch(rows):
 n=max(len(p['ids']) for p in rows);ids=torch.zeros((len(rows),n),dtype=torch.long);mask=torch.zeros_like(ids)
 for i,p in enumerate(rows):ids[i,:len(p['ids'])]=torch.tensor(p['ids']);mask[i,:len(p['ids'])]=1
 return {'input_ids':ids.cuda(),'attention_mask':mask.cuda()}
def predict(model,rows,micro):
 model.eval();result=np.empty(len(rows));indices=sorted(range(len(rows)),key=lambda i:len(rows[i]['ids']))
 with torch.no_grad():
  for k in range(0,len(indices),micro):
   ix=indices[k:k+micro];v=model(**batch([rows[i] for i in ix])).logits
   if not torch.isfinite(v).all():raise RuntimeError('Nonfinite inference logits')
   result[ix]=v.softmax(-1)[:,1].cpu().numpy()
 return result
def main(a):
 torch.set_num_threads(4);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
 torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True,warn_only=True)
 data=torch.load(BASE/'prepared.pt',weights_only=False,map_location='cpu');train=data['train'];label='y_relevance' if a.objective=='relevance' else 'y_support'
 out=BASE/(('smoke_' if a.smoke else '')+a.objective+'_'+str(a.seed));out.mkdir(exist_ok=True)
 if (out/'DONE.json').exists():print('Already done');return
 if (out/'step_log.jsonl').exists():raise RuntimeError('Refusing append to previous incomplete run')
 if a.objective=='shuffled':
  train=[dict(p) for p in train];ys=[p[label] for p in train];random.Random(a.seed).shuffle(ys)
  for p,y in zip(train,ys):p[label]=y
 if a.smoke:
  ordered=sorted(train,key=lambda p:len(p['ids']),reverse=True)
  train=ordered[:16]+random.Random(a.seed).sample(train,min(784,len(train)))
  a.epochs=1
 accum=16//a.micro;assert a.micro*accum==16
 steps_per_epoch=math.ceil(math.ceil(len(train)/a.micro)/accum)
 config=vars(a)|{'effective_batch':16,'gradient_accumulation':accum,'max_length':512,'precision':'fp32','lr':1e-5,'weight_decay':.01,'warmup_ratio':.1,'optimizer':'AdamW_foreach_false','checkpoint_selection':'dev_common_set_budget_then_predeclared_top2_fallback','deterministic':'warn_only','train_pairs':len(train),'total_steps':steps_per_epoch*a.epochs,'model':'/home/<user>/models/mdeberta-v3-base','dataset_sha256':json.loads((BASE/'split_manifest.json').read_text(encoding='utf-8'))['dataset_sha256'],'torch':torch.__version__,'visible_device':os.environ.get('CUDA_VISIBLE_DEVICES'),'gpu':torch.cuda.get_device_name(0)}
 write_json(out/'config.json',config)
 model=AutoModelForSequenceClassification.from_pretrained(config['model'],num_labels=2,local_files_only=True,dtype=torch.float32).cuda()
 assert all(p.dtype==torch.float32 for p in model.parameters()),'Expected every model parameter FP32'
 model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
 optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=.01,foreach=False)
 scheduler=get_linear_schedule_with_warmup(optimizer,int(.1*config['total_steps']),config['total_steps'])
 probe=next(p for n,p in model.named_parameters() if 'encoder.layer.0' in n and p.requires_grad);before=probe.detach().flatten()[:512].cpu().clone()
 step=0;best=(-1,);patience=0;start=time.monotonic();rng=random.Random(a.seed);history=[]
 for epoch in range(a.epochs):
  model.train();ix=list(range(len(train)));rng.shuffle(ix)
  # Local length sorting controls padding; shuffle batches so length is not a curriculum.
  chunks=[sorted(ix[k:k+256],key=lambda i:len(train[i]['ids'])) for k in range(0,len(ix),256)]
  batches=[c[k:k+a.micro] for c in chunks for k in range(0,len(c),a.micro)];rng.shuffle(batches)
  if a.smoke:
   longix=sorted(range(len(train)),key=lambda i:len(train[i]['ids']),reverse=True)[:16]
   batches=[longix[k:k+a.micro] for k in range(0,16,a.micro)]+batches
  for k in range(0,len(batches),accum):
   bs=batches[k:k+accum];n=sum(map(len,bs));optimizer.zero_grad(set_to_none=True);loss_sum=0.;tokens=0
   for bix in bs:
    rr=[train[i] for i in bix];inputs=batch(rr);logits=model(**inputs).logits
    loss=F.cross_entropy(logits,torch.tensor([p[label] for p in rr],device='cuda'),reduction='sum')/n
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
    loss.backward();loss_sum+=float(loss.detach());tokens+=int(inputs['attention_mask'].sum())
   grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
   if not torch.isfinite(grad):raise RuntimeError('Nonfinite gradients')
   optimizer.step();scheduler.step();step+=1
   if step==1 or step%10==0 or k+accum>=len(batches):
    elapsed=time.monotonic()-start
    emit(out/'step_log.jsonl',{'epoch':epoch+1,'step':step,'loss':loss_sum,'grad_norm':float(grad),'lr':scheduler.get_last_lr()[0],'elapsed_sec':elapsed,'mean_sec_per_step':elapsed/step,'peak_memory_mib':torch.cuda.max_memory_allocated()/2**20,'pairs':n,'tokens':tokens})
   if a.smoke and step>=50:break
  if a.smoke:
   delta=float((probe.detach().flatten()[:512].cpu()-before).abs().max())
   assert delta>0,'Backbone failed to update'
   dev=data['dev'][:128];scores=predict(model,dev,a.micro)
   write_json(out/'DONE.json',{'smoke_only':True,'steps':step,'seconds':time.monotonic()-start,'backbone_max_delta':delta,'peak_memory_mib':torch.cuda.max_memory_allocated()/2**20,'finite_dev_predictions':bool(np.isfinite(scores).all()),'micro':a.micro})
   return
  scores=predict(model,data['dev'],a.micro);metrics=classify(data['dev'],scores,.5)
  emit(out/'epoch_metrics.jsonl',{'epoch':epoch+1,'step':step,'elapsed_sec':time.monotonic()-start,'dev_support':metrics})
  history.append(metrics)
  key,chosen,grid=checkpoint_key(data['dev'],scores,data['meta'])
  own=classify([{**p,'y_support':p[label]} for p in data['dev']],scores,.5)
  emit(out/'checkpoint_selection.jsonl',{'epoch':epoch+1,'key':key,'selected_policy':chosen,'own_target':own})
  if key>best:
   best=key;patience=0
   torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch+1,'step':step,'rng_python':rng.getstate(),'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),'config':config},out/'best_checkpoint.pt')
   np.save(out/'best_dev_scores.npy',scores)
  else:patience+=1
  if patience>=2:break
 # Release optimizer slots before deserializing best full checkpoint.
 del optimizer,scheduler;torch.cuda.empty_cache()
 checkpoint=torch.load(out/'best_checkpoint.pt',map_location='cpu',weights_only=False);model.load_state_dict(checkpoint['model']);best_epoch=checkpoint['epoch'];del checkpoint
 dev_scores=np.load(out/'best_dev_scores.npy');threshold=choose_threshold(data['dev'],dev_scores)
 evaluate(data['dev'],dev_scores,data['meta'],threshold,out/'dev')
 scores=predict(model,data['test'],a.micro);evaluate(data['test'],scores,data['meta'],threshold,out/'test');np.save(out/'test_scores.npy',scores)
 export_run(data,dev_scores,scores,out,a.objective)
 from actual_predictions_stage4 import export as export_actual
 export_actual(model,predict,a.micro,out)
 write_json(out/'DONE.json',{'best_epoch':best_epoch,'threshold':threshold,'total_steps':step,'seconds':time.monotonic()-start,'peak_memory_mib':torch.cuda.max_memory_allocated()/2**20,'best_common_dev_key':best})
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--objective',choices=['support','relevance','shuffled'],required=True);parser.add_argument('--seed',type=int,default=42);parser.add_argument('--micro',type=int,default=4);parser.add_argument('--epochs',type=int,default=3);parser.add_argument('--smoke',action='store_true');a=parser.parse_args()
 try:main(a)
 except Exception as e:
  traceback.print_exc();out=BASE/(('smoke_' if a.smoke else '')+a.objective+'_'+str(a.seed));out.mkdir(exist_ok=True);write_json(out/'FAILED.json',{'error':str(e),'traceback':traceback.format_exc()});raise
