"""Small continued-training experiment with frozen manually reviewed train pairs.

Original checkpoints/data stay immutable. Expanded evaluation scores are exported
without treating missing labels as negative or inventing gold performance.
"""
import os,sys,time,json,hashlib,random,math,traceback,argparse
from pathlib import Path
import numpy as np,torch,sentencepiece as spm
from transformers import AutoModelForSequenceClassification,get_linear_schedule_with_warmup
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1];OLD=ROOT/'experiments/stage4_retraining';SMALL=ROOT/'experiments/stage4_verification_v1'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(OLD))
from train_stage4 import batch,predict
from policy_stage4 import checkpoint_key
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(obj):
    torch.set_num_threads(4);random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True,warn_only=True)
    out=BASE/f'pilot_{obj}_42';out.mkdir(exist_ok=True)
    if (out/'DONE.json').exists():return
    if (out/'step_log.jsonl').exists():raise RuntimeError('Refusing overwrite partial training')
    freeze=load(BASE/'TRAIN_SUPPLEMENT_FREEZE.json');assert sha(BASE/'train_supplement_reviewed.jsonl')==freeze['sha256']
    tasks={t['task_id']:t for t in load(BASE/'review_tasks.json')};supp=rows(BASE/'train_supplement_reviewed.jsonl')
    assert all(tasks[r['task_id']]['split']=='train' for r in supp)
    assert all(r['y_support'] in [0,1] and r['y_relevance'] in [0,1] for r in supp)
    sp=spm.SentencePieceProcessor(model_file='/home/<user>/models/mdeberta-v3-base/spm.model')
    def encode(q,t):
        qid=sp.encode(q)[:160];tid=sp.encode(t)[:509-len(qid)];return [1]+qid+[2]+tid+[2]
    original=torch.load(OLD/'prepared.pt',map_location='cpu',weights_only=False)
    # Common replay preserves the original support balance and is shared by objectives.
    rr=[];rng=random.Random(20261012)
    for label in [0,1]:
        pool=[r for r in original['train'] if r['y_support']==label];rng.shuffle(pool);rr.extend(pool[:512])
    oldids={r['candidate_id'] for r in rr};assert len(oldids)==1024
    training=rr+[r|{'ids':encode(r['question'],r['text'])} for r in supp]
    small=torch.load(SMALL/'prepared_unlabeled.pt',map_location='cpu',weights_only=False)
    gold={r['candidate_id']:r for r in rows(SMALL/'candidates_reviewed.jsonl')}
    dev=[r|gold[r['candidate_id']] for r in small['test'] if r['evaluation_split']=='dev_retrospective']
    test=[r|gold[r['candidate_id']] for r in small['test'] if r['evaluation_split']!='dev_retrospective']
    assert not {r['task_id'] for r in training}&{r['task_id'] for r in dev+test}
    parent=OLD/f'{obj}_42/best_checkpoint.pt';check=torch.load(parent,map_location='cpu',weights_only=False)
    model=AutoModelForSequenceClassification.from_pretrained(check['config']['model'],num_labels=2,local_files_only=True,dtype=torch.float32)
    model.load_state_dict(check['model']);del check;model.cuda()
    label='y_support' if obj=='support' else 'y_relevance'
    config={'objective':obj,'seed':42,'mode':'continued_training_pilot','epochs':3,'lr':5e-6,'effective_batch':16,'micro':4,
        'replay_pairs':len(rr),'new_reviewed_pairs':len(supp),'supplement_sha256':freeze['sha256'],
        'parent_checkpoint_sha256':sha(parent),'dev_source':'stage4_frozen_AI_reviewed_8_tasks','test_small_retrospective':True,
        'expanded_114_task_test_labels':'pending; scores exported, no gold metrics yet','replay_candidate_ids':sorted(oldids)}
    save(out/'config.json',config)
    optim=torch.optim.AdamW(model.parameters(),lr=5e-6,weight_decay=.01);steps=3*math.ceil(len(training)/16)
    schedule=get_linear_schedule_with_warmup(optim,max(1,int(.1*steps)),steps);step=0;best=(-1,);start=time.monotonic()
    for epoch in range(1,4):
        ix=list(range(len(training)));rng.shuffle(ix);model.train()
        for k in range(0,len(ix),16):
            indices=ix[k:k+16];optim.zero_grad(set_to_none=True);lossvalue=0
            for j in range(0,len(indices),4):
                rs=[training[i] for i in indices[j:j+4]];logits=model(**batch(rs)).logits
                loss=torch.nn.functional.cross_entropy(logits,torch.tensor([r[label] for r in rs],device='cuda'),reduction='sum')/len(indices)
                assert torch.isfinite(loss);loss.backward();lossvalue+=float(loss.detach())
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(grad)
            optim.step();schedule.step();step+=1
            if step==1 or step%10==0:
                event={'epoch':epoch,'step':step,'loss':lossvalue,'grad_norm':float(grad),'elapsed_sec':time.monotonic()-start,'peak_memory_mib':torch.cuda.max_memory_allocated()/2**20}
                with (out/'step_log.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(event)+'\n')
                print(json.dumps(event),flush=True)
        scores=predict(model,dev,4);key,policy,grid=checkpoint_key(dev,scores,small['meta'])
        save(out/f'epoch_{epoch}_dev.json',{'key':key,'policy':policy});np.save(out/f'epoch_{epoch}_dev_scores.npy',scores)
        if key>best:
            best=key;torch.save({'model':model.state_dict(),'config':config,'epoch':epoch,'step':step},out/'best_checkpoint.pt')
    del optim,schedule;torch.cuda.empty_cache();check=torch.load(out/'best_checkpoint.pt',map_location='cpu',weights_only=False);model.load_state_dict(check['model']);selected_epoch=check['epoch'];del check
    np.save(out/'small_dev_scores.npy',predict(model,dev,4));np.save(out/'small_test_scores.npy',predict(model,test,4))
    save(out/'TRAINING_DONE.json',{'steps':step,'selected_epoch':selected_epoch,'seconds':time.monotonic()-start})
    pool=rows(BASE/'candidates_unlabeled.jsonl');evalrows=[r for r in pool if r['split']!='train']
    # Score both input variants; no model scores are provided to the annotation pack.
    for variant in ['fragment','context']:
        data=[r|{'ids':encode(r['question'],r['text'] if variant=='fragment' else r['context_text'])} for r in evalrows]
        np.save(out/f'expanded_{variant}_scores.npy',predict(model,data,4))
        with (out/f'expanded_{variant}_index.jsonl').open('w',encoding='utf-8') as f:
            for r in evalrows:f.write(json.dumps({'candidate_id':r['candidate_id'],'task_id':r['task_id'],'split':r['split'],'rank':r['retrieval']['rank']})+'\n')
    save(out/'DONE.json',{'status':'PILOT_TRAINED_AND_EXPANDED_SCORES_EXPORTED','expanded_pairs':len(evalrows),'expanded_gold_evaluation_complete':False})
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--objective',choices=['support','relevance'],required=True);a=ap.parse_args()
    try:main(a.objective)
    except Exception:
        out=BASE/f'pilot_{a.objective}_42';out.mkdir(exist_ok=True);save(out/'FAILED.json',{'traceback':traceback.format_exc()});raise
