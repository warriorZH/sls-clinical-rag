"""Score unchanged parent models on identical expanded inputs after pilot exits."""
import os,sys,time,json,traceback,argparse
from pathlib import Path
import numpy as np,torch,sentencepiece as spm
from transformers import AutoModelForSequenceClassification
B=Path(__file__).resolve().parent;ROOT=B.parents[1];OLD=ROOT/'experiments/stage4_retraining'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(OLD))
from train_stage4 import predict
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def main(obj):
    while not (B/f'pilot_{obj}_42/DONE.json').exists():
        if (B/f'pilot_{obj}_42/FAILED.json').exists():raise RuntimeError('Pilot failed')
        time.sleep(30)
    torch.set_num_threads(4);out=B/f'parent_{obj}_42';out.mkdir(exist_ok=True)
    if (out/'DONE.json').exists():return
    check=torch.load(OLD/f'{obj}_42/best_checkpoint.pt',map_location='cpu',weights_only=False)
    model=AutoModelForSequenceClassification.from_pretrained(check['config']['model'],num_labels=2,local_files_only=True,dtype=torch.float32)
    model.load_state_dict(check['model']);del check;model.cuda()
    sp=spm.SentencePieceProcessor(model_file='/home/<user>/models/mdeberta-v3-base/spm.model')
    rr=[json.loads(s) for s in (B/'candidates_unlabeled.jsonl').read_text(encoding='utf-8').splitlines()];rr=[r for r in rr if r['split']!='train']
    for variant in ['fragment','context']:
        data=[]
        for r in rr:
            q=sp.encode(r['question'])[:160];t=sp.encode(r['text'] if variant=='fragment' else r['context_text'])[:509-len(q)]
            data.append(r|{'ids':[1]+q+[2]+t+[2]})
        np.save(out/f'expanded_{variant}_scores.npy',predict(model,data,4))
    save(out/'DONE.json',{'status':'PARENT_SCORES_EXPORTED','pairs':len(rr),'input_order':'same expanded index as pilot','expanded_gold_complete':False})
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--objective',required=True,choices=['support','relevance']);a=ap.parse_args()
    try:main(a.objective)
    except Exception:save(B/f'PARENT_{a.objective}_FAILED.json',{'traceback':traceback.format_exc()});raise
