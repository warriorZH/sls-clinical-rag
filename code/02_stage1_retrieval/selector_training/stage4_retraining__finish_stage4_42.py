"""Complete post-training exports/inference; never retrain seed 42."""
import argparse, json, hashlib, time, shutil
from pathlib import Path
import torch, numpy as np
from transformers import AutoModelForSequenceClassification
from train_stage4 import BASE, predict
from policy_stage4 import export_run
from actual_predictions_stage4 import export

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main(args):
    assert args.seed == 42
    torch.set_num_threads(4)
    out = BASE/f'{args.objective}_42'
    if (out/'DONE.json').exists(): return
    files = [out/'best_dev_scores.npy',out/'test_scores.npy']
    before = {p.name:sha(p) for p in files}
    failure = json.loads((out/'FAILED.json').read_text(encoding='utf-8'))
    assert "'ascii' codec" in failure['error']
    for name in ['FAILED.json','own_target_metrics.json','selected_policies.json']:
        source=out/name; backup=out/(name+'.before_resume')
        if source.exists() and not backup.exists(): shutil.copy2(source,backup)
    data = torch.load(BASE/'prepared.pt',weights_only=False,map_location='cpu')
    dev,test = [np.load(p) for p in files]
    assert len(dev)==len(data['dev']) and len(test)==len(data['test'])
    export_run(data,dev,test,out,args.objective)
    assert json.loads((out/'own_target_metrics.json.before_resume').read_text(encoding='utf-8')) == json.loads((out/'own_target_metrics.json').read_text(encoding='utf-8'))
    del data
    checkpoint = torch.load(out/'best_checkpoint.pt',weights_only=False,map_location='cpu')
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint['config']['model'], num_labels=2, local_files_only=True,dtype=torch.float32)
    model.load_state_dict(checkpoint['model']); epoch=checkpoint['epoch']; checkpoint_step=checkpoint['step']; del checkpoint
    model=model.cuda(); start=time.monotonic()
    export(model,predict,4,out)
    assert before == {p.name:sha(p) for p in files}
    step_rows=[json.loads(s) for s in (out/'step_log.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    done={'best_epoch':epoch,'checkpoint_step':checkpoint_step,'total_steps':step_rows[-1]['step'],
          'completion_mode':'POST_TRAINING_RESUME_NO_RETRAINING','original_failure_preserved':True,
          'original_score_hashes':before,'original_scores_unchanged':True,
          'post_training_resume_seconds':time.monotonic()-start,'actual_retrieval_pairs':9000}
    (out/'DONE.json').write_text(json.dumps(done,indent=2),encoding='utf-8')
    print(json.dumps(done),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--objective',choices=['support','relevance'],required=True);parser.add_argument('--seed',type=int,default=42)
    main(parser.parse_args())
