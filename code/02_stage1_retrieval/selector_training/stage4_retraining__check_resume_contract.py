"""Check repaired UTF-8 exports and frozen inputs without touching trained predictions."""
import json,hashlib,tempfile
from pathlib import Path
import numpy as np
from policy_stage4 import export_run
BASE=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''):h.update(chunk)
    return h.hexdigest()
state=json.loads((BASE/'resume_status.json').read_text(encoding='utf-8'))
for name,h in state['immutable_hashes'].items():assert sha(BASE/name)==h,name
dummy=[{'task_id':'编码测试','candidate_id':'编码测试__1','question':'病情说明','text':'辨证与治则材料',
        'group':'编码测试','source':'med','y_support':1,'y_relevance':1,'covers':['a1','a2'],'constructed':False},
       {'task_id':'编码测试','candidate_id':'编码测试__2','question':'病情说明','text':'不支持该病情的材料',
        'group':'编码测试','source':'med','y_support':0,'y_relevance':1,'covers':[],'constructed':False}]
data={'dev':dummy,'test':dummy,'meta':{'编码测试':{'group':'编码测试','source':'med','required':['a1','a2']}}}
with tempfile.TemporaryDirectory(prefix='utf8_check_',dir=BASE) as folder:
    export_run(data,np.array([.9,.1]),np.array([.9,.1]),Path(folder),'support')
    recovered=[json.loads(s) for s in (Path(folder)/'test_raw_predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    assert recovered[0]['question']=='病情说明'
original_scores={}
for objective in ['support','relevance']:
    out=BASE/f'{objective}_42';done=json.loads((out/'DONE.json').read_text(encoding='utf-8'))
    assert done['completion_mode']=='POST_TRAINING_RESUME_NO_RETRAINING'
    for name,h in done['original_score_hashes'].items():assert sha(out/name)==h
    predictions=[json.loads(s) for s in (out/'actual_retrieval_predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(predictions)==9000 and all(p['y_support'] is None for p in predictions)
    scores=np.load(out/'actual_retrieval_scores.npy');assert len(scores)==9000 and np.isfinite(scores).all()
    original_scores[objective]=done['original_score_hashes']
result={'status':'PASS','ASCII_locale_explicit_UTF8_exports':'PASS','frozen_inputs_hashes':state['immutable_hashes'],
        'seed42_original_scores_unchanged':original_scores,'seed42_actual_retrieval_pairs_per_model':9000,
        'seed42_retrained':False,'synthetic_IO_probe_not_experimental_data':True}
(BASE/'RESUME_CONTRACT_CHECK.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({'status':'PASS','frozen_data_unchanged':True,'seed42_not_retrained':True,'seed42_actual_pairs':9000}),flush=True)
