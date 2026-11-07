"""Freeze scope/points and exact visible model inputs without model scores."""
import json,re,hashlib,collections
from pathlib import Path
import sentencepiece as spm
B=Path(__file__).resolve().parent
def save(n,v):(B/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    if (B/'EXPANDED_REVIEW_INPUT_FREEZE.json').exists():return
    tasks=json.loads((B/'review_tasks.json').read_text(encoding='utf-8'));pool=[json.loads(s) for s in (B/'candidates_unlabeled.jsonl').read_text(encoding='utf-8').splitlines()]
    sp=spm.SentencePieceProcessor(model_file='/home/<user>/models/mdeberta-v3-base/spm.model')
    selected=[];excluded=[]
    for t in tasks:
        if t['split']=='train':continue
        if t['source']=='hallu' and (re.search('首选|应选用|应首洗|外搽|哪种药|药物|方剂',t['question']) or any(re.search('汤$|丸$|散$|油膏$|小茴香|吴茱萸|肉桂|附子',p['text']) for p in t['provisional_points'])):
            # A formula-mentioned mechanism question remains in scope, but formula
            # selection is excluded by its question and answer, before scores.
            if not re.search('病机|原因|证候',t['question']):
                excluded.append({'task_id':t['task_id'],'reason':'drug/formula selection, outside diagnostic explanation/treatment principles'});continue
        points=[]
        for p in t['provisional_points']:
            text=p['text'].strip();typ=p.get('type',t['question_type'])
            if t['question_type']=='principle' and t['source']=='med':
                text=re.sub(r'^(治疗原则是|治疗原则为|治则治法为|治则为|治疗应|治疗以)','',text)
                pieces=[x.strip('。；; ') for x in re.split('[，,；;]',text) if x.strip('。；; ')]
            elif re.fullmatch('[\u4e00-\u9fff、和及]{2,30}',text) and ('、' in text or '和' in text):
                pieces=[x for x in re.split('、|和',text) if len(x)>=2]
            else:pieces=[text]
            for v in pieces:
                points.append({'id':f'a{len(points)+1}','text':v,'type':typ,'parent_id':p['id'],'required':True,'authority':'inherited_source_points_deterministic_atomization_pending_semantic_audit'})
        if not points:excluded.append({'task_id':t['task_id'],'reason':'no source necessary points'});continue
        selected.append({k:t[k] for k in ['task_id','source','split','question_type','question']}|{'necessary_points':points,'scope_review':'score_blind_rule_and_source_review'})
    tids={t['task_id'] for t in selected};byid={t['task_id']:t for t in selected}
    save('expanded_review_tasks.json',selected);save('EXPANDED_SCOPE_EXCLUSIONS.json',excluded)
    out=[]
    for r in pool:
        if r['task_id'] not in tids:continue
        qids=sp.encode(r['question'])[:160]
        for variant in ['fragment','context']:
            evidence=r['text'] if variant=='fragment' else r['context_text'];eids=sp.encode(evidence)[:509-len(qids)]
            out.append({'candidate_id':r['candidate_id'],'task_id':r['task_id'],'variant':variant,'rank':r['retrieval']['rank'],
                'question':sp.decode(qids),'evidence':sp.decode(eids),'input_ids_sha256':hashlib.sha256(json.dumps([1]+qids+[2]+eids+[2]).encode()).hexdigest(),
                'input_truncated':len(sp.encode(r['question']))>160 or len(sp.encode(evidence))>len(eids),
                'source':byid[r['task_id']]['source'],'split':r['split'],'question_type':byid[r['task_id']]['question_type'],'necessary_points':byid[r['task_id']]['necessary_points'],
                'y_support':None,'y_relevance':None,'covers':None})
    with (B/'expanded_review_inputs.jsonl').open('w',encoding='utf-8') as f:
        for r in out:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    save('EXPANDED_REVIEW_INPUT_FREEZE.json',{'task_count':len(selected),'original_eval_tasks':114,'excluded_tasks':len(excluded),'input_pairs_with_variants':len(out),
        'split_counts':dict(collections.Counter(t['split'] for t in selected)),'source_type_counts':dict(collections.Counter(t['source']+'/'+t['question_type'] for t in selected)),
        'input_sha256':sha(B/'expanded_review_inputs.jsonl'),'tasks_sha256':sha(B/'expanded_review_tasks.json'),'prediction_scores_read':False,
        'reference_inserted':False,'labels_complete':False,'clinical_expert_gold':False})
    print('REVIEW_INPUTS_FROZEN',len(selected),len(out),flush=True)
if __name__=='__main__':main()
