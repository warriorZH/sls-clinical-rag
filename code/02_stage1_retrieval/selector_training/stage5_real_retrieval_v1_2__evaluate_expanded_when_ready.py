"""Finite provisional-label comparison; unknown labels retain explicit bounds.

Main tables are model-assisted diagnostics, not an audited expert benchmark.
No test-driven threshold tuning, no training, no favourable-case deletion.
"""
import json,time,hashlib,collections,csv,traceback
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parent
GROUPS={}
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def qmetric(rs,required,scores=None,m=100,tau=0):
    if scores is None:chosen=rs
    else:
        chosen=sorted(zip(rs,scores),key=lambda x:(-x[1],x[0]['candidate_id']))[:m];chosen=[r for r,s in chosen if s>=tau]
    cov=set().union(*(set(r['covers']) for r in chosen if not r['annotation_uncertain'])) if chosen else set()
    possible=set(cov)
    if any(r['annotation_uncertain'] for r in chosen):possible|=required
    n=len(chosen);neg=sum(not bool(r['covers']) and not r['annotation_uncertain'] for r in chosen);unknown=sum(bool(r['annotation_uncertain']) for r in chosen)
    return {'coverage_lower':len(cov&required)/len(required),'coverage_upper':len(possible&required)/len(required),
        'full_coverage_lower':int(required<=cov),'full_coverage_upper':int(required<=possible),
        'selected_count':n,'negative_count_confirmed_model_label':neg,'unknown_selected':unknown,'negative_budget_upper_count':neg+unknown,
        'empty':int(n==0),'conflict_selected':sum(r['condition_conflict'] for r in chosen),
        'partial_selected':sum(r['partial_support'] for r in chosen),'selected_ids':[r['candidate_id'] for r in chosen]}
def summarize(qs):
    n=sum(q['selected_count'] for q in qs)
    return {'questions':len(qs),**{k:float(np.mean([q[k] for q in qs])) for k in ['coverage_lower','coverage_upper','full_coverage_lower','full_coverage_upper','empty','selected_count']},
        'negative_rate_lower':sum(q['negative_count_confirmed_model_label'] for q in qs)/n if n else 0,
        'negative_rate_upper':sum(q['negative_budget_upper_count'] for q in qs)/n if n else 0,
        'uncertain_selected_rate':sum(q['unknown_selected'] for q in qs)/n if n else 0,
        'conflict_selected_rate':sum(q['conflict_selected'] for q in qs)/n if n else 0,
        'partial_selected_rate':sum(q['partial_selected'] for q in qs)/n if n else 0}
def boot(a,b):
    aa={q['task_id']:q for q in a};bb={q['task_id']:q for q in b};assert set(aa)==set(bb)
    groups=collections.defaultdict(list)
    for t in sorted(aa):groups[GROUPS.get(t,t)].append(aa[t]['coverage_lower']-bb[t]['coverage_lower'])
    sums=np.array([sum(v) for v in groups.values()]);counts=np.array([len(v) for v in groups.values()]);rng=np.random.default_rng(20261012)
    ix=rng.integers(len(groups),size=(5000,len(groups)));values=sums[ix].sum(1)/counts[ix].sum(1)
    return {'difference':float(sums.sum()/counts.sum()),'ci95':np.quantile(values,[.025,.975]).tolist(),'questions':len(aa),'groups':len(groups),'replicates':5000,
        'interpretation':'conditional on provisional labels and seed42 models; excludes annotator uncertainty'}
def main():
    while not all((B/f'ANNOTATION_DONE_{i}.json').exists() for i in [0,1]):
        if any((B/f'ANNOTATION_FAILED_{i}.json').exists() for i in [0,1]):raise RuntimeError('Annotation failed; no forced labels')
        time.sleep(30)
    inputs=rows(B/'expanded_review_inputs.jsonl');im={(r['candidate_id'],r['variant']):r for r in inputs}
    labels=rows(B/'expanded_labels_shard_0.jsonl')+rows(B/'expanded_labels_shard_1.jsonl');lm={}
    for r in labels:
        key=r['candidate_id'],r['variant'];assert key not in lm and key in im
        assert r['input_ids_sha256']==im[key]['input_ids_sha256'];assert r['evidence']==im[key]['evidence'];lm[key]=r
    assert set(im)==set(lm)
    merged=B/'expanded_labels_provisional.jsonl'
    with merged.open('w',encoding='utf-8') as f:
        for key in sorted(lm):f.write(json.dumps(lm[key],ensure_ascii=False)+'\n')
    freeze={'status':'MODEL_ASSISTED_LABELS_FROZEN_AUDIT_PENDING','sha256':h(merged),'pairs_with_variants':len(labels),
        'uncertain_pairs':sum(r['annotation_uncertain'] for r in labels),'clinical_expert_gold':False,'test_score_used_for_labeling':False}
    save(B/'EXPANDED_LABEL_FREEZE_PROVISIONAL.json',freeze)
    # Deterministic, score-blind audit pack covers folds/sources/variants, not only errors.
    strata=collections.defaultdict(list)
    for r in labels:strata[(r['split'],r['source'],r['variant'])].append(r)
    audit=[]
    for key,rs in sorted(strata.items()):
        rs.sort(key=lambda r:hashlib.sha256(('audit20261012:'+r['candidate_id']+r['variant']).encode()).hexdigest());audit.extend(rs[:8])
    save(B/'EXPANDED_BLIND_AUDIT_PACK.json',audit)
    tasks={t['task_id']:t for t in load(B/'expanded_review_tasks.json')}
    import torch
    old=torch.load(B.parents[1]/'experiments/stage4_retraining/prepared.pt',map_location='cpu',weights_only=False)
    GROUPS.update({tid:m['group'] for tid,m in old['meta'].items() if 'group' in m});del old
    original=rows(B/'pilot_support_42/expanded_fragment_index.jsonl');index={r['candidate_id']:i for i,r in enumerate(original)}
    assert len(index)==11400
    results=[];reports={};upper_reports={};policylocks={}
    for variant in ['fragment','context']:
        for k in [20,50,100]:
            groups=collections.defaultdict(list)
            for r in labels:
                if r['variant']==variant and r['rank']<=k:groups[r['task_id']].append(r)
            required={tid:{p['id'] for p in tasks[tid]['necessary_points']} for tid in groups}
            ceilings=[qmetric(rs,required[tid])|{'task_id':tid,'source':tasks[tid]['source'],'question_type':tasks[tid]['question_type'],'split':tasks[tid]['split']} for tid,rs in groups.items()]
            for subset in ['all','med','hallu','diagnosis','principle']:
                qs=[q for q in ceilings if q['split']=='test' and (subset=='all' or q['source']==subset or q['question_type']==subset)]
                if qs:results.append({'method':'pool_ceiling','variant':variant,'k':k,'subset':subset,'policy':'all_candidates',**summarize(qs)})
            upper_reports[variant+'/'+str(k)]=[q for q in ceilings if q['split']=='test']
            for method in ['bm25','parent_support','parent_relevance','pilot_support','pilot_relevance']:
                if method=='bm25':sc=lambda r:-r['rank']
                else:
                    version,obj=method.split('_');file=B/f'{version}_{obj}_42/expanded_{variant}_scores.npy';scores=np.load(file);assert len(scores)==len(index) and np.isfinite(scores).all()
                    sc=lambda r:float(scores[index[r['candidate_id']]])
                def qs_for(m,tau,fold):
                    return [qmetric(rs,required[tid],[sc(r) for r in rs],m,tau)|{'task_id':tid,'source':tasks[tid]['source'],'question_type':tasks[tid]['question_type'],'split':fold}
                        for tid,rs in groups.items() if tasks[tid]['split']==fold]
                policy=None
                if method!='bm25':
                    feasible=[]
                    for m in [1,2,4,6]:
                        for tau in np.arange(.05,1,.05):
                            mm=summarize(qs_for(m,float(tau),'dev'))
                            if mm['negative_rate_upper']<=.1 and mm['empty']<=.02 and mm['selected_count']<=4:
                                feasible.append((mm['coverage_lower'],mm['full_coverage_lower'],-mm['negative_rate_upper'],float(tau),m))
                    if feasible:
                        best=max(feasible);policy={'threshold':best[-2],'m':best[-1]}
                policylocks[method+'/'+variant+'/'+str(k)]=policy
                selections={'top_2':qs_for(2,-10000,'test')}
                if policy:selections['dev_10pct']=qs_for(policy['m'],policy['threshold'],'test')
                for pname,qs in selections.items():
                    reports[method+'/'+variant+'/'+str(k)+'/'+pname]=qs
                    for subset in ['all','med','hallu','diagnosis','principle']:
                        qq=[q for q in qs if subset=='all' or q['source']==subset or q['question_type']==subset]
                        if qq:results.append({'method':method,'variant':variant,'k':k,'subset':subset,'policy':pname,**summarize(qq)})
    save(B/'EXPANDED_PROVISIONAL_RESULTS.json',results);save(B/'EXPANDED_QUESTION_RESULTS.json',reports);save(B/'EXPANDED_POOL_QUESTION_CEILINGS.json',upper_reports);save(B/'EXPANDED_DEV_POLICIES.json',policylocks)
    bootstraps={}
    for k in [20,50,100]:
        for variant in ['fragment','context']:
            bootstraps[f'support_minus_relevance/{variant}/{k}']=boot(reports[f'parent_support/{variant}/{k}/top_2'],reports[f'parent_relevance/{variant}/{k}/top_2'])
        for method in ['pool_ceiling','parent_support','parent_relevance']:
            a=upper_reports['context/'+str(k)] if method=='pool_ceiling' else reports[f'{method}/context/{k}/top_2']
            b=upper_reports['fragment/'+str(k)] if method=='pool_ceiling' else reports[f'{method}/fragment/{k}/top_2']
            bootstraps[f'context_minus_fragment/{method}/{k}']=boot(a,b)
    save(B/'EXPANDED_PROVISIONAL_BOOTSTRAP.json',bootstraps)
    for variant in ['fragment','context']:
        for method in ['pool_ceiling','parent_support','parent_relevance']:
            for k in [50,100]:
                a=upper_reports[variant+'/'+str(k)] if method=='pool_ceiling' else reports[f'{method}/{variant}/{k}/top_2']
                b=upper_reports[variant+'/20'] if method=='pool_ceiling' else reports[f'{method}/{variant}/20/top_2']
                bootstraps[f'k{k}_minus_k20/{method}/{variant}']=boot(a,b)
    save(B/'EXPANDED_PROVISIONAL_BOOTSTRAP.json',bootstraps)
    with (B/'EXPANDED_PROVISIONAL_RESULTS.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    text='# 扩展核验：模型辅助标注的暂定比较\n\n逐条辅助标注已完成并冻结，但尚未完成AI/人工抽样裁决，不能称完整专家复核或最终金标。标注模型独立于DeBERTa，模型分数未用于标签；不确定标注保留覆盖和负例上下界，未删除失败问题。以下lower是给定非不确定模型标签的保守覆盖估计，不是临床正确率。\n\n|方法|输入|候选数|test题数|覆盖lower|覆盖upper|\n|---|---|---:|---:|---:|---:|\n'
    for r in results:
        if r['subset']=='all' and r['policy'] in ['all_candidates','top_2'] and r['method'] in ['pool_ceiling','parent_support','parent_relevance']:
            text+=f"|{r['method']}|{r['variant']}|{r['k']}|{r['questions']}|{r['coverage_lower']*100:.2f}%|{r['coverage_upper']*100:.2f}%|\n"
    text+='\n先完成EXPANDED_BLIND_AUDIT_PACK.json的抽样核对，再判断是否追加训练。上下文提高标签覆盖也可能来自模型辅助标签偏差；bootstrap不包含标注模型误差。当前不启动新的DeBERTa训练。\n'
    (B/'EXPANDED_PROVISIONAL_REPORT.md').write_text(text,encoding='utf-8')
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,method,title in [(axes[0],'pool_ceiling','Candidate-pool point coverage'),(axes[1],'parent_support','Support-42 Top-2 point coverage')]:
        for variant in ['fragment','context']:
            selected=[r for r in results if r['method']==method and r['variant']==variant and r['subset']=='all' and r['policy'] in ['all_candidates','top_2']]
            selected.sort(key=lambda r:r['k']);xs=[r['k'] for r in selected];lower=[r['coverage_lower']*100 for r in selected];upper=[r['coverage_upper']*100 for r in selected]
            ax.plot(xs,lower,marker='o',label=variant+' lower');ax.fill_between(xs,lower,upper,alpha=.12)
        ax.set_xlabel('Candidates K');ax.set_ylabel('Coverage (%)');ax.set_ylim(0,100);ax.set_title(title+'\nprovisional model labels; shading = uncertainty bounds');ax.legend()
    fig.tight_layout();fig.savefig(B/'EXPANDED_PROVISIONAL_COVERAGE.png',dpi=180);plt.close(fig)
    save(B/'EXPANDED_PROVISIONAL_EVALUATION_DONE.json',{'status':'PROVISIONAL_LABEL_EVALUATION_COMPLETE_BLIND_AUDIT_PENDING','clinical_expert_gold':False,'labels_sha256':h(merged)})
if __name__=='__main__':
    try:main()
    except Exception:save(B/'EXPANDED_EVALUATION_FAILED.json',{'traceback':traceback.format_exc()});raise
