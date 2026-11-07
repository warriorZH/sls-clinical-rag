"""Finite evaluation queue: frozen AI labels, dev-only policies, no answer insertion."""
import os,sys,json,time,hashlib,collections,csv,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1];TRAIN=ROOT/'experiments/stage4_retraining'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(TRAIN))
import numpy as np,torch
from evaluate_v2 import classify,per_question,summarize,write_json
from policy_stage4 import policies
from transformers import AutoModelForSequenceClassification

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')

def predict_cpu(model,rr):
    model.eval();result=np.empty(len(rr));order=sorted(range(len(rr)),key=lambda i:len(rr[i]['ids']))
    with torch.no_grad():
        for start in range(0,len(order),4):
            indices=order[start:start+4];n=max(len(rr[i]['ids']) for i in indices)
            ids=torch.zeros((len(indices),n),dtype=torch.long);mask=torch.zeros_like(ids)
            for j,i in enumerate(indices):ids[j,:len(rr[i]['ids'])]=torch.tensor(rr[i]['ids']);mask[j,:len(rr[i]['ids'])]=1
            logits=model(input_ids=ids,attention_mask=mask).logits
            assert torch.isfinite(logits).all();result[indices]=logits.softmax(-1)[:,1].numpy()
    return result

def score_existing(obj,seed,rr):
    original=TRAIN/f'{obj}_{seed}'
    predictions={r['candidate_id']:r for r in rows(original/'actual_retrieval_predictions.jsonl')}
    scores=np.empty(len(rr));missing=[]
    for i,row in enumerate(rr):
        old=predictions.get(row['candidate_id'])
        if old is not None:
            assert old['question']==row['question'] and old['text']==row['text']
            scores[i]=old['model_score']
        else:missing.append(i)
    if missing:
        checkpoint=torch.load(original/'best_checkpoint.pt',map_location='cpu',weights_only=False)
        model=AutoModelForSequenceClassification.from_pretrained(checkpoint['config']['model'],num_labels=2,local_files_only=True,dtype=torch.float32)
        model.load_state_dict(checkpoint['model']);del checkpoint
        scores[missing]=predict_cpu(model,[rr[i] for i in missing]);del model
    assert np.isfinite(scores).all()
    return scores,{'reused_original_GPU_predictions':len(rr)-len(missing),'new_sanitized_query_pairs_CPU_FP32':len(missing),
                   'training_data_sha256':load(original/'config.json')['dataset_sha256'],'weights_retrained':False}

def evaluate_scores(name,objective,seed,rr,meta,scores,provenance):
    dest=BASE/'runs'/name;dest.mkdir(parents=True,exist_ok=True)
    dev_indices=[i for i,r in enumerate(rr) if r['evaluation_split']=='dev_retrospective']
    test_indices=[i for i,r in enumerate(rr) if r['evaluation_split']!='dev_retrospective']
    dev=[rr[i] for i in dev_indices];test=[rr[i] for i in test_indices];ds=scores[dev_indices];ts=scores[test_indices]
    np.save(dest/'scores_all.npy',scores)
    with (dest/'raw_predictions.jsonl').open('w',encoding='utf-8') as f:
        for r,s in zip(rr,scores):f.write(json.dumps({k:v for k,v in r.items() if k!='ids'}|{'score':float(s)},ensure_ascii=False)+'\n')
    ownlabel='y_relevance' if objective=='relevance' else 'y_support'
    own_dev=[r|{'y_support':r[ownlabel]} for r in dev]
    own_tau=max([i/20 for i in range(1,20)],key=lambda t:(classify(own_dev,ds,t)['f1'],-abs(t-.5)))
    own={label:{'dev':classify([r|{'y_support':r[label]} for r in dev],ds,own_tau),
                'test':classify([r|{'y_support':r[label]} for r in test],ts,own_tau)} for label in ['y_support','y_relevance']}
    save(dest/'own_and_cross_target.json',{'own_target':ownlabel,'tau_dev_own_f1':own_tau,'metrics':own})
    chosen={};metric={};qreports={};candidate_map={r['candidate_id']:r for r in test}
    def add(key,tau,m,mode='threshold_top_m'):
        qs=per_question(test,ts,meta,tau,m,mode)
        for q in qs:
            q['selected_condition_conflicts']=sum(candidate_map[i]['condition_conflict_flag'] for i in q['selected_ids'])
            q['selected_partial_without_point']=sum(candidate_map[i]['partial_without_complete_point'] for i in q['selected_ids'])
        qreports[key]=qs
        metric[key]={}
        for subset in ['all','med','hallu']:
            qq=[q for q in qs if subset=='all' or q['source']==subset];mm=summarize(qq)
            n=sum(q['selected_count'] for q in qq)
            mm['selected_conflict_flag_rate']=sum(q['selected_condition_conflicts'] for q in qq)/n if n else 0
            mm['selected_partial_without_point_rate']=sum(q['selected_partial_without_point'] for q in qq)/n if n else 0
            metric[key][subset]=mm
        with (dest/f'{key}_questions.jsonl').open('w',encoding='utf-8') as f:
            for q in qs:f.write(json.dumps(q,ensure_ascii=False)+'\n')
    for budget in [.05,.1,.15]:
        policy,grid=policies(dev,ds,meta,budget);chosen[str(budget)]=policy;save(dest/f'dev_grid_{budget}.json',grid)
        if policy:add(f'actual_dev_budget_{budget}',policy['threshold'],policy['m'])
    for m in [1,2,4,6]:add(f'top_{m}',0,m,'top_m')
    add('actual_dev_own_f1_m4',own_tau,4)
    if objective in ['support','relevance']:
        reference=load(TRAIN/f'{objective}_{seed}/selected_policies.json').get('0.1')
        if reference:add('reference_dev_budget_transfer',reference['threshold'],reference['m'])
    ceiling=summarize(per_question(test,ts,meta,0,100000,'top_m'))
    save(dest/'selected_actual_dev_policies.json',chosen);save(dest/'set_metrics.json',metric)
    save(dest/'pool_ceiling.json',ceiling)
    save(dest/'DONE.json',{'status':'COMPLETE','objective':objective,'seed':seed,'provenance':provenance,
                            'label_sha256':sha(BASE/'candidates_reviewed.jsonl'),'test_tasks':8,'dev_tasks':8,
                            'human_expert_gold':False,'test_used_to_choose_policy':False})
    print('EVALUATED',name,'TOP2',metric['top_2'],flush=True)

def lexical_baselines(rr,meta):
    from baselines_stage4 import tokenize,features
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    original=torch.load(TRAIN/'prepared.pt',map_location='cpu',weights_only=False)
    corpus=list(dict.fromkeys(r['text'] for r in original['train_full']))
    tf=TfidfVectorizer(analyzer=tokenize,dtype=np.float32).fit(corpus)
    q=tf.transform([r['question'] for r in rr]);e=tf.transform([r['text'] for r in rr])
    scores={'tfidf':np.asarray(q.multiply(e).sum(axis=1)).ravel()}
    model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,random_state=42))
    model.fit(features(original['train']),[r['y_support'] for r in original['train']])
    scores['length_overlap']=model.predict_proba(features(rr))[:,1]
    scores['random']=np.array([int(hashlib.sha256(('42:'+r['candidate_id']).encode()).hexdigest()[:13],16)/16**13 for r in rr])
    devbm=[r['retrieval']['bm25_score'] for r in rr if r['evaluation_split']=='dev_retrospective']
    scale=max(1,float(np.quantile(devbm,.95)))
    scores['bm25_retriever']=1-np.exp(-np.array([r['retrieval']['bm25_score'] for r in rr])/scale)
    del original,model,tf
    for name,scoreset in scores.items():
        if not (BASE/'runs'/name/'DONE.json').exists():evaluate_scores(name,'baseline',42,rr,meta,scoreset,{'fit':'frozen reference train only; BM25 corpus unchanged','bm25_score_scale_actual_dev_unlabeled_q95':scale,'test_labels_used_to_fit':False})

def bootstrap(a,b,key):
    bm={q['task_id']:q for q in b};groups=collections.defaultdict(list)
    for q in a:groups[q['group']].append(q[key]-bm[q['task_id']][key])
    values=list(groups.values());tot=np.array([sum(v) for v in values]);n=np.array([len(v) for v in values]);rng=np.random.default_rng(20261011)
    samples=[]
    for _ in range(5000):
        ix=rng.integers(len(values),size=len(values));samples.append(tot[ix].sum()/n[ix].sum())
    return {'difference':float(tot.sum()/n.sum()),'ci95':np.quantile(samples,[.025,.975]).tolist(),'groups':len(values),'replicates':5000,'sampling_seed':20261011,'model_seed_variation_not_estimated_here':True}

def report():
    records=[];boot={};complete=[]
    runroot=BASE/'runs'
    for dest in sorted(runroot.iterdir()):
        if not (dest/'DONE.json').exists():continue
        info=load(dest/'DONE.json');complete.append(dest.name);metrics=load(dest/'set_metrics.json')
        for policy,subsets in metrics.items():
            for source,mm in subsets.items():records.append({'run':dest.name,'objective':info['objective'],'seed':info['seed'],'policy':policy,'source':source,**mm})
    for seed in [42,2026,3407]:
        sa=runroot/f'support_{seed}/top_2_questions.jsonl';sb=runroot/f'relevance_{seed}/top_2_questions.jsonl'
        if sa.exists() and sb.exists():
            aa,bb=rows(sa),rows(sb)
            for source in ['all','med','hallu']:
                a=[q for q in aa if source=='all' or q['source']==source];b=[q for q in bb if source=='all' or q['source']==source]
                boot[f'{seed}/{source}']={key:bootstrap(a,b,key) for key in ['coverage','full_coverage']}
    if records:
        with (BASE/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    aggregates=[]
    for obj in ['support','relevance']:
        for source in ['all','med','hallu']:
            subset=[r for r in records if r['objective']==obj and r['policy']=='top_2' and r['source']==source]
            aggregates.append({'objective':obj,'source':source,'completed_seed_count':len(subset),
                               'seeds':[r['seed'] for r in subset],
                               'coverage_mean':float(np.mean([r['coverage'] for r in subset])) if subset else None,
                               'coverage_seed_sd':float(np.std([r['coverage'] for r in subset],ddof=1)) if len(subset)>1 else None})
    save(BASE/'paired_bootstrap.json',boot);save(BASE/'three_seed_summary.json',aggregates)
    save(BASE/'evaluation_status.json',{'completed':complete,'clinical_expert_gold':False,'label_version':'AI_semantic_scope_review_v1'})
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4));means=[];names=[];errors=[]
    for item in aggregates:
        if item['coverage_mean'] is None:continue
        names.append(item['objective']+'/'+item['source']);means.append(item['coverage_mean']*100);errors.append((item['coverage_seed_sd'] or 0)*100)
    if names:
        ax.bar(range(len(names)),means,yerr=errors,capsize=3);ax.set_xticks(range(len(names)),names,rotation=20,ha='right');ax.set_ylabel('Point coverage (%)');ax.set_ylim(0,100);ax.set_title('Query-only pool / AI-reviewed labels; bars = available seeds')
        fig.tight_layout();fig.savefig(BASE/'actual_pool_coverage.png',dpi=180)
    plt.close(fig)
    text='# 真实检索核验进度\n\n'+f'已完成：{", ".join(complete)}。\n\n'
    text+='本页由后台有限队列更新。训练数据未修改，新增核验仅作评价；共8道dev和8道test，每题20条真实候选。test含6个未用医案及2个回顾性Hallu；标签为单人AI文本核验，不能称真人独立医学金标。\n\n'
    text+='主要策略在真实dev上按既定集合预算选择；无可行策略明确为空。固定Top-2作共同诊断，同时报告池上限、部分帮助与条件冲突。候选不含后补参考答案；原始问题缺失临床条件和教材标题丢失仍是限制。\n\n'
    text+='|目标|来源|已完成种子数|Top-2覆盖率均值|种子标准差|\n|---|---|---:|---:|---:|\n'
    for r in aggregates:
        if r['coverage_mean'] is not None:text+=f"|{r['objective']}|{r['source']}|{r['completed_seed_count']}|{r['coverage_mean']*100:.2f}%|{r['coverage_seed_sd']*100:.2f}%|\n" if r['coverage_seed_sd'] is not None else f"|{r['objective']}|{r['source']}|{r['completed_seed_count']}|{r['coverage_mean']*100:.2f}%|尚未估计|\n"
    (BASE/'RESULTS.md').write_text(text,encoding='utf-8')

def main():
    torch.set_num_threads(2)
    frozen=load(BASE/'LABEL_FREEZE.json');assert sha(BASE/'candidates_reviewed.jsonl')==frozen['label_file_sha256']
    labels={r['candidate_id']:r for r in rows(BASE/'candidates_reviewed.jsonl')}
    payload=torch.load(BASE/'prepared_unlabeled.pt',map_location='cpu',weights_only=False)
    rr=[]
    for r in payload['test']:
        label=labels[r['candidate_id']];assert r['question']==label['question'] and r['text']==label['text']
        rr.append(label|{'ids':r['ids']})
    meta=payload['meta'];del payload
    (BASE/'runs').mkdir(exist_ok=True)
    lexical_baselines(rr,meta);report()
    pending=[(obj,s) for s in [42,2026,3407] for obj in ['support','relevance']]
    while pending:
        for obj,seed in list(pending):
            dest=BASE/'runs'/f'{obj}_{seed}'
            if (dest/'DONE.json').exists():pending.remove((obj,seed));continue
            if not (TRAIN/f'{obj}_{seed}/DONE.json').exists():continue
            scores,provenance=score_existing(obj,seed,rr);evaluate_scores(f'{obj}_{seed}',obj,seed,rr,meta,scores,provenance)
            pending.remove((obj,seed));report()
        if pending:
            state=load(TRAIN/'resume_status.json')
            if state['status']=='FAILED':raise RuntimeError('Training queue failed; evaluations already completed preserved')
            save(BASE/'WAITING_FOR_TRAINING.json',{'pending':[f'{o}_{s}' for o,s in pending],'training_status':state['status']})
            time.sleep(60)
    assert sha(BASE/'candidates_reviewed.jsonl')==frozen['label_file_sha256']
    report();save(BASE/'DONE.json',{'status':'ALL_SIX_RUNS_EVALUATED_ON_FROZEN_QUERY_ONLY_POOL','label_sha256':frozen['label_file_sha256'],'human_expert_gold':False,'original_training_data_unchanged':True})
    print('ALL_SIX_ACTUAL_EVALUATIONS_COMPLETE',flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        (BASE/'EVALUATION_FAILED.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
